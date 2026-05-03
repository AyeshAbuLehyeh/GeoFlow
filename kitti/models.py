"""Model definitions for GeoFlow on KITTI."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

try:
    from .positional_encoding import PositionalEncoding2D
except ImportError:  # pragma: no cover - standalone execution fallback
    from positional_encoding import PositionalEncoding2D


class GeoFlowKITTI(nn.Module):
    """GeoFlow model for KITTI cross-view localization.

    Parameters
    ----------
    d_model:
        Latent feature width after projection.
    backbone:
        Timm backbone name.
    sat_size:
        Satellite crop size used to size positional encodings.
    predict_orientation:
        If ``True``, enables an additional orientation head.
    """

    def __init__(self, d_model: int = 128, backbone: str = "efficientnet_b0", sat_size: int = 512, predict_orientation: bool = False, pretrained: bool = True):
        super().__init__()
        self.predict_orientation = predict_orientation

        print(f"Initializing GeoFlowKITTI with backbone={backbone}, orientation_head={predict_orientation}.")

        try:
            self.sat_extractor = create_model(backbone, pretrained=pretrained, features_only=True)
            self.grd_extractor = create_model(backbone, pretrained=pretrained, features_only=True)
        except Exception as exc:  # pragma: no cover - runtime fallback path
            if pretrained:
                print(f"Warning: pretrained weights for '{backbone}' are unavailable ({exc}). Falling back to randomly initialized backbones.")
                self.sat_extractor = create_model(backbone, pretrained=False, features_only=True)
                self.grd_extractor = create_model(backbone, pretrained=False, features_only=True)
            else:
                raise

        self.feature_channels = self.sat_extractor.feature_info.channels()[-1]
        self.sat_feature_size = max(1, sat_size // 32)
        self.grd_feature_h = 256 // 32
        self.grd_feature_w = 1024 // 32

        self.sat_proj = nn.Conv2d(self.feature_channels, d_model, kernel_size=1)
        self.grd_proj = nn.Conv2d(self.feature_channels, d_model, kernel_size=1)

        self.coord_proj = nn.Sequential(
            nn.Linear(2, 8),
            nn.LayerNorm(8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 16),
            nn.LayerNorm(16),
            nn.ReLU(inplace=True),
        )

        self.sat_pos_embed = PositionalEncoding2D(d_model, max_h=max(20, self.sat_feature_size + 4), max_w=max(20, self.sat_feature_size + 4))
        self.grd_pos_embed = PositionalEncoding2D(d_model, max_h=max(12, self.grd_feature_h + 4), max_w=max(36, self.grd_feature_w + 4))

        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, dropout=0.1, batch_first=True)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        decoder_input_size = d_model + 16
        self.decoder_r = nn.Sequential(
            nn.Linear(decoder_input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )
        self.decoder_theta = nn.Sequential(
            nn.Linear(decoder_input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

        if self.predict_orientation:
            self.decoder_orientation = nn.Sequential(
                nn.Linear(decoder_input_size, 64),
                nn.LayerNorm(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(64, 2),
            )

    def forward(self, sat: torch.Tensor, grd: torch.Tensor, coord: torch.Tensor):
        sat_feat_maps = self.sat_extractor(sat)
        grd_feat_maps = self.grd_extractor(grd)

        sat_feats = self.sat_proj(sat_feat_maps[-1])
        grd_feats = self.grd_proj(grd_feat_maps[-1])

        sat_feats_with_pos = self.sat_pos_embed(sat_feats)
        grd_feats_with_pos = self.grd_pos_embed(grd_feats)

        sat_tokens = sat_feats_with_pos.flatten(2).permute(0, 2, 1)
        grd_tokens = grd_feats_with_pos.flatten(2).permute(0, 2, 1)

        fused_tokens, _ = self.cross_attention(query=grd_tokens, key=sat_tokens, value=sat_tokens)
        fused_vec = self.global_pool(fused_tokens.permute(0, 2, 1)).squeeze(-1)

        coord_token = self.coord_proj(coord)
        decoder_input = torch.cat([fused_vec, coord_token], dim=1)

        pred_log_r = self.decoder_r(decoder_input)
        pred_mu_r = pred_log_r[:, 0:1]
        pred_raw_var = pred_log_r[:, 1:2]

        pred_theta = self.decoder_theta(decoder_input)
        pred_mu_vec = pred_theta[:, 0:2]
        pred_raw_kappa = pred_theta[:, 2:3]

        pred_var = F.softplus(pred_raw_var) + 1e-6
        pred_kappa = F.softplus(pred_raw_kappa) + 1e-6

        if self.predict_orientation:
            pred_orientation_raw = self.decoder_orientation(decoder_input)
            pred_orientation = F.normalize(pred_orientation_raw, p=2, dim=1)
            return pred_mu_r, pred_var, pred_mu_vec, pred_kappa, pred_orientation

        return pred_mu_r, pred_var, pred_mu_vec, pred_kappa

    @torch.no_grad()
    def extract_features(self, sat: torch.Tensor, grd: torch.Tensor):
        self.eval()
        sat_feat_maps = self.sat_extractor(sat)
        grd_feat_maps = self.grd_extractor(grd)

        sat_feats = self.sat_proj(sat_feat_maps[-1])
        grd_feats = self.grd_proj(grd_feat_maps[-1])

        sat_feats_with_pos = self.sat_pos_embed(sat_feats)
        grd_feats_with_pos = self.grd_pos_embed(grd_feats)

        sat_tokens = sat_feats_with_pos.flatten(2).permute(0, 2, 1)
        grd_tokens = grd_feats_with_pos.flatten(2).permute(0, 2, 1)
        fused_tokens, _ = self.cross_attention(query=grd_tokens, key=sat_tokens, value=sat_tokens)
        fused_vec = self.global_pool(fused_tokens.permute(0, 2, 1)).squeeze(-1)
        return fused_vec, None

    def decode(self, fused_vec: torch.Tensor, f_map_sat, coord: torch.Tensor):
        coord_token = self.coord_proj(coord)
        decoder_input = torch.cat([fused_vec, coord_token], dim=1)

        pred_log_r = self.decoder_r(decoder_input)
        pred_mu_r = pred_log_r[:, 0:1]
        pred_raw_var = pred_log_r[:, 1:2]

        pred_theta = self.decoder_theta(decoder_input)
        pred_mu_vec = pred_theta[:, 0:2]
        pred_raw_kappa = pred_theta[:, 2:3]

        pred_var = F.softplus(pred_raw_var) + 1e-6
        pred_kappa = F.softplus(pred_raw_kappa) + 1e-6

        if self.predict_orientation:
            pred_orientation_raw = self.decoder_orientation(decoder_input)
            pred_orientation = F.normalize(pred_orientation_raw, p=2, dim=1)
            return pred_mu_r, pred_var, pred_mu_vec, pred_kappa, pred_orientation

        return pred_mu_r, pred_var, pred_mu_vec, pred_kappa


class GeoFlowKITTIOrientation(GeoFlowKITTI):
    def __init__(self, d_model: int = 128, backbone: str = "efficientnet_b0", sat_size: int = 512, pretrained: bool = True):
        super().__init__(d_model=d_model, backbone=backbone, sat_size=sat_size, predict_orientation=True, pretrained=pretrained)

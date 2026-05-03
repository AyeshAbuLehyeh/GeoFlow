"""Backward-compatible wrapper for the refactored orientation-aware model."""

from .losses import OrientationLoss
from .models import GeoFlowKITTIOrientation as GeoFlowKITTI

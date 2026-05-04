# [CVPR'26] GeoFlow: Real-Time Fine-Grained Cross-View Geolocalization via Iterative Flow Prediction

[Project Page](https://ayeshabulehyeh.github.io/geoflow_page/) · [ArXiv](https://arxiv.org/abs/2603.21943) · [Paper](https://arxiv.org/pdf/2603.21943)

<p align="center">
  <img src="assets/geoflow_architecture.png" width="900" alt="GeoFlow architecture">
</p>

GeoFlow is a lightweight framework for fine-grained cross-view geolocalization. The method formulates localization as an iterative flow refinement problem: the model extracts shared visual context once and then refines a coordinate estimate through lightweight updates.

The current public release focuses on the **KITTI** codebase. The **VIGOR** release will follow in a subsequent update.

---

## Setup

### 1) Clone the repository

```bash
git clone https://github.com/AyeshAbuLehyeh/GeoFlow.git
cd GeoFlow
```

### 2) Create and activate the conda environment

```bash
conda create -n geoflow python=3.9 -y
conda activate geoflow
```

### 3) Install the requirements

```bash
pip install -r requirements.txt
```

---

## KITTI dataset preparation

Download and structure the KITTI data according to [HighlyAccurate](https://github.com/YujiaoShi/HighlyAccurate).

After downloading, update the dataset root and file-list paths to match your local setup. The evaluation scripts already expose these paths as arguments, and the defaults can also be adjusted in [kitti/data.py](kitti/data.py).

Expected list files:

- `train_files.txt`
- `test1_files.txt`
- `test2_files.txt`

If your dataset is stored in a different location, change:

- `--root-dir`
- `--test-file`
- and, if needed, the defaults in [kitti/data.py](kitti/data.py)

---

## Repository structure

- **kitti/** — KITTI implementation with datasets, models, losses, and training/evaluation scripts
- **vigor/** — VIGOR implementation (coming soon)

---

## Training

### 1) Non-orientation model

Uses the ground-truth (or externally provided) camera orientation and predicts only the translation correction between ground and satellite views. (Known orientation) 

```bash
python -m kitti.train \
  --train-list ./train_files.txt \
  --batch-size 128 \
  --epochs 200 \
  --backbone efficientnet_b0 \
  --d-model 128 \
  --use-augmentation
```

### 2) Orientation-aware model

Jointly predicts both translation and camera orientation, so it can localize without assuming orientation is known beforehand. (Unknown orientation) 

```bash
python -m kitti.train_orient \
  --train-list ./train_files.txt \
  --batch-size 128 \
  --epochs 200 \
  --backbone efficientnet_b0 \
  --d-model 128
```


## Evaluation

### Non-orientation model

```bash
python -m kitti.eval \
  --model-path checkpoints/<run_name>/best.pth \
  --test-set test2 \
  --batch-size 16 \
  --num-iterations 5 \
  --num-random-starts 10 \
  --root-dir /path/to/KITTI \
  --test-file /path/to/test2_files.txt
```

### Orientation-aware model

```bash
python -m kitti.test_orient \
  --checkpoint-path checkpoints/<run_name>/best.pth \
  --batch-size 16 \
  --root-dir /path/to/KITTI \
  --test-file /path/to/test2_files.txt
```

Evaluation outputs are written to `inference_outputs/` as both text and JSON summaries.

---

## Citation

If GeoFlow is useful for your research, please cite:

```bibtex
@article{abulehyeh2026geoflow,
      title  = {{GeoFlow}: Real-Time Fine-Grained Cross-View Geolocalization via Iterative Flow Prediction},
      author = {Abu Lehyeh, Ayesh and Zhang, Xiaohan and Arrabi, Ahmad and Sultani, Waqas and Chen, Chen and Wshah, Safwan},
      journal={arXiv preprint arXiv:2603.21943},
      year={2026}
}
```


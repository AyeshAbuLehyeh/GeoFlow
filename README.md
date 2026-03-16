# GeoFlow (Real-Time Fine-Grained Cross-View Geolocalization via Iterative Flow Prediction)
Official implementation of GeoFlow, a lightweight and efficient framework for fine-grained cross-view geolocalization
**Accepted at CVPR 2026**

<p align="center">
  <img src="assets/geoflow_architecture.png" width="900">
</p>

---

## Overview

Fine-grained cross-view geolocalization aims to determine the precise geographic location of a **ground-level image** by matching it with satellite imagery. Existing approaches often rely on computationally heavy matching or dense correlation mechanisms, resulting in high memory usage and slow inference.

**GeoFlow** introduces a lightweight and efficient alternative by formulating localization as an **iterative coordinate refinement problem**. Instead of predicting the final location in a single step, GeoFlow progressively improves a coordinate estimate through **iterative flow prediction**.

The model first extracts shared visual context between the ground and aerial views and then repeatedly refines candidate coordinates using lightweight regression modules. Since the expensive visual feature extraction is performed only once, the refinement iterations incur minimal additional computational cost.

This design enables GeoFlow to achieve a strong balance between **accuracy, computational efficiency, and real-time inference speed**, making it suitable for **resource-constrained deployment scenarios such as robotics, UAV navigation, and embedded vision systems**.

---

## Paper

**GeoFlow: Real-Time Fine-Grained Cross-View Geolocalization via Iterative Flow Prediction**  
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR 2026**)

Paper and project page links will be added soon.

---

## Code

The full training and evaluation code will be released soon.

The repository will include:

- Training pipeline
- Evaluation scripts
- Pretrained models
- Dataset loaders for cross-view geolocalization benchmarks

---

## Datasets

GeoFlow is evaluated on standard cross-view geolocalization benchmarks:

- **KITTI**
- **VIGOR**

Instructions for dataset preparation will be provided soon.

---

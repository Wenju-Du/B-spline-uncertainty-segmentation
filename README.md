# B-spline Activations as Intrinsic Uncertainty Estimators in Medical Image Segmentation

Official implementation of our MICCAI 2026 paper:

**B-spline Activations as Intrinsic Uncertainty Estimators in Medical Image Segmentation**

This repository contains the core implementation of the proposed UKAN model and the B-spline activation based uncertainty estimation framework for medical image segmentation.

## Overview

The proposed method uses B-spline activations inside a KAN-based segmentation network as intrinsic uncertainty estimators. The core model is implemented as `UKAN` in `archs.py`, with a residual KAN prediction head that supports both segmentation prediction and B-spline basis activation extraction for uncertainty analysis.

Main components:

- `archs.py`: proposed UKAN model and residual B-spline/KAN prediction head.
- `kan.py`: KAN linear layers and spline basis computation.
- `train.py`: training script for the proposed method. Use `--arch UKAN`.
- `val.py`: validation script for trained checkpoints.
- `eval_selective_prediction.py`: image-level selective prediction evaluation.
- `eval_uncertainty_methods.py`: pixel-level uncertainty evidence analysis.
- `viz_bspline_activation.py`: B-spline activation visualization.

`archs_base.py` and `archs_edl.py` are kept for compatibility with existing optional branches in the current training/validation code. The proposed method uses `--arch UKAN`.

## Environment

Create a conda environment and install dependencies:

```bash
conda create -n bspline_seg python=3.10
conda activate bspline_seg
pip install torch torchvision numpy pandas opencv-python albumentations scikit-learn scipy medpy matplotlib tqdm pyyaml tensorboardX timm
```

CUDA is recommended. The current scripts call `.cuda()` directly, so CPU-only execution may require minor modifications.

## Dataset Preparation

Please organize each dataset as follows:

```text
inputs/
  dataset_name/
    images/
      case001.png
      case002.png
      ...
    masks/
      0/
        case001.png
        case002.png
        ...
```

Notes:

- Datasets are not included due to license restrictions.
- Images are expected to be `.png` files.
- Binary masks are expected under `masks/0/`.
- For most datasets, mask names should match image names, e.g. `case001.png`.
- For `dataset=busi`, the code expects masks named as `<image_id>_mask.png`.
- The train/validation split is generated internally with an 80/20 split using `--dataseed`.

## Training

Train the proposed UKAN model:

```bash
python train.py \
  --arch UKAN \
  --dataset dataset_name \
  --data_dir inputs \
  --output_dir outputs \
  --name dataset_UKAN \
  --epochs 200 \
  --batch_size 8 \
  --input_w 256 \
  --input_h 256 \
  --input_list 128,160,256 \
  --lr 1e-4 \
  --kan_lr 1e-2
```

Training outputs are saved to:

```text
outputs/
  dataset_UKAN/
    config.yml
    log.csv
    model.pth
    vis/
```

## Validation

Validate a trained model:

```bash
python val.py --name dataset_UKAN --output_dir outputs
```

The validation script loads:

```text
outputs/dataset_UKAN/config.yml
outputs/dataset_UKAN/model.pth
```

## Uncertainty Evaluation

Image-level selective prediction:

```bash
python eval_selective_prediction.py \
  --name dataset_UKAN \
  --output_dir outputs
```

Pixel-level uncertainty evidence analysis:

```bash
python eval_uncertainty_methods.py \
  --name dataset_UKAN \
  --output_dir outputs \
  --primary_method entropy
```

B-spline activation visualization:

```bash
python viz_bspline_activation.py \
  --name dataset_UKAN \
  --output_dir outputs
```

Optional visualization arguments:

```bash
--n_images 20
--fig2_image_idx 0
```

## Minimal Reproduction Workflow

```bash
python train.py --arch UKAN --dataset dataset_name --data_dir inputs --name dataset_UKAN
python val.py --name dataset_UKAN --output_dir outputs
python eval_selective_prediction.py --name dataset_UKAN --output_dir outputs
python eval_uncertainty_methods.py --name dataset_UKAN --output_dir outputs
python viz_bspline_activation.py --name dataset_UKAN --output_dir outputs
```

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{du2026bspline,
  title={B-spline Activations as Intrinsic Uncertainty Estimators in Medical Image Segmentation},
  author={Du, Wenju and others},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI},
  year={2026}
}
```

## Contact

For questions, please contact Wenju Du.

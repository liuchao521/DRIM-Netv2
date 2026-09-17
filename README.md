# DRIM-Net++

Official PyTorch implementation of **DRIM-Net++: Dual-Representation Inconsistency Modeling with Hyperspherical Evidence Learning for Generalizable Deepfake Detection**.

DRIM-Net++ combines:

- **DRCF**: Dual-Representation Contrast Fusion for multi-scale structural-textural inconsistency estimation.
- **IFP**: Inconsistency-Guided Frequency Perception for disagreement-conditioned frequency enhancement.
- **HIEL**: Hyperspherical Inconsistency Evidence Learning for cross-sample organization of normalized inconsistency descriptors.

The repository also retains the conference-version DRIM-Net model for reference.

## Repository layout

```text
.
|-- train.py                       # DRIM-Net++ training with HIEL
|-- evaluate.py                    # checkpoint evaluation 
|-- network/
|   |-- MainNet.py                 # DRIM-Net wrapper
|   |-- MainNet_IHR_V2.py          # DRIM-Net++ wrapper
|   |-- DINOV3/
|   |   |-- DRCF.py
|   |   |-- IFP.py
|   |   `-- dinov3_adapter.py
|   `-- models/efficientnet/
|       |-- DRIMNET.py             # DRIM-Net
|       `-- DRIMNET_IHR_V2.py      # DRIM-Net++ and HIEL
|-- data/splits/example.txt
|-- checkpoints/
`-- outputs/
```

## Installation

Create a Python environment and install PyTorch for the CUDA version on your machine. Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

`mmcv` may require a wheel matched to the installed PyTorch and CUDA versions. Follow the official OpenMMLab installation instructions if a source build is triggered.

## Pretrained backbone

The DINOv3 checkpoint is not redistributed in this repository. Obtain the official `dinov3_vits16plus` checkpoint under its original license and expose it through:

```bash
export DINOV3_WEIGHTS=/absolute/path/to/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
```

EfficientNet-B4 ImageNet weights are loaded by the bundled EfficientNet implementation on first use.

## Dataset lists

Training and evaluation use plain-text sample lists. Each line contains an image path and a binary label separated by whitespace:

```text
/absolute/path/to/real/000001.png 0
/absolute/path/to/fake/000001.png 1
```

Label `0` denotes real and label `1` denotes fake. Dataset files and face crops are not redistributed. See `data/README.md` for the expected preprocessing boundary.

## Training

```bash
python train.py \
  --train_txt_path data/splits/train.txt \
  --valid_txt_path data/splits/val.txt \
  --out_path outputs/drimnetpp \
  --device cuda:0 \
  --batch_size 64 \
  --epoches 40 \
  --lr 5e-4 \
  --dino_lr 3e-4 \
  --inc_cls_weight 0.1 \
  --align_weight 0.05 \
  --uniform_weight 0.05 \
  --reg_target inc
```

The default setting applies alignment and uniformity to the inconsistency embedding (`--reg_target inc`). The `fused` and `both` options are retained for controlled comparisons.

## Evaluation

Evaluate one checkpoint or every checkpoint in a directory:

```bash
python evaluate.py \
  --model_dir checkpoints/best_auc.pkl \
  --test_txt_path data/splits/test.txt \
  --device cuda:0 \
  --select_by auc \
  --results_file outputs/evaluation.txt
```

The script reports AUC, AP, EER, ACC, BACC, and threshold-scanned ACC/BACC. Use a validation-set threshold for final test reporting when the protocol requires a fixed threshold; threshold scanning on a test set is diagnostic only.

## Reported cross-dataset results

The following values are copied from the manuscript tables; checkpoints are not bundled in this repository.

| Method | FF++-C23 ACC/AUC | DFDC ACC/AUC | DFR ACC/AUC | Celeb-DFv2 ACC/AUC | WDF ACC/AUC | AVG ACC/AUC |
|---|---:|---:|---:|---:|---:|---:|
| DRIM-Net | 94.80 / 98.14 | 66.15 / 74.72 | 72.08 / 92.03 | 75.40 / 81.95 | 69.97 / 77.96 | 70.90 / 81.66 |
| DRIM-Net++ | 92.84 / 97.34 | 67.88 / 76.74 | 76.92 / 92.42 | 77.98 / 84.79 | 72.04 / 81.51 | 73.71 / 83.86 |

The average includes the source-domain FF++-C23 column, matching the manuscript table.

## Citation

The DRIM-Net++ citation will be updated after publication. See `CITATION.cff` for the current manuscript metadata.

## Third-party code and license

This release contains code derived from EfficientNet and DINOv3. Their original notices and license terms continue to apply. See `THIRD_PARTY_NOTICES.md`. A project-wide open-source license has not yet been selected; until one is added, no additional license is granted for the authors' code.


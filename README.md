# Refined Batch Normalization for Domain Adaptation

This project is an implementation of the paper "Reducing Divergence in Batch Normalization for Domain Adaptation."

## Project Structure
```
project/
├── configs/                # Configuration Files
│   └── default.yaml       # Default Configuration
├── data/                  # Dataset Directory
│   ├── office31/         # Office-31Dataset
│   ├── imageclef/        # ImageCLEF-DADataset
│   ├── officehome/       # Office-HomeDataset
│   └── visda2017/        # VisDA-2017Dataset
├── datasets/             # Dataset Loading Module
│   ├── office31.py      # Office-31Dataset类
│   ├── imageclef.py     # ImageCLEF-DADataset类
│   ├── officehome.py    # Office-HomeDataset类
│   ├── visda2017.py     # VisDA-2017Dataset类
│   └── transforms.py     # Data Transformation
├── models/              # Model Definitions
│   ├── backbone.py     # Feature Extractor (ResNet)
│   ├── classifier.py   # Classifier Head
│   ├── discriminator.py# Domain Discriminator
│   ├── rbn.py         # Refined Batch Normalization
│   └── cdan.py        # CDAN模型
├── utils/              # Utility Functions
│   ├── logger.py      # Logging Utilities
│   ├── losses.py      # Loss Functions
│   └── utils.py       # 通用Utility Functions
├── output/             # Output Directory
│   ├── checkpoints/   # Model Checkpoints
│   └── logs/          # Training Logs
├── scripts/           # Run Scripts
│   ├── train.sh      # Training Script
│   └── test.sh       # Testing Script
├── train.py          # Main Training Program
├── test.py           # Main Testing Program
├── requirements.txt  # Project Dependencies
└── README.md        # Project Documentation
```

## Environment Requirements

- Python >= 3.7
- PyTorch >= 1.7.0
- CUDA >= 10.1 (for GPU training)


## Dataset Preparation

### Office-31
Directory Structure:
```
data/office31/
├── amazon/
│   └── images/
│       ├── back_pack/
│       ├── bike/
│       └── ...
├── dslr/
│   └── images/
│       ├── back_pack/
│       ├── bike/
│       └── ...
└── webcam/
    └── images/
        ├── back_pack/
        ├── bike/
        └── ...
```

### ImageCLEF-DA
Directory Structure:
```
data/imageclef/
├── i/
│   ├── class1/
│   ├── class2/
│   └── ...
├── p/
│   ├── class1/
│   ├── class2/
│   └── ...
└── c/
    ├── class1/
    ├── class2/
    └── ...
```

### Office-Home
Directory Structure:
```
data/officehome/
├── Art/
│   ├── Alarm_Clock/
│   ├── Backpack/
│   └── ...
├── Clipart/
├── Product/
└── Real_World/
```

### VisDA-2017
Directory Structure:
```
data/visda2017/
├── train/
│   ├── aeroplane/
│   ├── bicycle/
│   └── ...
└── validation/
    ├── aeroplane/
    ├── bicycle/
    └── ...
```

## Usage

### 1. Configuration

Modify `configs/default.yaml` in the configuration parameters:
```yaml
data:
  dataset_type: 'office31'
  num_classes: 31
model:
  use_rbn: true
  replace_layer: 3
```

### 2. Training

```bash
bash scripts/train.sh
```

Direct execution:
```bash
python train.py \
    --source_data data/office31/amazon \
    --target_data data/office31/webcam \
    --dataset_type office31 \
    --num_classes 31 \
    --use_rbn \
    --replace_layer 3 \
    --epochs 50
```

### 3. Testing
```bash
bash scripts/test.sh
```
Direct execution:
```bash
python test.py \
    --test_data data/office31/webcam \
    --dataset_type office31 \
    --num_classes 31 \
    --model_path output/model_final.pth \
    --use_rbn \
    --replace_layer 3
```

## Main Features
- Replace the later-stage BN layers in the network with RBN (Refined Batch Normalization) to reduce cumulative estimation bias
- Based on the CDAN framework for domain adaptation.
- Use ResNet-50 as a feature extractor.
- Support multiple commonly used domain adaptation datasets.
Provide a complete training and evaluation process.

## Experimental Results

Accuracy on the Office-31Dataset：

| Method    | A → W | D → W | W → D | A → D | D → A | W → A | Avg  |
|-----------|-------|-------|-------|-------|-------|-------|------|
| CDAN      | 94.1  | 98.6  | 100.0 | 92.9  | 70.1  | 69.3  | 87.7 |
| CDAN+RBN  | 95.9  | 99.1  | 100.0 | 95.7  | 76.1 | 74.5  | 90.2 |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


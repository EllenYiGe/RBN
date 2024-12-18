# Refined Batch Normalization for Domain Adaptation

本项目是对论文 "Reducing Divergence in Batch Normalization for Domain Adaptation" 的实现。

## 项目结构
```
project/
├── configs/                # 配置文件
│   └── default.yaml       # 默认配置
├── data/                  # 数据集目录
│   ├── office31/         # Office-31数据集
│   ├── imageclef/        # ImageCLEF-DA数据集
│   ├── officehome/       # Office-Home数据集
│   └── visda2017/        # VisDA-2017数据集
├── datasets/             # 数据集加载模块
│   ├── office31.py      # Office-31数据集类
│   ├── imageclef.py     # ImageCLEF-DA数据集类
│   ├── officehome.py    # Office-Home数据集类
│   ├── visda2017.py     # VisDA-2017数据集类
│   └── transforms.py     # 数据变换
├── models/              # 模型定义
│   ├── backbone.py     # 特征提取器（ResNet）
│   ├── classifier.py   # 分类器头部
│   ├── discriminator.py# 域判别器
│   ├── rbn.py         # Refined Batch Normalization
│   └── cdan.py        # CDAN模型
├── utils/              # 工具函数
│   ├── logger.py      # 日志工具
│   ├── losses.py      # 损失函数
│   └── utils.py       # 通用工具函数
├── output/             # 输出目录
│   ├── checkpoints/   # 模型检查点
│   └── logs/          # 训练日志
├── scripts/           # 运行脚本
│   ├── train.sh      # 训练脚本
│   └── test.sh       # 测试脚本
├── train.py          # 训练主程序
├── test.py           # 测试主程序
├── requirements.txt  # 项目依赖
└── README.md        # 项目文档
```

## 环境要求

- Python >= 3.7
- PyTorch >= 1.7.0
- CUDA >= 10.1 (对于GPU训练)


## 数据集准备

### Office-31
目录结构：
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
目录结构：
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
目录结构：
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
目录结构：
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

## 使用方法

### 1. 配置

修改 `configs/default.yaml` 中的配置参数：
```yaml
data:
  dataset_type: 'office31'
  num_classes: 31
model:
  use_rbn: true
  replace_layer: 3
```

### 2. 训练

使用预设脚本：
```bash
bash scripts/train.sh
```

或直接运行：
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

### 3. 测试

使用预设脚本：
```bash
bash scripts/test.sh
```

或直接运行：
```bash
python test.py \
    --test_data data/office31/webcam \
    --dataset_type office31 \
    --num_classes 31 \
    --model_path output/model_final.pth \
    --use_rbn \
    --replace_layer 3
```

## 主要特性

- 使用RBN(Refined Batch Normalization)替换网络后期的BN层，减少估计偏移累积
- 基于CDAN框架进行域适配
- 使用ResNet-50作为特征提取器
- 支持多个常用域适应数据集
- 提供完整的训练和评估流程

## 实验结果

在Office-31数据集上的准确率：

| Method    | A → W | D → W | W → D | A → D | D → A | W → A | Avg  |
|-----------|-------|-------|-------|-------|-------|-------|------|
| CDAN      | 93.1  | 98.2  | 100.0 | 89.8  | 70.1  | 68.0  | 86.6 |
| CDAN+RBN  | 94.5  | 98.6  | 100.0 | 92.1  | 71.4  | 69.8  | 87.7 |

## 引用

如果您使用了本代码，请引用原论文：
```bibtex
@inproceedings{rbn2024,
  title={Reducing Divergence in Batch Normalization for Domain Adaptation},
  author={Author Names},
  booktitle={Proceedings of AAAI},
  year={2024}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。


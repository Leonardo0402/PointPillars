# PointPillars 简介

本项目是一个基于 PyTorch 的 [PointPillars](https://arxiv.org/abs/1812.05784) 实验性实现，用于在 KITTI 点云数据集上进行 3D 目标检测。代码主要面向初学者，结构清晰，便于理解。

## 主要模块
- **config.py**：集中定义网格范围、锚框尺寸等参数。
- **reader.py**：读取 KITTI 的点云与标签文件。
- **processor.py**：将原始点云划分为 Pillars，并生成训练所需的真值。
- **network.py**：包含 Pillar Feature Net、2D 卷积 Backbone 与检测头。
- **lossfunction.py**：实现焦点损失等训练损失。
- **train.py**：示例训练脚本。
- **inference.py**：推理脚本（仍在完善中）。

## 环境准备
1. 安装依赖
   ```bash
   pip install torch numpy tensorflow scikit-learn pybind11
   ```
2. 编译 C++ 扩展
   ```bash
   c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) point_pillars.cpp \
      -o point_pillars$(python3-config --extension-suffix)
   ```

## 数据准备
将 KITTI 数据集下载并解压到 `config.py` 中 `Parameters.kitti_path` 指定的位置（默认 `/home/neil/disk/kitti`），目录结构示例如下：
```
kitti
├── training
│   ├── velodyne    # 点云文件 (.bin)
│   └── label_2     # 标注文件 (.txt)
└── testing
    └── velodyne
```

## 运行示例
```bash
python train.py       # 开始训练
# 目前 inference.py 仍在开发中，可参考 train.py 中的模型调用方式
```

## 参考
- [KITTI 3D Object Detection 数据集简介](https://medium.com/test-ttile/kitti-3d-object-detection-dataset-d78a762b5a4)
- 原始论文：Alex H. Lang et al., *PointPillars: Fast Encoders for Object Detection from Point Clouds*, CVPR 2019.


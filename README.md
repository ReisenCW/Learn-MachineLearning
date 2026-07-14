# 机器学习自学仓库 Learn ML

## Reference
1. 机器学习: https://developers.google.cn/machine-learning/crash-course
2. Pytorch: FreeCodeCamp
   1. Youtube: https://www.youtube.com/watch?v=V_xro1bcAuA&t=4s (only first 4 chapters)
   2. Book version: https://www.learnpytorch.io/

## 目录结构

### 机器学习

1. `linear/linear.ipynb` 线性回归模型的实现, 使用numpy, pandas等基本库, 实现了包括 梯度下降, loss的计算, 指标的计算等
2. `classification/classification.ipynb` 分类模型的实现, 包括逻辑回归, softmax等


### PyTorch
1. `00_pytorch_fundamentals`: Pytorch基础知识, 包括张量操作、GPU加速、PyTorch核心语法等
2. `01_pytorch_workflow`: PyTorch工作流入门, 以线性回归为例, 涵盖数据准备、模型构建、训练循环、评估指标与模型保存
3. `02_neural_network_classification`: 神经网络分类, 包括二分类与多分类、激活函数（ReLU/Sigmoid/Softmax）、损失函数（BCE/CrossEntropy）
4. `03_pytorch_computer_vision`: 计算机视觉入门, 使用CNN进行图像分类（FashionMNIST数据集）
5. `04_pytorch_custom_data`: 自定义数据集, 包括DataLoader使用、数据增强（torchvision.transforms）、自定义Dataset类
6. `05_pytorch_going_modular`: 模块化, 演示将notebook代码拆分为可复用的Python脚本（data_setup, engine, model_builder, utils, train）
7. `06_pytorch_transfer_learning`: 迁移学习, 使用预训练模型（EfficientNet）进行特征提取(冻结+训练classifier)
8. `07_pytorch_experiment_tracking`: 实验追踪, 使用TensorBoard记录训练指标、对比不同实验、可视化模型图
9. `08_pytorch_paper_replicating`: 论文复现, 从头复现Vision Transformer（ViT）论文
10. `ex_pytorch_2_feature`: PyTorch2特性演示, 包括device的新增操作, `torch.compile`等

## 环境
1. 使用Python 3.12
2. PyTorch 2.11.0+cu128
3. uv管理环境


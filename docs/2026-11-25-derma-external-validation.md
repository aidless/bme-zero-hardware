# DermaMNIST 跨数据集外部验证（2026-11-25）

**结论**：同一套 ResNet18 迁移学习协议在 7 类 DermaMNIST 上取得 test accuracy 0.7751 [0.7601, 0.7900]、macro-AUC 0.9343 [0.9254, 0.9429]；ECE(15 bins)=0.0539、Brier=0.3200。协议可迁移到更难的多类任务，但与 PneumoniaMNIST 的二分类 AUC 0.9721 [0.9569, 0.9833] 不同口径，不能写成“超过 SOTA”。

## 任务与口径

- 目的：回应“换个数据集还行吗”，用第二个 MedMNIST 数据集做外部验证，并补概率校准。
- 数据来源：`from medmnist import DermaMNIST`，`split=train/val/test`，`download=True`，`root=./data`，官方划分。
- 类别映射 (来自 `train.info['label']`):
  - 0 = actinic keratoses and intraepithelial carcinoma
  - 1 = basal cell carcinoma
  - 2 = benign keratosis-like lesions
  - 3 = dermatofibroma
  - 4 = melanoma
  - 5 = melanocytic nevi
  - 6 = vascular lesions
- 图像通道：medmnist 3.0.2 的 DermaMNIST 返回原生 3 通道 RGB；脚本用条件 `Lambda(lambda x: x.repeat(3,1,1) if x.shape[0]==1 else x)`，灰度图 repeat、原生 RGB 保留。直接无条件 repeat 会得到 9 通道并在 `conv1` 报错，已实测并修正。

## 数据分布

| split | 样本数 | 类别0 | 类别1 | 类别2 | 类别3 | 类别4 | 类别5 | 类别6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 7007 | 228 | 359 | 769 | 80 | 779 | 4693 | 99 |
| val | 1003 | 33 | 52 | 110 | 12 | 111 | 671 | 14 |
| test | 2005 | 66 | 103 | 220 | 23 | 223 | 1341 | 29 |

train/val/test 合计 10015；类别 5（melanocytic nevi）占约 66.9%，类别 3 与 6 为小类。

## 训练协议与选模

- 权重：`~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`，ImageNet1K；torchvision `resnet18(weights=None)` 后加载该 state_dict，替换 `fc` 为 7 类。
- 预处理：`Resize(224) → ToTensor → 3 通道条件 Lambda`；第二阶段加 `RandomAffine(degrees=8, translate=(0.08,0.08))`。`ToTensor` 已归一化到 [0,1]，未再除以 255。
- 第一阶段冻结全部 backbone，仅训 head，4 epoch，Adam lr=1e-3。
- 第二阶段解冻 layer3/layer4，分层 LR：head 1e-3、layer3/layer4 各 1e-4；最多 10 epoch，patience=4；CrossEntropyLoss；batch=64；AMP（cuda）。
- 选模：每个 epoch 在 val 上算 one-vs-rest macro-AUC，取全局最佳；最终选中 `finetune epoch 2`，val accuracy=0.7737、val macro-AUC=0.9385。
- 训练 epoch 轨迹（val accuracy / val macro-AUC）：head 1–4 = 0.6989/0.8461、0.7059/0.8958、0.7178/0.9051、0.7308/0.9104；finetune 1–6 = 0.7617/0.9379、0.7737/0.9385、0.7627/0.9372、0.7348/0.9334、0.7468/0.9233、0.7557/0.9276。epoch 6 早停。

## Test 结果

| 指标 | 实测值 | 95% CI（1000×分层 bootstrap） |
|---|---:|---:|
| accuracy | 0.7751 | [0.7601, 0.7900] |
| macro-AUC（one-vs-rest） | 0.9343 | [0.9254, 0.9429] |
| ECE（15 bins） | 0.0539 | — |
| multiclass Brier | 0.3200 | — |

per-class AUC：0=0.9554、1=0.9622、2=0.8913、3=0.9375、4=0.8797、5=0.9267、6=0.9874。**最弱类为 4 melanoma（0.8797）**，次弱为 2 benign keratosis-like lesions（0.8913）。混淆矩阵显示 class 2 与 class 4 大量被判成占多数的 class 5，符合长尾分布下的少数类困难。

## 校准与对标

- ECE=0.0539，略高于常见的 0.05 参考线；reliability 图显示中高置信区间经验准确率低于置信度（过度自信）。未做温度缩放，因为任务是外部验证与校准度量，不是校准方法比较。
- 官方 [MedMNIST 主页](https://medmnist.com) 的 Benchmarking 表（2026-11-24 抓取）列出 DermaMNIST ResNet-18 (224)：AUC 0.920、ACC 0.754。我们的 0.9343/0.7751 数值高于该表，但**不写“超过 SOTA”**：本实验是 ImageNet 预训练迁移 + 分层微调；官方表为 benchmark 基线，页面未给出可对齐的预训练/训练细节；且本实验把 medmnist 默认 28×28 图 resize 到 224，而 MedMNIST+ 的 224 数据由原始 3×600×450 缩放生成（见 [ MedMNIST+ 说明](https://github.com/MedMNIST/MedMNIST/blob/main/on_medmnist_plus.md)）。两者输入来源与协议不同，只能做参考，不能同口径宣称优于。

## 跨数据集结论

PneumoniaMNIST 是二分类 AUC=0.9721 [0.9569,0.9833]；DermaMNIST 是 7 类 macro-AUC=0.9343 [0.9254,0.9429]。AUC 定义、类别数、难度和类别平衡都不同，两个区间不可直接相减。可下结论：同一迁移学习流程在更难的多类皮肤病任务上仍达到较高判别力（macro-AUC≈0.934）与中等准确度（≈0.775），具备跨数据集可迁移性；最弱类与过度自信是后续改进点。

## 环境与复现

- GPU：NVIDIA GeForce RTX 3060 Laptop 6GB；device=cuda。
- 版本：python 3.12.3、torch 2.14.1+cu130、torchvision 0.29.2+cu132、medmnist 3.0.2、scikit-learn 1.9.1、numpy 2.5.2、matplotlib 3.11.1。
- seed=42；cudnn deterministic=True、benchmark=False；bootstrap 用 `np.random.default_rng(42)` 分层重采样。
- 运行命令：`python3 -u bench_imaging_derma.py 2>&1 | tee data/derma_rigorous_run.log`。
- 端到端运行 292.5s。

## 产物

- 脚本：`bench_imaging_derma.py`
- 指标：`data/derma_rigorous_metrics.json`
- stdout 日志：`data/derma_rigorous_run.log`
- 可靠性图：`figures/derma_reliability.png`
- 混淆矩阵：`figures/derma_confusion.png`

`verifier:` JSON 中 macro-AUC 等于 7 个 per-class AUC 均值；混淆矩阵各行和逐类等于官方 test 类别计数；accuracy 与 macro-AUC 点估计均落在各自 bootstrap 95% CI 内；日志逐字包含 test 与校准数字；两张图均非空且尺寸分别为 1080×1000、1240×1080。四项断言已通过。

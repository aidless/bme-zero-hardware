# BME 零硬件实测 数据集下载

本项目所有数据来自公开来源。脚本本身会按需自动下载一部分；下面列出需要手动准备的数据及其来源。

## 1. PneumoniaMNIST（① 医学影像）
`mini_project.py` / `mini_project_cnn.py` / `adv_resnet.py` 通过 `medmnist` 自动下载：
```bash
pip install medmnist
# 首次运行脚本时自动下载，无需手动操作
```

## 2. ResNet18 ImageNet 预训练权重（adv_resnet.py）
torchvision 首次运行会自动下载到 `~/.cache/torch/hub/checkpoints/`。若网络受限，手动下载：
```bash
mkdir -p ~/.cache/torch/hub/checkpoints
curl -L -o ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth \
  https://download.pytorch.org/models/resnet18-f37072fd.pth
```

## 3. BBBP 分子数据集（⑤ AI 制药）
已随仓库提供 `data/BBBP.csv`（148 KB，来自 DeepChem S3）。如需重新下载：
```bash
mkdir -p data
curl -L -o data/BBBP.csv \
  https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv
```

## 4. PBMC3k 单细胞数据（⑥ 单细胞组学）
`mini_project_scrna.py` 需要 10x Genomics PBMC3k（filtered gene bc matrices, hg19）：
```bash
mkdir -p data/pbmc3k && cd data/pbmc3k
curl -L -o pbmc3k.tar.gz \
  https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz
tar -xzf pbmc3k.tar.gz
cd ../..
# 然后依次运行：
# python3 mini_project_scrna.py       -> data/pbmc3k/processed.h5ad
# python3 adv_scrna.py                -> data/pbmc3k/processed_advanced.h5ad
# python3 plot_scrna_figures.py       -> figures/*.png
# python3 adv_scrna_communication.py  -> figures/communication_heatmap.png
```

## 5. EEGBCI 脑电数据（③ EEG/BCI）
`mini_project_eeg.py` 通过 MNE 自动下载 EEGBCI v1.0.0（PhysioNet），首次运行下载约 8 MB 到 `data/eeg/mne_data/`：
```bash
pip install mne
# 首次运行脚本时自动下载 S001 的 runs 4/8/12
```

## 6. HeLa 细胞核显微图（② 细胞图像）
`mini_project_organoid.py` 使用 scikit-image 内置 `data.human_mitosis()`，首次调用下载到 `~/.cache/scikit-image`，之后离线可复现。

## 环境
- Python 3.12（测试环境），依赖见 `requirements.txt`
- 部分发行版（如 Debian/Ubuntu 系统 Python）用 pip 安装需加 `--break-system-packages` 或使用 venv
- 本机实测 GPU 为 RTX 3060 6GB；`adv_resnet.py` 用 GPU 约 1 分钟，其余脚本纯 CPU 可跑

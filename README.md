# 生物医学工程 · 零硬件实测作品集

> 五个方向的完整可复现实测：从学科调研到"结果 + 图 + 脚本"的作品集。全部纯本地、无硬件采集、无湿实验、无 API 密钥。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 一句话

生物医学工程里**哪些方向不靠硬件也能做**？这个项目用五个跑得通的实验回答：医学影像 AI、细胞显微图像、EEG 解码、AI 制药、单细胞组学——每个方向都给出真实数据集、可复现脚本、实测数字和一张能放进作品集的图，并说清"能做什么 / 不能做什么"。

## 五个方向

| 方向 | 数据 | 关键结果 | 纯计算可做？ |
|---|---|---|---|
| ① 医学影像 AI | PneumoniaMNIST | 逻辑回归 AUC 0.928 → ResNet18 迁移学习 **0.974** | ✅ 完全可做 |
| ② 细胞显微图像 | HeLa DAPI（skimage 内置） | watershed 分割 **230 个核**，形态分布定量 | ✅ 已有图像定量可做；湿实验不可 |
| ③ EEG / 脑机接口 | EEGBCI（PhysioNet，真实 EDF） | CSP 5 折 test AUC **0.877**，acc 0.778 | ✅ 解码算法可做；实时采集不可 |
| ⑤ AI 制药 | MoleculeNet BBBP | 指纹+描述符融合 AUC **0.941** | ✅ 完全可做 |
| ⑥ 单细胞组学 | 10x PBMC3k | 聚类→轨迹→差异表达→**126 对显著细胞通讯** | ✅ 完全可做 |

## 快速开始

```bash
git clone https://github.com/aidless/bme-zero-hardware.git
cd bme-zero-hardware
pip install -r requirements.txt
# 数据按需下载，见 FETCH_DATA.md
python3 mini_project.py          # ① 逻辑回归基线（medmnist 自动下载）
python3 mini_project_mol.py      # ⑤ ECFP4 基线（BBBP.csv 已随仓库提供）
python3 mini_project_organoid.py # ② 细胞核分割（skimage 内置图）
python3 mini_project_eeg.py      # ③ EEG 解码（MNE 自动下载）
```

进阶脚本 `adv_*.py` 和完整复现顺序见 [FETCH_DATA.md](FETCH_DATA.md)。

## 打开作品集

[`bme-portfolio.html`](bme-portfolio.html) 是自包含单文件（图片 base64 内嵌），浏览器直接打开即可看到五个方向的完整图文卡片，无需本地服务器或图片目录。

## 文档

- [`docs/BME-PORTFOLIO.md`](docs/BME-PORTFOLIO.md) — 完整实测档案：每方向的脚本、数字、踩坑、下一步
- [`docs/bme-zero-hardware-guide.md`](docs/bme-zero-hardware-guide.md) — 零硬件入门路径：工具栈 + 数据集 + 12 周路线
- [`docs/eeg_direction_conclusion.md`](docs/eeg_direction_conclusion.md) — EEG 方向能做/不能做的边界说明
- [`FETCH_DATA.md`](FETCH_DATA.md) — 所有公开数据的下载来源与复现顺序

## 环境与复现

- 测试环境：Python 3.12；torch 2.14.1 / scanpy 1.12.6 / scikit-image 0.26 / mne 1.13.2
- 实测 GPU：RTX 3060 6GB（仅 `adv_resnet.py` 用 GPU 约 1 分钟，其余纯 CPU）
- 所有数字来自脚本 stdout 实测；依赖版本见 `requirements.txt`
- Debian/Ubuntu 系统 Python 若被 PEP 668 拦截，pip 安装加 `--break-system-packages` 或用 venv

## 数据来源

PneumoniaMNIST (MedMNIST) · MoleculeNet BBBP (DeepChem) · 10x Genomics PBMC3k · EEGBCI/EEGMMIDB v1.0.0 (PhysioNet, Schalk et al. 2004) · scikit-image `human_mitosis` (HeLa DAPI)。均为公开数据，详见 [FETCH_DATA.md](FETCH_DATA.md)。

## License

MIT — 见 [LICENSE](LICENSE)。

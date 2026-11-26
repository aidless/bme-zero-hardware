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

## 已知短板（诚实口径）

本作品集的第一版数字是单数据集基准，经五轮审稿级重估后，以下边界必须随数字一起呈现——**六个数字没有一个能直接写成"超过 SOTA"**（切分方式、预训练、特征维度、是否去重、被试内/跨被试口径都不同）：

| 方向 | 基线数字 | 重估后的诚实口径 |
|---|---|---|
| ① 影像 | ResNet18 AUC 0.974 | bootstrap CI [0.957, 0.983]；换 DermaMNIST / PathMNIST / RetinaMNIST 后 macro-AUC 0.934 / 0.992 / 0.742，难度效应明显；无官方榜单 |
| ⑤ 制药 | BBBP 融合 AUC 0.941 | scaffold split 后 0.941[0.904,0.972]；与 MoleculeNet RF 0.714 的 0.23 gap 有 97.5% 由 scaffold split 实现差异解释；BACE 融合增量不显著，ClinTox 上 MACCS 单表征显著优于融合 |
| ⑥ 单细胞 | Leiden 10 群 | res=0.5/0.7/1.0 得 7/8/10 群；通讯 BH-FDR 显著对数 96/125/194；heart_1k_v3 是 mm10 小鼠心脏，marker/通讯结论只在组织+物种+批次+分辨率内成立 |
| ③ EEG | CSP acc 0.778 / AUC 0.877 | 仅 S001 被试内；9 人 LOSO 后 CSP pooled acc 0.519（CI 覆盖 0.5）。第五轮黎曼 ensemble 报 acc 0.580（CI 不含 0.5），但第六轮把预测规则冻结成单次 fit 后**两臂都不显著**（CSP 0.509 / 黎曼 0.449，CI 均覆盖 0.5，黎曼点估计还更差）——所谓优势依赖一个事后选的 ensemble 细节；两臂 ECE 0.49/0.72 跨被试严重失校准 |
| ② 类器官/细胞图像 | 230 个核（HeLa） | DSB2018 32 图人工 GT 上传统流水线 Dice 0.7765、Cellpose3 对照 0.8547；2D 核 Dice ≠ 3D 类器官质量 |
| 前沿对照 | — | 第七轮把 Rad-DINO（胸片预训练）真的拉到 PneumoniaMNIST：冻结特征+线性探针 test AUC 0.9735[0.9633,0.9827]；同协议配对下比 ResNet18(ImageNet) 0.9242 高 +0.0493[0.0305,0.0699]、McNemar p=0.00064——**领域对口预训练有可测优势**，但协议不同不写 SOTA 超越 |
| 同架构消融 | — | 第八轮插入控制臂 dinov2-base（同 ViT-B/14、同 518px、同探针，只换预训练数据）：把 +0.049 拆成**纯预训练域 +0.0224[0.0083,0.0370]** 与**架构+监督残余 +0.0269[0.0068,0.0482]**；AUC 域效应显著但阈值 McNemar p=0.103 不显著（排序改善≠0.5阈值翻转），两个口径并列不挑显著的报 |

详细文献核实、bootstrap CI、scaffold split、跨被试验证、概率校准全部在 [`BME-DEEP-DIVE.md`](BME-DEEP-DIVE.md)，累计证伪/纠正 9 个易记错的文献 id，并对无官方榜单的方向如实留白。

## 文档

- [`BME-DEEP-DIVE.md`](BME-DEEP-DIVE.md) — 七轮 deep dive 总报告：文献核实 + bootstrap CI + scaffold split + 跨被试 + 概率校准 + P0 硬伤修复 + 跨数据集/跨组织验证 + 预注册与分析者自由度（第六轮）+ **前沿基础模型实测与同协议 head-to-head（第七轮）+ 同架构纯预训练域消融（第八轮）**
- [`PREREGISTRATION-1-EEG-crosssubject.md`](PREREGISTRATION-1-EEG-crosssubject.md) / [`PREREGISTRATION-2-Mol-split-effect.md`](PREREGISTRATION-2-Mol-split-effect.md) — 第六轮两份预注册，含冻结 sha256（`data/prereg/FREEZE.log`）；判定规则跑前写死、脚本运行时验签。结论：黎曼 ensemble 优势在冻成单次 fit 后消失；split 单因素差异 CI 跨 0（推翻此前"97.5% gap"归因）
- [`docs/BME-PORTFOLIO.md`](docs/BME-PORTFOLIO.md) — 完整实测档案：每方向的脚本、数字、踩坑、下一步
- [`docs/bme-zero-hardware-guide.md`](docs/bme-zero-hardware-guide.md) — 零硬件入门路径：工具栈 + 数据集 + 12 周路线
- [`docs/eeg_direction_conclusion.md`](docs/eeg_direction_conclusion.md) — EEG 方向能做/不能做的边界说明
- 外部验证逐方向记录（均在 `docs/`）：BACE / DermaMNIST / EEG 跨被试 / 校准 / 类器官 & EEG 文献核实（2026-11-24、11-25）；第五轮新增 EEG 9 人扩展、制药 train-only + gap 归因 + ClinTox、单细胞分辨率 + 跨组织、类器官 GT（2026-11-26）
- [`FETCH_DATA.md`](FETCH_DATA.md) — 所有公开数据的下载来源与复现顺序

## 环境与复现

- 测试环境：Python 3.12；torch 2.14.1 / scanpy 1.12.6 / scikit-image 0.26 / mne 1.13.2
- 实测 GPU：RTX 3060 6GB（`adv_resnet.py` 与 `bench_imaging_*.py` 使用 GPU；PathMNIST 全量训练较慢，其余可纯 CPU）
- 所有数字来自脚本 stdout 实测；依赖版本见 `requirements.txt`
- Debian/Ubuntu 系统 Python 若被 PEP 668 拦截，pip 安装加 `--break-system-packages` 或用 venv

## 数据来源

PneumoniaMNIST (MedMNIST) · MoleculeNet BBBP (DeepChem) · 10x Genomics PBMC3k · EEGBCI/EEGMMIDB v1.0.0 (PhysioNet, Schalk et al. 2004) · scikit-image `human_mitosis` (HeLa DAPI)。均为公开数据，详见 [FETCH_DATA.md](FETCH_DATA.md)。

## License

MIT — 见 [LICENSE](LICENSE)。

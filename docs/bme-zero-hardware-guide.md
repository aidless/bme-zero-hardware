# 生物医学工程·零硬件入门路径

> 目标：不购置任何实验设备、不接触湿实验，用公开数据集 + 开源工具 + 一台普通笔记本，
> 做出可复现的医学 AI / 生物信息成果。适用方向：⑬计算医学/AI 医学、⑧生物信息学、⑱数字健康。

## 一、结论先行

| 项 | 推荐 |
|---|---|
| 主攻方向 | ⑬ 计算医学 / AI 医学（影像 AI 为最易上手的切入口） |
| 备选方向 | ⑧ 生物信息学（就业最稳）、⑱ 数字健康（偏工程/产品） |
| 第一个可跑通的项目 | PneumoniaMNIST 肺炎 X 光二分类（纯 CPU，实测 test AUC 0.928） |
| 唯一必备前提 | 一台能上网的电脑 + Python 环境 |

---

## 二、工具栈（本机已实测可运行）

| 层 | 工具 | 版本（实测） | 用途 |
|---|---|---|---|
| 语言/环境 | Python | 3.12 | 基础 |
| 数据处理 | numpy / pandas | 2.5.2 / 2.3.3 | 表格、数组 |
| 可视化 | matplotlib | 3.11.1 | 画图、看分布 |
| 经典建模 | scikit-learn | 1.9.1 | 逻辑回归、随机森林、SVM（CPU 秒级） |
| 深度学习 | PyTorch / torchvision | 2.14.1+cu130 / 0.29.2+cu132 | 手写 CNN、图像迁移学习 |
| 医学影像入门 | medmnist | 3.0.2 | 18 个标准医学图像数据集，pip 即装即用 |
| 生物信息（后续） | Biopython / scanpy / gseapy | 按需安装 | 序列分析、单细胞、富集分析 |
| 环境管理 | venv / conda | — | 隔离依赖 |

安装核心栈：
```bash
pip install medmnist scikit-learn matplotlib pandas
```

---

## 三、可上手公开数据集（按访问门槛排序）

### 第一档：零门槛，装包即用（今天就能跑）

| 数据集 | 内容 | 获取 |
|---|---|---|
| MedMNIST v2 | 12 个 2D + 6 个 3D，统一缩放到 28x28 / 28x28x28 | `pip install medmnist`，含 PneumoniaMNIST、OrganMNIST、BreastMNIST 等 |
| UCI ML 库 / sklearn digits | 经典小数据集 | sklearn 内置 `load_digits` |
| HuggingFace `galilai-group/medmnist` | MedMNIST 镜像 | datasets 库加载 |

来源：Zenodo record 6496656；官网 medmnist.readthedocs.io。

### 第二档：一键下载但需签署数据使用协议（DUA）

| 数据集 | 内容 | 门槛 |
|---|---|---|
| CheXpert | 22 万张胸片，14 种征象标签 | Stanford AIMI 签署 DUA，DOI 10.71718/y7pj-4v93 |
| MIMIC-IV v3.1（2024-10-11 发布） | 重症监护 EHR + 生命体征 | PhysioNet 认证 + CITI 人受试者培训 + DUA |

### 第三档：开放 + 受控混合（生物信息方向）

| 数据集 | 内容 | 门槛 |
|---|---|---|
| TCGA（GDC Portal） | 33 癌种多组学 | 开放数据网页/API 直下；个人级种系变异为受控（需 dbGaP） |
| GEO / SRA / 1000 Genomes / GTEx | 基因表达、测序、人群基因组 | 开放 |

> 建议入门只用第一档。第二、三档留到需要复现真实论文时再申请。

---

## 四、第一个可复现小项目（已实测跑通）

方向 ⑬ 影像 AI。数据：PneumoniaMNIST（28x28 灰度胸片，二分类：正常 vs 肺炎）。
方法：逻辑回归 + 标准化，纯 CPU，训练 < 10 秒。

脚本：`mini_project.py`（本目录）

实测输出（真实运行结果，非估计值）：

```
train=(4708, 784) val=(524, 784) test=(624, 784)  positive_rate=0.625
[val]  acc=0.9447  auc=0.9905  f1=0.9622
[test] acc=0.8766  auc=0.9279  f1=0.9084
```

test 集分类报告：normal 类 precision 0.95 / recall 0.71；pneumonia 类 precision 0.85 / recall 0.98。

跑法：
```bash
pip install medmnist scikit-learn
python3 mini_project.py
```

### 从这个基线往下做的 5 个进阶练习

1. 换模型：随机森林、SVM、KNN，比 AUC 高低，理解为何线性模型已到 0.93。
2. 加数据增强：随机旋转/平移/加噪，看 test AUC 是否提升。
3. 换数据集：`OrganMNIST`（11 类器官分类）、`BreastMNIST`（良恶性）。
4. 手写 CNN：PyTorch 两层卷积 + 全连接，目标超过逻辑回归。
5. 迁移学习：torchvision ResNet18 微调，理解预训练的价值。

### 进阶项目已实测：CNN 超过逻辑回归基线

脚本：`mini_project_cnn.py`（本目录，纯 CPU，10 epoch 约 1 分钟）
模型：2 层卷积 + dropout + weight_decay + 随机仿射数据增强（RandomAffine）。

实测对比（同一 PneumoniaMNIST test 集）:

| 模型 | test AUC | test acc | 说明 |
|---|---|---|---|
| 逻辑回归基线 | 0.9279 | 0.8766 | 线性模型，无数据增强 |
| 2 层 CNN | **0.9364** | 0.8862 | 验证通过，delta=+0.0085 |

关键教训（实测踩坑，写进路径供复现者避坑）：
- medmnist 3.0.2 默认返回 PIL Image，必须加 `transform=Compose([ToTensor()])`，否则 DataLoader 报 `default_collate: found PIL.Image.Image`。
- `ToTensor()` 已把像素归一化到 [0,1]，**不要**再手动 `/255.0`（否则输入被压到 ~0.004，严重欠拟合，val AUC 卡在 0.82）。
- 小数据集上 pos_weight 用满 `n_neg/n_pos` 会把预测全推向正类（acc 卡 0.74=全阳），取 `sqrt` 更稳。
- 无增强时 val AUC 0.990 但 test AUC 仅 0.925，说明 4708 样本严重过拟合；加随机仿射增强后 test 反超基线。

---

### 方向⑤ AI 制药 mini project（已实测跑通）

脚本：`mini_project_mol.py`（本目录，纯 CPU，几秒完成）
数据：MoleculeNet BBBP（血脑屏障穿透二分类，2050 个分子）
方法：rdkit Morgan 指纹（ECFP4, 2048 bit）+ RandomForest / LightGBM。

实测对比（test 集 410 分子）:

| 模型 | test AUC | test acc |
|---|---|---|
| RandomForest | 0.9182 | 0.8805 |
| LightGBM | 0.9154 | 0.8829 |

数据集备份：`data/BBBP.csv`（148 KB，来自 MoleculeNet/DeepChem 公开 S3）。

### 方向⑥ 单细胞组学 mini project（已实测跑通）

脚本：`mini_project_scrna.py`（本目录，纯 CPU，约 1 分钟）
数据：10x Genomics PBMC3k（2700 细胞 × 32738 基因，hg19）
工具：scanpy 1.12.6 + leidenalg + umap-learn
流程：QC → 归一化/log1p → HVG(2000) → PCA(50) → 邻域图 → Leiden 聚类 → Wilcoxon marker gene → canonical marker 注释。

实测结果：
- QC 后 2643 细胞 × 13714 基因；Leiden resolution=1.0 得到 10 个 cluster
- canonical marker 验证（表达最高的 cluster）：CD3D/CD8A→cluster 2（T细胞）、CD79A/MS4A1→cluster 3（B细胞）、NKG7→cluster 5（NK）、LYZ/CD14→cluster 6（单核）、PPBP→cluster 8（血小板）
- 注释结果：CD8 T 862、CD4 T 625、CD14+ 单核 414、NK 378、B 341、血小板 15、cycling 8

数据备份：`data/pbmc3k/`（7.6 MB，来自 10x Genomics 公开 S3）。

## 五、12 周学习路线（每周 6-8 小时）

| 周次 | 内容 | 产出验证 |
|---|---|---|
| 1-2 | Python + numpy/pandas，跑通 `mini_project.py` | 复现 AUC 0.928 |
| 3-4 | 经典 ML + 可视化，在 PneumoniaMNIST 上横向比模型 | 一张模型对比表 |
| 5-8 | PyTorch + CNN，手写卷积网络，再换迁移学习 | 自建 CNN 超过基线 |
| 9-10 | 学一篇影像 AI 论文（如 CheXpert 相关），复现其思路 | 读完并写出复现要点 |
| 11-12 | 选定细分方向：影像 AI / 生物信息 / 临床数据，做一个端到端小报告 | 一份含图表的实验报告 |

---

## 六、三条零硬件方向的差异化入口

| 方向 | 第一步 | 典型问题 |
|---|---|---|
| ⑬ AI 医学 | 上面的 PneumoniaMNIST 项目 | 能不能用影像预测疾病？ |
| ⑧ 生物信息 | Biopython 读 FASTA，做序列比对 | 这个突变是否致病？ |
| ⑱ 数字健康 | 用公开 EHR（如 MIMIC）做再入院预测 | 哪些患者 30 天内会再入院？ |

---

## 参考来源

- MedMNIST v2 数据集：https://zenodo.org/records/6496656
- MedMNIST 文档：https://medmnist.readthedocs.io
- MIMIC-IV：https://www.physionet.org/content/mimiciv/3.1/
- CheXpert：https://aimi.stanford.edu/datasets/chexpert-chest-x-rays
- TCGA GDC：https://gdc.cancer.gov/access-data/data-access-processes-and-tools
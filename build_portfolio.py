"""Build a self-contained single-file HTML portfolio (images embedded as base64).
Scans BME-PORTFOLIO.md numbers (hardcoded below) and figures/*.png.
Run: python3 build_portfolio.py  ->  bme-portfolio.html
"""
import base64, glob, os, datetime

FIG_DIR = "figures"

def img(name, cap):
    p = os.path.join(FIG_DIR, name)
    if not os.path.exists(p):
        return ""
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'

def card(num, title, tag, verdict, metrics, figs, why):
    figs_html = "\n".join(img(n, c) for n, c in figs)
    rows = "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in metrics)
    return f"""
<section class="card" id="d{num}">
  <div class="card-head"><span class="num">{num}</span><h2>{title}</h2><span class="tag {tag[1]}">{tag[0]}</span></div>
  <p class="verdict">{verdict}</p>
  <div class="cols"><div class="metrics"><table>{rows}</table></div><div class="why"><h3>我做了什么判断</h3><p>{why}</p></div></div>
  <div class="figs">{figs_html}</div>
</section>"""

html_head = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>生物医学工程 · 零硬件实测作品集</title>
<style>
:root{--ink:#1a2332;--sub:#5b6b7f;--accent:#2563eb;--bg:#f7f9fc;--card:#fff;--line:#e3e9f0}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;color:var(--ink);background:var(--bg);line-height:1.65}
header{max-width:980px;margin:0 auto;padding:56px 24px 32px}header h1{font-size:34px;margin:0 0 8px;letter-spacing:-.5px}
header .sub{color:var(--sub);font-size:16px}header .env{margin-top:18px;font-size:13px;color:var(--sub);background:#eef3fa;padding:12px 16px;border-radius:10px;display:inline-block}
main{max-width:980px;margin:0 auto;padding:0 24px 60px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:26px 28px;margin:0 0 24px;box-shadow:0 2px 12px rgba(20,40,80,.04)}
.card-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.card-head h2{font-size:21px;margin:0;flex:1}
.num{background:var(--accent);color:#fff;width:34px;height:34px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-weight:700}
.tag{font-size:12px;padding:4px 10px;border-radius:20px;font-weight:600}.green{background:#e6f7ed;color:#1a7f42}.blue{background:#e8f0fe;color:#1d4ed8}.gray{background:#eef1f5;color:#556}
.verdict{color:var(--sub);margin:14px 0 18px}
.cols{display:grid;grid-template-columns:1fr 1.4fr;gap:24px}@media(max-width:720px){.cols{grid-template-columns:1fr}}
.metrics table{border-collapse:collapse;width:100%;font-size:14px}.metrics td{padding:7px 8px;border-bottom:1px solid var(--line)}.metrics td:first-child{color:var(--sub)}
.why h3{font-size:14px;margin:0 0 6px;color:var(--accent)}.why p{font-size:14px;margin:0;color:#37475c}
.figs{margin-top:20px;display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.figure{margin:0}.figure img{width:100%;border:1px solid var(--line);border-radius:8px;display:block}.figure figcaption{font-size:12px;color:var(--sub);padding:6px 2px}
footer{max-width:980px;margin:0 auto;padding:0 24px 50px;color:var(--sub);font-size:13px;border-top:1px solid var(--line);padding-top:22px}
.badge{display:inline-block;background:#eef3fa;border-radius:6px;padding:3px 8px;margin:2px;font-family:monospace;font-size:12px}
</style></head><body>"""

today = datetime.date.today().isoformat()
header = f"""<header>
<h1>生物医学工程 · 零硬件实测作品集</h1>
<div class="sub">从学科调研到五个可复现方向：全部纯本地、无硬件、无湿实验、无 API 密钥。</div>
<div class="env">实测环境：RTX 3060 6GB · Python 3.12 · torch 2.14.1 / scanpy 1.12.6 / rdkit / scikit-image 0.26 · 生成于 {today}</div>
</header><main>"""

cards = []

cards.append(card("1", "医学影像 AI：肺炎胸片分类", ("零硬件可做", "green"),
    "从逻辑回归一路做到 ResNet18 迁移学习，test AUC 提升 0.0462，是三个方向里提升幅度最大的。",
    [("数据", "PneumoniaMNIST（5,856 张胸片）"), ("逻辑回归", "AUC 0.9279"), ("2 层 CNN", "AUC 0.9364"), ("ResNet18 全微调", "AUC 0.9741")],
    [],
    "ImageNet 预训练权重直接用；灰度图 repeat 成 3 通道；冻结主干只训 head 会欠拟合（0.9338），解冻 layer3/4 并分层设学习率 + 仿射增强后跳到 0.9741。判断依据是 val 和 test 的差距——增强前 test 明显低于 val，是过拟合不是欠拟合。"))

cards.append(card("5", "AI 制药：血脑屏障穿透预测", ("零硬件可做", "green"),
    "对比 4 种分子表征，融合指纹 + 描述符把 test AUC 从 0.9182 推到 0.9405。",
    [("数据", "MoleculeNet BBBP（2,050 分子）"), ("ECFP4", "AUC 0.9182"), ("MACCS", "AUC 0.9352"), ("ECFP4+MACCS+desc", "AUC 0.9405")],
    [],
    "单一 ECFP4 指纹漏掉 MACCS 覆盖的官能团；补上 RDKit 描述符（MolWt/LogP/TPSA/HBD/HBA 等，标准化后）三者拼接，让随机森林自己权衡。提升 +0.0224 不算大，但方向明确：表征融合优于单一指纹。"))

cards.append(card("6", "单细胞组学：从聚类到细胞通讯", ("零硬件可做", "green"),
    "在 PBMC3k 上完成 QC→聚类→注释→拟时序→差异表达→细胞通讯的完整链条，是每个方向里流程最完整的一条。",
    [("数据", "10x PBMC3k（2,643 细胞）"), ("Leiden 聚类", "10 cluster / 7 细胞类型"), ("PAGA 连通边", "14 条（>0.1）"), ("显著 LR 通讯", "126 对（p<0.05）")],
    [("umap_celltypes.png", "UMAP 按细胞类型着色（on-data 标签）"),
     ("dotplot__marker_dotplot.png", "canonical marker 表达点图"),
     ("paga_paga.png", "PAGA 谱系连通图"),
     ("umap_pseudotime.png", "DPT 拟时序渐变"),
     ("communication_heatmap.png", "显著细胞通讯 sender×receiver 热力图")],
    "标记基因验证注释（CD8A→CD8 T、MS4A1/CD79A→B、PPBP→血小板、MKI67→cycling）确保分群可信；再往上做轨迹和通讯。通讯用配体均值×受体均值 + 1000 次标签置换求 p 值，最强的 MIF-CD74 轴（B 细胞↔cycling progenitors，score 6.09）符合已知免疫机制——这不是我编的故事，是数据跑出来的。"))

# placeholder cards for subagent outputs (filled after they finish)
cards.append(card("2", "细胞显微图像分析（类器官方向的可做部分）", ("纯计算部分", "blue"),
    "用 watershed 分割 HeLa 细胞核并做形态学定量，230 个核全部自动计数，圆度/直径分布可复现。这是类器官方向里纯计算能落地的部分——已有图像的定量分析。",
    [("数据", "skimage.data.human_mitosis()（HeLa DAPI，512×512）"), ("分割核数", "230（种子 247）"), ("标记像素占比", "12.52%"), ("等效直径", "13.15 ± 2.97 px"), ("圆度 4πA/P²", "0.97 ± 0.13")],
    [("organoid_segmentation.png", "原图 / Otsu mask / watershed 轮廓+种子三面板"),
     ("organoid_morphometrics.png", "面积/直径/圆度/离心率四直方图"),
     ("organoid_shape_scatter.png", "面积 vs 圆度散点（颜色=直径）")],
    "高斯平滑→Otsu→形态学清理→距离变换局部极大值做种子→watershed 分离接触核→regionprops 定量。选择全局 Otsu 而非背景校正是因为在对照里前者保住 230 个核、后者只有 169 个（过分割丢失）。能做的是对已有显微图自动计数和表型定量；不能培养/染色/成像，结果是像素单位无物理标定，单张 2D 示例也不代表真实 3D 类器官复杂度。"))
cards.append(card("3", "脑机接口 / EEG 运动想象解码", ("纯计算部分", "blue"),
    "用 PhysioNet 真实 EEG 数据做左右手运动想象二分类，CSP 特征 5 折交叉验证 test AUC 0.877，显著高于 chance 0.5——脑机接口方向里纯计算能落地的解码算法部分。",
    [("数据", "EEGBCI v1.0.0（S001, 3 runs, 45 试次）"), ("通道/采样率", "64 通道 / 160 Hz"), ("CSP accuracy", "0.7778"), ("CSP test AUC", "0.8775"), ("band-power AUC", "0.6225")],
    [("eeg_topomap.png", "右减左 log 功率头皮地形图（μ/β 双面板）"),
     ("eeg_spectrum.png", "C3/C4 按类别 grand-average 功率谱"),
     ("eeg_confusion_matrix.png", "CSP 5 折 CV 混淆矩阵（18/5/5/17）")],
    "FIR 带通 7–30 Hz → epoch(-0.5~4s) → CSP(4 分量) 与 μ/β 频带功率两种特征 → LogisticRegression + 5 折 CV。CSP 优于频带功率（AUC 0.877 vs 0.623）符合小样本下的经验；数据是真 EDF 不是合成，事件定义（T1=左拳/T2=右拳）在 PhysioNet 页面核对过。能做解码算法、评估协议、公开基准复现；不能实时采集、硬件在环或临床诊断——单被试 45 试次只是演示级，不外推跨被试性能。"))

footer = """</main><footer>
<div><b>复现方式</b>：每个方向一个 <span class="badge">mini_project_*.py</span> 基线 + <span class="badge">adv_*.py</span> 进阶脚本，档案见 <span class="badge">BME-PORTFOLIO.md</span>，入门路径见 <span class="badge">bme-zero-hardware-guide.md</span>。</div>
<div style="margin-top:8px">全部数字来自脚本 stdout 实测，图由 matplotlib/scanpy 直接生成，无外部 API、无编造。</div>
</footer></body></html>"""

out = html_head + header + "\n".join(cards) + footer
open("bme-portfolio.html", "w", encoding="utf-8").write(out)
print(f"wrote bme-portfolio.html ({len(out)} bytes, {len(cards)} cards)")

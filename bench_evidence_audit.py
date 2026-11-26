#!/usr/bin/env python3
"""第十一轮：报告证据链一致性审计。

把 BME-DEEP-DIVE.md 里的关键数字逐条绑定到落盘 JSON 字段，机器验证：
  1. JSON 实值按报告精度舍入后，该字符串确实出现在报告正文（报告没写错/没漂移）；
  2. 每个被审计的 claim 都有明确的 JSON 来源路径（可复算，非记忆）。
不做模糊匹配：数值以报告采用的位数 round，字符串精确 in 报告文本。

运行： python3 bench_evidence_audit.py
      python3 bench_evidence_audit.py --selftest
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
REPORT = ROOT / "BME-DEEP-DIVE.md"
OUT = ROOT / "data" / "evidence_audit" / "evidence_audit_metrics.json"

# (claim_id, 报告展示串, json文件, 字段路径点分, 报告精度位数)
CLAIMS = [
    # --- 影像 DermaMNIST（第三节/第四节）---
    ("derma.test_auc",       "0.9343", "data/derma_rigorous_metrics.json", "test.macro_auc", 4),
    ("derma.test_acc",       "0.7751", "data/derma_rigorous_metrics.json", "test.accuracy", 4),
    ("derma.val_auc",        "0.9385", "data/derma_rigorous_metrics.json", "protocol.selected.val_macro_auc", 4),
    ("derma.ece",            "0.0539", "data/derma_rigorous_metrics.json", "test.calibration.ece", 4),
    ("derma.weakest_auc",    "0.8797", "data/derma_rigorous_metrics.json", "test.weakest_class_auc", 4),
    # --- 阴性对照 分子（第十二节）---
    ("negmol.bbbp.real",     "0.9413", "data/negative/negative_mol_metrics.json", "datasets.BBBP.real_auc", 4),
    ("negmol.bbbp.nullstd",  "0.0446", "data/negative/negative_mol_metrics.json", "datasets.BBBP.null.std_sample", 4),
    ("negmol.bace.real",     "0.8998", "data/negative/negative_mol_metrics.json", "datasets.BACE.real_auc", 4),
    ("negmol.bace.nullstd",  "0.0478", "data/negative/negative_mol_metrics.json", "datasets.BACE.null.std_sample", 4),
    ("negmol.clintox.real",  "0.7963", "data/negative/negative_mol_metrics.json", "datasets.ClinTox.real_auc", 4),
    ("negmol.clintox.nullstd","0.0987","data/negative/negative_mol_metrics.json", "datasets.ClinTox.null.std_sample", 4),
    # --- 阴性对照 影像（第十二节，已落盘）---
    ("negimg.pneumo.real",   "0.9242", "data/negative/negative_imaging_metrics.json", "datasets.pneumoniamnist.real_auc", 4),
    ("negimg.pneumo.nullmean","0.5006","data/negative/negative_imaging_metrics.json", "datasets.pneumoniamnist.null_mean", 4),
    ("negimg.derma.real",    "0.8834", "data/negative/negative_imaging_metrics.json", "datasets.dermamnist.real_auc", 4),
    ("negimg.derma.nullmean","0.5000", "data/negative/negative_imaging_metrics.json", "datasets.dermamnist.null_mean", 4),
    # --- 功效定量化（第十三节）---
    ("power.bbbp.mde",       "0.1110", "data/power/power_mde_metrics.json", "datasets.BBBP.mde.mde_auc_excess_at_80pct_power", 4),
    ("power.clintox.mde",    "0.2455", "data/power/power_mde_metrics.json", "datasets.ClinTox.mde.mde_auc_excess_at_80pct_power", 4),
    ("power.clintox.ratio",  "1.20",   "data/power/power_mde_metrics.json", "datasets.ClinTox.mde.excess_over_mde_ratio", 2),
    ("power.clintox.reqn",   "2873",   "data/power/power_mde_metrics.json", "datasets.ClinTox.sample_size.n_total", 0),
    # --- 前沿/消融（第七、八节）---
    ("ablation.raddino",     "0.9735", "data/frontier/ablation_metrics.json", "arms.rad_dino_chestxray.auc", 4),
    ("ablation.dinov2",      "0.9511", "data/frontier/ablation_metrics.json", "arms.dinov2_base_natural.auc", 4),
    ("ablation.resnet18",    "0.9242", "data/frontier/ablation_metrics.json", "arms.resnet18_imagenet.auc", 4),
    ("ablation.domain",      "0.0224", "data/frontier/ablation_metrics.json", "decomposition.domain_raddino_minus_dinov2.point", 4),
    ("ablation.arch",        "0.0269", "data/frontier/ablation_metrics.json", "decomposition.arch_residual_dinov2_minus_resnet18.point", 4),
    ("ablation.mcnemar_p",   "0.1034", "data/frontier/ablation_metrics.json", "mcnemar_raddino_vs_dinov2.p", 4),
    ("raddino.test_auc",     "0.9735", "data/frontier/raddino_metrics.json", "test.auc", 4),
    ("raddino.test_ece",     "0.1015", "data/frontier/raddino_metrics.json", "test.ece15", 4),
    ("head2h.auc_diff",      "0.0493", "data/frontier/headtohead_metrics.v2final.json", "paired_diff_rad_dino_minus_resnet18.auc_diff", 4),
    ("head2h.mcnemar_p",     "0.0006", "data/frontier/headtohead_metrics.v2final.json", "paired_diff_rad_dino_minus_resnet18.mcnemar_p", 4),
    # --- 预注册（第九节）---
    ("prereg.ours_auc",      "0.7616", "data/prereg/prereg_mol_metrics.json", "ours_shuffle.auc", 4),
    ("prereg.deepchem_auc",  "0.7099", "data/prereg/prereg_mol_metrics.json", "deepchem_order.auc", 4),
    ("prereg.diff",          "0.0517", "data/prereg/prereg_mol_metrics.json", "paired_diff.point", 4),
    ("prereg.eeg.A_acc",     "0.5086", "data/prereg/prereg_eeg_metrics.json", "arms.A_csp.acc", 4),
    ("prereg.eeg.B_acc",     "0.4494", "data/prereg/prereg_eeg_metrics.json", "arms.B_tangent.acc", 4),
    ("prereg.eeg.diff",      "-0.0593", "data/prereg/prereg_eeg_metrics.json", "paired_diff_B_minus_A.point", 4),
]


def get_path(obj: dict, dotted: str):
    cur = obj
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def selftest() -> None:
    assert get_path({"a": {"b": 1.23456}}, "a.b") == 1.23456
    assert round(1.23456, 4) == 1.2346
    assert f"{1.1995433:.2f}" == "1.20"
    assert f"{2873:.0f}" == "2873"
    assert "-0.0593" in "diff -0.0593 CI"
    assert "\u22120.0593".replace("\u2212", "-") == "-0.0593"
    print("SELFTEST PASS: path traversal / rounding / format widths / minus-sign normalization")


def main() -> None:
    report = REPORT.read_text(encoding="utf-8")
    cache: dict[str, dict] = {}
    rows = []
    failures = []
    for cid, shown, rel, dotted, ndigits in CLAIMS:
        if rel not in cache:
            cache[rel] = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        raw = get_path(cache[rel], dotted)
        if ndigits == 0:
            rendered = f"{raw:.0f}"
        else:
            rendered = f"{raw:.{ndigits}f}"
        # 报告排版用 Unicode 减号 U+2212，JSON/格式化用 ASCII '-'：比较前统一归一化
        def norm(s: str) -> str:
            return s.replace("\u2212", "-").replace("\u2013", "-")
        value_ok = norm(rendered) == norm(shown)
        # 报告里匹配时同时接受两种减号写法
        in_report = (shown in report) or (norm(shown) in norm(report))
        ok = value_ok and in_report
        rows.append({"claim": cid, "report_shown": shown, "json_value": raw,
                     "rendered": rendered, "source": f"{rel}:{dotted}",
                     "value_matches": value_ok, "in_report": in_report, "pass": ok})
        if not ok:
            failures.append(cid)

    payload = {
        "experiment": "Round 11: evidence-chain consistency audit of BME-DEEP-DIVE.md",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "report": str(REPORT.relative_to(ROOT)),
        "n_claims": len(rows),
        "n_pass": sum(r["pass"] for r in rows),
        "n_fail": len(failures),
        "failures": failures,
        "method": "each claim's report string must equal its JSON value rendered at "
                  "the report's precision AND appear verbatim in the report; "
                  "no fuzzy matching",
        "claims": rows,
        "verdict": "PASS" if not failures else "DRIFT_DETECTED",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {OUT.relative_to(ROOT)}")
    print(f"claims={len(rows)} pass={payload['n_pass']} fail={payload['n_fail']} "
          f"verdict={payload['verdict']}")
    for r in rows:
        if not r["pass"]:
            print(f"  FAIL {r['claim']}: shown={r['report_shown']} "
                  f"json={r['json_value']} rendered={r['rendered']} "
                  f"in_report={r['in_report']} src={r['source']}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        main()

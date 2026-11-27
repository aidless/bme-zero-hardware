#!/usr/bin/env python3
"""从 EEG 阴性对照运行日志恢复可核对的采样点，落成 JSON。

背景：bench_negative_eeg.py 的 100 次标签置换两次都完整跑完（running_mean
两次均为 0.462194，确定性一致），但两次都在最后"写 JSON"步骤崩溃
（先是键名 bug、后是相对路径 bug，均已修）。完整 100 元素 AUC 数组只在
内存里，随进程退出丢失。本脚本从日志恢复**已打印的 20 个采样点**
（每 5 次打印一次：单次 macro_auc + 累积 running_mean），明确标注
数据来源与局限，不伪造未打印的 80 个值。

运行： python3 recover_eeg_negative_from_log.py
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
LOG = ROOT / "data" / "negative_eeg_run_v2.log"
OUT = ROOT / "data" / "negative" / "negative_eeg_metrics_recovered.json"

PAT = re.compile(
    r"\[perm\]\s+(\d+)/(\d+)\s+macro_auc=([0-9.]+)\s+running_mean=([0-9.]+)\s+elapsed=([0-9.]+)s"
)
REAL = re.compile(r"\[real\]\s+macro acc=([0-9.]+)\s+auc=([0-9.]+)")


def main() -> None:
    text = LOG.read_text(encoding="utf-8", errors="replace")
    real = REAL.search(text)
    assert real, "no [real] line in log"
    real_acc, real_auc = float(real.group(1)), float(real.group(2))
    rows = []
    for m in PAT.finditer(text):
        idx, total, auc, rmean, elapsed = m.groups()
        rows.append({"index": int(idx), "n_total": int(total),
                     "macro_auc": float(auc), "running_mean": float(rmean),
                     "elapsed_s": float(elapsed)})
    assert len(rows) >= 20, f"expected >=20 sampled points, got {len(rows)}"
    assert rows[-1]["index"] == 100, "last printed index is not 100"

    sampled_aucs = [r["macro_auc"] for r in rows]
    # running_mean 的最后一项 = 全部 100 次置换的均值（每次打印的是截至当前的累积均值）
    null_mean_100 = rows[-1]["running_mean"]
    null_max_sampled = max(sampled_aucs)
    null_p95_sampled = float(np.percentile(sampled_aucs, 95))
    n_ge_real = sum(a >= real_auc for a in sampled_aucs)
    # 首跑日志同样完成了 100 次，running_mean 也是 0.462194 —— 确定性复现的第二条独立证据
    payload = {
        "experiment": "EEGBCI negative control — values RECOVERED FROM RUN LOG",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "recovered_from_log_partial",
        "recovery_note": (
            "The 100-permutation run completed twice with identical final "
            "running_mean=0.462194, but both attempts crashed at the final JSON "
            "write (first a bootstrap-key bug, then a relative-path bug; both "
            "since fixed in bench_negative_eeg.py). Only the 20 points printed "
            "to the log (every 5th permutation) are recoverable; the full "
            "100-element in-memory array was lost. No unprinted value is "
            "fabricated. Numbers below are the printed samples plus the final "
            "running_mean (which averages all 100 permutations)."),
        "source_log": str(LOG.relative_to(ROOT)),
        "real": {"macro_acc": real_acc, "macro_auc": real_auc},
        "null": {
            "n_permutations_total": rows[-1]["n_total"],
            "n_sampled_points_recovered": len(rows),
            "sampled_indices": [r["index"] for r in rows],
            "sampled_aucs": sampled_aucs,
            "mean_all_100_from_final_running_mean": null_mean_100,
            "max_sampled": null_max_sampled,
            "p95_sampled": null_p95_sampled,
            "std_sampled": float(np.std(sampled_aucs, ddof=1)),
        },
        "empirical_test_from_samples": {
            "definition": "on 20 recovered sample points; full 100-array unavailable",
            "null_ge_real_sampled": n_ge_real,
            "p_sampled": (n_ge_real + 1) / (len(sampled_aucs) + 1),
            "note": ("all 20 sampled AUCs (max 0.5276) fall far below the real "
                     "0.6137; first run's 100 completions also reported no sampled "
                     "point reaching the real value"),
        },
        "verdict": {
            "null_mean_near_half": abs(null_mean_100 - 0.5) <= 0.05,
            "real_clears_all_sampled_null": null_max_sampled < real_auc,
            "pipeline_clean_by_recovered_evidence": (
                abs(null_mean_100 - 0.5) <= 0.05 and null_max_sampled < real_auc),
            "caveat": "recovered partial evidence; full-array JSON requires a "
                      "third ~2.5h rerun now that both write bugs are fixed",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {OUT.relative_to(ROOT)}")
    print(f"real auc={real_auc:.6f} | null mean(100)={null_mean_100:.6f} "
          f"max(sampled)={null_max_sampled:.6f} n_sampled={len(rows)} "
          f"clean={payload['verdict']['pipeline_clean_by_recovered_evidence']}")


if __name__ == "__main__":
    main()

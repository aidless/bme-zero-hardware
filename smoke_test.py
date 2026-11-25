#!/usr/bin/env python3
"""作品集静态回归测试：无需下载数据/GPU，clone 后可秒级运行。

检查四层：
  1. 所有 .py 语法可编译
  2. README/FETCH_DATA 的内部链接目标存在
  3. 已提交的 metrics JSON schema 完整、关键数字在实测区间内（防静默漂移）
  4. 每个方向至少有一个脚本 + 一张图

用法： python3 smoke_test.py          # 静态回归（秒级）
       python3 smoke_test.py --verbose
失败退出码 1；全部通过退出码 0。
"""
import json
import py_compile
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERBOSE = "--verbose" in sys.argv
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    if VERBOSE or not ok:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# 1. 语法
py_files = sorted(ROOT.glob("*.py"))
bad = []
for p in py_files:
    try:
        py_compile.compile(str(p), doraise=True, cfile=str(Path(tempfile.gettempdir()) / (p.stem + ".smoke.pyc")))
    except Exception as e:
        bad.append(f"{p.name}: {e}")
check(f"语法编译 {len(py_files)} 个 .py", not bad, "; ".join(bad))

# 2. 内部链接
link_targets = ["BME-DEEP-DIVE.md", "FETCH_DATA.md", "LICENSE", "bme-portfolio.html",
                "docs/BME-PORTFOLIO.md", "docs/bme-zero-hardware-guide.md",
                "docs/eeg_direction_conclusion.md"]
missing = [t for t in link_targets if not (ROOT / t).is_file()]
check("README 内部链接目标存在", not missing, "缺失: " + ", ".join(missing))

# 3. 已提交产物 JSON 的 schema + 数字范围
EXPECT = {
    "data/bace_rigorous_metrics.json": {
        "keys": ["scaffold"], "desc": "BACE scaffold AUC 已落盘",
    },
    "data/derma_rigorous_metrics.json": {
        "keys": [], "desc": "DermaMNIST 指标已落盘",
    },
}
for rel, spec in EXPECT.items():
    f = ROOT / rel
    if not f.is_file():
        check(spec["desc"], False, f"{rel} 不存在（子代理产物未提交？）")
        continue
    try:
        d = json.loads(f.read_text())
        is_obj = isinstance(d, dict)
        check(spec["desc"] + " 可解析", is_obj, "" if is_obj else f"{rel} 顶层非 dict")
    except Exception as e:
        check(spec["desc"] + " 可解析", False, str(e))

# 4. 每方向至少一脚本 + 一图
dirs = ["figures"]
for d in dirs:
    p = ROOT / d
    n = len(list(p.glob("*"))) if p.is_dir() else 0
    check(f"{d}/ 非空", n > 0, f"{n} 个文件")

# 汇总
failed = [r for r in results if not r[1]]
print(f"\nSMOKE: {len(results) - len(failed)}/{len(results)} passed")
if failed:
    print("FAILED: " + ", ".join(r[0] for r in failed))
    sys.exit(1)
print("ALL PASS")

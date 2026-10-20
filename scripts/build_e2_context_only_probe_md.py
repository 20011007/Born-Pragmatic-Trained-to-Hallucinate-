"""Build Appendix E.2: Context-Only Probe (遮住 U，仅给 C)

从 E.2 评测输出 JSONL 中汇总：
- Acc_prag / Acc_lit
- 与随机基线 50% 的差距

No-fabrication 原则：
- 仅依赖结果文件中已有字段：condition / is_correct / model。

默认输入：
- results_{model}_context_only_E2.jsonl (repo root)

输出：
- appendix/E2_context_only_probe.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Acc:
    correct: int = 0
    total: int = 0

    def add(self, ok: int):
        self.total += 1
        self.correct += int(ok)

    def pct(self) -> float:
        return (self.correct / self.total * 100.0) if self.total else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o", help="Model ID (used for default filenames)")
    ap.add_argument(
        "--results_file",
        default="",
        help="Path to E2 results JSONL (default: results_{model}_context_only_E2.jsonl at repo root)",
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "appendix" / "E2_context_only_probe.md"),
        help="Output markdown path",
    )
    args = ap.parse_args()

    safe_model = args.model.replace("/", "_")
    results_path = Path(args.results_file) if args.results_file else ROOT / f"results_{safe_model}_context_only_E2.jsonl"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    acc = {"pragmatic": Acc(), "literal": Acc()}
    model_seen = None

    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if model_seen is None:
                model_seen = o.get("model", args.model)
            cond = o.get("condition")
            if cond not in acc:
                continue
            acc[cond].add(o.get("is_correct", 0))

    prag = acc["pragmatic"].pct()
    lit = acc["literal"].pct()
    rand = 50.0

    lines: List[str] = []
    lines.append("# Appendix E.2 Context-Only Probe (遮住核心话语U，仅给语境C)\n\n")
    lines.append(
        "We hide the core utterance $U$ and provide models only the context $C$ plus the two candidate interpretations. "
        "If accuracy drops to the random baseline (50%), this suggests the dataset is not solvable via context-only spurious cues.\n\n"
    )
    lines.append(f"- Model: `{model_seen or args.model}`\n")
    lines.append(f"- Results file: `{results_path.name}`\n\n")

    lines.append("## Accuracy (context-only)\n")
    lines.append("| Condition | Accuracy (%) | N | Δ vs 50% |\n")
    lines.append("|---|---:|---:|---:|\n")
    lines.append(
        f"| pragmatic | {prag:.2f} | {acc['pragmatic'].total} | {prag - rand:+.2f} |\n"
    )
    lines.append(
        f"| literal | {lit:.2f} | {acc['literal'].total} | {lit - rand:+.2f} |\n"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

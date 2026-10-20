"""Build Appendix E.1: Prompt Robustness for Strict Prompt variants.

No-fabrication principle:
- Reads evaluation result JSONL files produced by scripts in this repo.
- Computes accuracies only from stored fields (is_correct, condition).

Expected inputs:
- results_{model}_strict_strict_base.jsonl
- results_{model}_strict_strict_robot.jsonl
- results_{model}_strict_strict_judge.jsonl
- results_{model}_strict_strict_compiler.jsonl
(or same with mode=cot if you run with --use_cot)

Output:
- appendix/E1_prompt_robustness.md

Metrics:
- Acc_prag, Acc_lit per variant
- Range across variants (max-min)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


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


def read_acc(path: Path) -> Dict[str, Acc]:
    out = {"pragmatic": Acc(), "literal": Acc()}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            cond = o.get("condition")
            if cond not in out:
                continue
            out[cond].add(o.get("is_correct", 0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model ID (used for default filenames)")
    ap.add_argument("--mode", default="strict", help="strict or cot")
    ap.add_argument(
        "--results_dir",
        default=str(ROOT),
        help="Directory containing results_*.jsonl (default: repo root)",
    )
    ap.add_argument(
        "--variants",
        default="strict_base,strict_robot,strict_judge,strict_compiler",
        help="Comma-separated variant names",
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "appendix" / "E1_prompt_robustness.md"),
        help="Output markdown path",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    safe_model = args.model.replace("/", "_")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    rows: List[Tuple[str, float, float, int]] = []  # variant, acc_prag, acc_lit, n_total

    for v in variants:
        p = results_dir / f"results_{safe_model}_{args.mode}_{v}.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"Missing results file: {p}")
        acc = read_acc(p)
        n_total = acc["pragmatic"].total + acc["literal"].total
        rows.append((v, acc["pragmatic"].pct(), acc["literal"].pct(), n_total))

    prag_vals = [r[1] for r in rows]
    lit_vals = [r[2] for r in rows]

    out_lines: List[str] = []
    out_lines.append("# Appendix E.1 Prompt Robustness (Strict Prompt Variants)\n")
    out_lines.append("This appendix reports prompt-variant robustness for Strict Prompting.\n")
    out_lines.append("All numbers are computed from stored evaluation outputs (no fabrication).\n\n")

    out_lines.append(f"- Model: `{args.model}`\n")
    out_lines.append(f"- Mode: `{args.mode}`\n")
    out_lines.append(f"- Variants: {', '.join('`'+v+'`' for v in variants)}\n\n")

    out_lines.append("## Overall accuracy by prompt variant\n")
    out_lines.append("| Variant | Acc_prag (%) | Acc_lit (%) | N (total) |\n")
    out_lines.append("|---|---:|---:|---:|\n")
    for v, apct, lpct, n in rows:
        out_lines.append(f"| `{v}` | {apct:.2f} | {lpct:.2f} | {n} |\n")

    out_lines.append("\n## Robustness summary\n")
    out_lines.append(f"- Acc_prag range across variants: **{max(prag_vals)-min(prag_vals):.2f}** (max={max(prag_vals):.2f}, min={min(prag_vals):.2f})\n")
    out_lines.append(f"- Acc_lit range across variants: **{max(lit_vals)-min(lit_vals):.2f}** (max={max(lit_vals):.2f}, min={min(lit_vals):.2f})\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(out_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Generate Table 1 (Taxonomy) for the PaCE paper.

Inputs:
- final_data/PaCE_Benchmark_Full_v1.jsonl

Outputs:
- paper_artifacts/tables/table1_taxonomy.csv
- paper_artifacts/tables/table1_taxonomy.tex

What this script does:
- Counts the number of unique sub-categories per dimension.
- Picks one representative example per dimension (common_utterance + sub_category).
- Leaves placeholders for Definition / Theory Source (to be filled manually from draft).

Design notes:
- We keep the dimension order fixed for readability.
- Example selection is deterministic (first occurrence by file order).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import pandas as pd


DIM_ORDER = [
    "Dim1_Implicature",
    "Dim2_SpeechActs",
    "Dim3_Presupposition",
    "Dim4_Deixis",
]

DIM_SHORT = {
    "Dim1_Implicature": "Implicature",
    "Dim2_SpeechActs": "Speech Acts",
    "Dim3_Presupposition": "Presupposition",
    "Dim4_Deixis": "Deixis",
}


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_ws(s: str) -> str:
    return " ".join((s or "").split())


def is_good_example(utt: str) -> bool:
    """Heuristic for paper-friendly examples (deterministic).

    Prefer short, readable, sentence-like examples for the taxonomy table.
    """
    u = normalize_ws(utt)
    if not u:
        return False
    if len(u) < 5:
        return False
    if len(u) > 42:
        return False
    bad = ["http://", "https://", "<", ">"]
    if any(b in u.lower() for b in bad):
        return False
    # Prefer clear sentence cues
    has_cue = any(ch in u for ch in ["?", "!", "."]) or u[0].isupper()
    if not has_cue:
        return False
    if u.lower() in {"yes.", "no.", "ok.", "okay."}:
        return False
    return True


def pick_best_example(candidates: List[Tuple[str, str]]) -> Tuple[str, str]:
    """Pick best (utterance, subcat) deterministically."""
    good = [(normalize_ws(u), s) for (u, s) in candidates if is_good_example(u)]
    if not good:
        non_empty = [(normalize_ws(u), s) for (u, s) in candidates if normalize_ws(u)]
        if not non_empty:
            return "", ""
        non_empty.sort(key=lambda x: (len(x[0]), x[0]))
        return non_empty[0]

    def key_fn(item: Tuple[str, str]) -> Tuple[int, int, str]:
        u, _ = item
        # Prefer questions/exclamations slightly
        bonus = 0
        if "?" in u:
            bonus -= 2
        if "!" in u:
            bonus -= 1
        return (len(u) + bonus, len(u), u)

    good.sort(key=key_fn)
    return good[0]


def escape_latex(s: str) -> str:
    # Minimal escaping for table cells.
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def to_booktabs_tex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    header = " & ".join(escape_latex(c) for c in cols) + r" \\"

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    # Compact layout (conference-style). Adjust widths if needed.
    lines.append(r"\begin{tabular}{l r p{4.3cm} p{2.6cm} p{3.4cm} p{2.6cm}}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                v = ""
            if isinstance(v, float):
                v = f"{v:.4f}".rstrip("0").rstrip(".")
            v = str(v)
            vals.append(escape_latex(v))
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{escape_latex(caption)}}}")
    lines.append(rf"\label{{{escape_latex(label)}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=r"final_data\\PaCE_Benchmark_Full_v1.jsonl",
        help="Benchmark JSONL path (relative to repo root).",
    )
    parser.add_argument(
        "--outdir",
        default=r"paper_artifacts\\tables",
        help="Output directory.",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Also generate paper-optimized Table1 (short columns, better examples).",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(os.path.normpath(args.input))
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    dim_to_subcats: Dict[str, set] = defaultdict(set)
    dim_to_candidates: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for obj in iter_jsonl(input_path):
        dim = str(obj.get("dimension", "")).strip()
        sub = str(obj.get("sub_category", "")).strip()
        utt = normalize_ws(str(obj.get("common_utterance", "")))
        if not dim:
            continue
        if sub:
            dim_to_subcats[dim].add(sub)
        dim_to_candidates[dim].append((utt, sub))

    rows: List[dict] = []
    for dim in DIM_ORDER:
        short = DIM_SHORT.get(dim, dim)
        n_sub = len(dim_to_subcats.get(dim, set()))
        utt, sub = pick_best_example(dim_to_candidates.get(dim, []))
        rows.append(
            {
                "Dimension": short,
                "#Subcats": n_sub,
                "Definition (placeholder)": "",
                "Theory Source (placeholder)": "",
                "Example Utterance": utt,
                "Example Subcat": sub,
            }
        )

    df = pd.DataFrame(rows)

    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, "table1_taxonomy.csv")
    out_tex = os.path.join(args.outdir, "table1_taxonomy.tex")

    df.to_csv(out_csv, index=False, encoding="utf-8")

    tex = to_booktabs_tex(
        df,
        caption="PaCE taxonomy (definitions and theory sources are filled from the draft).",
        label="tab:pace-taxonomy",
    )
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)

    # Paper-optimized version (recommended for manuscript)
    if args.paper:
        df_paper = df.rename(
            columns={
                "#Subcats": "Subcats",
                "Definition (placeholder)": "Definition",
                "Theory Source (placeholder)": "Theory",
                "Example Utterance": "Example",
                "Example Subcat": "Subcat",
            }
        )
        out_csv_paper = os.path.join(args.outdir, "table1_taxonomy_paper.csv")
        out_tex_paper = os.path.join(args.outdir, "table1_taxonomy_paper.tex")
        df_paper.to_csv(out_csv_paper, index=False, encoding="utf-8")
        tex_paper = to_booktabs_tex(
            df_paper,
            caption="PaCE taxonomy. Definitions and theory sources are from the draft.",
            label="tab:pace-taxonomy",
        )
        with open(out_tex_paper, "w", encoding="utf-8") as f:
            f.write(tex_paper)
        print("Saved:", out_csv_paper)
        print("Saved:", out_tex_paper)

    print("Saved:", out_csv)
    print("Saved:", out_tex)
    print("\nPreview (auto-selected examples):\n")
    print(df.to_string(index=False))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())

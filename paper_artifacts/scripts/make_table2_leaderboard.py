"""Generate Table 2 (Main Leaderboard) in an ACL-friendly style.

Input:
- test/result/PaCE_Leaderboard_Summary.csv

Outputs:
- paper_artifacts/tables/table2_leaderboard_paper.csv
- paper_artifacts/tables/table2_leaderboard_paper.tex

Formatting choices (ACL-friendly):
- Sort by Overall (descending).
- Columns: Model | Setting | Overall | Prag | Lit | CSG
  where CSG = (Lit - Prag) in percentage points.
- Acc values printed as percentages with 1 decimal.
- CSG printed as signed percentage points with 1 decimal.
- Optional multirow merge for models that have both direct & cot.

Note:
- Some models only have one setting; they appear as a single row.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

import pandas as pd


def escape_latex(s: str) -> str:
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
    return "".join(repl.get(ch, ch) for ch in str(s))


def fmt_pct(x: float) -> str:
    # 0.8467 -> 84.7
    return f"{x * 100:.1f}"


def fmt_pp(x: float) -> str:
    # -0.0970 -> -9.7
    return f"{x * 100:+.1f}"


def to_booktabs_tex(df: pd.DataFrame, caption: str, label: str, *, use_multirow: bool) -> str:
    cols = list(df.columns)

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    # l for model, c for setting, r for numbers
    lines.append(r"\begin{tabular}{l c r r r r}")
    lines.append(r"\toprule")
    header = " & ".join(escape_latex(c) for c in cols) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    if not use_multirow:
        for _, row in df.iterrows():
            vals = [escape_latex(row[c]) for c in cols]
            lines.append(" & ".join(vals) + r" \\")
    else:
        # Multirow: merge Model cells for same model (expect up to 2 settings)
        for model, g in df.groupby("Model", sort=False):
            g2 = g.reset_index(drop=True)
            if len(g2) == 1:
                r0 = g2.loc[0]
                vals = [escape_latex(r0[c]) for c in cols]
                lines.append(" & ".join(vals) + r" \\")
                continue

            # Prefer ordering: direct then cot
            order = {"direct": 0, "cot": 1}
            g2 = g2.sort_values("Setting", key=lambda s: s.map(order).fillna(9)).reset_index(drop=True)

            first = True
            for i in range(len(g2)):
                r = g2.loc[i]
                row_cells = []
                if first:
                    row_cells.append(r"\multirow{" + str(len(g2)) + r"}{*}{" + escape_latex(model) + r"}")
                    first = False
                else:
                    row_cells.append("")

                for c in cols[1:]:
                    row_cells.append(escape_latex(r[c]))
                lines.append(" & ".join(row_cells) + r" \\")

            lines.append(r"\addlinespace[2pt]")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{escape_latex(caption)}}}")
    lines.append(rf"\label{{{escape_latex(label)}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=r"test\\result\\PaCE_Leaderboard_Summary.csv",
        help="Path to PaCE_Leaderboard_Summary.csv (relative to repo root).",
    )
    parser.add_argument(
        "--outdir",
        default=r"paper_artifacts\\tables",
        help="Output directory.",
    )
    parser.add_argument(
        "--with-dims",
        action="store_true",
        help="Include per-dimension Acc columns (Implicature/SpeechActs/Presupposition/Deixis) (optional).",
    )
    parser.add_argument(
        "--multirow",
        action="store_true",
        help="Use \\multirow to merge the Model column for direct/cot pairs (requires LaTeX multirow package).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    core = df[["Model", "Mode", "Overall Acc", "Pragmatic Acc", "Literal Acc", "GAP (Lit-Prag)"]].copy()
    core = core.rename(columns={"Mode": "Setting"})

    # Sorting: Overall desc, then model name
    core = core.sort_values(["Overall Acc", "Model", "Setting"], ascending=[False, True, True]).reset_index(drop=True)

    out = pd.DataFrame(
        {
            "Model": core["Model"].astype(str),
            "Setting": core["Setting"].astype(str),
            "Overall": core["Overall Acc"].astype(float).map(fmt_pct),
            "Prag": core["Pragmatic Acc"].astype(float).map(fmt_pct),
            "Lit": core["Literal Acc"].astype(float).map(fmt_pct),
            "CSG": core["GAP (Lit-Prag)"].astype(float).map(fmt_pp),
        }
    )

    if args.with_dims:
        dims = [
            ("Imp", "Implicature Acc"),
            ("SA", "SpeechActs Acc"),
            ("Pre", "Presupposition Acc"),
            ("Dei", "Deixis Acc"),
        ]
        for short, col in dims:
            if col in df.columns:
                # Need to align by the same row order as core
                out[short] = core.join(df[[col]], how="left")[col].astype(float).map(fmt_pct)

    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, "table2_leaderboard_paper.csv")
    out_tex = os.path.join(args.outdir, "table2_leaderboard_paper.tex")

    out.to_csv(out_csv, index=False, encoding="utf-8")

    caption = "Main leaderboard on PaCE. Accuracies are in %, and CSG is Lit−Prag in percentage points (pp)."
    tex = to_booktabs_tex(out, caption=caption, label="tab:pace-leaderboard", use_multirow=bool(args.multirow))
    if args.multirow:
        tex = "% Requires \\usepackage{multirow}\n" + tex

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)

    print("Saved:", out_csv)
    print("Saved:", out_tex)
    print("\nPreview (top 10):\n")
    print(out.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

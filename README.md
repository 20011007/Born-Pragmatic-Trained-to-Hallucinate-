# PaCE Dataset and Evaluation Usage Manual

This package contains the English PaCE benchmark and its paper-related evaluation materials, together with a separately stored Chinese supplemental experiment.

## Package contents

| Path | Contents |
|---|---|
| `final_data/PaCE_Benchmark_Full_v1.jsonl` | Canonical English benchmark: 3,215 items with unique IDs. Each item includes a pragmatic case and a literal case. |
| `test/result/data/` | 24 English per-item result files. Each file contains 6,430 records: one pragmatic and one literal record for each of the 3,215 benchmark IDs. |
| `test/result/PaCE_Leaderboard_Summary.csv` | A 17-row summary of the direct and chain-of-thought leaderboard settings. The `Filename` column points to the corresponding per-item result file. |
| `test/` | Threaded English evaluation runners for general, strict-prompt, and context-only settings. |
| `paper_artifacts/scripts/` | Scripts for building the paper taxonomy and leaderboard tables. |
| `paper_artifacts/tables/` | Prepared taxonomy table files in CSV and LaTeX formats. |
| `scripts/` | Scripts for building the Appendix E1 prompt-robustness and E2 context-only summary artifacts from evaluation results. |
| `supplemental/chinese/chinese.json` | Separate Chinese supplemental input: 500 records. Despite the extension, it is line-delimited JSON (one JSON object per line). |
| `supplemental/chinese/results_zhongwen_eval.jsonl` | Per-item results for the Chinese supplemental experiment: 500 records. |
| `supplemental/chinese/evaluate_model_zhongwen.py` | Runner for evaluating a model on the Chinese supplemental input. |

## English benchmark format

The canonical benchmark is JSONL. Each line is one item with these top-level fields:

- `id`: item identifier.
- `dimension`, `phenomenon`, `sub_category`: taxonomy labels.
- `common_utterance`: the utterance shared by the paired cases.
- `case_pragmatic`, `case_literal`: the two context-flip cases.

Each case contains `context`, `speaker`, `listener`, `correct_option`, and a distractor field. The pragmatic case uses `distractor_literal`; the literal case uses `distractor_pragmatic`. The gold response is given by `correct_option`.

The benchmark has four dimensions: Implicature (982 items), Speech Acts (505), Presupposition (781), and Deixis (947). Together they account for all 3,215 English items. The two cases form a paired evaluation; they should not be treated as two unrelated benchmark items.

English result files are also JSONL. A result record includes the item `id`, `dimension`, `sub_category`, `condition`, `model`, `mode`, `ground_truth`, `prediction`, `is_correct`, and `raw_response`. The result files retain the model response alongside the normalized prediction and correctness label.

## Chinese supplemental data

The Chinese experiment is provided separately so it can be used as an additional cross-lingual experiment without being merged into the English benchmark or its leaderboard. Its input records use the same paired-case structure. The Chinese result records include pragmatic and literal predictions, labels, and correctness fields.

The Chinese input contains 500 records and 496 distinct values in its `id` field. When matching input and result rows, use both `id` and `utterance`; the ID alone is not unique for this subset.

## Reusing the supplied results and tables

Run commands from the package root. The paper table builders use Python and pandas:

```bash
python paper_artifacts/scripts/make_table1_taxonomy.py --input final_data/PaCE_Benchmark_Full_v1.jsonl --outdir paper_artifacts/tables
python paper_artifacts/scripts/make_table2_leaderboard.py --csv test/result/PaCE_Leaderboard_Summary.csv --outdir paper_artifacts/tables
```

The Appendix E1 and E2 builders accept input and output paths. Run either with `--help` to see its available options:

```bash
python scripts/build_e1_prompt_robustness_table.py --help
python scripts/build_e2_context_only_probe_md.py --help
```

## Running model evaluations

The evaluation runners call an OpenAI-compatible model API. They require the `openai` and `tqdm` Python packages; the table builder scripts also require `pandas`. From the package root, use `--help` for a runner's current options:

```bash
python test/run_pace_eval_universal_threaded.py --help
python test/run_pace_eval_strict_variants_threaded.py --help
python test/run_pace_eval_context_only_threaded.py --help
python supplemental/chinese/evaluate_model_zhongwen.py --help
```

For the English runners, explicitly set `--input_file` to `final_data/PaCE_Benchmark_Full_v1.jsonl`. The runners accept model and endpoint configuration and write evaluation outputs to local result files. 

## Privacy and API configuration


## Dataset and software terms



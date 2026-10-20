"""E.2 Context-Only Probe (遮住核心话语 U，仅给语境 C)

目的（论文附录额外实验）：
- 把 PaCE item 的核心 utterance U（final.jsonl 的 common_utterance）遮住，只给 context C。
- 仍然保留 A/B 两个解释选项，让模型“盲猜”哪一个更符合语境。
- 如果遮住 U 后准确率接近随机水平（50%），可作为证据：
  1) 数据集不是靠语境的虚假相关性就能被轻易猜中；
  2) 模型在完整输入下的高准确率更可能来自对 U+C 的推理，而非泄露。

No-fabrication 原则：
- options/ground_truth 来自 `test/final.jsonl` 已存字段。
- 输出结果 JSONL 仅记录模型原始输出与解析得到的 A/B。

输出 JSONL schema（与仓库其它脚本保持对齐，便于复用汇总器）：
{ id, dimension, sub_category, condition, model, mode, probe,
  ground_truth, prediction, is_correct, raw_response }

注意：
- 该 probe 与 Strict Prompting 无关，因此 mode 固定为 `context_only`。
- 为了避免格式污染，默认让模型只输出 A 或 B。

"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import openai
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]


def _is_auth_error(e: Exception) -> bool:
    s = str(e)
    return ("Error code: 401" in s) or ("Authentication Error" in s) or ("token_not_found" in s)


SYSTEM_PROMPT = (
    "You are a careful evaluator. You will be given ONLY the context, and two candidate interpretations. "
    "The original utterance is hidden. Choose the option that is more plausible given the context alone. "
    "Reply ONLY with 'A' or 'B'."
)


def load_jsonl(path: Path, limit: int = 0) -> List[Dict]:
    data: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
            if limit and len(data) >= limit:
                break
    return data


def safe_model_name(model: str) -> str:
    return model.replace("/", "_")


def parse_response(text: str) -> str:
    if not text:
        return "UNKNOWN"

    t = text.strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()

    m = re.match(r"^[\s\W]*([AB])[\s\W]*$", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"Answer[:\s]*([AB])", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    if len(t) < 50:
        ms = re.findall(r"\b([AB])\b", t, re.IGNORECASE)
        if ms:
            return ms[-1].upper()

    return "UNKNOWN"


def build_context_only_prompt(item: Dict, condition: str) -> Tuple[str, str, str]:
    # 仍沿用 PaCE 选项构造，保证 GT 与主实验一致：
    # pragmatic: A=correct_option, B=distractor_literal, GT=A
    # literal:   A=distractor_pragmatic, B=correct_option, GT=B
    if condition == "pragmatic":
        context_text = item["case_pragmatic"]["context"]
        opt_a = item["case_pragmatic"]["correct_option"]
        opt_b = item["case_pragmatic"]["distractor_literal"]
        gt = "A"
    else:
        context_text = item["case_literal"]["context"]
        opt_a = item["case_literal"]["distractor_pragmatic"]
        opt_b = item["case_literal"]["correct_option"]
        gt = "B"

    user_prompt = (
        f"Context: {context_text}\n\n"
        "[UTTERANCE HIDDEN]\n\n"
        "Question: Which interpretation is more plausible given the context alone?\n\n"
        f"Option A: {opt_a}\n"
        f"Option B: {opt_b}\n\n"
        "Reply ONLY with 'A' or 'B'.\n\n"
        "Answer:"
    )
    return SYSTEM_PROMPT, user_prompt, gt


def iter_processed_keys(results_path: Path) -> Set[Tuple[str, str]]:
    processed: Set[Tuple[str, str]] = set()
    if not results_path.exists():
        return processed
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            processed.add((o.get("id"), o.get("condition")))
    return processed


def call_model(client: openai.OpenAI, model: str, sys_prompt: str, user_prompt: str) -> str:
    while True:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=5,
            )
            return resp.choices[0].message.content if resp.choices else ""
        except Exception as e:
            if _is_auth_error(e):
                print(
                    "⚠️ API Error: 401 Authentication failed. "
                    "Likely api_key does NOT match base_url (proxy token mismatch).\n"
                    "⚠️ Fix: use the key issued for the selected base_url (e.g., https://yunwu.ai/v1)."
                )
                raise
            print(f"⚠️ API Error: {e}, retrying...")
            time.sleep(2)


def process_one_item(
    item: Dict,
    conditions: Iterable[str],
    args: argparse.Namespace,
    client: openai.OpenAI | None,
) -> List[Dict]:
    out: List[Dict] = []

    for cond in conditions:
        sys_prompt, user_prompt, gt = build_context_only_prompt(item, cond)

        if args.dry_run:
            raw = "A"
        else:
            raw = call_model(client, args.model, sys_prompt, user_prompt)  # type: ignore[arg-type]

        pred = "A" if args.dry_run else parse_response(raw)

        out.append(
            {
                "id": item.get("id"),
                "dimension": item.get("dimension"),
                "sub_category": item.get("sub_category"),
                "condition": cond,
                "model": args.model,
                "mode": "context_only",
                "probe": "E2_context_only",
                "ground_truth": gt,
                "prediction": pred,
                "is_correct": 1 if pred == gt else 0,
                "raw_response": raw,
            }
        )

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt-4o", help="Model ID (default: gpt-4o)")
    ap.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="API Key (or set OPENAI_API_KEY)")
    ap.add_argument(
        "--base_url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL") or "https://yunwu.ai/v1",
        help="API Base URL (or set OPENAI_BASE_URL). Default: https://yunwu.ai/v1",
    )
    ap.add_argument(
        "--input_file",
        type=str,
        default=str(ROOT / "test" / "final.jsonl"),
        help="Path to final.jsonl",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSONL (default: results_{model}_context_only_E2.jsonl at repo root)",
    )
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only_condition", type=str, default="all", help="pragmatic|literal|all")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.only_condition not in {"pragmatic", "literal", "all"}:
        raise SystemExit("--only_condition must be pragmatic|literal|all")

    if not args.dry_run:
        if not args.api_key:
            raise SystemExit("Missing --api_key (or OPENAI_API_KEY)")
        if not args.base_url:
            raise SystemExit("Missing --base_url (or OPENAI_BASE_URL)")

    conditions = ["pragmatic", "literal"] if args.only_condition == "all" else [args.only_condition]

    model_safe = safe_model_name(args.model)
    results_path = Path(args.out) if args.out else ROOT / f"results_{model_safe}_context_only_E2.jsonl"

    data = load_jsonl(Path(args.input_file), limit=args.limit)
    if not data:
        raise SystemExit(f"No data loaded from: {args.input_file}")

    processed = iter_processed_keys(results_path)

    client = None if args.dry_run else openai.OpenAI(api_key=args.api_key, base_url=args.base_url)

    # 任务列表（item, cond）粒度，便于断点续跑
    tasks: List[Tuple[Dict, List[str]]] = []
    for item in data:
        missing = [c for c in conditions if (item.get("id"), c) not in processed]
        if missing:
            tasks.append((item, missing))

    if not tasks:
        print(f"No pending tasks. Results already complete: {results_path}")
        return

    results_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = results_path.with_suffix(results_path.suffix + ".lock")
    if lock_path.exists() and not args.dry_run:
        print(f"⚠️ Lock file exists: {lock_path}. If no other run is active, delete it.")

    # 追加写
    with results_path.open("a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            futures = [ex.submit(process_one_item, item, miss, args, client) for item, miss in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="E2 context-only"):
                rows = fut.result()
                for r in rows:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    fout.flush()


if __name__ == "__main__":
    main()

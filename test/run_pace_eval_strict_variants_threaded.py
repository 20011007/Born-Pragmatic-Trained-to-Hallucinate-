import json
import time
import re
import os
import argparse
from tqdm import tqdm
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed


def _is_auth_error(e: Exception) -> bool:
    s = str(e)
    return ("Error code: 401" in s) or ("Authentication Error" in s) or ("token_not_found" in s)

# ======================================================
# E.1 Prompt Robustness (Strict Prompt Variants)
#
# Goal (no fabrication): evaluate whether Strict Prompting gains are robust to
# 2-3 semantically equivalent personas.
#
# Variants:
# - strict_base: baseline strict prompt (from run_pace_experiments.py)
# - strict_robot: persona "robot" + strict literal constraints
# - strict_judge: persona "judge" + strict literal constraints
# - strict_compiler: persona "compiler" + strict literal constraints
#
# Output JSONL schema aligns with other result files:
# {id, dimension, sub_category, condition, model, mode, prompt_variant,
#  ground_truth, prediction, is_correct, raw_response}
#
# Note: In PaCE `final.jsonl`, options are built as:
# pragmatic:  A=correct_option, B=distractor_literal ; ground_truth=A
# literal:    A=distractor_pragmatic, B=correct_option ; ground_truth=B
# ======================================================

STRICT_BASE_SYSTEM_PROMPT = """You are a literal-minded AI assistant designed for rigorous logical and technical analysis.
Your task is to interpret the utterance STRICTLY based on its literal definition and the provided context.
- IGNORE all social implications, conversational norms, or polite indirectness.
- If the context is technical, legal, or logical, focus ONLY on the factual truth conditions.
- Do not \"read between the lines\". Do not hallucinate meanings that are not explicitly stated."""

STRICT_ROBOT_SYSTEM_PROMPT = """You are an autonomous robot language module.
Operate in STRICT literal interpretation mode:
- Use ONLY what is explicitly stated in the utterance and the provided context.
- IGNORE social implications, politeness, indirectness, and conversational norms.
- Do not infer unstated intentions.
Output compliance is mandatory."""

STRICT_JUDGE_SYSTEM_PROMPT = """You are a courtroom judge deciding interpretations under a strict evidential standard.
Rules:
- Admit only evidence explicitly present in the utterance and the provided context.
- IGNORE pragmatic implicatures, politeness, and conversational norms.
- Do not speculate beyond the record.
Output compliance is mandatory."""

STRICT_COMPILER_SYSTEM_PROMPT = """You are a deterministic compiler for natural language.
Compilation target: a strictly literal proposition/meaning.
Constraints:
- Use ONLY explicit lexical meaning + explicit context.
- IGNORE implicature, politeness, and unstated intentions.
- Do not add any new facts.
Output compliance is mandatory."""

PROMPT_VARIANTS = {
    "strict_base": STRICT_BASE_SYSTEM_PROMPT,
    "strict_robot": STRICT_ROBOT_SYSTEM_PROMPT,
    "strict_judge": STRICT_JUDGE_SYSTEM_PROMPT,
    "strict_compiler": STRICT_COMPILER_SYSTEM_PROMPT,
}


def construct_prompt(item, condition: str, prompt_variant: str, use_cot: bool = False):
    utterance = item["common_utterance"]

    # PaCE option construction (same as existing eval scripts)
    if condition == "pragmatic":
        context_text = item["case_pragmatic"]["context"]
        opt_a = item["case_pragmatic"]["correct_option"]
        opt_b = item["case_pragmatic"]["distractor_literal"]
        ground_truth = "A"
    else:
        context_text = item["case_literal"]["context"]
        opt_a = item["case_literal"]["distractor_pragmatic"]
        opt_b = item["case_literal"]["correct_option"]
        ground_truth = "B"

    system_prompt = PROMPT_VARIANTS[prompt_variant]

    if use_cot:
        system_prompt += "\nYou must think step-by-step before answering."
        user_prompt = (
            f"Context: {context_text}\n"
            f"Speaker says: \"{utterance}\"\n\n"
            f"Question: What does the speaker mean?\n\n"
            f"Option A: {opt_a}\n"
            f"Option B: {opt_b}\n\n"
            "First, analyze the context and the literal meaning step-by-step.\n"
            "Then, state your final answer.\n"
            "Format your output as:\n"
            "Reasoning: [Your reasoning]\n"
            "Answer: [Option Letter]\n\n"
            "You must reply ONLY with the single letter 'A' or 'B' on the last line."
        )
    else:
        user_prompt = (
            f"Context: {context_text}\n"
            f"Speaker says: \"{utterance}\"\n\n"
            f"Question: What does the speaker mean?\n\n"
            f"Option A: {opt_a}\n"
            f"Option B: {opt_b}\n\n"
            "Please select the best option. Reply ONLY with the letter \"A\" or \"B\". Do NOT output any explanation or reasoning.\n\n"
            "Answer:"
        )

    return system_prompt, user_prompt, ground_truth


def parse_response(response_text: str):
    if not response_text:
        return "UNKNOWN"
    text = response_text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    match = re.match(r"^[\s\W]*([AB])[\s\W]*$", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"Answer[:\s]*([AB])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"Option[:\s]*([AB])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    if len(text) < 50:
        matches = re.findall(r"\b([AB])\b", text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    return "UNKNOWN"


def process_one(item, conditions, prompt_variant, client, args):
    results = []
    for condition in conditions:
        sys_prompt, user_prompt, ground_truth = construct_prompt(
            item=item,
            condition=condition,
            prompt_variant=prompt_variant,
            use_cot=args.use_cot,
        )

        if args.dry_run:
            response_text = "A"  # deterministic placeholder; not used for real metrics
        else:
            max_tokens = 4096 if args.use_cot else 50
            while True:
                try:
                    resp = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                        max_tokens=max_tokens,
                    )
                    response_text = resp.choices[0].message.content if resp.choices else ""
                except Exception as e:
                    if _is_auth_error(e):
                        print(
                            "⚠️ API Error: 401 Authentication failed. "
                            "Likely --api_key does NOT match --base_url (proxy token mismatch).\n"
                            "⚠️ Fix: use the key issued for the selected base_url (e.g., https://yunwu.ai/v1)."
                        )
                        raise
                    print(f"⚠️ API Error: {e}, retrying...")
                    time.sleep(2)
                    continue

                prediction = parse_response(response_text)
                if prediction != "UNKNOWN":
                    break
                print(f"⚠️ Invalid: {prediction}, Text: {str(response_text)[:80]}...")
                time.sleep(1)

        prediction = parse_response(response_text) if not args.dry_run else "A"

        results.append(
            {
                "id": item["id"],
                "dimension": item.get("dimension"),
                "sub_category": item.get("sub_category"),
                "condition": condition,
                "model": args.model,
                "mode": "cot" if args.use_cot else "strict",
                "prompt_variant": prompt_variant,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "is_correct": 1 if prediction == ground_truth else 0,
                "raw_response": response_text,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model ID")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="API Key (or set OPENAI_API_KEY)")
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL") or "https://yunwu.ai/v1",
        help="API Base URL (or set OPENAI_BASE_URL). Default: https://yunwu.ai/v1",
    )
    parser.add_argument("--input_file", type=str, default="final.jsonl", help="Input file")
    parser.add_argument("--prompt_variant", type=str, default="all", help="Variant name or 'all'")
    parser.add_argument("--use_cot", action="store_true", help="Enable CoT")
    parser.add_argument("--threads", type=int, default=4, help="Threads")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of items (0 means no limit)")
    parser.add_argument("--dry_run", action="store_true", help="No API call; emits dummy outputs for pipeline test")
    args = parser.parse_args()

    if not args.dry_run:
        if not args.api_key:
            raise SystemExit("Missing --api_key (or OPENAI_API_KEY)")
        if not args.base_url:
            raise SystemExit("Missing --base_url (or OPENAI_BASE_URL)")

    client = None if args.dry_run else openai.OpenAI(api_key=args.api_key, base_url=args.base_url)

    variants = list(PROMPT_VARIANTS.keys()) if args.prompt_variant == "all" else [args.prompt_variant]
    for v in variants:
        if v not in PROMPT_VARIANTS:
            raise SystemExit(f"Unknown prompt_variant: {v}. Choose from {list(PROMPT_VARIANTS.keys())} or 'all'.")

    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    if args.limit and args.limit > 0:
        data = data[: args.limit]

    from collections import defaultdict

    for prompt_variant in variants:
        mode_tag = "cot" if args.use_cot else "strict"
        safe_model = args.model.replace("/", "_")
        output_file = f"results_{safe_model}_{mode_tag}_{prompt_variant}.jsonl"

        processed_keys = set()
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        res = json.loads(line)
                        processed_keys.add(f"{res['id']}_{res['condition']}")
                    except Exception:
                        pass

        item_group = defaultdict(list)
        for item in data:
            for condition in ["pragmatic", "literal"]:
                if f"{item['id']}_{condition}" not in processed_keys:
                    item_group[item["id"]].append(condition)

        items_for_thread = [(item, conds) for item in data if item["id"] in item_group for conds in [item_group[item["id"]]]]

        print(f"🚀 Evaluation: {args.model} | Variant: {prompt_variant} | Mode: {mode_tag}")
        print(f"💾 Saving to: {output_file} (Real-time)")
        print(f"🔄 Resuming... Skipping {len(processed_keys)} processed items.")

        with open(output_file, "a", encoding="utf-8") as f_out:
            with ThreadPoolExecutor(max_workers=args.threads) as executor:
                futures = [executor.submit(process_one, item, conds, prompt_variant, client, args) for item, conds in items_for_thread]
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"{prompt_variant}"):
                    try:
                        results = future.result()
                        for res in results:
                            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                        f_out.flush()
                    except Exception as e:
                        print(f"Thread Error: {e}")

    print("✅ Done!")


if __name__ == "__main__":
    main()

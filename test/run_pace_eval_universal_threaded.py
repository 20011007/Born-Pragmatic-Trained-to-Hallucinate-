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

# ================= 📝 Prompt 构造工厂 =================
def construct_prompt(item, condition, use_cot=False):
    utterance = item['common_utterance']
    if condition == "pragmatic":
        context_text = item['case_pragmatic']['context']
        opt_a = item['case_pragmatic']['correct_option']
        opt_b = item['case_pragmatic']['distractor_literal']
        ground_truth = "A"
    else:
        context_text = item['case_literal']['context']
        opt_a = item['case_literal']['distractor_pragmatic']
        opt_b = item['case_literal']['correct_option']
        ground_truth = "B"
    
    system_prompt = "You are a logic and language expert."
    
    if use_cot:
        system_prompt += " You must think step-by-step before answering."
        user_prompt = f"""Context: {context_text}\nSpeaker says: "{utterance}"\n\nQuestion: What does the speaker mean?\n\nOption A: {opt_a}\nOption B: {opt_b}\n\nFirst, analyze the context and the literal vs. implied meaning step-by-step.\nThen, state your final answer.\nFormat your output as:\nReasoning: [Your reasoning]\nAnswer: [Option Letter]\n\nYou must reply ONLY with the single letter 'A' or 'B' on the last line."""
    else:
        # Direct 模式：增强 Prompt，防止废话
        user_prompt = f"""Context: {context_text}\nSpeaker says: "{utterance}"\n\nQuestion: What does the speaker mean?\n\nOption A: {opt_a}\nOption B: {opt_b}\n\nPlease select the best option. Reply ONLY with the letter "A" or "B". Do NOT output any explanation or reasoning.\n\nAnswer:"""
    
    return system_prompt, user_prompt, ground_truth

def parse_response(response_text, use_cot=False):
    if not response_text:
        return "UNKNOWN"
    text = response_text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # 1. 优先匹配纯字母
    match = re.match(r"^[\s\W]*([AB])[\s\W]*$", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    # 2. 匹配 Answer: A 格式
    match = re.search(r"Answer[:\s]*([AB])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    # 3. 匹配 Option A 格式
    match = re.search(r"Option[:\s]*([AB])", text, re.IGNORECASE)
    if match: return match.group(1).upper()

    # 4. 暴力兜底：文本很短时，取最后一个出现的 A 或 B
    if len(text) < 50:
        matches = re.findall(r"\b([AB])\b", text, re.IGNORECASE)
        if matches: return matches[-1].upper()

    return "UNKNOWN"

def process_item(item, conditions, client, args):
    results = []
    for condition in conditions:
        sys_prompt, user_prompt, ground_truth = construct_prompt(item, condition, args.use_cot)
        
        # 动态 Token 设置：Claude 虽然很乖，但 Direct 模式建议给 100 以防万一
        max_tokens = 4096 if args.use_cot else 100
        
        while True:
            response_text = "ERROR"
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens 
                )
                
                # 注释掉 DEBUG print，防止干扰进度条
                # print("DEBUG resp:", resp) 

                # 提取 Content
                try:
                    response_text = resp.choices[0].message.content
                except:
                    response_text = None
                
                # 兜底：防止 API 返回非标准结构
                if not response_text:
                    response_text = str(resp)

            except Exception as e:
                if _is_auth_error(e):
                    print(
                        "⚠️ API Error: 401 Authentication failed. "
                        "This usually means your --api_key does NOT match your --base_url (proxy token mismatch).\n"
                        "⚠️ Fix: use the key issued for the selected base_url (you said to use https://yunwu.ai/v1)."
                    )
                    raise
                print(f"⚠️ API Error: {e}, retrying...")
                time.sleep(2)
                continue
            
            prediction = parse_response(response_text, args.use_cot)
            
            if response_text != "ERROR" and prediction != "UNKNOWN":
                break
            
            # 如果解析失败，打印预览
            print(f"⚠️ Invalid: {prediction}, Text: {str(response_text)[:50]}...")
            time.sleep(1)

        result = {
            "id": item['id'],
            "dimension": item.get('dimension'),
            "sub_category": item.get('sub_category'),
            "condition": condition,
            "model": args.model,
            "mode": "cot" if args.use_cot else "direct",
            "ground_truth": ground_truth,
            "prediction": prediction,
            "is_correct": 1 if prediction == ground_truth else 0,
            "raw_response": response_text
        }
        results.append(result)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model ID")
    parser.add_argument("--api_key", type=str, required=True, help="API Key")
    parser.add_argument("--base_url", type=str, required=True, help="API Base URL")
    parser.add_argument("--input_file", type=str, default="final.jsonl", help="Input file")
    parser.add_argument("--use_cot", action="store_true", help="Enable CoT")
    parser.add_argument("--threads", type=int, default=4, help="Threads")
    args = parser.parse_args()

    client = openai.OpenAI(api_key=args.api_key, base_url=args.base_url)
    mode_tag = "cot" if args.use_cot else "direct"
    safe_model_name = args.model.replace("/", "_")
    output_file = f"results_{safe_model_name}_{mode_tag}.jsonl"

    print(f"🚀 Evaluation: {args.model} | Mode: {mode_tag}")
    print(f"💾 Saving to: {output_file} (Real-time)")

    data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    # 断点续传逻辑
    processed_keys = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    res = json.loads(line)
                    processed_keys.add(f"{res['id']}_{res['condition']}")
                except: pass
    print(f"🔄 Resuming... Skipping {len(processed_keys)} processed items.")

    items_to_process = []
    from collections import defaultdict
    item_group = defaultdict(list)
    
    # 找出没跑完的任务
    for item in data:
        for condition in ["pragmatic", "literal"]:
            if f"{item['id']}_{condition}" not in processed_keys:
                item_group[item['id']].append(condition)
    
    items_for_thread = [(item, conds) for item in data if item['id'] in item_group for conds in [item_group[item['id']]]]

    # 【核心修改】实时写入文件
    with open(output_file, 'a', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_item, item, conds, client, args) for item, conds in items_for_thread]
            
            # 使用 tqdm 包装 futures
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                try:
                    results = future.result()
                    # 立即写入结果
                    for res in results:
                        f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f_out.flush() # 强制刷新缓冲区，确保保存
                except Exception as e:
                    print(f"Thread Error: {e}")

    print("✅ Done!")

if __name__ == "__main__":
    main()
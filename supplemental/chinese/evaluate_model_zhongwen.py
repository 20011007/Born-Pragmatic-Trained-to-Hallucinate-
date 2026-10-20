import json
import os
import argparse
from openai import OpenAI
from tqdm import tqdm
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置默认参数，也可以通过命令行传参
DEFAULT_MODEL = "gpt-4o"
DEFAULT_BASE_URL = "https://yunwu.ai/v1"
DEFAULT_INPUT_FILE = "chinese.json"
DEFAULT_OUTPUT_FILE = "results_zhongwen_eval.jsonl"

def get_client(base_url, api_key):
    # 增加 timeout 和 max_retries
    return OpenAI(base_url=base_url, api_key=api_key, timeout=90.0, max_retries=5)

def query_model(client, model, context, utterance, option_a, option_b):
    # 构建 Prompt
    prompt = f"""
阅读下面的背景（Context）和说话人的话语（Utterance）。请选择对话语最恰当的理解。

背景：{context}
说话人说：“{utterance}”

选项 A：{option_a}
选项 B：{option_b}

请仅回复字母 "A" 或 "B"。不要解释。
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        
        # 简单解析
        if content.startswith("A") or "A" in content[:2]: return "A"
        if content.startswith("B") or "B" in content[:2]: return "B"
        return "Error"
    except Exception as e:
        # print(f"Error querying model: {e}") # 减少错误输出，避免刷屏
        return "Error"

def process_item(item, client, model):
    # 处理单条数据的函数，用于多线程并发
    
    # -----------------------------------
    # 1. Pragmatic Case (语用场景)
    # -----------------------------------
    p_label = "A"
    p_ctx = item['case_pragmatic']['context']
    p_opt_a = item['case_pragmatic']['correct_option']
    p_opt_b = item['case_pragmatic']['distractor_literal']
    
    if random.random() > 0.5:
        p_opt_a, p_opt_b = p_opt_b, p_opt_a
        p_label = "B"

    pred_p = query_model(client, model, p_ctx, item['common_utterance'], p_opt_a, p_opt_b)
    is_correct_p = 1 if pred_p == p_label else 0
    
    # -----------------------------------
    # 2. Literal Case (字面场景)
    # -----------------------------------
    l_label = "A"
    l_ctx = item['case_literal']['context']
    l_opt_a = item['case_literal']['correct_option']
    l_opt_b = item['case_literal']['distractor_pragmatic']
    
    if random.random() > 0.5:
        l_opt_a, l_opt_b = l_opt_b, l_opt_a
        l_label = "B"

    pred_l = query_model(client, model, l_ctx, item['common_utterance'], l_opt_a, l_opt_b)
    is_correct_l = 1 if pred_l == l_label else 0

    return {
        "id": item['id'],
        "utterance": item['common_utterance'],
        "pragmatic_correct": is_correct_p,
        "literal_correct": is_correct_l,
        "prediction_pragmatic": pred_p,
        "prediction_literal": pred_l,
        "label_pragmatic": p_label,
        "label_literal": l_label
    }

def evaluate(args):
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") 
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    
    if not api_key:
        print("错误：未找到 API Key。请设置 OPENAI_API_KEY 环境变量或通过 --api_key 参数传入。")
        return

    client = get_client(base_url, api_key)
    
    data_path = os.path.join(os.path.dirname(__file__), args.input_file)
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f if line.strip()]
        
    print(f"🚀 开始评测中文数据 (并行模式)，共 {len(dataset)} 条...")
    print(f"Model: {args.model} | Threads: {args.threads}")
    
    output_path = os.path.join(os.path.dirname(__file__), args.output_file)
    # 清空输出文件
    with open(output_path, "w", encoding="utf-8") as f:
        pass

    results = []
    correct_p = 0
    correct_l = 0

    # 使用 ThreadPoolExecutor 进行并发
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # 提交所有任务
        futures = {executor.submit(process_item, item, client, args.model): item for item in dataset}
        
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(futures), total=len(dataset), desc="Testing"):
            try:
                res = future.result()
                results.append(res)
                correct_p += res['pragmatic_correct']
                correct_l += res['literal_correct']

                # 实时写入结果，防止程序崩溃丢失数据
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Task generated an exception: {e}")

    # 打印最终统计
    if len(results) > 0:
        acc_p = correct_p / len(results)
        acc_l = correct_l / len(results)
        
        print(f"\n📊 评测结果 ({args.model}):")
        print(f"语用场景准确率 (Pragmatic Accuracy): {acc_p:.2%}")
        print(f"字面场景准确率 (Literal Accuracy):   {acc_l:.2%}")
        print(f"语用幻觉偏差 (Hallucination Bias):   {acc_p - acc_l:.2%}")
    else:
        print("\n沒有成功产生结果。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PaCE Evaluation on Chinese Data (Threaded)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--api_key", type=str, default=None, help="API Key")
    parser.add_argument("--base_url", type=str, default=None, help="Base URL")
    parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE, help="Input file")
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE, help="Output file")
    parser.add_argument("--threads", type=int, default=5, help="Number of concurrent threads")
    
    args = parser.parse_args()
    evaluate(args)

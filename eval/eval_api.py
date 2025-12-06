import os, json, argparse, time, base64
from typing import Union
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class call_llm:
    """
    A unified class for performing inference using Qwen2.5VL models with API.
    """
    
    def __init__(self, model_id="qwen-vl-max-latest", api_key=None, base_url=None):

        print("Initializing API client...")
        self.model_id = model_id
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
    def encode_image(self, image_path):
        """将图片编码为base64格式"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def inference(self, text: str, image: Union[list, str], do_sample=False, temperature=0.7):
        """使用API进行多模态推理"""
        if isinstance(image, str):
            image = [image]

        # 将图片编码为base64
        base64_images = []
        for img_path in image:
            abs_path = os.path.abspath(img_path)
            if not os.path.exists(abs_path):
                print(f"Warning: Image not found: {abs_path}")
                continue
            base64_images.append(self.encode_image(abs_path))
        
        # 构建消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text}
                ] + [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64img}"
                        }
                    } for b64img in base64_images
                ]
            }
        ]
        
        # API推理
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature if do_sample else 0.0
        )
        
        return completion.choices[0].message.content if completion.choices else ""


def process_item(idx, item, model, image_base):
    """Process a single benchmark item."""
    # 使用统一函数获取图片路径列表
    image_paths = item.get("images", [])
    full_image_paths = []
    for rel_path in image_paths:
        abs_path = os.path.join(image_base, rel_path)
        if not os.path.exists(abs_path):
            print(f"Warning: Image not found: {abs_path}")
            continue
        full_image_paths.append(abs_path)

    if not full_image_paths:
        print(f"Item {idx+1}: No valid images found")
        return None

    # 获取question
    question = item["question"]
    
    item_type = item.get("type", "")
    if item_type in ["Fine-grained Grounding", "Correspondence-Pointing"]:
        question = question + ' Output the point coordinates in JSON format.' 

    # 使用模型进行推理
    result = model.inference(
        text=question,
        image=full_image_paths,
        do_sample=False,
        temperature=0.0
    )
    
    # 保存结果
    item["assistant"] = result
    item["prompt"] = question
    if 'idx' not in item:
        item['idx'] = idx
        
    print(f"Item {idx+1} completed")
    return item
            

def process_benchmark(model, benchmark_name, benchmark_path, image_base, result_dir, max_workers=8):
    """Process a single benchmark with multi-threading."""
    
    # Create result directory if it doesn't exist
    os.makedirs(result_dir, exist_ok=True)

    # Read benchmark data
    print(f"Loading benchmark from {benchmark_path}")
    with open(benchmark_path, "r") as f:
        data = json.load(f)

    # Check existing result file and get tested data
    tested_keys = set()
    model_name = model.model_id.replace("/", "_")
    eval_path = os.path.join(result_dir, f"eval_{model_name}.jsonl")
    if os.path.exists(eval_path):
        print(f"Found existing results file: {eval_path}")
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f: 
                try:
                    item = json.loads(line.strip())
                    image_paths = item.get("images", [])
                    question_key = item.get('question', '') + str(image_paths)
                    tested_keys.add(question_key)
                except json.JSONDecodeError:
                    continue

    # Prepare items to process
    items_to_process = []
    for idx in range(len(data)):
        item = data[idx]
        image_paths = item.get("images", [])
        question_key = item.get('question', '') + str(image_paths)
        if question_key not in tested_keys:
            items_to_process.append((idx, item))

    if not items_to_process:
        print(f"[{benchmark_name}] All items already processed!")
        return

    print(f"[{benchmark_name}] Total items to process: {len(items_to_process)}")

    # Thread lock for file writing
    write_lock = threading.Lock()
    results = []
    
    # Process items with multi-threading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(process_item, idx, item, model, image_base): (idx, item)
            for idx, item in items_to_process
        }
        
        # Process completed tasks
        for future in as_completed(future_to_item):
            idx, item = future_to_item[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    
                    # Save results incrementally with thread lock
                    with write_lock:
                        with open(eval_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    
                    print(f"[{benchmark_name}] Item {idx+1}/{len(data)} completed and saved")
                else:
                    print(f"[{benchmark_name}] Item {idx+1}/{len(data)} failed to process")
            except Exception as e:
                print(f"[{benchmark_name}] Item {idx+1}/{len(data)} generated an exception: {e}")

    print(f"[{benchmark_name}] Processing completed. Results saved to {eval_path}")
    print(f"[{benchmark_name}] Total processed: {len(results)} items")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate model performance via API')
    parser.add_argument('--model', type=str, default="qwen-vl-max-latest", help='API model name to evaluate')
    parser.add_argument('--api_key', type=str, required=True, help='API key for authentication')
    parser.add_argument('--base_url', type=str, required=True, help='Base URL for API endpoint')
    parser.add_argument('--benchmark_path', type=str, required=True, help='Path to the benchmark directory (e.g., /path/to/CrossPoint-Bench)')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory to save evaluation results')
    parser.add_argument('--max_workers', type=int, default=8, help='Number of parallel workers for API calls')
    args = parser.parse_args()

    # Validate benchmark directory structure
    benchmark_json = os.path.join(args.benchmark_path, f"CrossPoint-Bench.json")
    image_dir = os.path.join(args.benchmark_path, "image")
    
    if not os.path.exists(benchmark_json):
        print(f"Error: Benchmark file not found: {benchmark_json}")
        return
    
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        return
    
    # Initialize model
    print(f"Evaluating model: {args.model}")
    print(f"Benchmark: CrossPoint-Bench")
    print(f"Results will be saved to: {args.result_dir}")
    
    model = call_llm(model_id=args.model, api_key=args.api_key, base_url=args.base_url)
    
    # Process benchmark
    process_benchmark(model, benchmark_json, image_dir, args.result_dir, args.max_workers)
    


if __name__ == "__main__":
    main()

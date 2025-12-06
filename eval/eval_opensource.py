import os, json, argparse, time
from typing import Union
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
class QwenVL:
    """
    A unified class for performing inference using Qwen2.5VL models with vLLM.
    """
    
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", tensor_parallel_size=1, gpu_memory_utilization=0.75):
        """
        Initialize the vLLM model and processor.
        
        Args:
            model_id (str): Path or Hugging Face model identifier
            tensor_parallel_size (int): Number of GPUs to use for tensor parallelism
            gpu_memory_utilization (float): GPU memory utilization ratio (0.0-1.0)
        """
        print("Loading Checkpoint with vLLM...")
        self.model_id = model_id
        
        # Initialize vLLM model
        self.model = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={"image": 20, "video": 1},
        )
        
        # Initialize processor for chat template
        self.processor = AutoProcessor.from_pretrained(model_id)
        

        
    def inference(self, text: str, image: Union[list, str], do_sample=False, temperature=0.7):
        """Perform multimodal inference using vLLM"""
        if isinstance(image, str):
            image = [image]

        # Build message format
        content = []
        for img_path in image:
            abs_path = os.path.abspath(img_path)
            if not os.path.exists(abs_path):
                print(f"Warning: Image not found: {abs_path}")
                continue
            content.append({"type": "image", "image": abs_path})
        
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]

        # Generate prompt and process vision information
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        mm_data = {"image": image_inputs} if image_inputs is not None else {}

        # vLLM inference
        sampling_params = SamplingParams(max_tokens=768, temperature=temperature if do_sample else 0.0)
        llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
        outputs = self.model.generate([llm_inputs], sampling_params=sampling_params)
        return outputs[0].outputs[0].text if outputs else ""



def process_item(idx, item, model, image_base, benchmark_name=""):
    """Process a single benchmark item."""
    # Use unified function to get image path list
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

    # Get question
    question = item["question"]
    
    # For CrossPoint-Bench, only for qwen
    if benchmark_name == "CrossPoint-Bench":
        item_type = item.get("type", "")
        if item_type in ["Fine-grained Grounding", "Correspondence-Pointing"]:
            question = question + ' Output the point coordinates in JSON format.'
    
    # Use model for inference
    result = model.inference(
        text=question,
        image=full_image_paths,
        do_sample=False,
        temperature=0.0
    )
    
    # Save results
    item["assistant"] = result
    item["prompt"] = question
    if 'idx' not in item:
        item['idx'] = idx
        
    print(f"Item {idx+1} completed")
    return item
            


def process_benchmark(model, benchmark_name, benchmark_path, image_base, result_dir):
    """Process a single benchmark."""
    print(f"\n{'='*60}")
    print(f"Starting evaluation: {benchmark_name}")
    print(f"{'='*60}")
    
    # Create result directory if it doesn't exist
    os.makedirs(result_dir, exist_ok=True)

    # Read benchmark data
    print(f"Loading benchmark from {benchmark_path}")
    with open(benchmark_path, "r") as f:
        data = json.load(f)

    # Check existing result file and get tested data
    tested_keys = set()
    model_name = os.path.basename(model.model_id)
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


    # Process items
    results = []
    for idx in range(len(data)):
        item = data[idx]
        
        # Check if already processed
        image_paths = item.get("images", [])
        question_key = item.get('question', '') + str(image_paths)
        if question_key in tested_keys:
            print(f"Item {idx+1} already processed, skipping...")
            continue
            
        print(f"[{benchmark_name}] Processing item {idx+1}/{len(data)}")
        result = process_item(idx, item, model, image_base, benchmark_name)
        
        if result is not None:
            results.append(result)
            
            # Save results incrementally
            with open(eval_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            print(f"Saved result for item {idx+1}")
        else:
            print(f"Failed to process item {idx+1}")

    print(f"[{benchmark_name}] Processing completed. Results saved to {eval_path}")
    print(f"[{benchmark_name}] Total processed: {len(results)} items")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate Qwen2.5VL model performance')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint or HuggingFace model name')
    parser.add_argument('--benchmark_path', type=str, required=True, help='Path to the benchmark directory (e.g., /path/to/CrossPoint-Bench)')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory to save evaluation results')
    args = parser.parse_args()

    # Validate benchmark directory structure
    benchmark_name = os.path.basename(args.benchmark_path)
    benchmark_json = os.path.join(args.benchmark_path, f"{benchmark_name}.json")
    image_dir = os.path.join(args.benchmark_path, "image")
    
    if not os.path.exists(benchmark_json):
        print(f"Error: Benchmark file not found: {benchmark_json}")
        return
    
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        return
    
    # Initialize and load model
    print(f"Loading model: {args.model}")
    print(f"Benchmark: {benchmark_name}")
    print(f"Results will be saved to: {args.result_dir}")
    
    model = QwenVL(model_id=args.model)
    print("Model loaded successfully!")

    # Process benchmark
    process_benchmark(model, benchmark_name, benchmark_json, image_dir, args.result_dir)
    


if __name__ == "__main__":
    main()
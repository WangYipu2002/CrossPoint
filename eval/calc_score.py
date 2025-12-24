#!/usr/bin/env python3
"""
Evaluation script for cross-view correspondence tasks.

Features:
- Evaluates Coordinate-based and Multiple-choice tasks.
- Verifies coordinates against Segmentation Masks (Base64).
- Generates detailed hierarchical reports (Type -> Level).
- Outputs to console and text file (no Excel).
"""

import json
import base64
import numpy as np
from PIL import Image
import io
import os
import argparse
import glob
from typing import Dict, List, Any, Optional, Tuple

# ==========================================
# Core Evaluation Logic
# ==========================================

def decode_base64_image(base64_str: str) -> Optional[np.ndarray]:
    """Decode base64 encoded image string to numpy array."""
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return np.array(image)
    except Exception:
        return None

def is_point_in_mask(point: Dict[str, float], mask_image: np.ndarray, threshold: int = 128) -> bool:
    """
    Check if a point falls within the white region of a mask.
    
    Logic:
    1. Get absolute x, y from extraction.
    2. Check boundaries.
    3. Check pixel value at (y, x).
    """
    if mask_image is None:
        return False
    
    try:
        x, y = int(point['x']), int(point['y'])
        height, width = mask_image.shape[:2]
        
        # Boundary check
        if not (0 <= x < width and 0 <= y < height):
            return False
        
        # Handle grayscale vs RGB masks
        if len(mask_image.shape) == 3:
            # Assuming standard mask is white (255) on black (0)
            # Taking mean or just checking one channel usually works for binary masks
            mask_gray = np.mean(mask_image, axis=2)
        else:
            mask_gray = mask_image
            
        # Check pixel value
        return bool(mask_gray[y, x] > threshold)
        
    except Exception:
        return False

def normalize_option(answer: Any) -> Optional[str]:
    """Normalize multiple-choice answer to uppercase letter."""
    if not isinstance(answer, str):
        return None
    # Remove whitespace, parenthesis, take uppercase
    return answer.strip().upper().strip('()')

def evaluate_coordinate_task(item: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate coordinate-based grounding tasks."""
    result = {
        'idx': item.get('idx'),
        'type': item.get('type', "Unknown"),
        'level': item.get('level', "Unknown"), # Added level tracking
        'extraction_success': item.get('extraction_success', False),
        'score': 0,
        'details': {}
    }
    
    if not item.get('extraction_success', False):
        result['details']['error'] = 'Answer extraction failed'
        return result
    
    # Critical: Use absolute coordinates converted during extraction
    if 'extracted_answer_absolute' not in item:
        result['details']['error'] = 'Missing absolute coordinates'
        return result
    
    if 'answer' not in item or not isinstance(item['answer'], str):
        result['details']['error'] = 'Missing or invalid answer (mask) field'
        return result
    
    # Decode ground truth mask
    mask_image = decode_base64_image(item['answer'])
    if mask_image is None:
        result['details']['error'] = 'Failed to decode mask image'
        return result
    
    extracted_point = item['extracted_answer_absolute']
    
    # JUDGEMENT CALL: Is the point inside the mask?
    is_correct = is_point_in_mask(extracted_point, mask_image)
    
    result['score'] = 1 if is_correct else 0
    result['details'] = {
        'extracted_point': extracted_point,
        'is_in_mask': is_correct
    }
    
    return result

def evaluate_letter_task(item: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate multiple-choice tasks."""
    result = {
        'idx': item.get('idx'),
        'type': item.get('type', "Unknown"),
        'level': item.get('level', "Unknown"), # Added level tracking
        'extraction_success': item.get('extraction_success', False),
        'score': 0,
        'details': {}
    }
    
    if not item.get('extraction_success', False):
        result['details']['error'] = 'Answer extraction failed'
        return result
    
    if 'extracted_answer' not in item or 'answer' not in item:
        result['details']['error'] = 'Missing answer fields'
        return result
    
    extracted_letter = normalize_option(item['extracted_answer'])
    correct_letter = normalize_option(item['answer'])
    
    if extracted_letter is None or correct_letter is None:
        result['details']['error'] = 'Invalid answer format'
        return result
    
    is_correct = extracted_letter == correct_letter
    result['score'] = 1 if is_correct else 0
    result['details'] = {
        'extracted_answer': extracted_letter,
        'correct_answer': correct_letter,
        'is_correct': is_correct
    }
    
    return result

# ==========================================
# Statistics & Reporting
# ==========================================

def calculate_hierarchical_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate stats grouped by Type -> Level.
    Structure:
    {
        "Overall": {...},
        "Fine-grained Grounding": {
            "Overall": {...},
            "object": {...},
            "part": {...}
        },
        ...
    }
    """
    stats = {
        "Overall": {"total": 0, "correct": 0, "extracted": 0}
    }
    
    for item in results:
        task_type = item['type']
        level = item['level']
        score = item['score']
        is_extracted = item['extraction_success']
        
        # Initialize Type dict
        if task_type not in stats:
            stats[task_type] = {"Overall": {"total": 0, "correct": 0, "extracted": 0}}
        
        # Initialize Level dict
        if level not in stats[task_type]:
            stats[task_type][level] = {"total": 0, "correct": 0, "extracted": 0}
            
        # Update Global
        stats["Overall"]["total"] += 1
        if is_extracted: stats["Overall"]["extracted"] += 1
        if score == 1: stats["Overall"]["correct"] += 1
        
        # Update Type Overall
        stats[task_type]["Overall"]["total"] += 1
        if is_extracted: stats[task_type]["Overall"]["extracted"] += 1
        if score == 1: stats[task_type]["Overall"]["correct"] += 1
        
        # Update Level
        stats[task_type][level]["total"] += 1
        if is_extracted: stats[task_type][level]["extracted"] += 1
        if score == 1: stats[task_type][level]["correct"] += 1
        
    return stats

def format_report(model_name: str, stats: Dict[str, Any]) -> str:
    """Format statistics into a readable string table."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"Model: {model_name}")
    lines.append("=" * 60)
    lines.append(f"{'Category':<35} | {'Acc (%)':<10} | {'Corr/Tot':<10}")
    lines.append("-" * 60)
    
    def get_acc(d):
        return (d["correct"] / d["total"] * 100) if d["total"] > 0 else 0.0
    
    # 1. Overall
    overall = stats["Overall"]
    lines.append(f"{'OVERALL':<35} | {get_acc(overall):<10.2f} | {overall['correct']}/{overall['total']}")
    lines.append("-" * 60)
    
    # 2. Iterate Types
    # Filter out "Overall" key to iterate specific types
    task_types = [k for k in stats.keys() if k != "Overall"]
    
    for t_type in sorted(task_types):
        t_stats = stats[t_type]
        # Type Overall
        t_ov = t_stats["Overall"]
        lines.append(f"{f'[{t_type}]':<35} | {get_acc(t_ov):<10.2f} | {t_ov['correct']}/{t_ov['total']}")
        
        # Levels within Type
        levels = [k for k in t_stats.keys() if k != "Overall"]
        for lvl in sorted(levels):
            l_stats = t_stats[lvl]
            lines.append(f"{f'  - {lvl}':<35} | {get_acc(l_stats):<10.2f} | {l_stats['correct']}/{l_stats['total']}")
        lines.append("-" * 60)
        
    return "\n".join(lines)

def process_file(file_path: str) -> Optional[Tuple[List[Dict], Dict]]:
    """Process a single file and return raw results and stats."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
        
    processed_results = []
    
    for item in data:
        task_type = item.get('type', '')
        
        # Route to appropriate evaluator
        # Note: Ensure these type strings match your JSON data exactly
        if task_type in ['Fine-grained Grounding', 'Correspondence-Pointing', 'Level 1', 'Level 4']:
            res = evaluate_coordinate_task(item)
        else:
            res = evaluate_letter_task(item)
            
        processed_results.append(res)
        
    stats = calculate_hierarchical_stats(processed_results)
    return processed_results, stats

def main():
    parser = argparse.ArgumentParser(description='Evaluate extraction results with fine-grained stats')
    parser.add_argument('--extracted_root', type=str, required=True,
                       help='Directory containing extracted_*.json files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for report files')
    args = parser.parse_args()
    
    if not os.path.exists(args.extracted_root):
        print(f"Error: Directory not found: {args.extracted_root}")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    report_file_path = os.path.join(args.output_dir, "evaluation_report.txt")
    
    # Clear previous report
    with open(report_file_path, 'w', encoding='utf-8') as f:
        f.write("Evaluation Report\n")
    
    # Find files
    extracted_files = glob.glob(os.path.join(args.extracted_root, "extracted_*.json"))
    if not extracted_files:
        print("No extracted_*.json files found.")
        return

    print(f"Found {len(extracted_files)} files. Evaluating...\n")

    for file_path in sorted(extracted_files):
        filename = os.path.basename(file_path)
        model_name = filename.replace('extracted_', '').replace('.json', '')
        
        result_tuple = process_file(file_path)
        if not result_tuple:
            continue
            
        _, stats = result_tuple
        
        # Generate Report String
        report_str = format_report(model_name, stats)
        
        # 1. Print to Console
        print(report_str)
        print("\n")
        
        # 2. Append to File
        with open(report_file_path, 'a', encoding='utf-8') as f:
            f.write(report_str + "\n\n")
            
    print(f"✓ Full report saved to: {report_file_path}")

if __name__ == "__main__":
    main()
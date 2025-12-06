#!/usr/bin/env python3
"""
Evaluation script for cross-view correspondence tasks.

Evaluates two types of tasks:
- Coordinate-based: Fine-grained Grounding, Correspondence-Pointing
- Multiple-choice: Visibility Reasoning, Correspondence-Judgement
"""

import json
import base64
import numpy as np
from PIL import Image
import io
import os
import argparse
import glob
from typing import Dict, List, Any, Optional
import pandas as pd


def decode_base64_image(base64_str: str) -> Optional[np.ndarray]:
    """Decode base64 encoded image string to numpy array."""
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return np.array(image)
    except Exception:
        return None


def is_point_in_mask(point: Dict[str, float], mask_image: np.ndarray, threshold: int = 128) -> bool:
    """Check if a point falls within the white region of a mask."""
    if mask_image is None:
        return False
    
    try:
        x, y = int(point['x']), int(point['y'])
        height, width = mask_image.shape[:2]
        
        if not (0 <= x < width and 0 <= y < height):
            return False
        
        mask_gray = np.mean(mask_image, axis=2) if len(mask_image.shape) == 3 else mask_image
        return bool(mask_gray[y, x] > threshold)
        
    except Exception:
        return False


def normalize_option(answer: Any) -> Optional[str]:
    """Normalize multiple-choice answer to uppercase letter."""
    if not isinstance(answer, str):
        return None
    return answer.strip().upper().strip('()')


def evaluate_coordinate_task(item: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate coordinate-based grounding tasks."""
    result = {
        'idx': item.get('idx'),
        'type': item.get('type', ""),
        'extraction_success': item.get('extraction_success', False),
        'score': 0,
        'details': {}
    }
    
    if not item.get('extraction_success', False):
        result['details']['error'] = 'Answer extraction failed'
        return result
    
    if 'extracted_answer_absolute' not in item:
        result['details']['error'] = 'Missing absolute coordinates'
        return result
    
    if 'answer' not in item or not isinstance(item['answer'], str):
        result['details']['error'] = 'Missing or invalid answer field'
        return result
    
    mask_image = decode_base64_image(item['answer'])
    if mask_image is None:
        result['details']['error'] = 'Failed to decode mask image'
        return result
    
    extracted_point = item['extracted_answer_absolute']
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
        'type': item.get('type', ""),
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




def evaluate_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Evaluate a single extraction result file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None
    
    # Check for pre-computed results
    if isinstance(data, dict) and 'summary' in data:
        if 'type_statistics' in data or 'type_stats' in data:
            return data
        return None
    
    results = []
    type_stats = {}
    
    for item in data:
        if not isinstance(item, dict):
            continue
        
        task_type = item.get('type', '')
        if not task_type:
            continue
        
        if task_type not in type_stats:
            type_stats[task_type] = {'total': 0, 'correct': 0, 'extracted': 0}
        
        type_stats[task_type]['total'] += 1
        
        if item.get('extraction_success', False):
            type_stats[task_type]['extracted'] += 1
        
        # Evaluate based on task type
        if task_type in ['Fine-grained Grounding', 'Correspondence-Pointing']:
            result = evaluate_coordinate_task(item)
        else:
            result = evaluate_letter_task(item)
        
        results.append(result)
        
        if result['score'] == 1:
            type_stats[task_type]['correct'] += 1
    
    summary = _calculate_statistics(file_path, len(data), type_stats)
    
    return {
        'summary': summary,
        'type_stats': type_stats,
        'results': results
    }


def _calculate_statistics(file_path: str, total_items: int, type_stats: Dict) -> Dict[str, Any]:
    """Calculate evaluation statistics."""
    total_all = sum(stats['total'] for stats in type_stats.values())
    extracted_all = sum(stats['extracted'] for stats in type_stats.values())
    correct_all = sum(stats['correct'] for stats in type_stats.values())
    
    summary = {
        'file_path': file_path,
        'total_items': total_items,
        'type_statistics': {}
    }
    
    if total_all > 0:
        summary['overall_statistics'] = {
            'total': total_all,
            'extracted': extracted_all,
            'correct': correct_all,
            'extraction_rate': round(extracted_all / total_all * 100, 2),
            'accuracy': round(correct_all / extracted_all * 100, 2) if extracted_all > 0 else 0,
            'overall_score': round(correct_all / total_all * 100, 2)
        }
    
    for task_type, stats in type_stats.items():
        if stats['total'] > 0:
            summary['type_statistics'][task_type] = {
                'total': stats['total'],
                'extracted': stats['extracted'],
                'correct': stats['correct'],
                'extraction_rate': round(stats['extracted'] / stats['total'] * 100, 2),
                'accuracy': round(stats['correct'] / stats['extracted'] * 100, 2) if stats['extracted'] > 0 else 0,
                'overall_score': round(stats['correct'] / stats['total'] * 100, 2)
            }
    
    return summary


def generate_excel_report(benchmark_results: Dict[str, Dict], excel_file: str):
    """Generate Excel evaluation report with one sheet per benchmark."""
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        for benchmark_name, models_results in benchmark_results.items():
            all_types = set()
            for result in models_results.values():
                if result and 'summary' in result:
                    all_types.update(result['summary'].get('type_statistics', {}).keys())
            
            sorted_types = sorted(all_types)
            
            rows = []
            for model_name, result in models_results.items():
                if not (result and 'summary' in result):
                    continue
                    
                summary = result['summary']
                overall = summary.get('overall_statistics', {})
                type_stats = summary.get('type_statistics', {})
                
                row = {
                    'Model': model_name,
                    'Extraction Rate': overall.get('extraction_rate', 0),
                    'Overall Score': overall.get('overall_score', 0)
                }
                
                for task_type in sorted_types:
                    row[task_type] = type_stats.get(task_type, {}).get('overall_score', 0)
                
                rows.append(row)
            
            if rows:
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=benchmark_name[:31], index=False)


def _process_extracted_results(extracted_root: str) -> Dict[str, Any]:
    """Process extracted results and evaluate."""
    results = {}
    
    # Find all extracted_*.json files
    extracted_files = glob.glob(os.path.join(extracted_root, "extracted_*.json"))
    
    if not extracted_files:
        return {}
    
    for file_path in extracted_files:
        filename = os.path.basename(file_path)
        model_name = filename.replace('extracted_', '').replace('.json', '')
        
        result = evaluate_file(file_path)
        if result is not None:
            results[model_name] = result
            # Print summary
            if 'summary' in result and 'overall_statistics' in result['summary']:
                overall = result['summary']['overall_statistics']
                print(f"  {model_name}: {overall.get('overall_score', 0):.2f}% "
                      f"({overall.get('correct', 0)}/{overall.get('total', 0)})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate extraction results')
    parser.add_argument('--extracted_root', type=str, required=True,
                       help='Directory containing extracted_*.json files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for evaluation reports')
    args = parser.parse_args()
    
    if not os.path.exists(args.extracted_root):
        print(f"Error: Directory not found: {args.extracted_root}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = _process_extracted_results(args.extracted_root)
    
    if not results:
        print("Error: No results found")
        return
    
    # Generate Excel report
    benchmark_results = {"CrossPoint-Bench": results}
    excel_file = os.path.join(args.output_dir, "evaluation_summary.xlsx")
    generate_excel_report(benchmark_results, excel_file)
    
    print(f"✓ Report: {excel_file}")


if __name__ == "__main__":
    main()
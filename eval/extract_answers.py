#!/usr/bin/env python3
"""
Answer extraction script for cross-view correspondence evaluation.
Extracts and normalizes answers from model outputs for coordinate-based and multiple-choice tasks.
"""

import json
import re
import os
import argparse
from typing import Dict, List, Any, Tuple, Optional


class AnswerExtractor:
    """Extracts and processes answers from model evaluation results."""
    
    def __init__(self, single_file_path: str = None, coord_format: str = 'absolute'):
        if not single_file_path:
            raise ValueError("Must specify input file path (--file parameter)")
        
        # Coordinate format: 'absolute', 'relative_1000', or 'relative_1'
        self.coord_format = coord_format
        
        self.input_files = {}
        self._process_single_file(single_file_path)
        
        self.output_dir = self._get_output_directory(single_file_path)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results = {}
    
    def _get_output_directory(self, file_path: str) -> str:
        """Determine output directory based on input file path."""
        file_dir = os.path.dirname(file_path)
        if 'inference' in file_dir:
            return file_dir.replace('inference', 'extracted')
        
        parent_dir = os.path.dirname(file_dir)
        benchmark_name = os.path.basename(file_dir)
        return os.path.join(parent_dir, 'extracted', benchmark_name)
    
    def _process_single_file(self, file_path: str):
        """Validate input file and extract model name."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        if not file_path.endswith('.jsonl'):
            raise ValueError(f"File must be in JSONL format: {file_path}")
        
        filename = os.path.basename(file_path)
        if filename.startswith("eval_") and filename.endswith(".jsonl"):
            model_name = filename[5:-6]
        else:
            model_name = os.path.splitext(filename)[0]
        
        self.input_files[model_name] = file_path
        print(f"Model: {model_name} | Coord format: {self.coord_format}")
    
    def convert_coordinates_for_item(self, item: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Convert and add absolute coordinate field for coordinate-based tasks.
        
        Supports three coordinate formats:
        - 'absolute': Pixel coordinates (no conversion)
        - 'relative_1000': Normalized to [0, 1000] range
        - 'relative_1': Normalized to [0, 1] range
        """
        result = item.copy()
        
        if not item.get("extraction_success", False):
            return result
        
        extracted_answer = item.get("extracted_answer")
        if not isinstance(extracted_answer, dict):
            return result
        
        task_type = item.get("type","")
        if task_type not in ["Fine-grained Grounding", "Correspondence-Pointing"]:
            return result
        
        if "x" not in extracted_answer or "y" not in extracted_answer:
            return result
        
        x = extracted_answer["x"]
        y = extracted_answer["y"]
        
        # Convert coordinates based on format
        if self.coord_format == 'absolute':
            # Already in absolute pixel coordinates
            x_abs, y_abs = x, y
            
        elif self.coord_format == 'relative_1000':
            # Convert from [0, 1000] to absolute pixels
            image_width = item.get("image_width")
            image_height = item.get("image_height")
            if image_width is None or image_height is None:
                print(f"Warning: Missing image dimensions for idx {item.get('idx')}, using original coordinates")
                x_abs, y_abs = x, y
            else:
                x_abs = (x / 1000.0) * image_width
                y_abs = (y / 1000.0) * image_height
                
        elif self.coord_format == 'relative_1':
            # Convert from [0, 1] to absolute pixels
            image_width = item.get("image_width")
            image_height = item.get("image_height")
            if image_width is None or image_height is None:
                print(f"Warning: Missing image dimensions for idx {item.get('idx')}, using original coordinates")
                x_abs, y_abs = x, y
            else:
                x_abs = x * image_width
                y_abs = y * image_height
        else:
            print(f"Warning: Unknown coordinate format '{self.coord_format}', using original coordinates")
            x_abs, y_abs = x, y
        
        result["extracted_answer_absolute"] = {
            "x": x_abs,
            "y": y_abs
        }
        
        return result
    
    def extract_coordinates_from_json(self, assistant_text: str, model_name: str = "") -> Optional[Tuple[float, float]]:
        """Extract coordinates from JSON format in assistant response."""
        json_blocks = re.findall(r'```json\s*(.*?)\s*```', assistant_text, re.DOTALL | re.IGNORECASE)
        
        if not json_blocks:
            json_patterns = [
                r'\{[^{}]*"point_2d"[^{}]*\}',
                r'\{[^{}]*"point"[^{}]*\}',
                r'\{[^{}]*"(?:x|position|coordinates)"[^{}]*\}',
            ]
            for pattern in json_patterns:
                matches = re.findall(pattern, assistant_text, re.DOTALL)
                if matches:
                    json_blocks.extend(matches)
        
        for json_text in json_blocks:
            try:
                json_text = json_text.strip()
                json_text = re.sub(r'//.*?(?=\n|$)', '', json_text, flags=re.MULTILINE)
                json_text = re.sub(r'/\*.*?\*/', '', json_text, flags=re.DOTALL)
                
                data = json.loads(json_text)
                coords = self._extract_coords_from_data(data)
                if coords:
                    return coords
            except (json.JSONDecodeError, KeyError, IndexError, ValueError):
                continue
        
        return None
    
    def _extract_coords_from_data(self, data: Any) -> Optional[Tuple[float, float]]:
        """Extract coordinates from parsed JSON data."""
        if isinstance(data, list) and data and isinstance(data[0], dict):
            for key in ["point_2d", "point"]:
                if key in data[0]:
                    point = data[0][key]
                    if isinstance(point, list) and len(point) >= 2:
                        return float(point[0]), float(point[1])
        
        if isinstance(data, dict):
            for key in ["point_2d", "point"]:
                if key in data and isinstance(data[key], list) and len(data[key]) >= 2:
                    return float(data[key][0]), float(data[key][1])
            
            if "x" in data and "y" in data:
                if data["x"] is not None and data["y"] is not None:
                    return float(data["x"]), float(data["y"])
            
            for nested_key in ["position", "coordinates"]:
                if nested_key in data and isinstance(data[nested_key], dict):
                    nested = data[nested_key]
                    if "x" in nested and "y" in nested:
                        if nested["x"] is not None and nested["y"] is not None:
                            return float(nested["x"]), float(nested["y"])
        
        return None
    
    def extract_coordinates_from_text(self, assistant_text: str, model_name: str = "") -> Optional[Tuple[float, float]]:
        """Extract coordinates from text (fallback method)."""
        patterns = [
            (r'<point\s+x="([0-9.]+)"\s+y="([0-9.]+)"[^>]*>.*?</point>', False),
            (r'(?:coordinates?|position|location).*?(?:are|is)\s+([0-9.]+),\s*([0-9.]+)', False),
            (r'[xX]\s*:\s*([0-9.]+).*?[yY]\s*:\s*([0-9.]+)', False),
            (r'[yY]\s*:\s*([0-9.]+).*?[xX]\s*:\s*([0-9.]+)', True),  # Y first, swap needed
            (r'["\']?x["\']?\s*:\s*([0-9.]+).*?["\']?y["\']?\s*:\s*([0-9.]+)', False),
            (r'\[([0-9.]+),\s*([0-9.]+)\]', False),
            (r'\(([0-9.]+),\s*([0-9.]+)\)', False),
            (r'([0-9.]+),\s*([0-9.]+)', False),
        ]
        
        for pattern, swap_xy in patterns:
            matches = re.findall(pattern, assistant_text, re.IGNORECASE | re.DOTALL)
            if matches:
                try:
                    x_str, y_str = matches[0][0].rstrip('.'), matches[0][1].rstrip('.')
                    x, y = float(x_str), float(y_str)
                    return (y, x) if swap_xy else (x, y)
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def extract_option_from_text(self, assistant_text: str, question_text: str = "") -> Optional[str]:
        """Extract option (A, B, C, D) from text."""
        # Standard patterns
        patterns = [
            r'\\boxed\{(?:\\text\{)?([ABCD])(?:\..*?)?\}',
            r'\\\(?\s*\\boxed\{\\text\{([ABCD])\}\}\s*\\\)?',
            r'\$+\s*\\boxed\{\\text\{([ABCD])(?:\..*?)?\}\}\s*\$+',
            r'\((?:Choice\s+)?([ABCD])\)',
            r'\*\*Answer:\s*([ABCD])\*\*',
            r'\*\*([ABCD])\.\s*(?:Yes|No|Green dot|Yellow dot|Blue dot|Red dot)\*\*',
            r'(?:答案是|answer is|选择|choose|correct answer is)\s*[:\s]*\*?\*?\(?([ABCD])\)?',
            r'(?:^|\n)\s*\*?([ABCD])\*?\.\s*(?:Yes|No)',
            r'\*\*([ABCD])\*\*',
            r'(?:^|\s|Answer:\s*)\(?([ABCD])\)?\s*[\.:]',
            r'^\s*\(?([ABCD])\)?\s*$',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, assistant_text, re.IGNORECASE | re.MULTILINE)
            if matches:
                return matches[0].upper()
        
        # Final answer patterns (take last match)
        final_patterns = [
            r'(?:Final Answer|Therefore|So|Hence|Thus|The correct answer is|The answer is).*?(?:\\boxed\{([ABCD])\}|\*\*([ABCD])\*\*|\(?([ABCD])\)?)',
            r'Answer:\s*\(?([ABCD])\)?',
        ]
        
        for pattern in final_patterns:
            matches = re.findall(pattern, assistant_text, re.IGNORECASE | re.DOTALL)
            if matches:
                # Handle tuple matches from alternation groups
                match = matches[-1]
                result = match if isinstance(match, str) else next(m for m in match if m)
                return result.upper()
        
        # Try intelligent option mapping based on question content
        if question_text:
            option_mapping = self.parse_question_options(question_text)
            if option_mapping:
                assistant_lower = assistant_text.lower().strip()
                
                # Check if response starts with an option value
                for option, value in option_mapping.items():
                    if assistant_lower.startswith(value.lower()):
                        return option
                
                # Extract from bold descriptive text
                descriptive_patterns = [
                    r'\*\*(.*?)\*\*',
                    r'(?:Therefore|So|The correct answer is|The answer is).*?\*\*(.*?)\*\*',
                ]
                
                for pattern in descriptive_patterns:
                    matches = re.findall(pattern, assistant_text, re.IGNORECASE | re.DOTALL)
                    for match in matches:
                        extracted_text = match.strip().lower()
                        for option, value in option_mapping.items():
                            if extracted_text == value.strip().lower():
                                return option
        
        return None
    
    def parse_question_options(self, question_text: str) -> Dict[str, str]:
        """Parse option mapping from question text (e.g., 'A.Yes B.No' -> {'A': 'Yes', 'B': 'No'})."""
        option_mapping = {}
        pattern = r'([ABCD])\.([^A-D\n]+?)(?=\s*[ABCD]\.|$)'
        matches = re.findall(pattern, question_text, re.IGNORECASE | re.DOTALL)
        
        for option, value in matches:
            cleaned_value = re.sub(r'\s+', ' ', value.strip())
            option_mapping[option.upper()] = cleaned_value
        
        return option_mapping
    
    def extract_answer(self, item: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Extract answer based on task type."""
        result = item.copy()
        result["model"] = model_name
        result["extracted_answer"] = None
        result["extraction_success"] = False
        
        assistant_text = item.get("assistant", "")
        task_type = item.get("type","")
        question_text = item.get("question", "")
        
        # Coordinate-based tasks
        if task_type in ["Fine-grained Grounding", "Correspondence-Pointing"]:
            coords = self.extract_coordinates_from_json(assistant_text, model_name)
            if coords is None:
                coords = self.extract_coordinates_from_text(assistant_text, model_name)
            
            if coords:
                result["extracted_answer"] = {"x": coords[0], "y": coords[1]}
                result["extraction_success"] = True
        
        # Multiple-choice tasks
        else:
            option = self.extract_option_from_text(assistant_text, question_text)
            if option:
                result["extracted_answer"] = option
                result["extraction_success"] = True
        
        return result
    
    def process_file(self, model_name: str, filepath: str) -> List[Dict[str, Any]]:
        """Process a single JSONL file and extract answers."""
        results = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        result = self.extract_answer(item, model_name)
                        results.append(result)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}")
            return []
        
        success_count = sum(1 for r in results if r["extraction_success"])
        print(f"Extracted: {success_count}/{len(results)} answers")
        
        # Apply coordinate conversion
        converted_results = [
            self.convert_coordinates_for_item(result, model_name)
            for result in results
        ]
        
        # Sort by index
        converted_results.sort(key=lambda x: x.get("idx", 0))
        
        return converted_results
    
    def save_results(self):
        """Save extraction results to JSON files."""
        for model_name, model_results in self.results.items():
            if model_name in self.input_files:
                original_filename = os.path.basename(self.input_files[model_name])
                new_filename = original_filename.replace("eval_", "extracted_").replace(".jsonl", ".json")
            else:
                new_filename = f"extracted_{model_name}.json"
            
            model_file = os.path.join(self.output_dir, new_filename)
            with open(model_file, 'w', encoding='utf-8') as f:
                json.dump(model_results, f, ensure_ascii=False, indent=2)
            print(f"Saved: {model_file}")
        
        self.save_failed_extractions()
    
    def save_failed_extractions(self):
        """Save failed extraction entries for debugging."""
        failed_extractions = {}
        
        for model_name, model_results in self.results.items():
            failed_items = [
                {
                    "idx": item.get("idx"),
                    "type": item.get("type",""),
                    "question": item.get("question"),
                    "assistant": item.get("assistant")
                }
                for item in model_results
                if not item.get("extraction_success", False)
            ]
            
            if failed_items:
                failed_extractions[model_name] = failed_items
        
        if failed_extractions:
            failed_file = os.path.join(self.output_dir, "failed_extractions.json")
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_extractions, f, ensure_ascii=False, indent=2)
            
            total_failed = sum(len(items) for items in failed_extractions.values())
            print(f"Failed: {total_failed} items -> {failed_file}")
    
    def generate_summary(self):
        """Generate and save extraction statistics summary."""
        summary = {}
        
        for model_name, model_results in self.results.items():
            total = len(model_results)
            success = sum(1 for r in model_results if r["extraction_success"])
            
            type_stats = {}
            for result in model_results:
                task_type = result.get("type","")
                if task_type not in type_stats:
                    type_stats[task_type] = {"total": 0, "success": 0}
                type_stats[task_type]["total"] += 1
                if result["extraction_success"]:
                    type_stats[task_type]["success"] += 1
            
            summary[model_name] = {
                "total": total,
                "success": success,
                "success_rate": success / total if total > 0 else 0,
                "type_stats": type_stats
            }
        
        
        # Print summary
        for model_name, stats in summary.items():
            print(f"\n{model_name}: {stats['success']}/{stats['total']} ({stats['success_rate']:.1%})")
            for task_type, type_stat in stats['type_stats'].items():
                rate = type_stat['success'] / type_stat['total'] if type_stat['total'] > 0 else 0
                print(f"  {task_type}: {type_stat['success']}/{type_stat['total']} ({rate:.1%})")
    
    def run(self):
        """Run extraction process."""
        for model_name, filepath in self.input_files.items():
            model_results = self.process_file(model_name, filepath)
            self.results[model_name] = model_results
        
        self.save_results()
        self.generate_summary()


def main():
    """Main function for answer extraction."""
    parser = argparse.ArgumentParser(description="Extract answers from model evaluation results")
    parser.add_argument('--file', '-f', type=str, required=True,
                       help='Path to JSONL file to process')
    parser.add_argument('--coord_format', type=str, default='absolute',
                       choices=['absolute', 'relative_1000', 'relative_1'],
                       help='Coordinate format of the model output (default: absolute)')
    args = parser.parse_args()
    
    extractor = AnswerExtractor(single_file_path=args.file, coord_format=args.coord_format)
    
    if not extractor.input_files:
        print("Error: No files to process")
        return
    
    extractor.run()


if __name__ == "__main__":
    main()
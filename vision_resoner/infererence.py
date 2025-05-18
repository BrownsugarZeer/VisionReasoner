# test_qwen_vl_model.py
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from models.vision_reasoner_model import VisionReasonerModel
from utils import visualize_results_enhanced

def main():
    parser = argparse.ArgumentParser(description="Test unified vision model on a single image")
    parser.add_argument("--model_path", type=str, default='Ricky06662/VisionReasoner-7B', help="Path to the model")
    parser.add_argument("--image_path", type=str, default='assets/airplanes.png', help="Path to the input image")
    parser.add_argument("--query", type=str, default='How many airplanes are there in this image?', help="Query/instruction for the model")
    parser.add_argument("--task", type=str, choices=["auto", "detection", "segmentation", "counting", "qa"], 
                        default="auto", help="Task type (default: auto)")
    parser.add_argument("--output_path", type=str, default="result_visualization.png", help="Path to save the output visualization")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = VisionReasonerModel(args.model_path)
    
    # Load image
    print(f"Loading image from {args.image_path}...")
    image = Image.open(args.image_path).convert("RGB")
    
    # Determine task type
    task_type = args.task
    if task_type == "auto":
        result, task_type = model.process_single_image(image, args.query, return_task_type=True)
        print(f"Task type: {task_type}")
    elif task_type == "detection":
        result = model.detect_objects(image, args.query)
    elif task_type == "segmentation":
        result = model.segment_objects(image, args.query)
    elif task_type == "counting":
        result = model.count_objects(image, args.query)
    else:  # QA
        result = model.answer_question(image, args.query)
    
    # Print results
    print("\n===== Results =====")
    print("User question: ", args.query)
    
    print("response: ", result)
    if 'thinking' in result and result['thinking'].strip() != "":
        print("Thinking process: ", result['thinking'])
    
    if task_type == "detection":
        print(f"Total number of detected objects: {len(result['bboxes'])}")
    elif task_type == "segmentation":
        print(f"Total number of segmented objects: {len(result['bboxes'])}")
    elif task_type == "counting":
        print(f"Total number of interested objects is: {result['count']}")
    else:  # QA
        print(f"The answer is: {result['answer']}")
    
    # Visualize results with the new three-panel layout
    visualize_results_enhanced(image, result, task_type, args.output_path)
    print(f"\nResult visualization saved as '{args.output_path}'")



if __name__ == "__main__":
    main()
import os
import json
import glob
import numpy as np
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="folder path of output files")
    return parser.parse_args()

def calculate_metrics(output_dir):
    # get all output files
    output_files = sorted(glob.glob(os.path.join(output_dir, "output_*.json")))
    
    if not output_files:
        print(f"cannot find output files in {output_dir}")
        return
    
    # for accumulating all data
    all_counts = []
    
    # read and process all files
    for file_path in output_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        # process all items in each file
        for item in results:
            pred_count = item['pred_count']
            gt_count = item['gt_count']
            
            all_counts.append({
                'image_id': item['image_id'],
                'mae': abs(pred_count - gt_count),
                'rmse': (pred_count - gt_count) ** 2,
                'correct_count': pred_count == gt_count
            })
            
    # calculate mae and rmse
    mae = np.mean([item['mae'] for item in all_counts])
    rmse = np.sqrt(np.mean([item['rmse'] for item in all_counts]))
    correct_count = np.mean([item['correct_count'] for item in all_counts])
    # print the results
    print(f"test len: {len(all_counts)}")
    print(f"mae: {mae:.4f}")
    print(f"rmse: {rmse:.4f}")
    print(f"correct_count: {correct_count:.4f}")
    

if __name__ == "__main__":
    args = parse_args()
    calculate_metrics(args.output_dir)

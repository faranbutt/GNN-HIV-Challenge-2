import pandas as pd
from sklearn.metrics import roc_auc_score
import sys
import json
import argparse

def score_submission(submission_file, truth_file='data/test_labels.csv'):
    """
    Calculate ROC-AUC score for a submission.
    """
    try:
        submission = pd.read_csv(submission_file)
        truth = pd.read_csv(truth_file)
        merged = truth.merge(submission, on='graph_id')
        
        if merged.empty:
            print("CRITICAL ERROR: No matching graph_ids found between truth and submission!")
            return 0.0, 0
        if merged['probability'].isnull().any():
            print("ERROR: Submission contains NaN probabilities.")
            return 0.0, len(merged)
        score = roc_auc_score(merged['target'], merged['probability'])
        return score, len(merged)
    except Exception as e:
        print(f"Error during scoring: {e}")
        return 0.0, 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_file", help="Path to the submission CSV")
    parser.add_argument("--json", action="store_true", help="Output JSON for CI parsing")
    args = parser.parse_args()

    score, count = score_submission(args.submission_file)
    
 
    print(f"Evaluation completed on {count} samples.")
    print(f"Submission ROC-AUC Score: {score:.4f}")
    
   
    if args.json:
        print("---")
        print(json.dumps({"roc_auc": float(score), "n_samples": count}))
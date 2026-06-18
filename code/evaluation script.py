import pandas as pd
import numpy as np
import sys
import os


def evaluate_accuracy(ground_truth_path, predictions_path):
    # 1. Check if files exist
    if not os.path.exists(ground_truth_path):
        print(f"Error: Ground truth file not found at '{ground_truth_path}'")
        return
    if not os.path.exists(predictions_path):
        print(f"Error: Predictions file not found at '{predictions_path}'")
        return

    # 2. Load the datasets
    df_gt = pd.read_csv(ground_truth_path)
    df_pred = pd.read_csv(predictions_path)

    # 3. Align datasets based on image_file
    # Rename columns to avoid confusion during merge
    df_gt = df_gt.rename(columns={
        'verbatimDate': 'gt_date',
        'verbatimDate_confidence': 'gt_date_conf',
        'verbatimLocality': 'gt_locality',
        'verbatimLocality_confidence': 'gt_locality_conf'
    })

    df_pred = df_pred.rename(columns={
        'verbatimDate': 'pred_date',
        'verbatimDate_confidence': 'pred_date_conf',
        'verbatimLocality': 'pred_locality',
        'verbatimLocality_confidence': 'pred_locality_conf'
    })

    merged_df = pd.merge(df_gt, df_pred, on='image_file', how='inner')

    if len(merged_df) == 0:
        print("Warning: No overlapping image_file names found between the two files!")
        print(f"Ground Truth images sample: {list(df_gt['image_file'].head(3))}")
        print(f"Prediction images sample: {list(df_pred['image_file'].head(3))}")
        return

    print(f"Found {len(merged_df)} matching images out of {len(df_gt)} ground truth rows.")

    # 4. Standardize text helper function to prevent false negatives from spaces/casing
    def clean_text(val):
        if pd.isna(val):
            return "missing"
        s = str(val).strip().lower()
        return "missing" if s in ["nan", "", "missing"] else s

    # 5. Standardize numeric helper function for confidence columns
    def clean_numeric(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    # Apply standardization
    merged_df['gt_date_clean'] = merged_df['gt_date'].apply(clean_text)
    merged_df['pred_date_clean'] = merged_df['pred_date'].apply(clean_text)

    merged_df['gt_locality_clean'] = merged_df['gt_locality'].apply(clean_text)
    merged_df['pred_locality_clean'] = merged_df['pred_locality'].apply(clean_text)

    merged_df['gt_date_conf_clean'] = merged_df['gt_date_conf'].apply(clean_numeric)
    merged_df['pred_date_conf_clean'] = merged_df['pred_date_conf'].apply(clean_numeric)

    merged_df['gt_locality_conf_clean'] = merged_df['gt_locality_conf'].apply(clean_numeric)
    merged_df['pred_locality_conf_clean'] = merged_df['pred_locality_conf'].apply(clean_numeric)

    # 6. Calculate Accuracy for each column
    merged_df['date_correct'] = merged_df['gt_date_clean'] == merged_df['pred_date_clean']
    merged_df['date_conf_correct'] = np.isclose(merged_df['gt_date_conf_clean'], merged_df['pred_date_conf_clean'],
                                                atol=1e-3)

    merged_df['locality_correct'] = merged_df['gt_locality_clean'] == merged_df['pred_locality_clean']
    merged_df['locality_conf_correct'] = np.isclose(merged_df['gt_locality_conf_clean'],
                                                    merged_df['pred_locality_conf_clean'], atol=1e-3)

    # Full row match means all four columns are correct simultaneously
    merged_df['full_row_correct'] = (
            merged_df['date_correct'] &
            merged_df['date_conf_correct'] &
            merged_df['locality_correct'] &
            merged_df['locality_conf_correct']
    )

    total_rows = len(merged_df)
    acc_date = merged_df['date_correct'].sum() / total_rows
    acc_date_conf = merged_df['date_conf_correct'].sum() / total_rows
    acc_locality = merged_df['locality_correct'].sum() / total_rows
    acc_locality_conf = merged_df['locality_conf_correct'].sum() / total_rows
    acc_full_row = merged_df['full_row_correct'].sum() / total_rows

    # 7. Print Report Card
    print("\n" + "=" * 50)
    print("              ACCURACY REPORT CARD              ")
    print("=" * 50)
    print(f"1. verbatimDate Accuracy:              {acc_date:.2%}  ({merged_df['date_correct'].sum()}/{total_rows})")
    print(
        f"2. verbatimDate_confidence Accuracy:   {acc_date_conf:.2%}  ({merged_df['date_conf_correct'].sum()}/{total_rows})")
    print(
        f"3. verbatimLocality Accuracy:          {acc_locality:.2%}  ({merged_df['locality_correct'].sum()}/{total_rows})")
    print(
        f"4. verbatimLocality_confidence Acc:    {acc_locality_conf:.2%}  ({merged_df['locality_conf_correct'].sum()}/{total_rows})")
    print("-" * 50)
    print(
        f"👉 PERFECT FULL ROW ACCURACY:          {acc_full_row:.2%}  ({merged_df['full_row_correct'].sum()}/{total_rows})")
    print("=" * 50)

    # 8. Show sample errors for debugging
    errors = merged_df[~merged_df['full_row_correct']]
    if len(errors) > 0:
        print("\n--- Sample Discrepancies (Up to 3) ---")
        sample_errors = errors.head(3)
        for _, row in sample_errors.iterrows():
            print(f"\nImage: {row['image_file']}")
            if not row['date_correct']:
                print(f"  [Date Error]     True: '{row['gt_date']}' | Pred: '{row['pred_date']}'")
            if not row['locality_correct']:
                print(f"  [Locality Error] True: '{row['gt_locality']}' | Pred: '{row['pred_locality']}'")
            if not row['date_conf_correct'] or not row['locality_conf_correct']:
                print(
                    f"  [Confidence Err] True Conf: ({row['gt_date_conf']}, {row['gt_locality_conf']}) | Pred Conf: ({row['pred_date_conf']}, {row['pred_locality_conf']})")


if __name__ == "__main__":
    # You can change these file names depending on your setup
    evaluate_accuracy(ground_truth_path="/home/soffer/kaggle/MuseumSCAT/museumscat-specimen-collection-annotation-task/train.csv",
                      predictions_path="/home/soffer/kaggle/MuseumSCAT/working/submission_pre.csv")
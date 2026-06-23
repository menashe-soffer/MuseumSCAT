import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
from Levenshtein import distance
from sklearn.metrics import auc



def visualize_results(ground_truth_path, predictions_path, predictions_path_2, images_path, idx_list=None):
    # 1. Load and merge datasets
    if ground_truth_path is not None:
        df_gt = pd.read_csv(ground_truth_path).rename(
        columns={'verbatimDate': 'gt_date', 'verbatimLocality': 'gt_locality'})
    df_pred = pd.read_csv(predictions_path).rename(
        columns={'verbatimDate': 'pred_date', 'verbatimLocality': 'pred_locality'})
    df_pred_2 = pd.read_csv(predictions_path_2).rename(
        columns={'verbatimDate': 'pred_date_2', 'verbatimLocality': 'pred_locality_2'})
    if ground_truth_path is not None:
        merged_df = pd.merge(df_gt, df_pred, on='image_file', how='inner')
    else:
        merged_df = df_pred
    merged_df = pd.merge(merged_df, df_pred_2, on='image_file', how='inner')


    # 2. Filter the dataframe based on idx_list
    if idx_list is not None:
        subset = merged_df.iloc[idx_list]
    else:
        subset = merged_df.head(5)

    # 3. Iterate through each selected image
    for idx, row in subset.iterrows():
        # --- Print Ground Truth and Prediction ---
        print(f"\n{'=' * 30}")
        print(f"Index: {idx} | Image: {row['image_file']}")
        try:
            print(f"Ground Truth -> Date: '{row['gt_date']}', Locality: '{row['gt_locality']}'")
            gt_exists = True
        except:
            gt_exists = False # probably test, no gt
        print(f"Prediction   -> Date: '{row['pred_date']}', Locality: '{row['pred_locality']}'")

        if gt_exists and \
            (row['gt_date'].lower() in ['nan', 'missing']) and (row['gt_locality'].lower() in ['nan', 'missing']) \
                and (row['pred_date'].lower() in ['nan', 'missing']) and (row['pred_locality'].lower() in ['nan', 'missing']):
                continue


        # --- Open Matplotlib Window ---
        img_path = os.path.join(images_path, row['image_file'])
        try:
            img = Image.open(img_path)
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            [ax_.imshow(img) for ax_ in ax]
            try:
                ax[0].set_title(f"Image: {row['image_file']}\nGT: {row['gt_date']}, {row['gt_locality']}\nPred: {row['pred_date']}, {row['pred_locality']}")
                ax[1].set_title(f"Image: {row['image_file']}\nGT: {row['gt_date']}, {row['gt_locality']}\nPred: {row['pred_date_2']}, {row['pred_locality_2']}")
            except:
                ax[0].set_title(f"Image: {row['image_file']}\nPred: {row['pred_date']}, {row['pred_locality']}")
                ax[1].set_title(f"Image: {row['image_file']}\nPred: {row['pred_date_2']}, {row['pred_locality_2']}")
            plt.axis('off')
            plt.show(block=True)  # This will pause and open a new window for each image
        except Exception as e:
            print(f"Error loading image '{img_path}': {e}")


import pandas as pd
import numpy as np
import os
from Levenshtein import distance
from sklearn.metrics import auc


def calculate_normalized_edit_distance(s1, s2):
    s1, s2 = str(s1), str(s2)
    dist = distance(s1, s2)
    # Normalize by the length of the longer string
    return dist / max(len(s1), len(s2), 1)


def calculate_aurc(df, error_col, confidence_col):
    # 1. Sort by confidence (highest to lowest)
    df = df.sort_values(by=confidence_col, ascending=False)

    # 2. Calculate cumulative coverage and cumulative error
    # Coverage is the proportion of samples at a given confidence threshold
    coverage = np.arange(1, len(df) + 1) / len(df)

    # Risk is the mean error of the subset of predictions kept
    # We take the cumulative mean of the errors
    cumulative_error = df[error_col].cumsum()
    risk = cumulative_error / np.arange(1, len(df) + 1)

    # 3. Calculate AUC of the Risk-Coverage curve
    return auc(coverage, risk)


def evaluate_competition_metrics(ground_truth_path, predictions_path):
    # Load and merge (keep original renaming logic)
    df_gt = pd.read_csv(ground_truth_path)
    df_pred = pd.read_csv(predictions_path)

    # Merge on image_file
    merged_df = pd.merge(df_gt, df_pred, on='image_file', suffixes=('_gt', '_pred'))

    # Calculate errors (Normalized Edit Distance)
    merged_df['date_error'] = [calculate_normalized_edit_distance(g, p)
                               for g, p in zip(merged_df['verbatimDate_gt'], merged_df['verbatimDate_pred'])]

    merged_df['locality_error'] = [calculate_normalized_edit_distance(g, p)
                                   for g, p in
                                   zip(merged_df['verbatimLocality_gt'], merged_df['verbatimLocality_pred'])]

    # Calculate AURC for both
    aurc_date = calculate_aurc(merged_df, 'date_error', 'verbatimDate_confidence_pred')
    aurc_locality = calculate_aurc(merged_df, 'locality_error', 'verbatimLocality_confidence_pred')

    print(f"--- Competition Metrics ---")
    print(f"AURC Date:     {aurc_date:.4f}")
    print(f"AURC Locality: {aurc_locality:.4f}")
    print(f"Final Score:   {(aurc_date + aurc_locality) / 2:.4f}")


# Call this instead of your previous evaluation function


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
    # evaluate_accuracy(ground_truth_path="/home/soffer/kaggle/MuseumSCAT/museumscat-specimen-collection-annotation-task/train.csv",
    #                   predictions_path="/home/soffer/kaggle/MuseumSCAT/working/submission_pre.csv")
    evaluate_competition_metrics(ground_truth_path="/home/soffer/kaggle/MuseumSCAT/museumscat-specimen-collection-annotation-task/train.csv",
                        predictions_path="/home/soffer/kaggle/MuseumSCAT/working/submission_post_train.csv")
    evaluate_competition_metrics(ground_truth_path="/home/soffer/kaggle/MuseumSCAT/museumscat-specimen-collection-annotation-task/train.csv",
                        predictions_path="/home/soffer/kaggle/MuseumSCAT/working/submission_spelling_train.csv")

    visualize_results(ground_truth_path=None,#"/home/soffer/kaggle/MuseumSCAT/museumscat-specimen-collection-annotation-task/train.csv",
                      predictions_path="/home/soffer/kaggle/MuseumSCAT/working/submission_post_test.csv",
                      predictions_path_2="/home/soffer/kaggle/MuseumSCAT/working/submission_spelling_test.csv",
                      images_path="/home/soffer/kaggle/MuseumSCAT/museumscat-specimen-collection-annotation-task/images", idx_list=np.arange(125))
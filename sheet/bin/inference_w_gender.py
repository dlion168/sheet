import argparse
import pandas as pd
from sheet.evaluation.metrics import calculate
from pathlib import Path

def merge_and_calculate_metrics(results_csv_path, meta_csv_path):
    """
    Merge the `gender` column from bvcc_test_gen.csv into results.csv and calculate metrics.
    """
    # Load the CSV files
    results_df = pd.read_csv(results_csv_path)
    bvcc_df = pd.read_csv(meta_csv_path)

    # Drop duplicates in bvcc_df to ensure unique `wav_path` rows
    bvcc_df = bvcc_df[['wav_path', 'gender']].drop_duplicates()

    # Merge the `gender` column into results.csv
    merged_df = results_df.merge(bvcc_df, on='wav_path', how='left')
    print(merged_df)
    # Save the updated CSV with the merged gender column
    # merged_df.to_csv(output_csv_path, index=False)
    # print(f"Merged CSV saved to {output_csv_path}")

    # Calculate utterance-level metrics
    calculate_metrics(merged_df)

def calculate_metrics(merged_df):
    """
    Calculate and display utterance-level metrics from the merged DataFrame.
    """
    genders = merged_df['gender'].unique()
    for gender in genders:
        if pd.isna(gender):
            continue
        
        print(f"\nMetrics for Gender: {gender}")
        
        # Filter DataFrame by gender
        gender_df = merged_df[merged_df['gender'] == gender]
        
        # Extract the required columns for metrics calculation
        true_mean_scores = gender_df['avg_score'].values
        predict_mean_scores = gender_df['answer'].values

        # Group by system_id to calculate system-level metrics
        grouped = gender_df.groupby('system_id')
        true_sys_mean_scores = grouped['avg_score'].mean().values
        predict_sys_mean_scores = grouped['answer'].mean().values

        # Calculate metrics using the given `calculate` function
        results = calculate(
            true_mean_scores,
            predict_mean_scores,
            true_sys_mean_scores,
            predict_sys_mean_scores,
        )

        # Display results for this gender
        print(
            f'[UTT][ Gender = {gender} ][ MSE = {results["utt_MSE"]:.3f} | LCC = {results["utt_LCC"]:.3f} | SRCC = {results["utt_SRCC"]:.3f} | KTAU = {results["utt_KTAU"]:.3f} ] '
            f'[SYS][ Gender = {gender} ][ MSE = {results["sys_MSE"]:.3f} | LCC = {results["sys_LCC"]:.4f} | SRCC = {results["sys_SRCC"]:.4f} | KTAU = {results["sys_KTAU"]:.3f} ]'
        )

def main():
    parser = argparse.ArgumentParser(
        description="Merge gender column and calculate utterance-level metrics for speech processing."
    )
    parser.add_argument(
        "--results_csv", type=str, required=True, help="Path to the results CSV file."
    )
    parser.add_argument(
        "--meta_csv", type=str, required=True, help="Path to the meta data CSV file with gender column."
    )
    args = parser.parse_args()

    # Call the function to merge data and calculate metrics
    merge_and_calculate_metrics(args.results_csv, args.meta_csv)

if __name__ == "__main__":
    main()

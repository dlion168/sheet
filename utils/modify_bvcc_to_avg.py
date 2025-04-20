import argparse
import pandas as pd
import random

def random_listener(series):
    """
    從當前群組中的 listener_id 值中隨機選一個，作為代表。
    """
    return random.choice(series.tolist())

def average_scores_with_random_listener(in_csv_path, out_csv_path):
    # 讀取原始CSV檔
    df = pd.read_csv(in_csv_path)

    # 要忽略 listener_id，因此在 groupby 時不把 listener_id 包含在分群欄位
    group_cols = [
        'wav_path',
        'system_id',
        'sample_id',
        'phoneme',
        'cluster',
        'reference'
    ]

    # 用 groupby 分群（不含 listener_id），並分別對 score 做 "mean"、對 listener_id 做 "random pick"
    df_result = df.groupby(group_cols, as_index=False).agg({
        'score': 'mean'
    })
    
    # 寫入新的 CSV
    df_result.to_csv(out_csv_path, index=False)

def main():
    parser = argparse.ArgumentParser(description='Compute average score, ignoring listener_id difference, and pick a random listener.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to the output CSV file.')
    args = parser.parse_args()

    average_scores_with_random_listener(args.input_csv, args.output_csv)
    print(f"Done! The averaged CSV is saved to {args.output_csv}.")

if __name__ == "__main__":
    main()

import pandas as pd

# 載入原始 results.csv
results_csv_path = "/bathrooms/ycevan/sheet/egs/sr_mos/exp/ssl-mos-wav2vec2-listener-1337/results/checkpoint-best/train/results.csv"
df = pd.read_csv(results_csv_path)

# 取得檔名中的 sample ID（不含路徑），以及對應的預測分數
df["sample_id_from_path"] = df["wav_path"].apply(lambda x: x.split("/")[-1].replace(".wav", ""))
df_answer = df[["sample_id_from_path", "answer"]]

# 輸出成 answer.txt 格式
answer_txt_path = "/bathrooms/ycevan/sheet/egs/sr_mos/exp/ssl-mos-wav2vec2-listener-1337/results/checkpoint-best/train/answer.txt"
df_answer.to_csv(answer_txt_path, index=False, header=False)

df_answer.head()

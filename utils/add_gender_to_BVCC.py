import os
import hashlib
import csv
import pandas as pd

# 定義年份與性別對應表
gender_mapping = {
    "BC2008": "Male",
    "BC2009": "Male",
    "BC2010": "Male",
    "BC2011": "Female",
    "BC2013": "Female",
    "BC2016": "Female",
}

# 初始化 hash 系統對應的 dict
hashed_system_to_year_gender = {}

# 讀取 secret_sys_mappings.txt
mappings_file = "/bathrooms/ycevan/VoiceMOS2022/main/secret_utt_mappings.txt"

with open(mappings_file, "r", encoding="utf-8") as file:
    for line in file:
        hashed_system, system_name = line.strip().split()
        if system_name.startswith("VCC2016") or system_name.startswith("VCC2018"):
            gender = "Female" if "TF" in system_name else "Male"
            year = "Unknown"
        elif system_name.startswith("VCC2020"):
            gender = "Female" if "TEF" in system_name else "Male"
            year = "2020"
        elif system_name.startswith("ESPnet"):
            gender = "Female"
            year = "Unknown"
        elif system_name.startswith("BC"):
            year = system_name[:6]  # 提取年份
            gender = gender_mapping.get(system_name[:6], "Unknown")
        else:
            gender = "Unknown"
            year = "Unknown"
        
        print(hashed_system.split("-")[1].replace(".wav", ""))
        hashed_system_to_year_gender[hashed_system.split("-")[1].replace(".wav", "")] = {
            "year": year,
            "gender": gender
        }

# 讀取 bvcc_test.csv 並更新性別
bvcc_test_file = "/bathrooms/ycevan/sheet/egs/bvcc/data/bvcc_test.csv"
output_file = "/bathrooms/ycevan/sheet/egs/bvcc/data/bvcc_test_gen.csv"

# 讀取 CSV 並處理
updated_rows = []
with open(bvcc_test_file, "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    fieldnames = reader.fieldnames + ["gender"]

    for row in reader:
        hashed_system = row["sample_id"]

        # 找出對應性別
        gender = hashed_system_to_year_gender.get(hashed_system).get("gender", "Unknown")
        

        row["gender"] = gender
        updated_rows.append(row)

# 寫入新的 CSV
with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(updated_rows)

print(f"處理完成，結果已儲存到 {output_file}")
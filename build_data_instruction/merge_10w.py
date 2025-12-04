import json
import os
import random

# JSON ÎÄ¼þ¼ÐÂ·¾¶
json_folder_path = "/data/user4/cww/Data_Construction/output7_newformat_Qwen_cww/novel_ele_10way/vanilla_ele_50000/base_all"
output_file_path = "/output7_newformat_Qwen_cww/novel_ele_10way/vanilla_ele_50000/base_all_merge_1wx12.json"
json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]

# ³õÊ¼»¯ºÏ²¢Êý¾ÝµÄÁÐ±í
all_data = []
for json_file in json_files:
    file_path = os.path.join(json_folder_path, json_file)
    with open(file_path, 'r') as f:
        data = json.load(f)
        # ´òÂÒÊý¾Ý
        random.shuffle(data)
        # ÌáÈ¡Ç° 1 ÍòÌõ¼ÇÂ¼
        selected_data = data[:10000]
        # ºÏ²¢µ½×ÜÊý¾ÝÁÐ±íÖÐ
        all_data.extend(selected_data)

# ´òÂÒËùÓÐºÏ²¢ºóµÄÊý¾Ý
random.shuffle(all_data)

# ÖØÐÂ±àºÅ
for idx, item in enumerate(all_data):
    item['id'] = f"identity_{idx + 1}"

# Ð´ÈëÐÂµÄ JSON ÎÄ¼þ
with open(output_file_path, 'w') as output_file:
    json.dump(all_data, output_file, indent=4)

print(f"success {output_file_path}. len:{len(all_data)}")

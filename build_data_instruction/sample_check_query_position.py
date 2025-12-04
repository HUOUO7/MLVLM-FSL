import json
import os
import random
import re
from collections import Counter
import random
def check_query_pos(cur_file, save_file, n_way, per_num):
    with open(cur_file,'r') as f:
        data = json.load(f)
    print(len(data))
    random.shuffle(data)
    count_dict = {i: 0 for i in range(1, n_way + 1)}
    new_data=[]
    for item in data:
        # user
        user_value = item['conversations'][0]['value']
        ans_start_index = user_value.find("Output is one of [") + len("Output is one of [")
        ans_end_index = user_value.find("]. Make sure")
        ans_list = user_value[ans_start_index:ans_end_index]
        ans_items = ans_list.split(",")
        ans_list = [ans_item.replace("'","").replace('"','').strip() for ans_item in ans_items]
        # print(ans_list)
        # assistant
        ass_value = item['conversations'][1]['value'].strip().replace("'","").replace('"','').replace('.','').lower()
        # print(ass_value)
        for i in range(len(ans_items)):
            ans_item = ans_items[i].replace("'","").replace('"','').lower().strip()
            # print(ans_item)
            # print(ans_item)
            if ass_value == ans_item:
                if count_dict[i+1] < per_num:
                    count_dict[i+1]+=1
                    new_data.append(item)
                break
    print(count_dict)
    print(len(new_data))
    with open(save_file, 'w') as f:
        json.dump(new_data,f,indent=4)

    return

root  = "/data/user4/cww/Data_Construction/output7_newformat_Qwen_cww/dogs/vanilla_fsl_50000/5way_novel"
save_root  = "/data/user4/cww/Data_Construction/output7_newformat_Qwen_cww/dogs/vanilla_fsl_50000/5way_novel_balance_1000"
n_way = 5
per_num = 1000
files = os.listdir(root)
os.makedirs(save_root,exist_ok=True)
for file in files:
    print(f"============{file}=================")
    cur_file = os.path.join(root, file)
    save_file = os.path.join(save_root, file)
    check_query_pos(cur_file, save_file,n_way, per_num)
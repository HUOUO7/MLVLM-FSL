# Look directly at the main function

import argparse
import itertools
import json
import os
import random
from functools import partial
import open_clip
import torch
from tqdm import tqdm
import re
from sentence_transformers import SentenceTransformer
import csv

class FSL_Dataset(torch.utils.data.Dataset):

    def __init__(self, test):
        self.test = json.load(open(test))

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        identity_id, input_text, ground_truth = self.test[idx]['id'], self.test[idx]['conversations'][0]['value'], \
            self.test[idx]['conversations'][1]['value']

        return {
            'identity_id': identity_id,
            'input_text': input_text,
            'ground_truth': ground_truth
        }


def collate_fn(inputs):
    identity_ids = [_['identity_id'] for _ in inputs]
    input_texts = [_['input_text'] for _ in inputs]
    ground_truths = [_['ground_truth'] for _ in inputs]

    return identity_ids, input_texts, ground_truths


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0


    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def parse_image_path(path):
    parts = path.split('/')
    class_name = parts[-2]
    img_name_with_ext = parts[-1]
    img_name = img_name_with_ext.split('.')[0]
    return class_name, img_name


def evaluate_exact_match_accuracy(results_file, dataset_name):
    results = json.load(open(results_file))

    # ===============direct match===============
    dir_exact_match = 0
    dir_error_match = 0
    occur_match = 0
    for i in tqdm(range(len(results)), desc='Direct Processing'):
        output = str(results[i]['Answer']).strip().lower().replace(" ", "")
        truth = str(results[i]['ground_truth']).strip().lower().replace(" ", "")
        output = re.sub(r'[^\w\s]', '', output)
        truth = re.sub(r'[^\w\s]', '', truth)
        if str(results[i]['Answer']).lower().replace(".", "").strip() == str(
                results[i]['ground_truth']).lower().replace(".", "").strip():
            dir_exact_match += 1
        else:
            dir_error_match += 1
        if truth in output:
            occur_match += 1

    infer_num = len(results)
    # ??????
    dir_accuracy = round(dir_exact_match / infer_num, 6)
    occur_accuracy = round(occur_match / infer_num, 6)
    print(
        f'For the {dataset_name} dataset, the total amount of inference data is: {infer_num}, the number of correct inferences is: {dir_exact_match}, the number of incorrect inferences is: {dir_error_match}, and the dir_accuracy rate is: {dir_accuracy}, occur_accuracy is {occur_accuracy}.')
    return infer_num, dir_exact_match, dir_error_match, dir_accuracy, occur_accuracy


def find_description(class_name, img_name):
    global description_ref_data
    if class_name in description_index:
        return description_index[class_name].get(img_name)
    return None

def find_qwen_idx(identity_id):
    global qwen_ref_data
    if identity_id in qwen_index:
        return qwen_index[identity_id]
    return None


def truncate_description(description, max_length=350):
    truncated_description = {}
    for key, value in description.items():
        if len(value) > max_length:
            truncated_description[key] = value[:max_length] + '...'
        else:
            truncated_description[key] = value
    return truncated_description

def preprocess(text):
    return text.replace('.', '').lower()

if __name__ == '__main__':

# Given the Qwen output file to be processed(Qwen_ref_file), the pre-generated description file(description_ref_file), and the path to save(save_folder)
# Given the attributes name in the description_ref_file(attributes_names)

# The final output file contains the sample subscript that is most similar to the query sample on attribute i in each instruction,
# and the sample subscript order after similarity aggregation

    data_to_write = []
    save_folder = "/data/user4/cww/sentence-transformers-master/test_result/0807_CUB_output_v1_pseudo_all_index.csv" # save folder path
    description_ref_file = "/data/user4/HUOJIAN/MOE-FGFSL/data/data_json/CUB/v1/novel_attributes_global_description.json"  # description generated file
    qwen_ref_file = "/data/user4/HUOJIAN/Qwen-VL/cww_test/output_qwen_5000_balance/0801_fsl4-pseudo4500/CUB.json"  # the first output file of Qwen-VL
    attributes_names = ['color_pattern', 'body_shape_and_size', 'feather_characteristics', 'behavioral_characteristics',
                     'overall_description'] # the attributes name in the description_ref_file
    LM_path = "/data/user4/cww/sentence-transformers-master/model/all-MiniLM-L6-v2/all-MiniLM-L6-v2"


    with open(description_ref_file, 'r', encoding='utf-8') as file:
        description_ref_data = json.load(file)

    with open(qwen_ref_file, 'r', encoding='utf-8') as file:
        qwen_ref_data = json.load(file)

    global description_index
    description_index = {
        entry["class_name"]: entry["description_data"]
        for entry in description_ref_data
    }
    global qwen_index
    qwen_index = {
        entry["id"]: entry["Answer"]
        for entry in qwen_ref_data
    }

    model = SentenceTransformer(LM_path)
    for i, one_data in tqdm(enumerate(qwen_ref_data), total=len(qwen_ref_data),
                                                              desc=f'Processing Cars',
                                                              dynamic_ncols=True):
        identity_id = one_data['id']
        row = {"id": identity_id}
        input_text = str(one_data["Question"])
        ground_truth = preprocess(one_data["ground_truth"])
        row["ground_truth"] = ground_truth
        print("======ground_truth=====")
        print(ground_truth)
        print("=======extract img_path=====")
        print(input_text)
        image_paths = re.findall(r'<img>(.*?)</img>', input_text)

        # extract classname
        print("=======extract classname=====")
        categories = re.findall(r'What is this\?\s(.*?)\.', input_text)
        categories = [preprocess(category) for category in categories]
        row["categories"] = ', '.join(categories)
        ground_idx = categories.index(ground_truth) + 1
        row["ground_idx"] = ground_idx

        for category in categories:
            print(category)

        qwen_output = preprocess(one_data["Answer"])

        if qwen_output in categories:
            qwen_idx_find_in = categories.index(qwen_output) + 1
        else:
            qwen_idx_find_in = qwen_output
        row["qwen_idx"] = qwen_idx_find_in

        descriptions_one_episode = []
# step 1 =retrieve ref description
        for i in range(len(image_paths)):
            image_path = image_paths[i]
            class_name, img_name = parse_image_path(image_path)
            if class_name and img_name:
                description = find_description(class_name, img_name)
                if description:
                    # print(f"Description for {img_name} in {class_name}:")
                    # print(json.dumps(description, indent=4))
                    truncated_description = truncate_description(description)
                    descriptions_one_episode.append(truncated_description)
                else:
                    print(f"No description found for {img_name} in {class_name}.")
            else:
                print("Invalid image path.")
# step2=cal similarity
        sum_array = [0, 0, 0, 0, 0]


        # the attributes name in the description_ref_file
        for attr in attributes_names:

            sentences = []
            for i in range(len(descriptions_one_episode)):
                sentences.append(descriptions_one_episode[i][attr])
            # For attribute i, find the index of the description most similar to query
            embeddings = model.encode(sentences)
            similarities = model.similarity(embeddings, embeddings)
            last_row = similarities[-1, :-1]
            # For attribute i, find the index of the description most similar to query
            most_similar_index = torch.argmax(last_row).item()
            row[attr] = most_similar_index + 1
            # Aggregate the similarity of each attribute i (sum)
            sum_array = [x + y for x, y in zip(last_row, sum_array)]
        sum_tensor = torch.tensor(sum_array)
        # Find the subscript of the aggregated description that is most similar to query
        most_similar_index_avg = torch.argmax(sum_tensor).item()
        row['all_avg'] = most_similar_index_avg + 1

        sorted_indices = torch.argsort(sum_tensor, descending=True).tolist()
        sorted_indices = [idx + 1 for idx in sorted_indices]
        row['all_avg_index'] = '-'.join(map(str, sorted_indices))
        print(row['all_avg_index'])


        data_to_write.append(row)
        print(row)

# step 3 = write to csv file
with open(save_folder, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['id'] + attributes_names + [ 'all_avg', 'all_avg_index', 'qwen_idx', 'ground_idx', 'categories', 'ground_truth']  #The name of the field to write to save file

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for data in data_to_write:
        writer.writerow(data)

print("CSV file has been created successfully.")



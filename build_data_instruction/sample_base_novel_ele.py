import random
import logging
import torch
from torch.utils.data import DataLoader, Sampler, Dataset
from torchvision import datasets, transforms
from collections import Counter
from PIL import Image
import zipfile
import os
from vision_datasets import Usages, ManifestDataset, DatasetHub
import tqdm
from vision_benchmark.datasets import class_map, template_map
import json
import re
def headCapitalize(text):
    words = text.split()
    if words:
        words[0] = words[0].capitalize()
    modified_text = " ".join(words)
    modified_text = modified_text.strip()
    return modified_text

def find_split_imgs(query_img_path, rotate_dir, split_dir):
    extra_part = os.path.relpath(query_img_path,split_dir)
    classname , imgname = extra_part.split('/')
    current_rotate_dir = os.path.join(rotate_dir, classname)
    rotate_imgs = []
    for item in os.listdir(current_rotate_dir):
        name, ext = os.path.splitext(imgname)
        if item.startswith(name):
            item_path = os.path.join(current_rotate_dir,item)
            rotate_imgs.append(item_path)
    return rotate_imgs

def sampled_json(data_root, dataset_dir, split_dir, class_dict, dataset_name, n_way, k_shot, k_query, rotate_dir, num_episodes = None):

    # num_classes = len(class_dict)
    # classes = list(class_dict.keys())
    classes = os.listdir(split_dir)
    num_classes = len(classes)
    # print(classes)
    if n_way > num_classes:
        n_way = num_classes
    if num_episodes is None :
        num_episodes = max(10000, num_classes * 10)   # how many tasks to sample

    class_item_list = {}
    for single_class in classes:
        single_class_dir = os.path.join(split_dir,single_class)
        items = os.listdir(single_class_dir)
        if single_class not in class_item_list:
            class_item_list[single_class] = []
        class_item_list[single_class].extend(items)

    sum_shot = k_shot + k_query
    all_samples = []                                       # store sampled sample
    class_occur = Counter()
    class_list = list(class_item_list) # record each class ocurence
    for _ in tqdm.tqdm(range(num_episodes)):
        selected_ways = random.sample(class_list,n_way)
        class_occur.update(selected_ways)
        support_way_shot_dict = {}
        query_way_shot_dict = {}
        for selected_way in selected_ways:
            support_way_shot_dict[selected_way] = []
            query_way_shot_dict[selected_way] = []
            selected_way_shots = class_item_list[selected_way]
            selected_shots = random.sample(selected_way_shots,sum_shot)
            support_shot = selected_shots[: k_shot]
            query_shot = selected_shots[k_shot: ]
            support_way_shot_dict[selected_way].extend(support_shot)
            query_way_shot_dict[selected_way].extend(query_shot)

        output_text = ""
        label_list = []
        for way in support_way_shot_dict:
            print(way)
            print(class_dict)
            label = class_dict[way]
            label = headCapitalize(label)
            for shot in support_way_shot_dict[way]:
                img_path = os.path.join(split_dir, way, shot)
                # split_img_list = find_split_imgs(img_path, rotate_dir, split_dir)
                # split_img_list.append(img_path)
                # random_image = random.choice(split_img_list)
                # output_text += f"<img_path>{random_image}<img_path> What is this? {label}."
                output_text += f"<img_path>{img_path}<img_path> What is this? {label}."
            label_list.append(label)
        for query_way in query_way_shot_dict:
            query_text = ''
            for query_shot in query_way_shot_dict[query_way]:
                query_label = class_dict[query_way]
                query_label = headCapitalize(query_label)
                query_img_path = os.path.join(split_dir,query_way,query_shot)
                # split_img_list = find_split_imgs(query_img_path, rotate_dir, split_dir)
                # split_img_list.append(query_img_path)
                # random_image = random.choice(split_img_list)
                query_text = output_text
                # query_text += f"So what is this?<img_path>{query_img_path}<img_path>.What is this? <img_path>{split_img_list[0]}<img_path>.What is this? <img_path>{split_img_list[1]}<img_path>.What is this? <img_path>{split_img_list[2]}<img_path>.What is this? <img_path>{split_img_list[3]}<img_path>.Output is one of {label_list}."
                # query_text += f"So what is this?<img_path>{random_image}<img_path>.Output is one of {label_list}.Make sure your output is in the answer list, with no spelling mistakes."
                query_text += f"So what is this?<img_path>{query_img_path}<img_path>.Output is one of {label_list}.Make sure your output is in the answer list, with no spelling mistakes."
                all_samples.append({
                    'input': query_text,
                    'output': f'{query_label}.',
                })

    for class_idx, occur in class_occur.items():
        print(f"Class {class_idx}: {occur} occurrences")

    print("yes")
    len_sample = len(all_samples)
    json.dump(all_samples, open(output_json, 'w'), indent=4)
    print(f"success write {output_json},len:{len_sample}")

def handle_class(dataset_name,split_dir):
    print(dataset_name)
    ## handle label
    if dataset_name == 'CUB':
        items = os.listdir(split_dir)
        processed_names_dict = {}
        for item in items:
            processed_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', item.split('.', 1)[1])
            processed_name = processed_name.replace('_', ' ')
            processed_names_dict[item] = processed_name

    elif dataset_name == 'CIFAR_FS':
        items = os.listdir(split_dir)
        processed_names_dict = {}
        for item in items:
            processed_name = item.replace('_', ' ')
            processed_names_dict[item] = processed_name

    elif dataset_name == 'Mini-ImageNet' or 'tiered_imagenet':
        items = os.listdir(split_dir)
        processed_names_dict = {}
        with open('/data/user4/CWW_Datasets/FSL/Mini-ImageNet/mini_classmap.json', 'r') as file:
            classmap = json.load(file)
        for item in items:
            try:
                value = classmap[item]
            except KeyError:
                print(f"{item} didnt find")
            processed_name = value.replace('_',' ')
            processed_names_dict[item] = processed_name
        # print(processed_names_dict)

    return processed_names_dict

n_way = 5
k_shot = 1
k_query = 1
data_root = '/data/user4/CWW_Datasets/FSL/'
img_saved_path = '/data/user4/CWW_Datasets/classification'
split = 'novel_all'  #base or novel_all
pseudo_flag = False #True or False
rotate_flag = False #True or False
exchange_flag = False
num = 10000
flag = 'vanilla'
# all_datasets_name = ['CUB', 'CIFAR_FS', 'MINI', 'TIERED']
# all_datasets_name = ['tiered_imagenet']
# all_datasets_name = ['food101', 'country211', 'stanford_cars','flower102', 'eurosat_clip', 'caltech_101', 'mnist', 'fer2013', 'fgvc_aircraft', 'iiit_pets', 'resisc45', 'dtd', 'gtsrb']
all_datasets_name = ['oxford_iiit_pets_20211007', 'eurosat_clip_20210930', 'dtd_20211007', 'food_101_20211007', 'resisc45_clip_20210924', 'country211_20210924', 'oxford_flower_102_20211007', 'stanford_cars_20211007', 'mnist_20211008', 'fgvc_aircraft_2013b_variants102_20211007', 'fer_2013_20211008', 'gtsrb_20210923']
to_simple = {'oxford_iiit_pets_20211007':'iiit_pets', 'eurosat_clip_20210930':'eurosat_clip', 'dtd_20211007':'dtd', 'food_101_20211007':'food101', 'resisc45_clip_20210924':'resisc45', 'country211_20210924':'country211','oxford_flower_102_20211007':'flower102', 'stanford_cars_20211007':'stanford_cars', 'caltech_101_20211007':'caltech_101', 'mnist_20211008':'mnist', 'fgvc_aircraft_2013b_variants102_20211007':'fgvc_aircraft', 'fer_2013_20211008':'fer2013', 'gtsrb_20210923':'gtsrb'}
# for dataset in datasets:
#     data_dir = f"/data/user4/cww/MobileVLM_main/data/finetune_data/{dataset}/base/"
#     output_file = f"/data/user4/cww/MobileVLM_main/data/finetune_data/{dataset}/FLAMINGO_Elevater_zero_shot_FT.json"
#     label_file = f"/data/user4/cww/Data_Construction/fsl_data/elevater_label_new/{dataset}_base_label.json"
#     build_data_for_flamingo(data_dir, output_file, label_file, dataset)

for dataset_name in all_datasets_name:
    dataset_dir = f"/data/user4/CWW_Datasets/Elevater_data/classification/{dataset_name}/{split}/"
    split_dir = f"/data/user4/CWW_Datasets/Elevater_data/classification/{dataset_name}/{split}/"
    # dataset_dir = f"/data/user4/cww/MobileVLM_main/data/finetune_data/{dataset_name}/{split}/"
    # split_dir = f"/data/user4/cww/MobileVLM_main/data/finetune_data/{dataset_name}/{split}/"
    rotate_dir = os.path.join(dataset_dir, 'rotate')
    dataname = to_simple[dataset_name]
    if exchange_flag == True:
        flag = 'exchange'
        with open(f"/home/user4/cww/Elevater_Toolkit_IC-main/A_Qwen_format/aug_label/exchange_label/exchange_{dataname}_{split}.json","r")as f:
            class_dict = json.load(f)
            # print(class_dict)
    elif pseudo_flag == True:
        flag = 'pseudo'
        with open(f"/data/user4/cww/Data_Construction/fsl_data/pseudo_elevater_label_new/{dataname}_pseudo_{split}_label.json","r")as f:
            class_dict = json.load(f)
    else:
        with open(f"/data/user4/cww/Data_Construction/fsl_data/elevater_label_new/{dataname}_{split}_label.json") as f:
            class_dict = json.load(f)
    out_dir = f'/data/user4/cww/Data_Construction/output7_newformat_Qwen_cww/novel_ele/{flag}_ele_{num}/{split}/'
    os.makedirs(out_dir, exist_ok=True)
    output_json = f'{out_dir}{dataset_name}_{split}_{n_way}way{k_shot}shot{k_query}query.json'


    try:
        sampled_json(data_root, dataset_dir, split_dir, class_dict, dataset_name, n_way, k_shot, k_query, rotate_dir,num)

    except FileNotFoundError as e:
        print(e)
        continue
    logging.shutdown()


    # if exchange_flag == True:
    #     flag = 'exchange'
    #     with open(f"/home/user4/cww/Elevater_Toolkit_IC-main/output7_newformat_Qwen_cww/exchange_label/exchange_{dataset_name}_{split}_label.json","r")as f:
    #         class_dict = json.load(f)
    #         # print(class_dict)
    # elif pseudo_flag == True:
    #     flag = 'pseudo'
    #     with open(f"/home/user4/cww/Elevater_Toolkit_IC-main/output7_newformat_Qwen_cww/pseudo_label/pseudo_{dataset_name}_{split}_label.json","r")as f:
    #         class_dict = json.load(f)
    # else:
    #     class_dict = handle_class(dataset_name, split_dir)
    # dict1 = handle_class(dataset_name, split_dir)
    # dict2 = class_dict
    # print(dict1)
    # print(dict2)
    # rotate_dir = os.path.join(dataset_dir,'rotate')
    # dict3 = {}
    #
    # for key1, value1 in dict1.items():
    #     value2 = headCapitalize(value1)
    #     if value2 in dict2:
    #         dict3[key1] = dict2[value2]
    #
    # output_path = f"/home/user4/cww/Elevater_Toolkit_IC-main/A_Qwen_format/aug_label/{flag}_label/{flag}_{dataset_name}_{split}.json"
    # with open(output_path, "w") as json_file:
    #     json.dump(dict3, json_file, indent=4)

    # print("write:", output_path)









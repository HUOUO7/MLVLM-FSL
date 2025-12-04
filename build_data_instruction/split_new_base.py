import os
import shutil
import random
import json
import numpy

def split_base_new(name,root_folder, num_classes, train_index, test_index):
    # new dataset path
    train_file = train_index['files_for_local_usage'][0].strip().replace(".zip","")
    test_file = test_index['files_for_local_usage'][0].strip().replace(".zip","")
    print(train_file)
    train_dir = os.path.join('/data/user4/CWW_Datasets/Elevater_data/',root_folder, train_file)
    test_dir = os.path.join('/data/user4/CWW_Datasets/Elevater_data/',root_folder, test_file)
    # random choose base or novel set
    files_and_directories = os.listdir(train_dir)
    num_classes = len(files_and_directories)
    new_files = numpy.random.permutation(files_and_directories)
    split = int(num_classes * 0.7)
    base_set = new_files[:split]
    novel_set = new_files[split:]

    print("========starting split========")
    print(f"{name}-base_set:{base_set}")
    print(f"{name}-novel_set:{novel_set}")
    print(f"{num_classes}:{len(base_set)}:{len(novel_set)}")
    # copy to
    base_dir = os.path.join('/data/user4/CWW_Datasets/Elevater_data/',root_folder,'base')
    novel_dir = os.path.join('/data/user4/CWW_Datasets/Elevater_data/',root_folder,'novel')
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    if os.path.exists(novel_dir):
        shutil.rmtree(novel_dir)
    os.makedirs(base_dir, exist_ok=False)
    os.makedirs(novel_dir, exist_ok=False)
    for base_class in base_set:
        current_train_dir = os.path.join(train_dir,base_class)
        current_test_dir = os.path.join(test_dir,base_class)
        current_base_dir = os.path.join(base_dir, base_class)
        if os.path.exists(current_base_dir):
            shutil.rmtree(current_base_dir)
        shutil.copytree(current_train_dir, current_base_dir)
        for item in os.listdir(current_test_dir):
            item_path = os.path.join(current_test_dir,item)
            if os.path.isdir(item_path):
                for subitem in os.listdir(item_path):
                    test_item = os.path.join(item_path,subitem)
                    dst_item = os.path.join(current_base_dir,item,subitem)
                    shutil.copy2(test_item, dst_item)
            else:
                test_item = os.path.join(current_test_dir,item)
                dst_item = os.path.join(current_base_dir, item)
                shutil.copy2(test_item,dst_item)
    for novel_class in novel_set:
        current_train_dir = os.path.join(train_dir,novel_class)
        current_test_dir = os.path.join(test_dir,novel_class)
        current_novel_dir = os.path.join(novel_dir, novel_class)
        if os.path.exists(current_novel_dir):
            shutil.rmtree(current_novel_dir)
        shutil.copytree(current_train_dir, current_novel_dir)
        for item in os.listdir(current_test_dir):
            item_path = os.path.join(current_test_dir, item)
            if os.path.isdir(item_path):
                for subitem in os.listdir(item_path):
                    test_item = os.path.join(item_path,subitem)
                    dst_item = os.path.join(current_novel_dir,item,subitem)
                    shutil.copy2(test_item, dst_item)
            else:
                test_item = os.path.join(current_test_dir,item)
                dst_item = os.path.join(current_novel_dir, item)
                shutil.copy2(test_item,dst_item)
    print(f"========{name}  split done========")



def get_dataset_info(data_root, dataset_json, img_saved_path, dataset_name):
    with open(dataset_json, 'r') as json_file:
        datasets_info = json.load(json_file)
    for dataset_info in datasets_info:
        name = dataset_info["name"]
        if name == dataset_name:
            root_folder = dataset_info["root_folder"]  # classification/cifar... dataset root path
            num_classes = dataset_info["num_classes"]
            train_index = dataset_info["train"]
            print(name)
            if name == "gtsrb":
                test_index = dataset_info['val']
            else:
                test_index = dataset_info['test']
            split_base_new(name, root_folder, num_classes, train_index, test_index)



# some base info to get datasetnames
data_root = '/data/user4/CWW_Datasets/Elevater_data'
dataset_json = '/data/user4/CWW_Datasets/Elevater_data/DataDownload/classification/resources/vision_split_dataset.json'
img_saved_path = '/data/user4/CWW_Datasets/classification'
allname = []
with open(dataset_json, 'r') as json_file:
    datasets_info = json.load(json_file)
for dataset_info in datasets_info:
    allname.append(dataset_info["name"])
print(allname)
count = 0
for dataset_name in allname:
    get_dataset_info(data_root, dataset_json, img_saved_path,dataset_name)
    count += 1
print(f"all {count} datasets split done")




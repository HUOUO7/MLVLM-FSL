##=================== 调用mllm的api或者直接用模型权重生成文本 ================##

# prompt1：对于一个细粒度狗类数据集，如果我想让你从四个属性方面对狗类数据集中的各种图片进行描述的话，你会选择哪四个属性？
#
# prompt2：如果给你这样一个细粒度狗类数据集中的一张图片，我让你从四个属性方面来对图片进行描述的话，你觉得我该怎样提出prompt来让你回答？请你给出一个示例。
#
# prompt3：（给出数据集的每张图片）。"请根据以下四个属性并从全局整体上描述这张狗类图片，每个方面用一段话回答：
# 1.体型与大小：描述狗的总体大小（如小型、中型、大型）以及身体结构（如瘦长、健壮、矮胖）等等。
# 2.毛发长度与类型：描述狗的毛发类型，如短毛、长毛、卷毛、丝滑等。
# 3.颜色与斑纹：描述狗的毛色及其分布模式（如纯色、斑点、条纹、虎斑等）。
# 4.头部与面部特征：描述狗的头部形状（如圆形、楔形、扁平）、耳朵类型（如直立、垂耳、半垂耳）、鼻子颜色和眼睛形状等。
# 5.全局描述：基于以上四个属性方的回答，根据这些属性描述生成一段对这张狗类图片的总体的简洁描述。"

##==================================================================##


##============================ 与V0相比的改进之处 ========================##
# 1. 为了避免生成的文本中包含非ascii字符（主要针对中文），增加了一个循环，最多尝试10次，如果10次都没有生成ascii字符（纯英文）的文本，则将生成的文本中的非ascii字符过滤掉。
# 2. 限制了生成的文本长度不超过100个字符。
##==================================================================##


import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import torch


def get_description(model, tokenizer, query):
    num_try = 0
    while True:
        res, _ = model.chat(tokenizer, query=query, history=None)
        num_try += 1
        if res.isascii():
            break
        if num_try > 10:
            res = ''.join(filter(str.isascii, res))
            break
    assert res is not None
    return res


def generate_attributes_global_description(model, tokenizer, FG_dataset_path, split, save_path):
    for s in split:
        data_json = []
        split_path = os.path.join(FG_dataset_path, s)
        print(f"Processing {split_path}...")
        for class_name in tqdm(os.listdir(split_path), desc='Processing'):
            class_path = os.path.join(split_path, class_name)
            data_json.append({'class_name': class_name, 'description_data': {}})
            class_dict = next((item for item in data_json if item['class_name'] == class_name), None)
            assert class_dict is not None

            for file in tqdm(os.listdir(class_path), desc='Processing'):
                if file.endswith('.jpg'):
                    image_path = os.path.join(class_path, file)
                    image_name = file.split('.')[0]

                    shape_and_size_query = f'Picture : <img>{image_path}</img>\n Please describe in detail the dog\'s overall size (e.g., small, medium, large) and body structure (e.g., lanky, stocky, stocky), etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    shape_and_size_res = get_description(model, tokenizer, shape_and_size_query)
                    print(f"Shape and Size: {shape_and_size_res}")

                    hair_length_and_type_query = f'Picture : <img>{image_path}</img>\n Please describe the dog\'s hair type, such as short hair, long hair, curly hair, silky hair, etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    hair_length_and_type_res = get_description(model, tokenizer, hair_length_and_type_query)
                    print(f"Hair length and type: {hair_length_and_type_res}")

                    color_and_markings_query = f'Picture : <img>{image_path}</img>\n Please describe the dog\'s coat color and its distribution pattern (such as solid color, spots, stripes, tabby, etc.). Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    color_and_markings_res = get_description(model, tokenizer, color_and_markings_query)
                    print(f"Color and markings: {color_and_markings_res}")

                    head_and_facial_features_query = f'Picture : <img>{image_path}</img>\n Please describe the dog\'s head shape (e.g., round, wedge-shaped, flat), ear type (e.g., upright, lop, semi-lop), nose color, eye shape, etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    head_and_facial_features_res = get_description(model, tokenizer,
                                                                      head_and_facial_features_query)
                    print(f"Head and facial features: {head_and_facial_features_res}")

                    overall_query = f'Picture : <img>{image_path}</img>\n Please describe the dog picture as a whole based on the following four attributes: 1. Body type and size: the dog\'s overall size (e.g., small, medium, large) and body structure (e.g., slender, stocky, short and stocky), etc. 2. Hair length and type: the dog\'s hair type, such as short hair, long hair, curly hair, silky hair, etc. 3. Color and markings: the dog\'s hair color and its distribution pattern (e.g., solid color, spots, stripes, tiger stripes, etc.). 4. Head and facial features: the dog\'s head shape (e.g., round, wedge-shaped, flat), ear type (e.g., erect, drooping, semi-drooping), nose color, and eye shape, etc. Please describe the dog picture as a whole based on the above four attributes and give a rich paragraph answer. Make sure your answer does not exceed 100 words.'
                    overall_description = get_description(model, tokenizer, overall_query)
                    print(f"Overall description: {overall_description}")

                    class_dict['description_data'].update({
                        image_name: {
                            'shape_and_size': shape_and_size_res,
                            'hair_length_and_type': hair_length_and_type_res,
                            'color_and_markings': color_and_markings_res,
                            'head_and_facial_features': head_and_facial_features_res,
                            'overall_description': overall_description
                        }
                    })

                    # json格式化输出
        with open(f'{save_path}/{s}_attributes_global_description.json', 'w') as f:
            json.dump(data_json, f, indent=4)


if __name__ == '__main__':
    device = torch.device("cuda:5")

    tokenizer = AutoTokenizer.from_pretrained("/data/user4/HUOJIAN/Qwen-VL/Qwen/Qwen-VL-Chat",
                                              trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        "/data/user4/HUOJIAN/Qwen-VL/Qwen/Qwen-VL-Chat",  # path to the output directory
        device_map=device,
        trust_remote_code=True
    ).eval()

    FG_dataset_name = 'Stanford_Dogs'
    FG_dataset_path = '/data/user4/CWW_Datasets/stanford_dogs_datasets'
    # split = ['novel']
    split = ['base', 'novel']
    save_path = f'/data/user4/HUOJIAN/MOE-FGFSL/data/data_json/{FG_dataset_name}/v1'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    generate_attributes_global_description(model, tokenizer, FG_dataset_path, split, save_path)

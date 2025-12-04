##===================调用mllm的api或者直接用模型权重生成文本================##

# prompt1：对于一个细粒度鸟类数据集，如果我想让你从四个属性方面对鸟类数据集中的各种图片进行描述的话，你会选择哪四个属性？
#
# prompt2：如果给你这样一个细粒度鸟类数据集中的一张图片，我让你从四个属性方面来对图片进行描述的话，你觉得我该怎样提出prompt来让你回答？请你给出一个示例。
#
# prompt3：（给出数据集的每张图片）。"请根据以下四个属性并从全局整体上描述这张鸟类图片，每个方面用一段话回答：
# 1.颜色模式：描述鸟类身体各部分（如头部、背部、腹部、翅膀和尾巴）的主要颜色和颜色分布。
# 2.体型和形状：描述鸟类的整体大小和体型特征，包括头部、喙、翅膀、尾巴和腿的形状。
# 3.羽毛特征：描述鸟类羽毛的类型和结构，包括羽毛的长度、纹理和排列方式。
# 4.行为特征：描述鸟类在图片中的行为和姿态，例如站立、飞行、觅食或其他活动。"
# 5.全局描述：基于以上四个属性方的回答，根据这些属性描述生成一段对这张鸟类图片的总体的简洁描述。

##==================================================================##

import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import torch


def generate_attributes_global_description(model, tokenizer, FG_dataset_path, split, save_path):
    # response, history = model.chat(tokenizer, query='Picture : <img>/data/user4/CWW_Datasets/FSL/CUB/base/005.Crested_Auklet/Crested_Auklet_0001_794941.jpg</img>\n Please describe this bird picture based on the following four attributes and as a whole: 1. Color pattern: Describe the main color and color distribution of each part of the bird\'s body (such as the head, back, abdomen, wings and tail). 2. Body shape and shape: Describe the overall size and body characteristics of the bird, including the shape of the head, beak, wings, tail and legs. 3. Feather characteristics: Describe the type and structure of the bird\'s feathers, including the length, texture and arrangement of the feathers. 4. Behavioral characteristics: Describe the behavior and posture of the bird in the picture, such as standing, flying, foraging or other activities. 5. Overall description: Make an overall description based on the picture and the description of the above four attributes. Please give a rich paragraph answer for each aspect in the format of "1. Color pattern: 2. Body shape and size: 3. Feather characteristics: 4. Behavioral characteristics: 5. Overall description:".', history=None)

    for s in split:
        data_json = []
        split_path = os.path.join(FG_dataset_path, s)
        for class_name in tqdm(os.listdir(split_path), desc='Processing'):
            class_path = os.path.join(split_path, class_name)
            data_json.append({'class_name': class_name, 'description_data': {}})
            class_dict = next((item for item in data_json if item['class_name'] == class_name), None)
            assert class_dict is not None

            for file in tqdm(os.listdir(class_path), desc='Processing'):
                if file.endswith('.jpg'):
                    image_path = os.path.join(class_path, file)
                    image_name = file.split('.')[0]

                    color_pattern_res, _ = model.chat(tokenizer,
                                                      query=f'Picture : <img>{image_path}</img>\n Please describe in detail the main colors and color distribution of each part of the bird\'s body (such as the head, back, abdomen, wings and tail, etc.) shown in this picture from the perspective of color pattern. Please give a rich paragraph answer.',
                                                      history=None)
                    print(f"Color pattern: {color_pattern_res}")

                    body_shape_and_size_res, _ = model.chat(tokenizer,
                                                            query=f'Picture : <img>{image_path}</img>\n Please describe the size and shape of the bird in this picture in detail, including the shape of its head, beak, wings, tail, legs, etc. Please give a rich paragraph answer.',
                                                            history=None)
                    print(f"Body shape and size: {body_shape_and_size_res}")

                    feather_characteristics_res, _ = model.chat(tokenizer,
                                                                query=f'Picture : <img>{image_path}</img>\n Please describe the type and structure of the feathers of the bird shown in this picture in detail from the perspective of feather characteristics, including the length, texture, and arrangement of the feathers, etc. Please give a rich paragraph answer.',
                                                                history=None)
                    print(f"Feather characteristics: {feather_characteristics_res}")

                    behavioral_characteristics, _ = model.chat(tokenizer,
                                                               query=f'Picture : <img>{image_path}</img>\n Please describe the behavior and posture of the bird shown in this picture in detail from the perspective of behavioral characteristics, such as standing, flying, foraging or other activities, etc. Please give a rich paragraph answer.',
                                                               history=None)
                    print(f"Behavioral characteristics: {behavioral_characteristics}")

                    overall_description, _ = model.chat(tokenizer,
                                                        query=f'Picture : <img>{image_path}</img>\n Please describe the bird picture as a whole based on the following four attributes: 1. Color pattern: the main color and color distribution of the bird\'s body parts (such as the head, back, abdomen, wings and tail). 2. Size and shape: the overall size and shape of the bird, including the shape of the head, beak, wings, tail and legs. 3. Feather characteristics: the type and structure of the bird\'s feathers, including the length, texture and arrangement of the feathers. 4. Behavioral characteristics: the behavior and posture of the bird in the picture, such as standing, flying, foraging or other activities. Please describe the bird picture as a whole based on the above four attributes and give a rich paragraph answer.',
                                                        history=None)
                    print(f"Overall description: {overall_description}")

                    class_dict['description_data'].update({
                        image_name: {
                            'color_pattern': color_pattern_res,
                            'body_shape_and_size': body_shape_and_size_res,
                            'feather_characteristics': feather_characteristics_res,
                            'behavioral_characteristics': behavioral_characteristics,
                            'overall_description': overall_description
                        }
                    })

                    # json格式化输出
        with open(f'{save_path}/{s}_attributes_global_description.json', 'w') as f:
            json.dump(data_json, f, indent=4)


if __name__ == '__main__':
    device = torch.device("cuda:7")

    tokenizer = AutoTokenizer.from_pretrained("/data/user4/HUOJIAN/Qwen-VL/Qwen/Qwen-VL-Chat",
                                              trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        "/data/user4/HUOJIAN/Qwen-VL/Qwen/Qwen-VL-Chat",  # path to the output directory
        device_map=device,
        trust_remote_code=True
    ).eval()

    FG_dataset_name = 'CUB'
    FG_dataset_path = '/data/user4/CWW_Datasets/FSL/CUB'
    split = ['base', 'novel', 'val']
    save_path = f'/data/user4/HUOJIAN/MOE-FGFSL/data/data_json/{FG_dataset_name}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    generate_attributes_global_description(model, tokenizer, FG_dataset_path, split, save_path)

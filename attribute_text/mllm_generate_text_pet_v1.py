##=================== 调用mllm的api或者直接用模型权重生成文本 ================##

# prompt1：对于一个细粒度宠物类数据集，如果我想让你从四个属性方面对宠物类数据集中的各种图片进行描述的话，你会选择哪四个属性？
#
# prompt2：如果给你这样一个细粒度宠物类数据集中的一张图片，我让你从四个属性方面来对图片进行描述的话，你觉得我该怎样提出prompt来让你回答？请你给出一个示例。
#
# prompt3：（给出数据集的每张图片）。"请根据以下四个属性并从全局整体上描述这张宠物类图片，每个方面用一段话回答：
# 1.毛色与图案：描述宠物的毛发颜色、毛发图案类型（斑点、条纹等）、光泽度、长度及任何独特标记等等。
# 2.体型与体态：描述宠物的体型大小、身体比例、肌肉状态、四肢特点及体态特点等等。
# 3.面部特征：描述宠物的眼睛颜色与形状、鼻头与口鼻轮廓、胡须的独特性及面部结构所体现的表情特征等等。
# 4.环境和附属物：描绘宠物所在背景（室内、室外、自然环境等），佩戴的附属物（项圈、衣物等）、互动对象及场景中的季节性或生活相关的元素等等。
# 5.全局描述：基于以上四个属性方的回答，根据这些属性描述生成一段对这张宠物类图片的总体的简洁描述。"

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

                    coat_color_and_pattern_query = f'Picture : <img>{image_path}</img>\n Please describe in detail the pet\'s coat color, coat pattern type (spots, stripes, etc.), glossiness, length, and any unique markings, etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    coat_color_and_pattern_res = get_description(model, tokenizer, coat_color_and_pattern_query)
                    print(f"Coat color and pattern: {coat_color_and_pattern_res}")

                    size_and_posture_query = f'Picture : <img>{image_path}</img>\n Please describe the pet\'s size, body proportions, muscle condition, limb characteristics, and body characteristics, etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    size_and_posture_res = get_description(model, tokenizer, size_and_posture_query)
                    print(f"Hair length and type: {size_and_posture_res}")

                    facial_features_query = f'Picture : <img>{image_path}</img>\n Please describe the pet\'s eye color and shape, nose and muzzle contours, whisker uniqueness, and facial expression characteristics reflected in the structure of the face, etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    facial_features_res = get_description(model, tokenizer, facial_features_query)
                    print(f"Color and markings: {facial_features_res}")

                    environment_and_accessories_query = f'Picture : <img>{image_path}</img>\n Please describe the pet\'s background (indoor, outdoor, natural environment, etc.), accessories worn (collars, clothing, etc.), interactive objects, and seasonal or life-related elements in the scene, etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    environment_and_accessories_res = get_description(model, tokenizer,
                                                                      environment_and_accessories_query)
                    print(f"Head and facial features: {environment_and_accessories_res}")

                    overall_query = f'Picture : <img>{image_path}</img>\n Please describe the pet picture as a whole based on the following four attributes: 1. Coat color and pattern: the pet\'s coat color, coat pattern type (spots, stripes, etc.), glossiness, length, and any unique markings, etc. 2. Size and posture: the pet\'s body size, body proportions, muscle condition, limb characteristics, and posture characteristics, etc. 3. Facial features: the pet\'s eye color and shape, nose and muzzle contours, whisker uniqueness, and facial expression characteristics reflected in the structure of the face, etc. 4. Environment and accessories: the pet\'s background (indoor, outdoor, natural environment, etc.), accessories worn (collars, clothing, etc.), interactive objects, and seasonal or life-related elements in the scene, etc. Please describe the pet picture as a whole based on the above four attributes and give a rich paragraph answer. Make sure your answer does not exceed 100 words.'
                    overall_description = get_description(model, tokenizer, overall_query)
                    print(f"Overall description: {overall_description}")

                    class_dict['description_data'].update({
                        image_name: {
                            'coat_color_and_pattern': coat_color_and_pattern_res,
                            'size_and_posture': size_and_posture_res,
                            'facial_features': facial_features_res,
                            'environment_and_accessories': environment_and_accessories_res,
                            'overall_description': overall_description
                        }
                    })

                    # json格式化输出
        with open(f'{save_path}/{s}_attributes_global_description.json', 'w') as f:
            json.dump(data_json, f, indent=4)


if __name__ == '__main__':
    device = torch.device("cuda:3")

    tokenizer = AutoTokenizer.from_pretrained("/data/user4/HUOJIAN/Qwen-VL/Qwen/Qwen-VL-Chat",
                                              trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        "/data/user4/HUOJIAN/Qwen-VL/Qwen/Qwen-VL-Chat",  # path to the output directory
        device_map=device,
        trust_remote_code=True
    ).eval()

    FG_dataset_name = 'Oxford_Pets'
    FG_dataset_path = '/data/user4/CWW_Datasets/Elevater_data/classification/oxford_iiit_pets_20211007'
    # split = ['novel']
    split = ['base', 'novel']
    save_path = f'/data/user4/HUOJIAN/MOE-FGFSL/data/data_json/{FG_dataset_name}/v1'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    generate_attributes_global_description(model, tokenizer, FG_dataset_path, split, save_path)

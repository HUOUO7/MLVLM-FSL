##=================== 调用mllm的api或者直接用模型权重生成文本 ================##

# prompt1：对于一个细粒度小汽车类数据集，如果我想让你从四个属性方面对小汽车类数据集中的各种图片进行描述的话，你会选择哪四个属性？
#
# prompt2：如果给你这样一个细粒度小汽车类数据集中的一张图片，我让你从四个属性方面来对图片进行描述的话，你觉得我该怎样提出prompt来让你回答？请你给出一个示例。
#
# prompt3：（给出数据集的每张图片）。"请根据以下四个属性并从全局整体上描述这张小汽车类图片，每个方面用一段话回答：
# 1.车身外形和尺寸：描述车辆的整体类别（如轿车、SUV、跑车等），尺寸级别，以及车身的长度、宽度、高度等基本尺寸。
# 2.前脸设计：描述车辆前脸设计，可以涵盖进气格栅的形状与风格、前大灯和日行灯的设计、保险杠的造型，以及品牌标识等等。
# 3.轮毂和轮胎：描述轮毂的材质、尺寸、轮辐设计，以及轮胎的型号、尺寸和花纹等等。
# 4.车身细节和装饰：描述车辆的侧裙、后扰流板、车窗轮廓、后视镜特色、排气系统设计、车顶附件（如行李架）以及车身的个性化装饰（贴花、装饰条、尾灯设计等）等等。
# 5.全局描述：基于以上四个属性方的回答，根据这些属性描述生成一段对这张车辆类图片的总体的简洁描述。"

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

                    body_shape_and_size_query = f'Picture : <img>{image_path}</img>\n Please describe in detail the the category of the vehicle (such as sedan, SUV, sports car, etc.), size level, and basic dimensions such as length, width, height, etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    body_shape_and_size_res = get_description(model, tokenizer, body_shape_and_size_query)
                    print(f"Body shape and size: {body_shape_and_size_res}")

                    front_face_design_query = f'Picture : <img>{image_path}</img>\n Please describe the front face design of the vehicle, which can include the shape and style of the air intake grille, the design of the headlights and daytime running lights, the shape of the bumper, and the brand logo, etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    front_face_desig_res = get_description(model, tokenizer, front_face_design_query)
                    print(f"Front face design: {front_face_desig_res}")

                    wheels_and_tires_query = f'Picture : <img>{image_path}</img>\n Please describe the material, size, spoke design of the wheels, as well as the model, size and pattern of the tires, etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    wheels_and_tires_res = get_description(model, tokenizer, wheels_and_tires_query)
                    print(f"Wheels and tires: {wheels_and_tires_res}")

                    body_details_and_decoration_query = f'Picture : <img>{image_path}</img>\n Please describe the vehicle\'s side skirts, rear spoiler, window profile, rearview mirror features, exhaust system design, roof accessories (such as luggage racks), and personalized decoration of the body (decals, decorative strips, taillight design, etc.), etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    body_details_and_decoration_res = get_description(model, tokenizer,
                                                                      body_details_and_decoration_query)
                    print(f"Body details and decoration: {body_details_and_decoration_res}")

                    overall_query = f'Picture : <img>{image_path}</img>\n Please describe the vehicle picture as a whole based on the following four attributes: 1. Body shape and size: the type of vehicle (such as sedan, SUV, sports car, etc.), size level, and basic dimensions such as length, width, height, etc. 2. Front face design: the front face design of the vehicle, which can include the shape and style of the air intake grille, the design of the headlights and daytime running lights, the shape of the bumper, and the brand logo, etc. 3. Wheels and tires: the material, size, spoke design of the wheels, as well as the model, size and pattern of the tires, etc. 4. Body details and decoration: the vehicle\'s side skirts, rear spoiler, window profile, rearview mirror features, exhaust system design, roof accessories (such as luggage racks), and personalized decoration of the body (decals, decorative strips, taillight design, etc.), etc. Please describe the vehicle picture as a whole based on the above four attributes and give a rich paragraph answer. Make sure your answer does not exceed 100 words.'
                    overall_description = get_description(model, tokenizer, overall_query)
                    print(f"Overall description: {overall_description}")

                    class_dict['description_data'].update({
                        image_name: {
                            'body_shape_and_size': body_shape_and_size_res,
                            'front_face_design': front_face_desig_res,
                            'wheels_and_tires': wheels_and_tires_res,
                            'body_details_and_decoration': body_details_and_decoration_res,
                            'overall_description': overall_description
                        }
                    })

                    # json格式化输出
        with open(f'{save_path}/{s}_attributes_global_description.json', 'w') as f:
            json.dump(data_json, f, indent=4)


if __name__ == '__main__':
    device = torch.device("cuda:1")

    tokenizer = AutoTokenizer.from_pretrained("/data/user4/HUOJIAN/Qwen-VL/Qwen/Qwen-VL-Chat",
                                              trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        "/data/user4/HUOJIAN/Qwen-VL/Qwen/Qwen-VL-Chat",  # path to the output directory
        device_map=device,
        trust_remote_code=True
    ).eval()

    FG_dataset_name = 'Stanford_Cars'
    FG_dataset_path = '/data/user4/CWW_Datasets/Elevater_data/classification/stanford_cars_20211007'
    # split = ['novel']
    split = ['base', 'val']
    # split = ['base', 'novel', 'val']
    save_path = f'/data/user4/HUOJIAN/MOE-FGFSL/data/data_json/{FG_dataset_name}/v1'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    generate_attributes_global_description(model, tokenizer, FG_dataset_path, split, save_path)

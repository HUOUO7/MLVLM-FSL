##=================== 调用mllm的api或者直接用模型权重生成文本 ================##

# prompt1：对于一个细粒度飞机类数据集，如果我想让你从四个属性方面对飞机类数据集中的各种图片进行描述的话，你会选择哪四个属性？
#
# prompt2：如果给你这样一个细粒度飞机类数据集中的一张图片，我让你从四个属性方面来对图片进行描述的话，你觉得我该怎样提出prompt来让你回答？请你给出一个示例。
#
# prompt3：（给出数据集的每张图片）。"请根据以下四个属性并从全局整体上描述这张飞机类图片，每个方面用一段话回答：
# 1.机翼形态：描述机翼的形状和布局。如机翼的位置（如高翼、中翼或低翼布局）、机翼数量（单翼或双翼等）、以及翼尖形状（方形、圆弧形等）或者是否具有前掠翼或后掠翼等等。
# 2.机身设计：描述机身的长度、宽度、轮廓线、是否有特殊的结构设计（如鸭式布局、无尾布局）、机身的流线型程度以及是否有明显的机体凸起（如驾驶舱形状）等等。
# 3.发动机配置：描述飞机的推进系统，包括发动机的数量、位置（如翼下、尾部）、类型（涡轮喷气、涡扇、螺旋桨）以及发动机的排列方式（如并排、串联）等等。
# 4.尾翼配置：描述垂直尾翼和水平尾翼的形状、大小、数量（如T型尾翼、V型尾翼）以及它们相对于机身的位置等等。
# 5.全局描述：基于以上四个属性方的回答，根据这些属性描述生成一段对这张飞机类图片的总体的简洁描述。"

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

                    wing_shape_query = f'Picture : <img>{image_path}</img>\n Please describe in detail the shape and layout of the wing. Such as the position of the wing (such as high wing, mid-wing or low wing layout), the number of wings (single wing or double wing, etc.), the shape of the wing tip (square, arc, etc.), or whether it has forward-swept wings or swept wings, etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    wing_shape_res = get_description(model, tokenizer, wing_shape_query)
                    print(f"Wing shape: {wing_shape_res}")

                    fuselage_design_query = f'Picture : <img>{image_path}</img>\n Please describe the length, width, contour line of the fuselage, whether there is a special structural design (such as canard layout, tailless layout), the streamlined degree of the fuselage, and whether there is a significant body bulge (such as the shape of the cockpit), etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    fuselage_design_res = get_description(model, tokenizer, fuselage_design_query)
                    print(f"Fuselage design: {fuselage_design_res}")

                    engine_configuration_query = f'Picture : <img>{image_path}</img>\n Please describe the aircraft\'s propulsion system, including the number, location (such as under the wing, tail), type (turbojet, turbofan, propeller) of the engine, and the arrangement of the engine (such as side by side, series), etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    engine_configuration_res = get_description(model, tokenizer, engine_configuration_query)
                    print(f"Engine configuration: {engine_configuration_res}")

                    tail_configuration_query = f'Picture : <img>{image_path}</img>\n Please describe the shape, size, number (such as T-tail, V-tail) of the vertical tail and horizontal tail, and their position relative to the fuselage, etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    tail_configuration_res = get_description(model, tokenizer,
                                                                      tail_configuration_query)
                    print(f"Tail configuration: {tail_configuration_res}")

                    overall_query = f'Picture : <img>{image_path}</img>\n Please describe the aircraft picture as a whole based on the following four attributes: 1. Wing shape: the shape and layout of the wing. Such as the position of the wing (such as high wing, mid-wing or low wing layout), the number of wings (single wing or double wing, etc.), the shape of the wing tip (square, arc, etc.) or whether it has forward-swept wings or swept wings, etc. 2. Fuselage design: the length, width, contour line of the fuselage, whether there is a special structural design (such as canard layout, tailless layout), the streamlined degree of the fuselage, and whether there is a significant body bulge (such as the shape of the cockpit), etc. 3. Engine configuration: the aircraft\'s propulsion system, including the number, location (such as under the wing, tail), type (turbojet, turbofan, propeller) of the engine, and the arrangement of the engine (such as side by side, in series), etc. 4. Tail configuration: the shape, size, number (such as T-tail, V-tail) of the vertical tail and horizontal tail, and their position relative to the fuselage, etc. Please describe the aircraft picture as a whole based on the above four attributes and give a rich paragraph answer. Make sure your answer does not exceed 100 words.'
                    overall_description = get_description(model, tokenizer, overall_query)
                    print(f"Overall description: {overall_description}")

                    class_dict['description_data'].update({
                        image_name: {
                            'wing_shape': wing_shape_res,
                            'fuselage_design': fuselage_design_res,
                            'engine_configuration': engine_configuration_res,
                            'tail_configuration': tail_configuration_res,
                            'overall_description': overall_description
                        }
                    })

                    # json格式化输出
        with open(f'{save_path}/{s}_attributes_global_description.json', 'w') as f:
            json.dump(data_json, f, indent=4)


if __name__ == '__main__':
    device = torch.device("cuda:2")

    tokenizer = AutoTokenizer.from_pretrained("/data/user4/HUOJIAN/Qwen-VL/Qwen/Qwen-VL-Chat",
                                              trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        "/data/user4/HUOJIAN/Qwen-VL/Qwen/Qwen-VL-Chat",  # path to the output directory
        device_map=device,
        trust_remote_code=True
    ).eval()

    FG_dataset_name = 'Fgvc_Aircraft'
    FG_dataset_path = '/data/user4/CWW_Datasets/Elevater_data/classification/fgvc_aircraft_2013b_variants102_20211007'
    # split = ['novel']
    split = ['base', 'novel']
    save_path = f'/data/user4/HUOJIAN/MOE-FGFSL/data/data_json/{FG_dataset_name}/v1'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    generate_attributes_global_description(model, tokenizer, FG_dataset_path, split, save_path)

##=================== 调用mllm的api或者直接用模型权重生成文本 ================##

# prompt1：对于一个通用类数据集，如果我想让你从四个属性方面对通用类数据集中的各种图片进行描述的话，你会选择哪四个属性？
#
# prompt2：如果给你这样一个通用类数据集中的一张图片，我让你从四个属性方面来对图片进行描述的话，你觉得我该怎样提出prompt来让你回答？请你给出一个示例。
#
# prompt3：（给出数据集的每张图片）。"请根据以下四个属性并从全局整体上描述这张通用类图片，每个方面用一段话回答：
# 1.颜色和纹理：提供对图像配色方案的详细分析，提及主色和任何显着的对比度或渐变。描述图像中存在的纹理，注意它是光滑的、粗糙的、有图案的还是均匀的。考虑颜色和纹理如何影响图像的整体外观。
# 2.形状：识别和描述图像中存在的几何形状。讨论物体的轮廓和边缘，注意它们的复杂性、对称性和相互关系。突出显示任何独特或不寻常的形状及其在图像上下文中的潜在意义。
# 3.活动：分析图像中描绘的任何动态元素或动作。描述物体之间的任何运动或相互作用，考虑速度、方向和强度等方面。解释这些活动是如何描绘的，以及它们可能传达什么样的情感或叙述。
# 4.场景：描述图像中捕捉到的设置或环境。明确场景是室内还是室外，并确定关键元素，例如自然特征（例如树木、水）或人造结构（例如建筑物、车辆）。讨论场景的灯光、氛围和情绪，并考虑元素的排列如何影响观看者的感知。
# 5.全局描述：基于以上四个属性方的回答，根据这些属性描述生成一段对这张通用类图片的总体的简洁描述。"

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


def generate_attributes_global_description(model, tokenizer, split_path, save_path, split_base_name):
    data_json = []

    class_name_list = []

    print(f"Processing {split_path}...")

    for image_path in tqdm(open(split_path, 'r').readlines(), desc='Processing'):
        image_path = image_path.strip()
        if image_path.endswith('.jpg') or image_path.endswith('.png'):
            class_name = image_path.split('/')[-2]
            if class_name not in class_name_list:
                class_name_list.append(class_name)
                data_json.append({'class_name': class_name, 'description_data': {}})

            class_dict = next((item for item in data_json if item['class_name'] == class_name), None)
            assert class_dict is not None

            image_name = image_path.split('/')[-1].split('.')[0]

            color_and_texture_query = f'Picture : <img>{image_path}</img>\n Please provide a detailed analysis of the image\'s color scheme, mentioning the dominant colors and any notable contrasts or gradients. Describe the texture present in the image, noting whether it is smooth, rough, patterned, or uniform. Consider how color and texture contribute to the overall appearance of the image. Please give a concise answer. Keep your answer to no more than 100 words.'
            color_and_texture_res = get_description(model, tokenizer, color_and_texture_query)
            print(f"Color and Texture: {color_and_texture_res}")

            shape_query = f'Picture : <img>{image_path}</img>\n Please identify and describe the geometric shapes present in the image. Discuss the contours and edges of objects, noting their complexity, symmetry, and relationship to each other. Highlight any distinctive or unusual shapes and their potential significance in the context of the image. Please give a concise answer. Keep your answer to no more than 100 words.'
            shape_res = get_description(model, tokenizer, shape_query)
            print(f"Shape: {shape_res}")

            activity_query = f'Picture : <img>{image_path}</img>\n Please analyze any dynamic elements or actions depicted in the image. Describe any movement or interaction between objects, considering aspects like speed, direction, and intensity. Explain how these activities are portrayed and what emotions or narratives they might convey. Please give a concise answer. Keep your answer to no more than 100 words.'
            activity_res = get_description(model, tokenizer, activity_query)
            print(f"Activity: {activity_res}")

            scene_query = f'Picture : <img>{image_path}</img>\n Please describe the setting or environment captured in the image. Specify whether it is an indoor or outdoor scene and identify key elements such as natural features (e.g., trees, water) or man-made structures (e.g., buildings, vehicles). Discuss the lighting, atmosphere, and mood of the scene, and consider how the arrangement of elements affects the viewer\'s perception. Please give a concise answer. Keep your answer to no more than 100 words.'
            scene_res = get_description(model, tokenizer, scene_query)
            print(f"Scene: {scene_res}")

            overall_query = f'Picture : <img>{image_path}</img>\n Please describe the picture as a whole based on the following four attributes: 1. Color and Texture: a detailed analysis of the image\'s color scheme, mentioning the dominant colors and any notable contrasts or gradients. Describe the texture present in the image, noting whether it is smooth, rough, patterned, or uniform. Consider how color and texture contribute to the overall appearance of the image. 2. Shape: please identify and describe the geometric shapes present in the image. Discuss the contours and edges of objects, noting their complexity, symmetry, and relationship to each other. Highlight any distinctive or unusual shapes and their potential significance in the context of the image. 3. Activity: please analyze any dynamic elements or actions depicted in the image. Describe any movement or interaction between objects, considering aspects like speed, direction, and intensity. Explain how these activities are portrayed and what emotions or narratives they might convey. 4. Scene: please describe the setting or environment captured in the image. Specify whether it is an indoor or outdoor scene and identify key elements such as natural features (e.g., trees, water) or man-made structures (e.g., buildings, vehicles). Discuss the lighting, atmosphere, and mood of the scene, and consider how the arrangement of elements affects the viewer\'s perception. Please describe the picture as a whole based on the above four attributes. Please give a concise answer. Keep your answer to no more than 100 words.'
            overall_description = get_description(model, tokenizer, overall_query)
            print(f"Overall description: {overall_description}")

            class_dict['description_data'].update({
                image_name: {
                    'color_and_texture': color_and_texture_res,
                    'shape': shape_res,
                    'activity': activity_res,
                    'scene': scene_res,
                    'overall_description': overall_description
                }
            })

    # json格式化输出
    with open(f'{save_path}/{split_base_name}_attributes_global_description.json', 'w') as f:
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

    FG_dataset_name = 'TIERED'

    split_path = "/data/user4/HUOJIAN/model-FGFSL/attribute_text/tiered_unique_image_paths3.txt"
    split_base_name = os.path.basename(split_path).split('.')[0]

    save_path = f'/data/user4/HUOJIAN/model-FGFSL/data/data_json/{FG_dataset_name}/v1'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    generate_attributes_global_description(model, tokenizer, split_path, save_path, split_base_name)

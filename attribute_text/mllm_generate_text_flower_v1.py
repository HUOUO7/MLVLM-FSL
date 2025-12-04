##=================== 调用mllm的api或者直接用模型权重生成文本 ================##

# prompt1：对于一个细粒度花类数据集，如果我想让你从四个属性方面对花类数据集中的各种图片进行描述的话，你会选择哪四个属性？
#
# prompt2：如果给你这样一个细粒度花类数据集中的一张图片，我让你从四个属性方面来对图片进行描述的话，你觉得我该怎样提出prompt来让你回答？请你给出一个示例。
#
# prompt3：（给出数据集的每张图片）。"请根据以下四个属性并从全局整体上描述这张花类图片，每个方面用一段话回答：
# 1.花瓣形状与大小：描述花瓣的形状（椭圆形、圆形、尖形等）和大小、花瓣的长度、宽度以及它们之间的比例关系等等。
# 2.颜色及图案：描述花的颜色及其分布模式，是哪种或哪几种颜色组成的单色、双色或多色，以及是否存在斑点、条纹或渐变色等等。
# 3.花萼形状与大小：描述花萼的形态和尺寸（如三角形、卵形等）以及相对于花瓣的大小比例等等。
# 4.结构与排列：描述花序类型（如伞形花序、穗状花序等）、花朵在植株上的排列方式、花朵是否单独开放或成簇，以及花朵中心（如雄蕊和雌蕊的排列）的特征等等。
# 5.全局描述：基于以上四个属性方的回答，根据这些属性描述生成一段对这张花类图片的总体的简洁描述。"

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

                    petal_shape_and_size_query = f'Picture : <img>{image_path}</img>\n Please describe in detail the the shape (oval, round, pointed, etc.) and size of the petals, the length and width of the petals, and the proportional relationship between them, etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    petal_shape_and_size_res = get_description(model, tokenizer, petal_shape_and_size_query)
                    print(f"Petal shape and size: {petal_shape_and_size_res}")

                    color_and_pattern_query = f'Picture : <img>{image_path}</img>\n Please describe the color of the flower and its distribution pattern, whether it is a single color, two colors or multiple colors, and whether there are spots, stripes or gradient colors, etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    color_and_pattern_res = get_description(model, tokenizer, color_and_pattern_query)
                    print(f"Color and pattern: {color_and_pattern_res}")

                    calyx_shape_and_size_query = f'Picture : <img>{image_path}</img>\n Please describe the shape and size of the calyx (such as triangle, oval, etc.) and the size ratio relative to the petals, etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    calyx_shape_and_size_res = get_description(model, tokenizer, calyx_shape_and_size_query)
                    print(f"Calyx shape and size: {calyx_shape_and_size_res}")

                    structure_and_arrangement_query = f'Picture : <img>{image_path}</img>\n Please describe the inflorescence type (such as umbel, spike, etc.), the arrangement of flowers on the plant, whether the flowers bloom alone or in clusters, and the characteristics of the center of the flower (such as the arrangement of stamens and pistils), etc. Please give a rich paragraph answer and keep your answer to no more than 100 words.'
                    structure_and_arrangement_res = get_description(model, tokenizer,
                                                                      structure_and_arrangement_query)
                    print(f"Structure and arrangement: {structure_and_arrangement_res}")

                    overall_query = f'Picture : <img>{image_path}</img>\n Please describe the flower picture as a whole based on the following four attributes: 1. Petal shape and size: the shape (oval, round, pointed, etc.) and size of the petals, the length and width of the petals, and the proportional relationship between them, etc. 2. Color and pattern: the color of the flower and its distribution pattern, which color or colors are composed of single color, double color or multi-color, and whether there are spots, stripes or gradient colors, etc. 3. Calyx shape and size: the shape and size of the calyx (such as triangle, oval, etc.) and the size ratio relative to the petals, etc. 4. Structure and arrangement: inflorescence type (such as umbel, spike, etc.), the arrangement of flowers on the plant, whether the flowers bloom alone or in clusters, and the characteristics of the center of the flower (such as the arrangement of stamens and pistils), etc. Please describe the flower picture as a whole based on the above four attributes and give a rich paragraph answer. Make sure your answer does not exceed 100 words.'
                    overall_description = get_description(model, tokenizer, overall_query)
                    print(f"Overall description: {overall_description}")

                    class_dict['description_data'].update({
                        image_name: {
                            'petal_shape_and_size': petal_shape_and_size_res,
                            'color_and_pattern': color_and_pattern_res,
                            'calyx_shape_and_size': calyx_shape_and_size_res,
                            'structure_and_arrangement': structure_and_arrangement_res,
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

    FG_dataset_name = 'Oxford_Flower'
    FG_dataset_path = '/data/user4/CWW_Datasets/Elevater_data/classification/oxford_flower_102_20211007'
    # split = ['novel']
    # split = ['base', 'val']
    # split = ['base', 'novel', 'val']
    split = ['base', 'novel']
    save_path = f'/data/user4/HUOJIAN/MOE-FGFSL/data/data_json/{FG_dataset_name}/v1'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    generate_attributes_global_description(model, tokenizer, FG_dataset_path, split, save_path)

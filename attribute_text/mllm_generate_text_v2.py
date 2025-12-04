import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import torch


# 改进之处:
# 1.修改 get_description 函数以支持批量处理。
# 2.在 generate_attributes_global_description 函数中，收集所有查询并一次性发送给模型进行处理。

def get_description(model, tokenizer, queries):
    results = []
    for query in queries:
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
        results.append(res)
    return results

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

            queries = []
            image_names = []
            for file in tqdm(os.listdir(class_path), desc='Processing'):
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_path = os.path.join(class_path, file)
                    image_name = file.split('.')[0]
                    image_names.append(image_name)

                    color_and_texture_query = f'Picture : <img>{image_path}</img>\n Please provide a detailed analysis of the image\'s color scheme, mentioning the dominant colors and any notable contrasts or gradients. Describe the texture present in the image, noting whether it is smooth, rough, patterned, or uniform. Consider how color and texture contribute to the overall appearance of the image. Please give a concise answer. Keep your answer to no more than 100 words.'
                    shape_query = f'Picture : <img>{image_path}</img>\n Please identify and describe the geometric shapes present in the image. Discuss the contours and edges of objects, noting their complexity, symmetry, and relationship to each other. Highlight any distinctive or unusual shapes and their potential significance in the context of the image. Please give a concise answer. Keep your answer to no more than 100 words.'
                    activity_query = f'Picture : <img>{image_path}</img>\n Please analyze any dynamic elements or actions depicted in the image. Describe any movement or interaction between objects, considering aspects like speed, direction, and intensity. Explain how these activities are portrayed and what emotions or narratives they might convey. Please give a concise answer. Keep your answer to no more than 100 words.'
                    scene_query = f'Picture : <img>{image_path}</img>\n Please describe the setting or environment captured in the image. Specify whether it is an indoor or outdoor scene and identify key elements such as natural features (e.g., trees, water) or man-made structures (e.g., buildings, vehicles). Discuss the lighting, atmosphere, and mood of the scene, and consider how the arrangement of elements affects the viewer\'s perception. Please give a concise answer. Keep your answer to no more than 100 words.'
                    overall_query = f'Picture : <img>{image_path}</img>\n Please describe the picture as a whole based on the following four attributes: 1. Color and Texture: a detailed analysis of the image\'s color scheme, mentioning the dominant colors and any notable contrasts or gradients. Describe the texture present in the image, noting whether it is smooth, rough, patterned, or uniform. Consider how color and texture contribute to the overall appearance of the image. 2. Shape: please identify and describe the geometric shapes present in the image. Discuss the contours and edges of objects, noting their complexity, symmetry, and relationship to each other. Highlight any distinctive or unusual shapes and their potential significance in the context of the image. 3. Activity: please analyze any dynamic elements or actions depicted in the image. Describe any movement or interaction between objects, considering aspects like speed, direction, and intensity. Explain how these activities are portrayed and what emotions or narratives they might convey. 4. Scene: please describe the setting or environment captured in the image. Specify whether it is an indoor or outdoor scene and identify key elements such as natural features (e.g., trees, water) or man-made structures (e.g., buildings, vehicles). Discuss the lighting, atmosphere, and mood of the scene, and consider how the arrangement of elements affects the viewer\'s perception. Please describe the picture as a whole based on the above four attributes. Please give a concise answer. Keep your answer to no more than 100 words.'

                    queries.extend([color_and_texture_query, shape_query, activity_query, scene_query, overall_query])

            results = get_description(model, tokenizer, queries)
            for i, image_name in enumerate(image_names):
                class_dict['description_data'].update({
                    image_name: {
                        'color_and_texture': results[i * 5],
                        'shape': results[i * 5 + 1],
                        'activity': results[i * 5 + 2],
                        'scene': results[i * 5 + 3],
                        'overall_description': results[i * 5 + 4]
                    }
                })

            with open(f'{save_path}/{s}_attributes_global_description.json', 'w') as f:
                json.dump(data_json, f, indent=4)

if __name__ == '__main__':
    device = torch.device("cuda:6")

    tokenizer = AutoTokenizer.from_pretrained("/data/user4/HUOJIAN/Qwen-VL/Qwen/Qwen-VL-Chat",
                                              trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        "/data/user4/HUOJIAN/Qwen-VL/Qwen/Qwen-VL-Chat",  # path to the output directory
        device_map=device,
        trust_remote_code=True
    ).eval()

    FG_dataset_name = 'FC100'
    FG_dataset_path = '/data/user4/CWW_Datasets/FSL/FC100'
    split = ['base', 'novel','val']
    save_path = f'/data/user4/HUOJIAN/MOE-FGFSL/data/data_json/{FG_dataset_name}/v1'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    generate_attributes_global_description(model, tokenizer, FG_dataset_path, split, save_path)
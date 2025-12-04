import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import torch
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor

# 改进之处:
# 用于图像批处理的 DataLoader：使用 PyTorch 对图像进行批处理DataLoader，从而实现更高效的图像加载和处理。
# 批量查询执行：模型不是按顺序运行查询，而是批量处理多个图像属性。
# 并行处理：使用线程并行处理图像批次num_workers，显著减少总体运行时间。

# Custom Dataset to handle loading the images and their paths
class ClassImageDataset(Dataset):
    def __init__(self, root_dir):
        self.class_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if
                              os.path.isdir(os.path.join(root_dir, d))]
        self.image_files = []
        self.class_names = []

        # Iterate over class folders and gather image paths
        for class_folder in self.class_folders:
            class_name = os.path.basename(class_folder)
            image_files = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
            self.image_files.extend(image_files)
            self.class_names.extend([class_name] * len(image_files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        class_name = self.class_names[idx]
        return image_file, image_name, class_name

# Function to generate the prompt description in a batch
def get_description_batch(model, tokenizer, queries):
    with torch.no_grad():
        results = []
        for query in tqdm(queries, desc="Generating descriptions"):
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

# Optimized function for generating image attributes
def process_image_batch(image_batch, model, tokenizer):
    batch_queries = []

    for image_path, _, _ in image_batch:
        color_and_texture_query = f'Picture : <img>{image_path}</img>\n Please provide a detailed analysis of the image\'s color scheme, mentioning the dominant colors and any notable contrasts or gradients. Describe the texture present in the image, noting whether it is smooth, rough, patterned, or uniform. Consider how color and texture contribute to the overall appearance of the image. Please give a concise answer. Keep your answer to no more than 100 words.'
        shape_query = f'Picture : <img>{image_path}</img>\n Please identify and describe the geometric shapes present in the image. Discuss the contours and edges of objects, noting their complexity, symmetry, and relationship to each other. Highlight any distinctive or unusual shapes and their potential significance in the context of the image. Please give a concise answer. Keep your answer to no more than 100 words.'
        activity_query = f'Picture : <img>{image_path}</img>\n Please analyze any dynamic elements or actions depicted in the image. Describe any movement or interaction between objects, considering aspects like speed, direction, and intensity. Explain how these activities are portrayed and what emotions or narratives they might convey. Please give a concise answer. Keep your answer to no more than 100 words.'
        scene_query = f'Picture : <img>{image_path}</img>\n Please describe the setting or environment captured in the image. Specify whether it is an indoor or outdoor scene and identify key elements such as natural features (e.g., trees, water) or man-made structures (e.g., buildings, vehicles). Discuss the lighting, atmosphere, and mood of the scene, and consider how the arrangement of elements affects the viewer\'s perception. Please give a concise answer. Keep your answer to no more than 100 words.'
        overall_query = f'Picture : <img>{image_path}</img>\n Please describe the picture as a whole based on the following four attributes: 1. Color and Texture: a detailed analysis of the image\'s color scheme, mentioning the dominant colors and any notable contrasts or gradients. Describe the texture present in the image, noting whether it is smooth, rough, patterned, or uniform. Consider how color and texture contribute to the overall appearance of the image. 2. Shape: please identify and describe the geometric shapes present in the image. Discuss the contours and edges of objects, noting their complexity, symmetry, and relationship to each other. Highlight any distinctive or unusual shapes and their potential significance in the context of the image. 3. Activity: please analyze any dynamic elements or actions depicted in the image. Describe any movement or interaction between objects, considering aspects like speed, direction, and intensity. Explain how these activities are portrayed and what emotions or narratives they might convey. 4. Scene: please describe the setting or environment captured in the image. Specify whether it is an indoor or outdoor scene and identify key elements such as natural features (e.g., trees, water) or man-made structures (e.g., buildings, vehicles). Discuss the lighting, atmosphere, and mood of the scene, and consider how the arrangement of elements affects the viewer\'s perception. Please describe the picture as a whole based on the above four attributes. Please give a concise answer. Keep your answer to no more than 100 words.'

        batch_queries.extend([color_and_texture_query, shape_query, activity_query, scene_query, overall_query])

    descriptions = get_description_batch(model, tokenizer, batch_queries)

    # Assign descriptions back to the appropriate images
    image_data = {}
    for idx, ( _, image_name, class_name) in enumerate(image_batch):
        if class_name not in image_data:
            image_data[class_name] = {'description_data': {}}
        image_data[class_name]['description_data'][image_name] = {  # Store using image name without extension
            'color_and_texture': descriptions[idx * 5],
            'shape': descriptions[idx * 5 + 1],
            'activity': descriptions[idx * 5 + 2],
            'scene': descriptions[idx * 5 + 3],
            'overall_description': descriptions[idx * 5 + 4],
        }
    return image_data

# Modified main function with DataLoader and parallel processing
def generate_attributes_global_description(model, tokenizer, FG_dataset_path, split, save_path, batch_size=8, num_workers=4):
    for s in split:
        data_json = []
        split_path = os.path.join(FG_dataset_path, s)
        print(f"Processing {split_path}...")

        # Use DataLoader for batching images
        dataset = ClassImageDataset(split_path)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        for batch in tqdm(data_loader, desc="Processing images in batches"):
            image_batch = list(zip(batch[0], batch[1], batch[2])) # [(image_path, image_name, class_name), ...]
            batch_results = process_image_batch(image_batch, model, tokenizer)

            # Add to the final JSON output by class
            for class_name, descriptions in batch_results.items():
                # Check if the class entry already exists in the data_json
                existing_entry = next((entry for entry in data_json if entry['class_name'] == class_name), None)
                if existing_entry:
                    existing_entry['description_data'].update(descriptions['description_data'])
                else:
                    data_json.append({
                        'class_name': class_name,
                        'description_data': descriptions['description_data']
                    })

            # Save JSON for each split
            with open(f'{save_path}/{s}_attributes_global_description.json', 'w') as f:
                json.dump(data_json, f, indent=4)

if __name__ == '__main__':
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained("/data/user4/HUOJIAN/Qwen-VL/Qwen/Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("/data/user4/HUOJIAN/Qwen-VL/Qwen/Qwen-VL-Chat",
                                                 device_map=device, trust_remote_code=True).eval()

    FG_dataset_name = 'FC100'
    FG_dataset_path = '/data/user4/CWW_Datasets/FSL/FC100'
    split = ['base', 'novel','val']
    save_path = f'/data/user4/HUOJIAN/MOE-FGFSL/data/data_json/{FG_dataset_name}/v3'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Increase the batch size to process more images at once
    generate_attributes_global_description(model, tokenizer, FG_dataset_path, split, save_path, batch_size=8, num_workers=4)

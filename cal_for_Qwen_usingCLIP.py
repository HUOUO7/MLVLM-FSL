# Text similarity between qwen's output and the set of candidate answers using CLIP
# Look main function
import os.path

import torch

from PIL import Image
import open_clip
import pandas as pd
from tqdm import tqdm
import json
import re


def cal_similarity(datasetname, file_dir, output_dir, model_name, model_path, device,n_way):
    global str_record
    # print("===cal similarity====")
    # print(file_dir)
    # print(os.path.exists(file_dir))
    with open(file_dir, 'r') as file:
        data = json.load(file)
    # print(len(data))
    # clip_dict = {0:0,1:0,2:0,3:0,4:0}
    clip_dict = {i: 0 for i in range(0, n_way)}
    pattern = re.compile(r'[^a-zA-Z0-9]')
    with open(output_dir, "w") as f:
        total_samples = len(data)
        correct_predictions = 0
        print(f"data length:{total_samples}")
        f.write(f"data length:{total_samples}\n")



        # if datasetname=='CIFAR_FS':
        #     answer_list = ['sweet pepper', 'bicycle', 'rose', 'whale', 'telephone', 'snail', 'baby', 'woman', 'table', 'fox', 'rocket', 'worm', 'bed', 'poppy', 'man', 'plain', 'wardrobe', 'pickup truck', 'chimpanzee', 'leopard']
        # elif datasetname=='CUB':
        #     answer_list = ['Prairie Warbler', 'Nighthawk', 'Groove billed Ani', 'European Goldfinch', 'Ring billed Gull', 'Frigatebird', 'Nashville Warbler', 'Wilson Warbler', 'Seaside Sparrow', 'Cerulean Warbler', 'White crowned Sparrow', 'Le Conte Sparrow', 'Barn Swallow', 'Rhinoceros Auklet', 'Yellow headed Blackbird', 'Louisiana Waterthrush', 'Northern Flicker', 'Hooded Oriole', 'Olive sided Flycatcher', 'Mangrove Cuckoo', 'Brown Creeper', 'Great Grey Shrike', 'Common Yellowthroat', 'Green Kingfisher', 'Ruby throated Hummingbird', 'Summer Tanager', 'Red faced Cormorant', 'Fox Sparrow', 'Pomarine Jaeger', 'Red legged Kittiwake', 'Chipping Sparrow', 'Black throated Blue Warbler', 'Brown Pelican', 'Painted Bunting', 'Dark eyed Junco', 'Common Tern', 'House Wren', 'Green tailed Towhee', 'American Pipit', 'Pine Grosbeak', 'White eyed Vireo', 'Downy Woodpecker', 'Yellow breasted Chat', 'Pied billed Grebe', 'Western Meadowlark', 'Kentucky Warbler', 'Glaucous winged Gull', 'Pileated Woodpecker', 'Blue headed Vireo', 'White necked Raven']
        # elif datasetname=='TIERED':
        #     answer_list = ['menu', 'spaghetti squash', 'cheeseburger', 'trifle', 'hay', 'washbasin', 'artichoke', 'mortar', 'gar', 'Greater Swiss Mountain dog', 'water tower', 'consomme', 'mushroom', 'kuvasz', 'espresso', 'ladybug', 'sandbar', 'promontory', 'turnstile', 'ice lolly', 'caldron', 'ice cream', 'eel', 'hammerhead', 'ground beetle', 'Bernese mountain dog', 'custard apple', 'komondor', 'affenpinscher', 'sulphur butterfly', 'tench', 'banana', 'electric ray', 'Tibetan mastiff', 'Great Dane', 'cucumber', 'chainlink fence', 'bee', 'briard', 'ladle', 'malinois', 'admiral', 'wine bottle', 'bathtub', 'burrito', 'coral reef', 'walking stick', 'EntleBucher', 'whiskey jug', 'face powder', 'kelpie', 'geyser', 'Granny Smith', 'grille', 'vase', 'rhinoceros beetle', 'alp', 'chocolate sauce', 'schipperke', 'fig', 'sturgeon', 'strawberry', 'great white shark', 'butternut squash', 'water jug', 'volcano', 'cicada', 'zucchini', 'barrel', 'lionfish', 'malamute', 'monarch', 'tiger shark', 'pineapple', 'Border collie', 'ringlet', 'hot pot', 'lemon', 'damselfly', 'picket fence', 'pizza', 'valley', 'coho', 'worm fence', 'red wine', 'head cabbage', 'dough', 'Rottweiler', 'breakwater', 'barracouta', 'cauliflower', 'stingray', 'goldfish', 'cardoon', 'Siberian husky', 'tiger beetle', 'bull mastiff', 'guacamole', 'ant', 'cup', 'dung beetle', 'mantis', 'lakeside', 'beer bottle', 'coffee mug', 'cabbage butterfly', 'Bouvier des Flandres', 'Old English sheepdog', 'weevil', 'Eskimo dog', 'water bottle', 'cockroach', 'lycaenid', 'acorn squash', 'German shepherd', 'pitcher', 'puffer', 'dam', 'jackfruit', 'pill bottle', 'Appenzeller', 'cliff', 'eggnog', 'leafhopper', 'coffeepot', 'boxer', 'groenendael', 'French bulldog', 'long-horned beetle', 'Doberman', 'Saint Bernard', 'plate', 'orange', 'carbonara', 'stone wall', 'sliding door', 'miniature pinscher', 'broccoli', 'pop bottle', 'rain barrel', 'potpie', 'collie', 'lacewing', 'bell pepper', 'pomegranate', 'fly', 'cricket', 'dragonfly', 'teapot', 'beaker', 'bucket', 'leaf beetle', 'grasshopper', 'bannister', 'Shetland sheepdog', 'tub', 'seashore', 'anemone fish', 'rock beauty', 'hotdog']
        # elif datasetname=='MINI':
        #     answer_list = ['trifle', 'bookshop', 'mixing bowl', 'lion', 'crate', 'electric guitar', 'nematode', 'black-footed ferret', 'cuirass', 'vase', 'scoreboard', 'malamute', 'king crab', 'ant', 'African hunting dog', 'school bus', 'dalmatian', 'theater curtain', 'hourglass', 'golden retriever']
        # ans_list_len = len(answer_list)
        # print(f"answer_list length:{ans_list_len}\n")
        # f.write(f"answer_list length:{ans_list_len}\n")

        counter = 0

        with torch.no_grad():
            # answers = tokenizer(answer_list).to(device)
            # answer_features = model.encode_text(answers)
            # answer_features /= answer_features.norm(dim=-1, keepdim=True)
            for _, item in tqdm(enumerate(data), total=total_samples,desc=f'processing {datasetname} '):
                question = item["Question"].strip()
                if question.find("Answer_list: ["):
                    start_index = question.find("Output is one of [") + len("Output is one of [")
                    end_index = question.find("].")
                    answer_list_str = question[start_index:end_index]
                    answer_list = [item.strip()[1:-1] for item in answer_list_str.split(",")]
                    ans_new_list = []
                    for ans in answer_list:
                        ans = ans.replace("'","").replace('"','')
                        ans_new_list.append(ans)
                    print(ans_new_list)
                    answers = tokenizer(ans_new_list).to(device)
                    answer_features = model.encode_text(answers)
                    answer_features /= answer_features.norm(dim=-1, keepdim=True)

                else:
                    print("don not have answer list")
                output = item["Answer"].strip()
                output = re.sub(r'[^\w\s]', '', output)
                truth = item["ground_truth"].strip()
                truth = re.sub(r'[^\w\s]', '', truth)
                query = tokenizer([output]).to(device)
                query_features = model.encode_text(query)
                query_features /= query_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * query_features @ answer_features.T).softmax(dim=-1)
                # similarity_matrices.append(similarity.cpu().numpy())
                # print(similarity)
                max_indices = torch.argmax(similarity, dim=1)
                answer_clip = answer_list[max_indices]
                clip_dict[int(max_indices)] += 1
                answer_clip = re.sub(r'[^\w\s]', '', answer_clip)
                print(f"answer:{answer_clip}  ; truth:{truth}\n")
                f.write(f"answer:{answer_clip}  ; truth:{truth}\n")

                if answer_clip.lower() == truth.lower().replace(".",""):
                    counter += 1
            accuracy_clip = counter / total_samples
            print(f"CLIP_Accuracy: {counter} / {total_samples} = {accuracy_clip}")
            f.write(f"CLIP_Accuracy: {counter} / {total_samples} = {accuracy_clip}\n")
            f.write(f"{clip_dict}")
            print(clip_dict)


        # ===========occurence accuracy================
        dir_exact_match = 0
        correct_predictions = 0
        for item in data:
            output = item["Answer"].strip().lower().replace(" ", "")
            output = re.sub(r'[^\w\s]', '', output)
            truth = item["ground_truth"].strip().lower().replace(" ", "")
            truth = re.sub(r'[^\w\s]', '', truth)
            if str(item["Answer"]).lower().replace(".", "").strip() == str(item["ground_truth"]).lower().replace(".","").strip():
                dir_exact_match += 1
            if truth in output:
                correct_predictions += 1


        dir_exact_match_acc = dir_exact_match / total_samples
        correct_predictions_acc = correct_predictions /total_samples
        accuracy_occur = correct_predictions / total_samples
        print(f"direct_find_truth_Accuracy: {accuracy_occur}")
        f.write(f"direct_find_truth_Accuracy: {accuracy_occur}\n")
        str_record += f"{output_dir}\n {datasetname}\n occur:{accuracy_occur}; {model_name}:{accuracy_clip}({counter} / {total_samples}) \n"
        str_record += f"Qwen-dir:{dir_exact_match_acc}({dir_exact_match}); Qwen-occr:{correct_predictions_acc}({correct_predictions}) \n"
        str_record += f"{clip_dict}"



# ===================main====================
# ===step 1=======
# Given datasetnames(dataset names),modelnames(CLIP L-14 or CLIP BIGG-14)
# Given the files(The directory(section/part) of the model output file)
# str_record is the string to record the results
str_record = ""
model2 = 'Qwen-Chat-int4' #1_7_B 7B
device = torch.device("cuda:2")

# datasetnames = ['CIFAR_FS','CUB','TIERED','MINI', 'fgvc_aircraft', 'oxford_flower_102', 'oxford_iiit_pets','stanford_cars']
datasetnames = ['oxford_flower_102', 'oxford_iiit_pets', 'stanford_cars', 'fgvc_aircraft', 'CUB','MINI','TIERED','CIFAR_FS','3way_novel_balance_500','7way_novel_balance_500','5way_novel_balance_1000','Dogs','FGVC','Flowers','Cars']
modelnames = ['ViT-L-14'] #['ViT-L-14','ViT-bigG-14']
files = [
    'ele_vanilia_3500_0609_random_self_1_history_2multi'
]
n_way = 5
try:
    for file in files:
        for model_name in modelnames:
            print(model_name)
            if model_name == 'ViT-bigG-14':
                model_path = '/data/user4/HUOJIAN/open_clip/clip/CLIP-VIT-bigG/open_clip_pytorch_model.bin'
                save_name = 'BIGG'
            elif model_name == 'ViT-L-14':
                model_path = '/data/user4/cww/MobileVLM/checkpoint/CLIP-ViT-L-14-laion2B-s32B-b82K/open_clip_pytorch_model.bin'
                save_name = 'L14'
            model, _, preprocess = open_clip.create_model_and_transforms(model_name,
                                                                         pretrained=model_path)
            tokenizer = open_clip.get_tokenizer(model_name)
            model.to(device)
            for datasetname in datasetnames:
                # if datasetname =="3way_novel_balance_500":
                #     n_way=3
                # elif datasetname == "7way_novel_balance_500":
                #     n_way=7
                # elif datasetname =="5way_novel_balance_1000":
                #     n_way=5
                print(datasetname)
                file_dir = f"/data/user4/HUOJIAN/Qwen-VL/cww_test/output_qwen_5000_balance/{file}/{datasetname}.json"
                if not os.path.exists(file_dir):
                    print("this?")
                    continue
                output_dir = f'/data/user4/HUOJIAN/Qwen-VL/cww_test/output_qwen_5000_balance/{file}/{datasetname}_Qwen_cal_acc_{save_name}.json'
                cal_similarity(datasetname, file_dir,output_dir,model_name,model_path,device,n_way)
    print(str_record)
except FileNotFoundError as e:
    print(str_record)
    print(datasetname)




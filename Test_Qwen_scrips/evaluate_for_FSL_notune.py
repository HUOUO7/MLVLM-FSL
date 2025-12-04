import argparse
import itertools
import json
import os
import random
from functools import partial
import open_clip
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
import re
class FSL_Dataset(torch.utils.data.Dataset):

    def __init__(self, test):
        self.test = json.load(open(test))

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        identity_id, input_text, ground_truth = self.test[idx]['id'], self.test[idx]['conversations'][0]['value'], \
            self.test[idx]['conversations'][1]['value']

        return {
            'identity_id': identity_id,
            'input_text': input_text,
            'ground_truth': ground_truth
        }


def collate_fn(inputs):
    identity_ids = [_['identity_id'] for _ in inputs]
    input_texts = [_['input_text'] for _ in inputs]
    ground_truths = [_['ground_truth'] for _ in inputs]

    return identity_ids, input_texts, ground_truths


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def clip_cal_similarity(datasetname, file_dir, output_dir, model_name, model_path, device):
    with open(file_dir, 'r') as file:
        data = json.load(file)
    pattern = re.compile(r'[^a-zA-Z0-9]')
    with open(output_dir, "w") as f:
        total_samples = len(data)
        f.write(f"data length:{total_samples}\n")
        counter = 0

        with torch.no_grad():
            for _, item in tqdm(enumerate(data), total=total_samples,desc=f'CLIP processing {datasetname} '):
                question = item["Question"].strip()
                if question.find("Answer_list: ["):
                    start_index = question.find("Output is one of [") + len("Output is one of [")
                    end_index = question.find("].Make sure")
                    answer_list_str = question[start_index:end_index]
                    answer_list = [item.strip()[1:-1] for item in answer_list_str.split(",")]
                    ans_new_list = []
                    for ans in answer_list:
                        ans = ans.replace("'","").replace('"','')
                        ans_new_list.append(ans)
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
                max_indices = torch.argmax(similarity, dim=1)
                answer_clip = answer_list[max_indices]
                f.write(f"answer:{answer_clip}  ; truth:{truth}\n")

                if answer_clip.lower() == truth.lower().replace(".",""):
                    counter += 1
            accuracy_clip = counter / total_samples
            print(f"CLIP_Accuracy: {counter} / {total_samples} = {accuracy_clip}")
            f.write(f"CLIP_Accuracy: {counter} / {total_samples} = {accuracy_clip}\n")
        # ===========occurence accuracy================
        occur_correct_predictions = 0
        for item in data:
            output = item["Answer"].strip().lower().replace(" ", "")
            output = re.sub(r'[^\w\s]', '', output)
            truth = item["ground_truth"].strip().lower().replace(" ", "")
            truth = re.sub(r'[^\w\s]', '', truth)

            if truth in output:
                occur_correct_predictions += 1

        accuracy_occur = occur_correct_predictions / total_samples
        print(f"direct_find_truth_Accuracy: {accuracy_occur}")
        f.write(f"direct_find_truth_Accuracy: {accuracy_occur}\n")
    return counter, accuracy_clip, occur_correct_predictions, accuracy_occur
        # str_record += f"{output_dir}\n {datasetname}\n occur:{accuracy_occur}; {model_name}:{accuracy_clip} \n"
        # with open("/data/user4/cww/LLaVA/checkpoint/LLaVA.evaluation/eval_test_tmp/str_tmp_0521.txt", 'w') as f:
        #     f.write(str_record)

def evaluate_exact_match_accuracy(results_file, dataset_name):
    results = json.load(open(results_file))

    #===============direct match===============
    dir_exact_match = 0
    dir_error_match = 0
    occur_match = 0
    for i in tqdm(range(len(results)), desc='Direct Processing'):
        output = str(results[i]['Answer']).strip().lower().replace(" ","")
        truth = str(results[i]['ground_truth']).strip().lower().replace(" ","")
        output = re.sub(r'[^\w\s]', '', output)
        truth = re.sub(r'[^\w\s]', '', truth)
        if str(results[i]['Answer']).lower().replace(".","").strip() == str(results[i]['ground_truth']).lower().replace(".","").strip():
            dir_exact_match += 1
        else:
            dir_error_match += 1
        if truth in output:
            occur_match += 1

    infer_num = len(results)
    # 保留六位小数
    dir_accuracy = round(dir_exact_match / infer_num, 6)
    occur_accuracy = round(occur_match / infer_num, 6)
    print(
        f'For the {dataset_name} dataset, the total amount of inference data is: {infer_num}, the number of correct inferences is: {dir_exact_match}, the number of incorrect inferences is: {dir_error_match}, and the dir_accuracy rate is: {dir_accuracy}, occur_accuracy is {occur_accuracy}.')
    return infer_num, dir_exact_match, dir_error_match, dir_accuracy, occur_accuracy
    # ============CLIP match=============
    # modelnames = ['ViT-L-14', 'ViT-bigG-14']
    # for model_name in modelnames:
    #     print(model_name)
    #     if model_name == 'ViT-bigG-14':
    #         model_path = '/data/user4/HUOJIAN/open_clip/clip/CLIP-VIT-bigG/open_clip_pytorch_model.bin'
    #         save_name = 'BIGG'
    #     elif model_name == 'ViT-L-14':
    #         model_path = '/data/user4/cww/MobileVLM/checkpoint/CLIP-ViT-L-14-laion2B-s32B-b82K/open_clip_pytorch_model.bin'
    #         save_name = 'L14'
    #     model, _, preprocess = open_clip.create_model_and_transforms(model_name,
    #                                                                  pretrained=model_path)
    #     tokenizer = open_clip.get_tokenizer(model_name)
    #     device = torch.device("cuda:1")
    #     model.to(device)
    #     tmp_str = f"{dataset_name}_{save_name}.json"
    #     output_dir = os.path.join(evaluate_save_folder, tmp_str)
    #     counter, accuracy_clip, occur_correct_predictions, accuracy_occur = clip_cal_similarity(dataset_name, results_file, output_dir, model_name, model_path, device)
    #
    # return infer_num, dir_exact_match, dir_error_match, dir_accuracy, counter, accuracy_clip, occur_correct_predictions, accuracy_occur


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--evaluate_config_path', type=str, default='')
    parser.add_argument('--evaluate_save_folder', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ref_history_path',type=str,default='')
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    # 注意如果是使用微调后的Lora/qlora模型的话，需要使用AutoPeftModelForCausalLM函数来加载模型，否则使用AutoModelForCausalLM函数来加载
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, device_map='cuda', trust_remote_code=True).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint,
                                              trust_remote_code=True)
    evaluate_save_folder = args.evaluate_save_folder
    if not os.path.exists(evaluate_save_folder):
        os.makedirs(evaluate_save_folder, exist_ok=True)

    evaluate_results = []
    # history1 = [
    #     ("Picture 1: <img>/data/user4/15425.jpg</img>\n What is this? Cup cake.\n\
    #     Picture 2:<img>/data/user4/image_0011.jpg</img>\n What is this? Flamingo.\n\
    #     Picture 3: <img>/data/user4/pug_12.jpg</img>\n What is this? Pug.\n\
    #     Picture 4: <img>/data/user4/Siamese_17.jpg</img>\n What is this? Siamese.\n\
    #     Picture 5: <img>/data/user4/beagle_3.jpg</img>\n What is this? Beagle.\n\
    #     Picture 6: <img>/data/user4/pug_1.jpg</img>\n So what is this?\n\
    #     Output is one of ['Cup cake', 'Flamingo', 'Pug', 'Siamese', 'Beagle'].", "Pug."),
    #     ("Picture 1: <img>/data/user4/AnnualCrop_12.jpg</img>\n What is this? Annual crop.\n\
    #     Picture 2: <img>/data/user4/Highway_11.jpg</img>\n What is this? Highway.\n\
    #     Picture 3: <img>/data/user4/River_15.jpg</img>\n What is this? River.\n\
    #     Picture 4: <img>/data/user4/baseball_diamond_001.jpg</img>\n What is this? Baseball diamond.\n\
    #     Picture 5: <img>/data/user4/railway_station_017.jpg</img>\n What is this? Railway station.\n\
    #     Picture 6: <img>/data/user4/AnnualCrop_21.jpg</img>\n So what is this?\n\
    #     Output is one of ['Annual crop', 'Highway', 'River', 'Baseball diamond', 'Railway station'].", "Annual crop.")
    # ]

    # history1 = [
    #     ("Picture 1: <img>/data/user4/15425.jpg</img>\n What is this? Cup cake.\n\
    #         Picture 2:<img>/data/user4/image_0011.jpg</img>\n What is this? Flamingo.\n\
    #         Picture 3: <img>/data/user4/pug_12.jpg</img>\n What is this? Pug.\n\
    #         Picture 4: <img>/data/user4/pug_1.jpg</img>\n So what is this?\n\
    #         Output is one of ['Cup cake', 'Flamingo', 'Pug'].", "Pug."),
    #     ("Picture 1: <img>/data/user4/AnnualCrop_12.jpg</img>\n What is this? Annual crop.\n\
    #         Picture 2: <img>/data/user4/Highway_11.jpg</img>\n What is this? Highway.\n\
    #         Picture 3: <img>/data/user4/River_15.jpg</img>\n What is this? River.\n\
    #         Picture 4: <img>/data/user4/AnnualCrop_21.jpg</img>\n So what is this?\n\
    #         Output is one of ['Annual crop', 'Highway', 'River'].", "Annual crop.")
    # ]

    with open(args.evaluate_config_path, "r") as file:
        dataset_configs = json.load(file)



    for dataset_config in dataset_configs:
        random.seed(args.seed)

        dataset_name = dataset_config['dataset_name']
        test_path = dataset_config['json_path']



        dataset = FSL_Dataset(
            test=test_path,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn),
        )

        outputs = []
        for _, (identity_ids, input_texts, ground_truths) in tqdm(enumerate(dataloader), total=len(dataloader),
                                                                  desc=f'Processing {dataset_name}',
                                                                  dynamic_ncols=True):
            answers = []
            history1 = None



            for input_text in input_texts:
                response, history2 = model.chat(tokenizer, query=input_text, history=history1, append_history=False)

                answers.append(response)


            for identity_id, input_text, answer, ground_truth in zip(identity_ids, input_texts, answers, ground_truths):
                outputs.append({
                    'id': identity_id,
                    'Question': input_text,
                    'Answer': answer,
                    'ground_truth': ground_truth,
                })
                print(f"answer:{answer} ; ground_truth:{ground_truth}")

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f"Evaluating {dataset_name} ...")

            results_file = os.path.join(evaluate_save_folder, f'{dataset_name}.json')
            json.dump(merged_outputs, open(results_file, 'w'), ensure_ascii=False)

            infer_num, dir_exact_match, dir_error_match, dir_accuracy, occur_accuracy = evaluate_exact_match_accuracy(results_file, dataset_name)
            evaluate_results.append({
                'dataset_name': dataset_name,
                'infer_num': infer_num,
                'dir_accuracy': dir_accuracy,
                'occur_accuracy': occur_accuracy,
                'dir_exact_match': dir_exact_match,
                'dir_error_match': dir_error_match
                # 这一块需要的话还可以添加一个存储匹配失败的数据的列表
            })
            # infer_num, dir_exact_match, dir_error_match, dir_accuracy, counter, accuracy_clip, occur_correct_predictions, accuracy_occur = evaluate_exact_match_accuracy(results_file, dataset_name)
            # evaluate_results.append({
            #     'dataset_name': dataset_name,
            #     'infer_num': infer_num,
            #     'dir_accuracy': dir_accuracy,
            #     'clip_accuracy': accuracy_clip,
            #     'occur_accuracy': accuracy_occur,
            #     'dir_exact_match': dir_exact_match,
            #     'clip_exact_match': counter,
            #     'occur_exact_match': occur_correct_predictions
            #     # 这一块需要的话还可以添加一个存储匹配失败的数据的列表
            # })


            # 保存评估结果
            evaluate_results_file = os.path.join(evaluate_save_folder, 'evaluate_accuracy_results-1.json')
            json.dump(evaluate_results, open(evaluate_results_file, 'w'), ensure_ascii=False)

        torch.distributed.barrier()

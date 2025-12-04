import argparse
import itertools
import json
import os
import random
from functools import partial
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM


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


def evaluate_exact_match_accuracy(results_file, dataset_name):
    results = json.load(open(results_file))

    #===============direct match===============
    dir_exact_match = 0
    dir_error_match = 0
    occur_match = 0
    for i in tqdm(range(len(results)), desc='Direct Processing'):
        output = str(results[i]['Answer']).strip().lower().replace(" ", "")
        truth = str(results[i]['ground_truth']).strip().lower().replace(" ", "")
        output = re.sub(r'[^\w\s]', '', output)
        truth = re.sub(r'[^\w\s]', '', truth)
        if str(results[i]['Answer']).lower().replace(".", "").strip() == str(
                results[i]['ground_truth']).lower().replace(".", "").strip():
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



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--evaluate_config_path', type=str, default='')
    parser.add_argument('--evaluate_save_folder', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ref_history_path', type=str, default='')
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    # 注意如果是使用微调后的Lora/qlora模型的话，需要使用AutoPeftModelForCausalLM函数来加载模型，否则使用AutoModelForCausalLM函数来加载
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.checkpoint, device_map='cuda', trust_remote_code=True).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint,
                                              trust_remote_code=True)

    evaluate_save_folder = args.evaluate_save_folder
    if not os.path.exists(evaluate_save_folder):
        os.makedirs(evaluate_save_folder, exist_ok=True)

    evaluate_results = []

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
            # history1 = []
            # history_tmp = eval(random.choice(ref_lines).strip())
            # history1.append(history_tmp)

            for input_text in input_texts:
                response, _ = model.chat(tokenizer, query=input_text, history=history1)
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

            infer_num, dir_exact_match, dir_error_match, dir_accuracy, occur_accuracy = evaluate_exact_match_accuracy(
                results_file, dataset_name)
            evaluate_results.append({
                'dataset_name': dataset_name,
                'infer_num': infer_num,
                'dir_accuracy': dir_accuracy,
                'occur_accuracy': occur_accuracy,
                'dir_exact_match': dir_exact_match,
                'dir_error_match': dir_error_match
                # 这一块需要的话还可以添加一个存储匹配失败的数据的列表
            })

            # 保存评估结果
            evaluate_results_file = os.path.join(evaluate_save_folder, 'evaluate_accuracy_results-1.json')
            json.dump(evaluate_results, open(evaluate_results_file, 'w'), ensure_ascii=False)

        torch.distributed.barrier()

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk, load_dataset, concatenate_datasets
import torch
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pretrain.modeling_flash_llama import LlamaForCausalLM
from collections import defaultdict

def init_args():
    parser = argparse.ArgumentParser()   
    parser.add_argument("--input", type=str, nargs="+", help="Input dataset")
    parser.add_argument('--output', type=str, help="Output dataset") 
    parser.add_argument("--model", type=str, default='EleutherAI/pythia-160m-deduped', help="Model name or path")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1, help="Workers.") 
    parser.add_argument('--save_workers', type=int, default=16, help="Save workers.")
    parser.add_argument('--fp16', action='store_true', help='Use fp16 mixed precision training.')
    parser.add_argument('--bf16', action='store_true', help='Use bf16 mixed precision training.') 
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum number of tokens the model can handle') 
    parser.add_argument('--text_field', type=str, default='text', help='Text field name')
    parser.add_argument("--tokens_field", type=str, default="input_ids", help="Store tokenized data in this field")
    parser.add_argument('--field', type=str, default='avg_loglikelihood', help='Field name')
    parser.add_argument('--domain_field', type=str, default='source_domain', help='Field name to distinguish domains')
    parser.add_argument("--shard", type=int, nargs=2, default=[1, 0])
    args = parser.parse_args()
    return args

class Processor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=(torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None))
        self.model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=(torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None))
        self.model.eval()
        self.model.to(self.device)
        
    def __getstate__(self):
        return self.args

    def __setstate__(self, state):
        self.__init__(state)

    # 计算给定文本序列的平均对数似然
    @torch.inference_mode()
    def log_likelihood(self, sents):
        if self.args.tokens_field in sents:
            seqs = sents[self.args.tokens_field]
        else:
            ## 使用tokenizer将输入文本转换为Token ID Hugging Face 的 tokenizer 能够处理列表，而不是像tokenizer.encode()单个样本处理，这样可以减少对 tokenizer 的调用次数。
            seqs = self.tokenizer(sents[self.args.text_field]).input_ids  # <fix>, 之前代码没有input_ids，seqs就会同时有input_ids和attention_mask
        
        max_length = min(max(len(x) for x in seqs), self.args.max_tokens + 1)
        
        bsz = len(seqs)

        input_ids = torch.zeros(bsz, max_length, dtype=torch.long)
        attention_mask = torch.zeros(bsz, max_length, dtype=torch.long)

        for i, x in enumerate(seqs):
            input_ids[i,:len(x)] = torch.tensor(x[:max_length], dtype=torch.long)
            attention_mask[i,:len(x)] = 1

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        targets = input_ids[:,1:]
        input_ids = input_ids[:,:-1]
        target_attention_mask = attention_mask[:,1:]

        # 使用log softmax计算log-likelihood
        logprobs = self.model(input_ids).logits.float().log_softmax(dim=-1)

        # 按批次和tokens长度计算对数似然，最后返回当前batch中对应输入句子的平均对数似然
        # targets.unsqueeze(-1)：将目标标签(实际的单词ID)的维度扩展一个维度，这样张量变成了形状为 (batch_size, sequence_length, 1)。
        # torch.gather(logprobs, -1, targets.unsqueeze(-1))：torch.gather函数根据targets的索引从logprobs中提取对应的对数概率。结果是一个形状与targets相同的张量，其中每个元素是目标标签对应的对数概率。
        # .squeeze(-1)：去掉最后一个维度，将张量形状还原为 (batch_size, sequence_length)，便于后续操作。
        loglikelihood = torch.gather(logprobs, -1, targets.unsqueeze(-1)).squeeze(-1)
        # target_attention_mask：这是一个掩码（mask），指示哪些位置是有效的（通常是1）和哪些位置是无效的（通常是0），例如可能用于处理填充。
        # target_attention_mask.float()：将掩码转换为浮点类型，便于进行逐元素乘法。
        # loglikelihood * target_attention_mask.float()：用掩码来过滤对数似然（log-likelihood），即无效位置的对数似然被置零，只保留有效位置的对数似然。
        loglikelihood = loglikelihood * target_attention_mask.float()
        # torch.sum(loglikelihood, dim=-1)：沿着序列长度的维度（dim=-1）求和，得到每个批次中所有有效位置上的对数似然的总和。
        # torch.sum(target_attention_mask, dim=-1)：沿着序列长度的维度（dim=-1）求和，得到每个批次中有效位置的数量。
        # .clamp(min=1)：确保有效位置的数量至少为1，以避免在后续除法计算中出现除以零的情况。
        # /：用得到的有效位置数量对总的对数似然进行归一化，即计算平均对数似然。
        avg_loglikelihood = torch.sum(loglikelihood, dim=-1) / torch.sum(target_attention_mask, dim=-1).clamp(min=1)

        return avg_loglikelihood.cpu().tolist()

    def __call__(self, items):
        output = {}
        avg_loglikelihood = self.log_likelihood(items)
        output[self.args.field] = avg_loglikelihood
        return output

def compute_ppls(dataset, field, domain_field):
    total_ppl = []
    domain_ppl = defaultdict(list)

    # 计算每个条目的 PPL
    for item in dataset:
        item_avg_loglikelihood = float(item[field])
        item_ppl = np.exp(-item_avg_loglikelihood)  # 先计算一次

        total_ppl.append(item_ppl)
        if domain_field == "single_label":
            domain_ppl[item['domain'][domain_field]].append(item_ppl)
        else:
            domain_ppl[item[domain_field]].append(item_ppl)

    # 计算整体平均 PPL
    average_ppl = np.mean(total_ppl)

    # 计算每个 domain 的平均 PPL（字典推导表达式）
    domain_average_ppl = {domain: np.mean(values) for domain, values in domain_ppl.items()}

    return average_ppl, domain_average_ppl

def main():
    args = init_args()
    
    dataset = concatenate_datasets([load_from_disk(path) for path in tqdm(args.input)])
    dataset = dataset.shard(args.shard[0], args.shard[1], contiguous=True)
    print(args)

    # 如果 avg_loglikelihood 字段不存在，使用 Processor 类处理数据集（批处理）。
    if args.field not in dataset.column_names:
        dataset = dataset.map(Processor(args), batched=True, batch_size=args.batch_size, num_proc=args.num_workers, keep_in_memory=True)

    # 计算平均 PPL
    average_ppl, domain_average_ppl = compute_ppls(dataset, args.field, args.domain_field)
    print(f"Average ppl: {average_ppl:.2f}")    
    for domain, avg_ppl in domain_average_ppl.items():
        print(f"Average ppl for domain '{domain}': {avg_ppl:.2f}")

    # 保存处理后的数据集
    dataset.save_to_disk(args.output, num_proc=args.save_workers)

if __name__ == "__main__":
    main()
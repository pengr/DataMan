import os
import re
import time
import json
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from multiprocessing import Process, Queue
from tqdm import tqdm
from datasets import load_from_disk, load_dataset, concatenate_datasets
import functools

# Request Templates
REQUEST_TEMPLATES = {
    'en': {
        'all_rating': """Please score the text on fourteen evaluation criteria and specify its domain:
Text: {text}
Domain:_
[1]Accuracy:_/5
[2]Coherence:_/5
[3]Language Consistency:_/5
[4]Semantic Density:_/5
[5]Knowledge Novelty:_/5
[6]Topic Focus:_/5
[7]Creativity:_/5
[8]Professionalism:_/5
[9]Style Consistency:_/5
[10]Grammatical Diversity:_/5
[11]Structural Standardization:_/5
[12]Originality:_/5
[13]Sensitivity:_/5
[14]Overall Score:_/5""",
        'score_only': """Please give an overall score for the text:
Text: {text}
Overall Score:_/5""",
        'detprompt_score_only': """Please give an overall score for the text:
Text: {text}
[1]Accuracy
[2]Coherence
[3]Language Consistency
[4]Semantic Density
[5]Knowledge Novelty
[6]Topic Focus
[7]Creativity
[8]Professionalism
[9]Style Consistency
[10]Grammatical Diversity
[11]Structural Standardization
[12]Originality
[13]Sensitivity
Overall Score:_/5""",
        'domain_only': """Please specify an domain type for the text:
Text: {text}
Domain:_""",
        'detprompt_domain_only': """Please specify an domain type for the text:
Text: {text}
Domain Types: [A]Medicine [B]Finance [C]Law [D]Education [E]Technology [F]Entertainment [G]Mathematics [H]Coding [I]Government [J]Culture [K]Transportation [L]Retail E-commerce [M]Telecommunication [N]Agriculture [O]Other
Domain:_"""
    },
    'zh': {
        'all_rating': """请给出文本的十四项评价指标分数和领域类别:
文本: {text}
领域:_
[1]准确性:_/5
[2]连贯性:_/5
[3]语种一致性:_/5
[4]信息含量:_/5
[5]知识新颖性:_/5
[6]专注度:_/5
[7]创意:_/5
[8]专业性:_/5
[9]风格一致性:_/5
[10]语法多样性:_/5
[11]结构规范性:_/5
[12]内容原创性:_/5
[13]敏感度:_/5
[14]总评:_/5""",
        'score_only': """请给出文本的整数总评分:
文本: {text}
总评:_/5""",
        'detprompt_score_only': """请给出文本的整数总评分:
文本: {text}
[1]准确性
[2]连贯性
[3]语种一致性
[4]信息含量
[5]知识新颖性
[6]专注度
[7]创意
[8]专业性
[9]风格一致性
[10]语法多样性
[11]结构规范性
[12]内容原创性
[13]敏感度
总评:_/5""",
        'domain_only': """请给出文本的领域类型:
文本: {text}
领域:_""",
        'detprompt_domain_only': """请给出文本的领域类型:
文本: {text}
领域类型：[A]医疗 [B]金融 [C]法律 [D]教育 [E]科技 [F]娱乐 [G]数学 [H]代码 [I]政务 [J]文化 [K]交通 [L]零售电商 [M]电信 [N]农业 [O]其他
领域:_"""
    }
}
# Regular expression: matches ellipses surrounded by double quotes, matches URL items starting with ![](//) characters, score, domain type, raw text, N/A
ELLIPSIS_PATTERN = r'^["“”]…+["“”]|^["“”]\.\.\.+["“”]'
LINK_PATTERN = r"!\[\]\([\/]?[^\)]+\)\n"
TEXT_PATTERNS = {
    'en': {
        'all_rating': r'\nText: (.*?)(\n)+Domain',
        'score_only': r'\nText: (.*?)(\n)+Overall Score',
        'domain_only': r'\nText: (.*?)(\n)+Domain',
    },
    'zh': {
        'all_rating': r'\n文本：(.*?)(\n)+领域',
        'score_only': r'\n文本：(.*?)(\n)+总评',
        'domain_only': r'\n文本：(.*?)(\n)+领域',
    }
}

def fix_seed(seed=1024):
    """Fix random seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)

def regular_filtering(text):
    """Apply regular expressions to filter text."""
    text = re.sub(ELLIPSIS_PATTERN, '', text)
    text = re.sub(LINK_PATTERN, '', text)
    return text

def preprocess_worker(tokenizer, llama_tokenizer, args, examples, indices=None):
    source = os.path.splitext(os.path.basename(args.input_path))[0]                        # 获取文件名
    additional_fields = ["text", "document_index", "source_domain", "length"] + (["ppl"] if args.data_format == "ppl" else []) \
        + (["quality_score", "toxicity"] if args.data_format == "ChineseWebText2" else [])
    result = {field: [] for field in additional_fields}

    num_examples = len(examples[next(iter(examples))])
    for i in range(num_examples):
        # 1. Load original text
        if args.data_format == "ppl":
            text, source_domain, quality_score, ppl = examples['txt'][i], examples.get('source')[i] if 'source' in examples else source, None, examples['ppl'][i]
        elif args.data_format == "qwen":
            text, source_domain, quality_score, ppl = examples.get('text', examples.get('content'))[i], examples.get('source')[i] if 'source' in examples else source, None, None
        elif args.data_format == "slimpajama":
            text, source_domain, quality_score, ppl = examples['text'][i], examples['meta'][i]["redpajama_set_name"].replace("RedPajama", ""), None, None
        elif args.data_format == "commoncrawl":
            text, source_domain, quality_score, ppl = examples['text'][i], source, None, None
        elif args.data_format == "ChineseWebText2":
            text, source_domain, quality_score, ppl = examples['text'][i], examples['domain'][i]["single_label"], \
                examples['quality_score'][i], None

        # 2. Regualr fitering ellipses surrounded by double quotes, matches URL items starting with ![](//) characters, 
        text = regular_filtering(text)

        # 3. Fast implementation of trucate origin text
        if len(text) > args.char_truncate_max_length:
            text = tokenizer.tokenize(text[:args.char_truncate_max_length//2]) + tokenizer.tokenize(text[-args.char_truncate_max_length//2:])
        else:
            text = tokenizer.tokenize(text)
        if len(text) > args.truncate_max_length:    # 若截断中间文本，则在中间添加"..."表示有省略句子 
            text = text[:args.truncate_max_length//2] + ["..."] + text[-args.truncate_max_length//2:]
        text_len_qwen = len(text)
        text = tokenizer.convert_tokens_to_string(text).strip()
        text_len = len(llama_tokenizer.tokenize(text)) if llama_tokenizer else text_len_qwen  # 若LlamaTokenizer存在，则用它来标记化（正则过滤, 截断的文本）获得token长度

        # 4. Put json formatted text into shared queue, text是正则过滤, 截断的文本
        result["text"].append(text)
        result["document_index"].append(f"{source_domain}_{indices[i] + 1}")
        result["source_domain"].append(source_domain)
        result["length"].append(text_len)
        if ppl is not None:
            result["ppl"].append("%.2f" % ppl)
        if quality_score is not None:
            result["quality_score"].append(quality_score)
    return result

def inference_worker(args, tokenizer, llm, request_template, sampling_params, additional_fields, model_type, examples):
    # prepare chat texts
    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": request_template.format(text=text)}],
            tokenize=False,
            add_generation_prompt=True
        )
        for text in examples['text']
    ]  

    # generate output
    outputs = llm.generate(chat_texts, sampling_params)
    examples.update({field: [] for field in additional_fields})

    idx = 0
    for output in outputs:
        output_text = output.outputs[0].text.strip()
        if model_type == 'score_only':
            examples["overall_score"].append(output_text)
        elif model_type == 'domain_only':
            examples["application_domain"].append(output_text)
        elif model_type == 'all_rating':
            metrics = output_text.split("\n")
            if len(metrics) != len(additional_fields):  # 若metrics不符合additional_fields 的数量要求
                original_fields = list(set(examples.keys()) - set(additional_fields)) 
                for field in original_fields:  
                    del examples[field][idx]
                continue  
            for field, metric in zip(additional_fields, metrics):
                examples[field].append(metric)
            idx = idx + 1
    return examples

def worker_process(args, w_id, processed_shard, tokenizer):
    # Initialize LLM based on gpu_id
    os.environ.update({'CUDA_VISIBLE_DEVICES': str(w_id)})
    llm = LLM(model=args.model_name_or_path, tokenizer=args.model_name_or_path, tensor_parallel_size=1)

    # Set REQUEST_TEMPLATE
    request_template = REQUEST_TEMPLATES[args.lang][args.model_type]

    # Pass the default decoding hyperparameters
    sampling_params = SamplingParams(temperature=args.temperature, seed=args.seed, stop_token_ids=[151643, 151645], max_tokens=args.max_tokens)
    additional_fields_dict = {
        'score_only': ["overall_score"],
        'domain_only': ["application_domain"],
        'all_rating': [
            "application_domain", "accuracy", "coherence", "language_consistency", 
            "semantic_density", "knowledge_novelty", "topic_focus", "creativity",
            "professionalism", "style_consistency", "grammatical_diversity", 
            "structural_standardization", "originality", "sensitivity", "overall_score"
        ]
    }
    model_type = args.model_type#.split('_')[-1]
    additional_fields = additional_fields_dict[model_type]

    # TODO: Dataset.map的每一批都要给inference_worker传入这一系列参数，会不会耗费时间？
    result = processed_shard.map(
        functools.partial(inference_worker, args, tokenizer, llm, request_template, sampling_params, additional_fields, model_type),
        batched=True,
        batch_size=args.batch_size,
        remove_columns=args.remove_inference_column_names,
        num_proc=1
    )
    return result

def main(args):
    """Multi-process (multi-CPU) text preprocessing & Standalone (single-node & multi-GPU) vllm inference"""
    dirs = [os.path.dirname(os.path.abspath(path)) for path in [args.processed_path, args.inferenced_path]]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

    fix_seed(args.seed)
    start_time = time.time()

    # 只初始化一次Qwen2 tokenizer, LLaMA2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left", use_fast=args.use_fast)
    llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_tokenizer_path, padding_side="left", use_fast=args.use_fast) if args.llama_tokenizer_path else None
    
    # preprocess
    processed_dataset = None
    if os.path.exists(args.processed_path):
        print(f"{args.processed_path} already exists. Skipping preprocessing.")
    else:
        print(f"Loading '{args.input_path}'...")
        if args.json:
            dataset = load_dataset("json", data_files=args.input_path, split="train")
        else:
            dataset = load_from_disk(args.input_path)
        print(f"Loaded '{args.input_path}'")
        dataset = dataset.shard(args.preprocess_shard[0], args.preprocess_shard[1], contiguous=True)

        processed_dataset = dataset.map(
            functools.partial(preprocess_worker, tokenizer, llama_tokenizer, args),
            batched=True,
            batch_size=args.batch_size,
            remove_columns=args.remove_preprocess_column_names,
            num_proc=args.num_cpu_workers,
            with_indices=True
        )

    # inference
    if os.path.exists(args.inferenced_path):
        print(f"{args.inferenced_path} already exists. Skipping inferencing.")
    else:
        print(f"Loading '{args.processed_path}'...")
        if args.json and processed_dataset is None:
            processed_dataset = load_dataset("json", data_files=args.processed_path, split="train")
        elif processed_dataset is None:
            processed_dataset = load_from_disk(args.processed_path)
        print(f"Loaded '{args.processed_path}'")

        assert args.inference_shard[0] <= 8, "Inference Dataset处理片数必须小于等于8，以确保 args.inference_shard[1] 不超过 CUDA_VISIBLE_DEVICES 的最大值7"
        assert args.num_gpu_workers==1, "Only Support Single-GPU RUN"
        print(f"Processing dataset '{args.inference_shard[1]}'")
        processed_shard = processed_dataset.shard(args.inference_shard[0], args.inference_shard[1], contiguous=True)
        inference_dataset = worker_process(args, args.inference_shard[1], processed_shard, tokenizer)

        # 收集结果并保存
        print(f"Saving to '{args.inferenced_path}'...")
        inference_dataset.save_to_disk(f"{args.inferenced_path}", num_proc=args.num_cpu_workers)
        print(f"Saved to '{args.inferenced_path}'")
    
    end_time = time.time()
    print(f"All completed, cost {end_time - start_time} s.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="multiprocess preprocess && standalone inference")
    parser.add_argument('--model_name_or_path', type=str, help='model or tokenizer path.')
    parser.add_argument('--input_path', type=str, help='input data path.')
    parser.add_argument('--processed_path', type=str, help='multiprocess preprocessed data path.')
    parser.add_argument('--inferenced_path', type=str, help='standalone inferenced data path.')
    parser.add_argument('--llama_tokenizer_path', type=str, default=None, help='llama tokenzier path.')
    parser.add_argument('--lang', default="zh", type=str, choices=['zh', 'en'])
    parser.add_argument('--num_cpu_workers', default=64, type=int, help='number of processes in multi-processing')
    parser.add_argument('--num_gpu_workers', default=1, type=int, help='number of processes in standalone inference')
    parser.add_argument('--max_tokens', default=1, type=int, help='Maximum number of output tokens')
    parser.add_argument('--seed', default=1024, type=int, help='seed size')
    parser.add_argument('--temperature', default=0.0, type=float, help='generation temperature')
    parser.add_argument('--truncate_max_length', default=1896, type=int, help='maximum truncated token length, zh_score:1896, en_score:1894')
    parser.add_argument('--char_truncate_max_length', default=60000, type=int, help='maximum truncated char length, zh_score:60000, en_score:20000')
    parser.add_argument('--data_format', default="qwen", type=str, help='data format', choices=['ppl', 'qwen', 'slimpajama', 'commoncrawl', 'ChineseWebText2'])
    parser.add_argument('--model_type', default='score_only', type=str, help='use which model to inference', choices=['all_rating', 'score_only', 'detprompt_score_only', 'domain_only', 'detprompt_domain_only'])
    parser.add_argument("--json", action="store_true", help="Input is json dataset.")
    parser.add_argument("--use_fast", action="store_true", help="Use fast tokenizer.")
    parser.add_argument("--remove_preprocess_column_names", type=str, nargs="+", default=None, help="Columns to remove in preprocess.")
    parser.add_argument("--remove_inference_column_names", type=str, nargs="+", default=None, help="Columns to remove in inference.")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for the map function.")
    parser.add_argument("--preprocess_shard", type=int, nargs=2, default=[1, 0], help="Pick a shard of the preprocessed dataset")
    parser.add_argument("--inference_shard", type=int, nargs=2, default=[1, 0], help="Pick a shard of the inferenced dataset")
    args = parser.parse_args()
    main(args)
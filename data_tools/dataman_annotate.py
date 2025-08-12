import os
import re
import math
import time
import json
import socket
import logging
import argparse
import subprocess
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import multiprocessing
from multiprocessing import Process, Queue, Lock
import torch.multiprocessing as mp
from tqdm import tqdm

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
        'domain': """请给出文本的领域类型:
文本: {text}
领域:_""",
        'detprompt_domain': """请给出文本的领域类型:
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
    assert isinstance(seed, int)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def set_request_config(lang, model_type):
    """Set global request template based on language and model type."""
    global REQUEST_TEMPLATE, TEXT_PATTERN_COMPILED
    REQUEST_TEMPLATE = REQUEST_TEMPLATES[lang][model_type]

def regular_filtering(text):
    """Apply regular expressions to filter text."""
    text = re.sub(ELLIPSIS_PATTERN, '', text)
    text = re.sub(LINK_PATTERN, '', text)
    return text

class PreprocessWriter:
    def __init__(self, num_cpu_workers, output_path, batch_size):
        self.num_cpu_workers = num_cpu_workers
        self.output_path = output_path
        self.batch_size = batch_size
        self.finished_workers = 0
        self.lock = Lock()

    def write(self, queue):
        while self.finished_workers < self.num_cpu_workers:
            batch = []
            for _ in range(self.batch_size):
                data = queue.get()
                if data is None:
                    self.finished_workers += 1
                    break
                batch.append(data)
            if batch:
                with self.lock:
                    with open(self.output_path, 'a', encoding='utf-8') as wf:
                        wf.writelines(batch)

class InferenceWriter:
    def __init__(self, num_gpu_workers, output_path):
        self.num_gpu_workers = num_gpu_workers
        self.output_path = output_path
        self.finished_workers = 0
        self.lock = mp.Lock()
        
    def write(self, queue):
        while self.finished_workers < self.num_gpu_workers:
            results = queue.get()
            if results is None:
                with self.lock:
                    self.finished_workers += 1
            else:
                with self.lock:
                    with open(self.output_path, 'a', encoding='utf-8') as wf:
                        wf.writelines(results)

def preprocess_worker(param, queue):
    """Worker functions for multiprocessing"""
    try:
        model_name_or_path, llama_tokenizer_path, input_path, truncate_max_length, char_truncate_max_length, start_position, end_position, worker_id, data_format = param

        # Initialize Qwen2 and Llama tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", use_fast=True)
        llama_tokenizer = AutoTokenizer.from_pretrained(llama_tokenizer_path, padding_side="left", use_fast=True) if llama_tokenizer_path else None

        source, _ = os.path.splitext(os.path.basename(input_path))
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as rf:
            for _ in range(start_position):  # Skip the first start_position line directly   
                try:
                    next(rf)
                except StopIteration:
                    logging.warning(f"Start position {start_position} exceeds file length for worker {worker_id}.")
                    break

            for lineID, line in tqdm(enumerate(rf, start_position), total=end_position-start_position, desc=f"Worker {worker_id} Processing {start_position}-{end_position}"):
                if lineID >= end_position or not line:
                    break
                
                # 1. Load original text
                if data_format == "ppl":  
                    source, data = line.strip().split("\t")
                    text, source_domain, ppl = json.loads(data)['txt'], data.get('source', source), json.loads(data)['ppl']
                elif data_format == "qwen":
                    data = json.loads(line.strip())
                    text, source_domain, ppl = data.get('text', data.get('content')), data.get('source', source), None
                elif data_format == "slimpajama":
                    data = json.loads(line.strip())
                    text, source_domain, ppl = data.get('text'), data.get('meta')["redpajama_set_name"].replace("RedPajama", ""), None
                elif data_format == "commoncrawl":
                    data = json.loads(line.strip())
                    text, source_domain, ppl = data.get('text'), source, None

                # 2. Regualr fitering ellipses surrounded by double quotes, matches URL items starting with ![](//) characters, 
                text = regular_filtering(text)

                # 3. Fast implementation of trucate origin text
                if len(text) > char_truncate_max_length:
                    text = tokenizer.tokenize(text[:char_truncate_max_length//2]) + tokenizer.tokenize(text[-char_truncate_max_length//2:])
                else:
                    text = tokenizer.tokenize(text)
                if len(text) > truncate_max_length:    # 若截断中间文本，则在中间添加"..."表示有省略句子 
                    text = text[:truncate_max_length//2] + ["..."] + text[-truncate_max_length//2:]
                text_len_qwen = len(text)
                text = tokenizer.convert_tokens_to_string(text).strip()
                text_len = len(llama_tokenizer.tokenize(text)) if llama_tokenizer else text_len_qwen  # 若LlamaTokenizer存在，则用它来标记化（正则过滤, 截断的文本）获得token长度

                # 4. Put json formatted text into shared queue, text是正则过滤, 截断的文本
                result = {"text": text, "id": f"{source_domain}_{lineID + 1}", "source_domain": f"{source_domain}", "length": f"{text_len}"}
                if ppl is not None:
                    result["ppl"] = "%.2f" % ppl
                queue.put(json.dumps(result, ensure_ascii=False) + '\n')
            # 5. When the process ends, notify the writer, add an None flag
            queue.put(None)
            logging.info(f"Preprocess Worker {worker_id} completed.")
    except Exception as e:
        logging.error(f"Error in preprocess worker {worker_id}: {e}")


def inference_worker(param, queue):
    """Worker functions for Standalone inference"""
    try:
        model_name_or_path, seed, max_tokens, temperature, input_path, start_position, end_position, worker_id, data_format, lang, model_type, batch_size = param
        
        # Initialize Qwen2 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", use_fast=True)
        # Dynamically set CUDA_VISIBLE_DEVICES according to the incoming gpu_id
        os.environ['CUDA_VISIBLE_DEVICES'] = str(worker_id)
        # 在每个 Worker 中调用 set_request_config
        set_request_config(lang, model_type)
        # Initialize LLM, Add retry logic when countering network error, 这里tokenizer是qwen2, 仅用于vllm inference
        for attempt in range(3):
            try:
                llm = LLM(model=model_name_or_path, tokenizer=model_name_or_path, tensor_parallel_size=1)
                break
            except socket.gaierror:
                if attempt < 2:
                    time.sleep(5)
                else:
                    raise
        # Pass the default decoding hyperparameters
        sampling_params = SamplingParams(temperature=temperature, seed=seed, stop_token_ids=[151643, 151645], max_tokens=max_tokens)

        with open(input_path, 'r', encoding='utf-8') as rf:
            for _ in range(start_position):   # Skip the first start_position line directly  
                try:
                    next(rf)
                except StopIteration:
                    logging.warning(f"Start position {start_position} exceeds file length for worker {worker_id}.")
                    break
            
            for lineID in tqdm(range(start_position, end_position, batch_size), total=((end_position-start_position)//batch_size)+1, desc=f"Worker {worker_id} Inferencing {start_position}-{end_position}"):
                chat_texts, texts, text_ids, source_domains, text_lens, ppls = [], [], [], [], [], []
                for _ in range(batch_size):
                    if lineID >= end_position:
                        break
                    line = next(rf)
                    lineID += 1
                    if not line: break
                    data = json.loads(line.strip())
                    # 1. 正则过滤, 截断的文本
                    text = data['text']
                    texts.append(text)
                    # 2. 正则过滤, 截断，Qwen2 Request文本，可兼容多种Qwen2 Request类型
                    request = REQUEST_TEMPLATE.format(text=text)
                    messages = [{"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": request}]
                    # 3. 正则过滤, 截断, chat格式化，Qwen2 Request文本
                    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    chat_texts.append(chat_text)
                    # 4. 文本id，来源领域，以及文本tokens（取决于有无Llama tokenizer）
                    text_ids.append(data['id'])
                    source_domains.append(data['source_domain'])
                    text_lens.append(data['length'])
                    if data_format == "ppl":
                        ppls.append(data['ppl'])

                outputs = llm.generate(chat_texts, sampling_params)
                results = []
                result_dicts = [{} for _ in range(len(outputs))]  # 预先分配

                for i, output in enumerate(outputs):
                    output_text = output.outputs[0].text.strip()
                    result_dict = result_dicts[i]

                    # 添加其他共同字段
                    result_dict.update({
                        "text": texts[i],
                        "length": text_lens[i],
                        "source_domain": source_domains[i],
                        "id": text_ids[i],
                    })

                    # 根据 model_type 和 output_text 来设置 json 字段
                    if model_type.endswith('score_only'):
                        result_dict["overall_score"] = output_text
                    elif model_type.endswith('domain'):
                        result_dict["application_domain"] = output_text
                    elif model_type.endswith('all_rating'):
                        metrics = output_text.split("\n")
                        metric_names = [
                            "application_domain", "accuracy", "coherence", "language_consistency", "semantic_density",
                            "knowledge_novelty", "topic_focus", "creativity",
                            "professionalism", "style_consistency", "grammatical_diversity", 
                            "structural_standardization", "originality", "sensitivity", "overall_score"
                        ]
                        result_dict.update(dict(zip(metric_names, metrics)))
                    
                # 批量序列化 JSON
                results = [json.dumps(result_dict, ensure_ascii=False) + '\n' for result_dict in result_dicts]
                queue.put(results)
        queue.put(None)
        logging.info(f"Inference worker {worker_id} completed.")
    except Exception as e:
        logging.error(f"Error in inference worker {worker_id}: {str(e)}")


def multiprocess_preprocess(args, input_path, output_path):
    """Multi-process (multi-CPU) text preprocessing"""
    start_time = time.time()
    multiprocessing.set_start_method('spawn', force=True)  # 在主模块中设置“spawn”,防止资源竞争,使得Tokenizer中use_fast可为True

    # Create shared queue, shared variable and Writerinstance, then start writing
    queue = Queue(maxsize=args.batch_size)
    preprocess_writer = PreprocessWriter(args.num_cpu_workers, output_path, args.batch_size)
    writer_task = Process(target=preprocess_writer.write, args=(queue,))
    writer_task.start()

    def count_lines(filename):
        result = subprocess.run(['python', '/mnt/nas/pengru.pr/DataMan/utils/num_of_lines.py',
                                 '--input_file', filename], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if 'Number of lines:' in line:
                return int(line.split(':')[-1].strip())
        raise RuntimeError("Failed to determine the number of lines from output")
    total_lines = count_lines(input_path)
    chunk_size = math.ceil(total_lines / args.num_cpu_workers)
    positions = [chunk_size * i for i in range(args.num_cpu_workers)]
    positions.append(total_lines) 
    worker_params = [(args.model_name_or_path,
                      args.llama_tokenizer_path,
                      input_path, 
                      args.truncate_max_length,
                      args.char_truncate_max_length,
                      positions[i], 
                      positions[i+1], 
                      i,
                      args.data_format) for i in range(args.num_cpu_workers)]

    print(f"Starting {len(worker_params)} preprocess workers.")
    preprocess_tasks = [Process(target=preprocess_worker, args=(param, queue)) for param in worker_params]

    for task in preprocess_tasks:
        task.start()
    for task in preprocess_tasks:
        task.join()
    writer_task.join()

    end_time = time.time()
    print(f"Successfully multi-preprocessed {input_path} to {output_path}, cost {end_time-start_time} s.")


def standalone_inference(args, input_path, output_path):
    """Standalone (single-node & multi-GPU) vllm inference"""
    start_time = time.time()
    mp.set_start_method('spawn', force=True)   # 在主模块中设置“spawn”,防止资源竞争,使得Tokenizer中use_fast可为True

    def count_lines(filename):
        result = subprocess.run(['python', '/mnt/nas/pengru.pr/DataMan/utils/num_of_lines.py',
                                 '--input_file', filename], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if 'Number of lines:' in line:
                return int(line.split(':')[-1].strip())
        raise RuntimeError("Failed to determine the number of lines from output")
    total_lines = count_lines(input_path)
    chunk_size = math.ceil(total_lines / args.num_gpu_workers)
    positions = [chunk_size * i for i in range(args.num_gpu_workers)]
    positions.append(total_lines) 
    worker_params = [(args.model_name_or_path,
                      args.seed,
                      args.max_tokens,
                      args.temperature,
                      input_path, 
                      positions[i], 
                      positions[i+1], 
                      i,
                      args.data_format,
                      args.lang, 
                      args.model_type,
                      args.batch_size) for i in range(args.num_gpu_workers)]

    # Create shared queue, shared variable and Writerinstance, then start writing,这部分代码不能放到上面去，会报错
    queue = mp.Queue(maxsize=args.batch_size)
    inference_writer = InferenceWriter(args.num_gpu_workers, output_path)
    writer_task = Process(target=inference_writer.write, args=(queue,))
    writer_task.start()

    print(f"Starting {len(worker_params)} inference workers.")
    inference_tasks = [mp.Process(target=inference_worker, args=(param, queue)) for param in worker_params]
    
    for task in inference_tasks:
        task.start()
    for task in inference_tasks:
        task.join()
    writer_task.join()

    end_time = time.time()
    print(f"Successfully standlone-inference {input_path} to {output_path}, cost {end_time-start_time} s.")


def main(args):
    dirs = [os.path.dirname(os.path.abspath(path)) for path in [args.processed_path, args.inferenced_path]]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

    fix_seed(args.seed)
    start_time = time.time()
    if os.path.exists(args.processed_path):
        print(f"{args.processed_path} already exists. Skipping preprocessing.")
    else:
        multiprocess_preprocess(args, input_path=args.input_path, output_path=args.processed_path)
    if os.path.exists(args.inferenced_path):
        print(f"{args.inferenced_path} already exists. Skipping inferencing.")
    else:
        standalone_inference(args, input_path=args.processed_path, output_path=args.inferenced_path)
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
    parser.add_argument('--data_format', default="qwen", type=str, help='data format', choices=['ppl', 'qwen', 'slimpajama', 'commoncrawl'])
    parser.add_argument('--model_type', default='score_only', type=str, help='use which model to inference', choices=['all_rating', 'score_only', 'detprompt_score_only', 'domain_only', 'detprompt_domain_only'])
    parser.add_argument("--batch_size", type=int, default=10000, help="Maximum storage size for shared queue.")
    args = parser.parse_args()
    main(args)
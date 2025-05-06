import os
import re
import time
import json
import math
import random
import logging
import argparse
import gevent
# from gevent import monkey
# monkey.patch_all()
from gevent.pool import Pool
import requests
import jsonlines
import subprocess
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# Constants: Base directory path, OpenAI Request configuration, Directory of source files
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAX_API_RETRY = 3
LLM_MIT_RETRY_SLEEP = 5
SOURCE_FILES_DIR = f'{base_path}/data/Qwen-datasets-100K'

# Logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Initialize tokenizer
TOKENIZER = AutoTokenizer.from_pretrained(f'{base_path}/checkpoints/sft/base/Qwen2-0_5B_exp7/', padding_side="right", use_fast=True)

# Request templates, Category dict and Patterns
REQUEST_CONFIG = {
    'en': {
        "GPT_REQUEST_TEMPLATE": """Please carefully read and analyze the following text, score it based on fourteen evaluation criteria and their respective scoring definitions. 
Additionally, select the most appropriate category from the fifteen domain types that best matches the content of the text. Let's think step by step.
Text: {text}
Domain Types: [A]Medicine [B]Finance [C]Law [D]Education [E]Technology [F]Entertainment [G]Mathematics [H]Coding [I]Government [J]Culture [K]Transportation [L]Retail E-commerce [M]Telecommunication [N]Agriculture [O]Other
Evaluation Criteria:
[1]Textual Accuracy: The higher the score, the fewer grammar, referential, and spelling errors the text contains, and the more accurate its expression. _/5
[2]Semantic Coherence: The higher the score, the more fluent the content is expressed, and the stronger its logical coherence. _/5
[3]Language Consistency: The higher the score, the more consistent the use of language in the text, with less mixing of languages. _/5
[4]Ratio of Valid Semantic Content: The higher the score, the greater the proportion of valid information in the text, with less irrelevant or redundant information. _/5
[5]Novelty of Knowledge: The higher the score, the more novel and cutting-edge the knowledge provided by the text, with more insightful views on the industry or topic. _/5
[6]Topic Focus: The higher the score, the more the text content focuses on the topic, with less deviation from the main theme. _/5
[7]Creative Expression Ratio: The higher the score, the more creative elements are shown in the text's expression. _/5
[8]Professionalism: The higher the score, the more professional terminology appears in the text, with more accurate use of terms and more professional domain-specific expression. _/5
[9]Style Consistency: The higher the score, the more consistent the style of the text, with proper and appropriate style transitions. _/5
[10]Richness of Grammatical Structures: The higher the score, the more varied and correct the grammatical structures used in the text, showing a richer language expression ability. _/5
[11]Standardization of Structure: The higher the score, the clearer the structure followed by the text and the more standardized its format. _/5
[12]Non-repetitiveness of Content: The higher the score, the fewer repetitions and similar content in the text. _/5
[13]Appropriateness of Sensitive Content: The higher the score, the more appropriately sensitive topics are handled in the text, with less inappropriate content. _/5
[14]Overall Score: The higher the score, the better the comprehensive evaluation of the text, with superior performance in all aspects.""",
    "QWEN_REQUEST_TEMPLATE": """Please score the text on fourteen evaluation criteria and specify its domain:
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
"QWEN_SCORE_REQUEST_TEMPLATE": """Please give an overall score for the text:
Text: {text}
Overall Score:_/5""",
"QWEN_DETPROMPT_SCORE_REQUEST_TEMPLATE": """Please give an overall score for the text:
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
"QWEN_DOMAIN_REQUEST_TEMPLATE": """Please specify an domain type for the text:
Text: {text}
Domain:_""",
"QWEN_DETPROMPT_DOMAIN_REQUEST_TEMPLATE": """Please specify an domain type for the text:
Text: {text}
Domain Types: [A]Medicine [B]Finance [C]Law [D]Education [E]Technology [F]Entertainment [G]Mathematics [H]Coding [I]Government [J]Culture [K]Transportation [L]Retail E-commerce [M]Telecommunication [N]Agriculture [O]Other
Domain:_""",
## 适用于gpt-4-turbo
# "SCORE_PATTERN": r"(?:\[[1-9][0-4]?\]|\b[1-9][0-4]?\.|\b[1-9][0-4]?\))[^:]+:\**\s\**(\d+(?:\.\d+)?)/5|\b[1-9][0-4]?\.[^:]+\s\((\d+(?:\.\d+)?)/5\)",
## 适用于gpt-4o
"SCORE_PATTERN": r"(?:\[[1-9][0-4]?\]|\b[1-9][0-4]?\.|\b[1-9][0-4]?\)).*(?:\*+|\s+|\(+)(\d+(\.\d+)?)/5",
# "SCORE_PATTERN": 
"CATEGORY_DICT": {"A":"Medicine","B":"Finance","C":"Law","D":"Education","E":"Technology","F":"Entertainment","G":"Mathematics","H":"Coding",
                    "I":"Government","J":"Culture","K":"Transportation","L":"Retail E-commerce","M":"Telecommunication","N":"Agriculture","O":"Other"},
"CATEGORY_PATTERN": r'\[\s*([A-Z])\s*\]|(?<=\s)([A-Z])(?=\])(?!\]\])|\bDomain Type:\**\s*\**([A-Z])\s*(?=[\(\)\*\-])',
"TEXT_PATTERN": r'\nText: (.*?)(\n)+Domain Types',
"NA_PATTERN": r'(?:\*\s|:\s)N/A'
    },
'zh': {
        "GPT_REQUEST_TEMPLATE": """请仔细阅读和分析这段文本，根据十四项评价指标及其相应的分数定义给文本打分。同时，从十五个领域类型中选择最符合文本内容的一个类别。让我们一步步地思考。
文本：{text}
领域类型：[A]医疗 [B]金融 [C]法律 [D]教育 [E]科技 [F]娱乐 [G]数学 [H]代码 [I]政务 [J]文化 [K]交通 [L]零售电商 [M]电信 [N]农业 [O]其他
评价指标：
[1]文本准确性：分数越高，文本的语法、指代和拼写错误越少，表达越准确。 _/5
[2]语义连贯性：分数越高，文本内容表达越流畅，逻辑连贯性越强。 _/5
[3]语种统一性：分数越高，文本使用的语种越一致，杂糅语言的使用越少。 _/5
[4]有效语义内容比例：分数越高，文本所含有效信息的比例越多，无关或冗余信息越少。 _/5
[5]知识新颖性：分数越高，文本提供的知识越新颖和越前沿，对行业或主题的见解越有洞察力。 _/5
[6]主题专注度：分数越高，文本内容越专注于主题，越少偏离主线。 _/5
[7]创意表达比例：分数越高，文本在表达中展现的创意成分越多。 _/5
[8]专业程度：分数越高，文本中出现的专业术语越多，术语使用越准确，领域内表达越专业。 _/5
[9]风格统一度：分数越高，文本的风格越统一，风格转换得当且恰当。 _/5
[10]语法结构丰富性：分数越高，文本采用的语法结构越多样且正确，表现出更丰富的语言表达能力。 _/5
[11]结构规范程度：分数越高，文本遵循的结构越明确，格式越规范。 _/5
[12]内容不重复度：分数越高，文本中的重复和相似内容越减少。 _/5
[13]敏感内容适宜度：分数越高，文本中对敏感话题的处理越得体，越少出现不适宜内容。 _/5
[14]总评分：分数越高，文本综合评价越高，各方面表现越优秀。""",
    "QWEN_REQUEST_TEMPLATE": """请给出文本的十四项评价指标分数和领域类别:
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
"QWEN_SCORE_REQUEST_TEMPLATE": """请给出文本的整数总评分:
文本: {text}
总评:_/5""",
"QWEN_DETPROMPT_SCORE_REQUEST_TEMPLATE": """请给出文本的整数总评分:
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
"QWEN_DOMAIN_REQUEST_TEMPLATE": """请给出文本的领域类型:
文本: {text}
领域:_""",
"QWEN_DETPROMPT_DOMAIN_REQUEST_TEMPLATE": """请给出文本的领域类型:
文本: {text}
领域类型：[A]医疗 [B]金融 [C]法律 [D]教育 [E]科技 [F]娱乐 [G]数学 [H]代码 [I]政务 [J]文化 [K]交通 [L]零售电商 [M]电信 [N]农业 [O]其他
领域:_""",
"SCORE_PATTERN": r"\[[1-9][0-4]?][^：]+：(\d+(?:\.\d+)?)/5",
"CATEGORY_DICT": {"A":"医疗","B":"金融","C":"法律","D":"教育","E":"科技","F":"娱乐","G":"数学","H":"代码",
                    "I":"政务","J":"文化","K":"交通","L":"零售电商","M":"电信","N":"农业","O":"其他"},
"CATEGORY_PATTERN": '领域类型.*?[\[\]]?([A-Z])[\]\[]?',
"TEXT_PATTERN": r'\n文本：(.*?)(\n)+领域类型',
"NA_PATTERN": r'：N/A'
    },
}
# Regular expression: matches ellipses surrounded by double quotes, matches URL items starting with ![](//) characters, score, domain type, raw text, N/A
ELLIPSIS_PATTERN = r'^["“”]…+["“”]|^["“”]\.\.\.+["“”]'
LINK_PATTERN = r"!\[\]\([\/]?[^\)]+\)\n"


def fix_seed(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def regualr_fitering(text):
    text = re.sub(ELLIPSIS_PATTERN, '', text)
    text = re.sub(LINK_PATTERN, '', text)
    return text

def truncate_tokens(text, max_length, keep_tokens):
    tokens = TOKENIZER.tokenize(text)
    if len(tokens) > max_length:
        tokens = tokens[:keep_tokens] + tokens[-keep_tokens:]
        return TOKENIZER.convert_tokens_to_string(tokens), True
    return TOKENIZER.convert_tokens_to_string(tokens), False

def count_lines(filename):
    result = subprocess.run(['python', 'DataMan/utils/num_of_lines.py',
                                '--input_file', filename], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        if 'Number of lines:' in line:
            return int(line.split(':')[-1].strip())
    raise RuntimeError("Failed to determine the number of lines from output")

def unsample(lines, upsampled_size):
    return random.choices(lines, k=upsampled_size) if lines else []

def set_request_config(lang):
    if lang not in REQUEST_CONFIG:
        raise ValueError(f"Unsupported language: {lang}")

    request_config = REQUEST_CONFIG[lang]

    global GPT_REQUEST_TEMPLATE, QWEN_REQUEST_TEMPLATE, QWEN_SCORE_REQUEST_TEMPLATE, QWEN_DETPROMPT_SCORE_REQUEST_TEMPLATE, \
        QWEN_DOMAIN_REQUEST_TEMPLATE, QWEN_DETPROMPT_DOMAIN_REQUEST_TEMPLATE, SCORE_PATTERN, CATEGORY_DICT, CATEGORY_PATTERN, TEXT_PATTERN, NA_PATTERN

    GPT_REQUEST_TEMPLATE = request_config["GPT_REQUEST_TEMPLATE"]
    QWEN_REQUEST_TEMPLATE = request_config["QWEN_REQUEST_TEMPLATE"]
    QWEN_SCORE_REQUEST_TEMPLATE = request_config["QWEN_SCORE_REQUEST_TEMPLATE"]
    QWEN_DETPROMPT_SCORE_REQUEST_TEMPLATE = request_config["QWEN_DETPROMPT_SCORE_REQUEST_TEMPLATE"]
    QWEN_DOMAIN_REQUEST_TEMPLATE = request_config["QWEN_DOMAIN_REQUEST_TEMPLATE"]
    QWEN_DETPROMPT_DOMAIN_REQUEST_TEMPLATE = request_config["QWEN_DETPROMPT_DOMAIN_REQUEST_TEMPLATE"]
    SCORE_PATTERN = request_config["SCORE_PATTERN"]
    CATEGORY_DICT = request_config["CATEGORY_DICT"]
    CATEGORY_PATTERN = request_config["CATEGORY_PATTERN"]
    TEXT_PATTERN = request_config["TEXT_PATTERN"]
    NA_PATTERN = request_config["NA_PATTERN"]


def create_transformed_data(index, request, response=None, gpt_model=None, data_type='gpt'):
    data_structures = {
        'gpt': {
            "index": index,
            "request_json": {
                "model": gpt_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert to evaluate the text quality with high accuracy and confidence. Don't hesitate to use the full range of the score scale, including extreme scores if the text warrants it."
                    },
                    {
                        "role": "user",
                        "content": request
                    },
                ],
                "temperature": 0.0,
                "max_tokens": 4096,
            }
        },
        'qwen': {
            "id": str(index),
            "conversations": [
                {"from": "user", "value": request},
                {"from": "assistant", "value": response},
            ]
        },
        'qwen2': {
            "type": "chatml",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": request},
                {"role": "assistant", "content": response}
            ],
            "source": str(index)
        }
    }

    if data_type in data_structures:
        return data_structures[data_type]
    
    raise ValueError("Unsupported data type. Please choose 'gpt', 'qwen', or 'qwen2'.")


def prepare_gpt_request(input_path, output_path, gpt_model, truncate, trucate_max_length, sample_size, exist_output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    text_path = os.path.join(os.path.dirname(output_path), os.path.basename(output_path).replace('_request', ''))
    source, extension = os.path.splitext(os.path.basename(input_path))

    # Collect existing line numbers
    existing_indices = set()
    if exist_output_path:
        for path in exist_output_path:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    index = data.get('index')
                    _, num = index.rsplit('_', 1)
                    existing_indices.add(int(num)-1)   # line numbers start from 1, but set need to start from 0

    # Handle truncation and random selection
    if truncate:  
        total_lines = count_lines(input_path)
        assert total_lines >= sample_size, "Not enough lines to sample."
        available_line_numbers = list(set(range(total_lines)) - existing_indices)
        selected_line_numbers = sorted(random.sample(available_line_numbers, sample_size))

        with open(input_path, 'r', encoding='utf-8') as inp_f, open(output_path, 'a', encoding='utf-8') as out_f, open(text_path, 'w', encoding='utf-8') as txt_f: 
            for i, line in enumerate(inp_f):
                if i not in selected_line_numbers:  
                    continue  

                data = json.loads(line)
                text = data.get('text', data.get('content'))           
                text = regualr_fitering(text)
                truncated_text, _ = truncate_tokens(text, trucate_max_length, trucate_max_length//2)
                request = GPT_REQUEST_TEMPLATE.format(text=truncated_text)
                transformed_data = create_transformed_data(f"{source}_{i+1}", request, gpt_model=gpt_model, data_type='gpt')
                txt_f.write(json.dumps(truncated_text, ensure_ascii=False) + '\n')
                out_f.write(json.dumps(transformed_data, ensure_ascii=False) + '\n')
            print(f"Transformed file saved to {output_path}")
    # Handle case without truncation
    else:
        with open(input_path, 'r', encoding='utf-8') as inp_f, open(output_path, 'w', encoding='utf-8') as out_f, open(text_path, 'w', encoding='utf-8') as txt_f: 
            for i, line in enumerate(inp_f):
                if extension == ".jsonl":      
                    data = json.loads(line)
                elif extension == ".csv":       
                    source = line.split('\t')[0]
                    data = json.loads(line.split('\t')[1])
                text = data.get('text', data.get('content', data.get('txt'))) 
                text = regualr_fitering(text)
                request = GPT_REQUEST_TEMPLATE.format(text=text)
                transformed_data = create_transformed_data(f"{source}_{i+1}", request, gpt_model=gpt_model, data_type='gpt')
                txt_f.write(json.dumps(text, ensure_ascii=False) + '\n')
                out_f.write(json.dumps(transformed_data, ensure_ascii=False) + '\n')            
            print(f"Transformed file saved to {output_path}")



def process_gpt_response(input_path, output_path, gpt_model):
    # Prepare output directory, request and successfully processed path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    request_path = output_path.replace('failed', 're_request')
    success_path = output_path.replace('failed', 'success')

    # Failed response, request processed by GPT4 again, successful response
    fail_lines, request_lines, success_lines = [], [], []
    with open(input_path, 'r', encoding='utf-8') as inp_f, open(output_path, 'a', encoding='utf-8') as fail_f,\
    open(request_path, 'a', encoding='utf-8') as req_f, open(success_path, 'a', encoding='utf-8') as suc_f:
        for i, line in enumerate(inp_f):
            data = json.loads(line)
            index = data["index"]

            # Fail response1: the text begins with double quotes + ellipsis and url link, replace these characters to make new request
            request = data['request_json']['messages'][1]['content']
            if re.search(ELLIPSIS_PATTERN, request):
                request = re.sub(ELLIPSIS_PATTERN, '', request)
            if re.search(LINK_PATTERN, request):
                request = re.sub(LINK_PATTERN, '', request)
            transformed_data = create_transformed_data(index, request, gpt_model=gpt_model, data_type='gpt')
            
            # Fail response2: Calling GPT API to respond failed
            if (data['status'] != "success") or ('data' not in data['response']) or (not isinstance(data['response']['data'], dict)):
                print(f"{index} GPT API调用失败。")
                fail_lines.append(f"GPT API调用失败。{index}")
                request_lines.append(transformed_data)
                continue

            # Note: There will be a response only after successfully calling the GPT API
            gpt_response = data['response']['data']["response"]['choices'][0]["message"]['content']

            # Fail response3: N/A" in the response
            if re.search(NA_PATTERN, gpt_response):
                print(f"{index} 存在N/A。")
                fail_lines.append(f"存在N/A。{index} {gpt_response}")
                request_lines.append(transformed_data)
                continue
            
            scoresA = re.findall(SCORE_PATTERN, gpt_response)
            scores = [next(filter(None, item)) for item in scoresA]
            # Fail response4: No scores output in response
            if scores == None or scores == []:  
                print(f"{index} 没有匹配到评分。")
                fail_lines.append(f"没有匹配到评分。{index} {gpt_response}")
                request_lines.append(transformed_data)
                continue
            # Fail response5: incomplete (less than 14) scores output in the response
            elif len(scores) < 14:   
                print(f"{index} 评分不完整，仅有{len(scores)}项。")
                fail_lines.append(f"评分不完整，仅有{len(scores)}项。{index} {gpt_response}")
                request_lines.append(transformed_data)
                continue
            # Fail response6: too many (greater than 14) scores output in the response
            elif len(scores) > 14 and len(scores) != 28:   
                print(f"{index} 评分抽取有误，有过多{len(scores)}项。")
                fail_lines.append(f"评分抽取有误，有过多{len(scores)}项。{index} {gpt_response}")
                request_lines.append(transformed_data)
                continue
            elif len(scores) == 28:
                first_half, second_half = scores[:14], scores[14:]
                # Success response, 若刚好是相同的两套分数，则只保留一组
                if first_half == second_half:
                    scores = first_half
                # Fail response, 若不是相同的两套分数，则还是失败
                else:
                    print(f"{index} 评分抽取有误，有过多{len(scores)}项。")
                    fail_lines.append(f"评分抽取有误，有过多{len(scores)}项。{index} {gpt_response}")
                    request_lines.append(transformed_data)
                    continue
            assert len(scores) == 14, "分数项不是14个"

            categories = re.findall(CATEGORY_PATTERN, gpt_response)
            categories = [next(filter(None, item)) for item in categories]
            categories = list(set(categories))  # 防止文本重复确定领域类型
            # Fail response7: No categories output in response
            if categories == None or categories == []:
                print(f"{index} 没有匹配到领域类型。")
                fail_lines.append(f"没有匹配到领域类型。{index} {gpt_response}")
                request_lines.append(transformed_data)
                continue   
            # Fail response8: multiple categories (not typical domain) in response
            elif len(categories) > 1:
                print(f"{index} 输出{len(categories)}个领域类型，{categories}。")
                fail_lines.append(f"输出{len(categories)}个领域类型，{categories}。{index} {gpt_response}")
                request_lines.append(transformed_data)
                continue   
            else:
                categories = [CATEGORY_DICT[categories[0]]]
            assert len(categories) == 1, "分数项不是1个"

            success_lines.append(data)

        for l in fail_lines:
            fail_f.write(json.dumps(l, ensure_ascii=False) + '\n')
        print(f"Failed {len(fail_lines)} sentence in {output_path}")
        for l in request_lines:
            req_f.write(json.dumps(l, ensure_ascii=False) + '\n')
        print(f"Re-requested {len(request_lines)} sentence in {request_path}")
        for l in success_lines:
            suc_f.write(json.dumps(l, ensure_ascii=False) + '\n')
        print(f"Successed {len(success_lines)} sentence in {success_path}")
        print(f"Processed file {input_path}")


def prepare_qwen_request(input_path, output_path, qwen_version, text_version, sample_size, valid_size, test_size, seed, quantile):
    # Prepare output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    source_data = defaultdict(list)

    def if_detail_text(org_text, prompt_template):
        prompt_template_tokens = TOKENIZER.tokenize(prompt_template.replace("{text}", ""))
        new_trucate_max_length = 2048 - len(prompt_template_tokens)
        truncated_text, _ = truncate_tokens(org_text, new_trucate_max_length, new_trucate_max_length//2)
        return truncated_text

    if text_version == "detText":
        source_indexes = defaultdict(list)
        with open(input_path, 'r', encoding='utf-8') as rf:
            for line in tqdm(rf, desc='Building index'):
                data = json.loads(line)
                index = data["index"].rpartition('_')
                source, line_index = index[0], int(index[-1])
                source_indexes[source].append(line_index)

        line_texts = {}
        for source, indexes in tqdm(source_indexes.items(), desc='Extracting lines'):
            file_path = os.path.join(SOURCE_FILES_DIR, f'{source}.jsonl')
            with open(file_path, 'r', encoding='utf-8') as file:
                for i, line in enumerate(file, start=1):
                    if i in indexes:
                        data = json.loads(line)
                        line_text = data.get('text', data.get('content'))
                        line_texts[source+f"_{i}"] = line_text

        # 根据input_path文件的顺序重建text_list
        text_list = []
        with open(input_path, 'r', encoding='utf-8') as rf:
            for line in tqdm(rf, desc='Reordering text list'):
                data = json.loads(line)
                text_list.append(line_texts[data["index"]])
        text_list = iter(text_list)

    with open(input_path, 'r', encoding='utf-8') as rf:
        total_input_lines = sum(1 for _ in rf)
        rf.seek(0) 
        for line in tqdm(rf, total=total_input_lines, desc="Processing"):
            if text_version == "detText":
                org_text = next(text_list, None)
            data = json.loads(line)
            index = data["index"]
            source, line_index = index.rpartition('_')[0], int(index.rpartition('_')[-1])
            gpt_request = data['request_json']['messages'][1]['content']
            gpt_response = data['response']['data']["response"]['choices'][0]["message"]['content']
            scores = re.findall(SCORE_PATTERN, gpt_response)
            scores = [next(filter(None, item)) for item in scores]
            # Round each score, Below 3 points -> round down, above 3 points -> round up
            scores = [math.floor(float(score)) if float(score) < 3 else math.ceil(float(score)) for score in scores]
            if len(scores) == 28:
                first_half, second_half = scores[:14], scores[14:]
                if first_half == second_half:
                    scores = first_half
                else:
                    continue
            overall_score = scores[-1]
            categories = re.findall(CATEGORY_PATTERN, gpt_response)
            categories = [next(filter(None, item)) for item in categories]
            categories = list(set(categories))   # Prevent text from duplicating and detPromptermine field type
            category = categories[0]
            # Match the original truncated text from resuqet.jsonl. If the text is not matched, discard it~
            match= re.search(TEXT_PATTERN, gpt_request, re.DOTALL)
            if match:
                text = match.group(1).strip()
            else:
                continue
            if qwen_version in ['qwen2', 'qwen2_balance_q']:
                qwen_response = "\n".join(categories + [str(score) for score in scores])
                qwen_request = QWEN_REQUEST_TEMPLATE.format(text=text)
            elif qwen_version in ['qwen2_score', 'qwen2_score_balance_q']:
                qwen_response = f"{overall_score}"
                if text_version == "detText":
                    text = if_detail_text(org_text, QWEN_SCORE_REQUEST_TEMPLATE)
                qwen_request = QWEN_SCORE_REQUEST_TEMPLATE.format(text=text)
            elif qwen_version in ['qwen2_detPrompt_score', 'qwen2_detPrompt_score_balance_q']:
                qwen_response = f"{overall_score}"
                if text_version == "detText":
                    text = if_detail_text(org_text, QWEN_DETPROMPT_SCORE_REQUEST_TEMPLATE)
                qwen_request = QWEN_DETPROMPT_SCORE_REQUEST_TEMPLATE.format(text=text)
            elif qwen_version in ['qwen2_domain', 'qwen2_domain_balance_q']:
                qwen_response = f"{category}"
                if text_version == "detText":
                    text = if_detail_text(org_text, QWEN_DOMAIN_REQUEST_TEMPLATE)
                qwen_request = QWEN_DOMAIN_REQUEST_TEMPLATE.format(text=text)
            elif qwen_version in ['qwen2_detPrompt_domain', 'qwen2_detPrompt_domain_balance_q']:
                qwen_response = f"{category}"
                if text_version == "detText":
                    text = if_detail_text(org_text, QWEN_DETPROMPT_DOMAIN_REQUEST_TEMPLATE)
                qwen_request = QWEN_DETPROMPT_DOMAIN_REQUEST_TEMPLATE.format(text=text)
            transformed_data = create_transformed_data(index, qwen_request, response=qwen_response, data_type=qwen_version.split("_")[0])
            transformed_data = json.dumps(transformed_data, ensure_ascii=False) + "," if re.match(r'^qwen(?!2)', qwen_version) \
                else json.dumps(transformed_data, ensure_ascii=False) if re.match(r'^qwen2', qwen_version) else None
            source_data[source].append((transformed_data, overall_score))
    print(f"All source data loaded")

    # Split the training set and test set at the source granularity
    base_filename, file_extension = os.path.splitext(output_path)
    if qwen_version.endswith("balance_q"):
        for k in range(1, quantile+1):
            finetune_train_path_k, finetune_valid_path_k, finetune_test_path_k = f"{base_filename}{k}_train{file_extension}", f"{base_filename}{k}_valid{file_extension}", \
                f"{base_filename}{k}_test{file_extension}"
            train_set_k, valid_set_k, test_set_k = [], [], []
            
            for source, tuples_list in source_data.items():
                lines, overall_scores = zip(*tuples_list) 
                ID_lines, test_lines, ID_overall_scores, test_overall_scores = train_test_split(lines, overall_scores, 
                                                                                                test_size=(test_size/len(lines)), random_state=seed, shuffle=True)
                train_lines, valid_lines, train_overall_scores, valid_overall_scores = train_test_split(ID_lines, ID_overall_scores, 
                                                                                        test_size=(valid_size/(len(lines)-test_size)), random_state=seed, shuffle=True)
                sample_indices = random.sample(range(len(train_lines)), min(sample_size, len(train_lines)))
                train_lines = [train_lines[i] for i in sample_indices]
                train_overall_scores = [train_overall_scores[i] for i in sample_indices]

                # Up-sampling Operation
                low_score_lines = [line for line, score in zip(train_lines, train_overall_scores) if score < 3]
                high_score_lines = [line for line, score in zip(train_lines, train_overall_scores) if score >= 3]
                low_score_size = len(low_score_lines)
                high_score_size = len(high_score_lines)

                # 这里很特殊，因为当高分样本远多于低分样本时，它两差值非常大，哪怕1/quantile的差值都远大于低分样本数，
                # 这时候random.choices(lines, k=min(upsampled_size, len(lines)))，永远都是低分样本数少，
                # 因此采样数如果等于一成不变的低分样本数，那设置几等分的上采样比例就没有意义了
                # 故就改成两者差值与低分样本的较小值
                if high_score_size >= low_score_size:
                    difference_size = high_score_size - low_score_size
                    upsampled_size = min(difference_size // quantile, low_score_size) *k
                    upsampled_lines = unsample(low_score_lines, upsampled_size)
                else:
                    difference_size = low_score_size - high_score_size
                    upsampled_size  = min(difference_size // quantile, high_score_size)*k
                    upsampled_lines = unsample(high_score_lines, upsampled_size)
                train_lines.extend(upsampled_lines)  # add sampled line into orignal traning set
                test_set_k.extend(test_lines)
                valid_set_k.extend(valid_lines)
                train_set_k.extend(train_lines)
                print(f"Finished {source}, Train data size: {len(train_lines)}, Valid data size: {len(valid_lines)}, Test data size: {len(test_lines)}")

            with open(finetune_train_path_k, 'w', encoding='utf-8') as train_f, open(finetune_valid_path_k, 'w', encoding='utf-8') as valid_f, \
                open(finetune_test_path_k, 'w', encoding='utf-8') as test_f:
                train_f.write("["+("\n".join(train_set_k)).rstrip(",")+"]" if re.match(r'^qwen(?!2)', qwen_version) else "\n".join(train_set_k) if re.match(r'^qwen2', qwen_version) else None)
                valid_f.write("["+("\n".join(valid_set_k)).rstrip(",")+"]" if re.match(r'^qwen(?!2)', qwen_version) else "\n".join(valid_set_k) if re.match(r'^qwen2', qwen_version) else None)
                test_f.write("["+("\n".join(test_set_k)).rstrip(",")+"]" if re.match(r'^qwen(?!2)', qwen_version) else "\n".join(test_set_k) if re.match(r'^qwen2', qwen_version) else None)
            print(f"Transformed file saved to {output_path}")
    else:
        finetune_train_path, finetune_valid_path, finetune_test_path = f"{base_filename}_train{file_extension}", f"{base_filename}_valid{file_extension}", \
            f"{base_filename}_test{file_extension}"
        train_set, valid_set, test_set = [], [], []

        for source, tuples_list in source_data.items():
            lines, _ = zip(*tuples_list) 
            ID_lines, test_lines = train_test_split(lines, test_size=(test_size/len(lines)), random_state=seed, shuffle=True)
            train_lines, valid_lines = train_test_split(ID_lines, test_size=(valid_size/(len(lines)-test_size)), random_state=seed, shuffle=True)
            train_lines = random.sample(train_lines, min(sample_size, len(train_lines)))
            test_set.extend(test_lines)
            valid_set.extend(valid_lines)
            train_set.extend(train_lines)
            print(f"Finished {source}, Train data size: {len(train_lines)}, Valid data size: {len(valid_lines)}, Test data size: {len(test_lines)}")

        with open(finetune_train_path, 'w', encoding='utf-8') as train_f, open(finetune_valid_path, 'w', encoding='utf-8') as valid_f, \
            open(finetune_test_path, 'w', encoding='utf-8') as test_f:
            train_f.write("["+("\n".join(train_set)).rstrip(",")+"]" if re.match(r'^qwen(?!2)', qwen_version) else "\n".join(train_set) \
                if re.match(r'^qwen2', qwen_version) else None)
            valid_f.write("["+("\n".join(valid_set)).rstrip(",")+"]" if re.match(r'^qwen(?!2)', qwen_version) else "\n".join(valid_set) \
                if re.match(r'^qwen2', qwen_version) else None)
            test_f.write("["+("\n".join(test_set)).rstrip(",")+"]" if re.match(r'^qwen(?!2)', qwen_version) else "\n".join(test_set) \
                if re.match(r'^qwen2', qwen_version) else None)
        print(f"Transformed file saved to {output_path}")


def main(args):
    fix_seed(seed=args.seed)
    set_request_config(args.lang)
    if args.process_type == 'prepare_gpt_request':  
        prepare_gpt_request(args.input_path, args.output_path, args.gpt_model, args.truncate, args.trucate_max_length, args.sample_size, args.exist_output_path)
    elif args.process_type == 'request_openai':
        request_openai(args.input_path, args.output_path, args.max_attempt, args.log_items, args.batch_size)
    elif args.process_type == 'process_gpt_response':  
        process_gpt_response(args.input_path, args.output_path, args.gpt_model)
    elif args.process_type == 'prepare_qwen_request':  
        prepare_qwen_request(args.input_path, args.output_path, args.qwen_version, args.text_version, args.sample_size, args.valid_size, args.test_size, args.seed, args.quantile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='input jsonl file.')
    parser.add_argument('--output_path', type=str, required=True, help='output jsonl file.')
    parser.add_argument('--exist_output_path', type=str, nargs="+", help='may already exist output files')
    parser.add_argument('--lang', default='en', help='language name', choices=['en', 'zh'])
    parser.add_argument('--qwen_version', default="qwen", type=str, choices=['qwen2', 'qwen2_balance_q',
    'qwen2_score', 'qwen2_score_balance_q', 'qwen2_detPrompt_score', 'qwen2_detPrompt_score_balance_q',
    'qwen2_domain', 'qwen2_domain_balance_q', 'qwen2_detPrompt_domain', 'qwen2_detPrompt_domain_balance_q'])
    parser.add_argument('--text_version', default="text", type=str, choices=['text', 'detText'])
    parser.add_argument('--truncate', default=False, action="store_true", help='truncate requests fed to gpt')
    parser.add_argument('--trucate_max_length', default=1896, type=int, help='maximum truncated token length')
    parser.add_argument('--sample_size', default=4000, type=int, help='data size per source')
    parser.add_argument('--test_size', default=100, type=int, help='test size per source')
    parser.add_argument('--valid_size', default=100, type=int, help='valid size per source')
    parser.add_argument('--seed', default=1024, type=int, help='seed size')
    parser.add_argument('--quantile', default=0, type=int, help='upsample the training set with imbalanced samples by several equal parts')
    parser.add_argument('--max_attempt', type=int, default=3, help='max attempt times per request.')
    parser.add_argument('--log_items', type=int, default=1, help='log when how many items request.')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size when request.')
    parser.add_argument('--process_type', type=str, required=True, help='process type', choices=['prepare_gpt_request', 'process_gpt_response', 'prepare_qwen_request'])
    args = parser.parse_args()
    main(args)
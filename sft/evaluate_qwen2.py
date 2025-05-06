import os
import sys
import json
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def fix_seed(seed=1024):
    assert isinstance(seed, int)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        sources, messages, responses = [], [], []
        for line in f.readlines():
            data = json.loads(line)
            sources.append(data['source'].rpartition('_')[0])
            messages.append(data['messages'][:2])
            responses.append(data['messages'][-1])
    return sources, messages, responses


def compute_domain_metrics(y_true, y_pred):
    ## Domain Acc
    accuracy = accuracy_score(y_true, y_pred)

    ## Domain F1
    f1 =  f1_score(y_true, y_pred, average='macro')

    # Domain sub-class Acc
    CATEGORY_DICT = {"A":"医疗","B":"金融","C":"法律","D":"教育","E":"科技","F":"娱乐","G":"数学","H":"代码",
                     "I":"政务","J":"文化","K":"交通","L":"零售电商","M":"电信","N":"农业","O":"其他"}
    domains = set(y_true)
    domain_accuracies = {}
    for domain in sorted(domains):
        indices = [i for i, label in enumerate(y_true) if label == domain]
        domain_true = [y_true[i] for i in indices]
        domain_pred = [y_pred[i] for i in indices]
        domain_accuracy = accuracy_score(domain_true, domain_pred)
        domain_accuracies[CATEGORY_DICT[domain]+"Acc"] = round(domain_accuracy,3)

    # Variance of domain sub-classification accuracy:
    variance = np.var(list(domain_accuracies.values()))

    metrics = {'Domain Accuracy': f"{accuracy:.3f}", 
               'Domain Variance': f"{variance:.4f}", 
               'Domain F1 Score': f"{f1:.3f}", 
              }
    metrics.update(domain_accuracies) 
    return metrics


def compute_score_metrics(prompts, y_true, y_pred, print_false_case=False):
    # Acc
    acc_five = accuracy_score(y_true, y_pred)
    true_two, pred_two = np.where(y_true < 3, 0, 1), np.where(y_pred < 3, 0, 1)
    acc_two = accuracy_score(true_two, pred_two)

    # F1 scores
    f1_five = f1_score(y_true, y_pred, average='macro')
    f1_two = f1_score(true_two, pred_two, average='macro')

    # Acc of Positive Sample, Negative Sample
    cm = confusion_matrix(true_two, pred_two)
    if cm.shape == (2, 2):
        denominator_0 = cm[0, 0] + cm[0, 1]
        denominator_1 = cm[1, 0] + cm[1, 1]
        # Avoid division by zero by checking if denominator is not zero
        acc_neg = cm[0, 0] / denominator_0 if denominator_0 > 0 else -1000
        acc_pos = cm[1, 1] / denominator_1 if denominator_1 > 0 else -1000
        acc_diff = -1000 if (denominator_0 <= 0 or denominator_1 <= 0) else acc_neg - acc_pos
    else:
        acc_neg = acc_pos = acc_diff = np.nan  # Using np.nan for undefined cases
        print("Confusion matrix is not 2x2. Please check your data.")

    # Miss Rate of Negative Sample 
    prob_neg_false1 = np.sum((y_true < 2) & (y_pred >= 3)) / np.sum(y_true < 3)
    prob_neg_false2 = np.sum((y_true >= 2) & (y_true < 3) & (y_pred > 3)) / np.sum(y_true < 3)
    prob_neg_false3 = np.sum((y_true >= 2) & (y_true < 3) & (y_pred == 3)) / np.sum(y_true < 3)

    # Misjudge Rate of Positive Sample 
    prob_pos_false1 = np.sum((y_true >= 4) & (y_pred < 3)) / np.sum(y_true >= 3)
    prob_pos_false2 = np.sum((y_true >= 3) & (y_true < 4) & (y_pred < 3)) / np.sum(y_true >= 3)

    if print_false_case:
        # Case Study of Missing Negative Texts 
        miss_neg_indices = [i for i, (true_v, pred_v) in enumerate(zip(y_true, y_pred)) if true_v < 2 and pred_v >= 3]
        miss_neg_texts = [prompts[i] for i in miss_neg_indices]
        miss_neg_true_v = [y_true[i] for i in miss_neg_indices]
        miss_neg_pred_v = [y_pred[i] for i in miss_neg_indices]
        print("Missing Negative Texts:")
        for text, true_v, pred_v in zip(miss_neg_texts, miss_neg_true_v, miss_neg_pred_v):
            print(json.dumps(text, ensure_ascii=False))
            print(f"{true_v},{pred_v}"  + '\n')
        
        # Case Study of Misjuding Positive Texts
        misjudge_pos_indices = [i for i, (true_v, pred_v) in enumerate(zip(y_true, y_pred)) if true_v >= 4 and pred_v <= 2]
        misjudge_pos_texts = [prompts[i] for i in misjudge_pos_indices]
        misjudge_pos_true_v = [y_true[i] for i in misjudge_pos_indices]
        misjudge_pos_pred_v = [y_pred[i] for i in misjudge_pos_indices]
        print("Misjuding Positive Texts:")
        for text, true_v, pred_v in zip(misjudge_pos_texts, misjudge_pos_true_v, misjudge_pos_pred_v):
            print(json.dumps(text, ensure_ascii=False))
            print(f"{true_v},{pred_v}"+"\n")
    
    # Corr
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)

    metrics = {'Five-class Accuracy': f"{acc_five:.3f}",
               'Two-class Accuracy': f"{acc_two:.3f}",
               'Positive Sample Accuracy': f"{acc_pos:.4f}",
               'Negative Sample Accuracy': f"{acc_neg:.4f}", 
               'Negative Sample Miss Rate1': f"{prob_neg_false1:.4f}", 
               'Negative Sample Miss Rate2': f"{prob_neg_false2:.4f}", 
               'Negative Sample Miss Rate3': f"{prob_neg_false3:.4f}", 
               'Positive Sample Misjudge Rate1': f"{prob_pos_false1:.4f}",
               'Positive Sample Misjudge Rate2': f"{prob_pos_false2:.4f}",
               'Pearson\'s Correlation': f"{pearson_corr:.3f}", 
               'Spearman\'s Correlation': f"{spearman_corr:.3f}",
               'Five-class F1 Score': f"{f1_five:.3f}",
               'Two-class F1 Score': f"{f1_two:.3f}",
            }
    return metrics


def report_all_rating_result(true_responses, pred_responses):
    true_scores = np.array([list(map(float, item.split("\n")[1:])) for item in true_responses], dtype=np.float16).T
    pred_scores = np.array([list(map(float, item.split("\n")[1:])) for item in pred_responses], dtype=np.float16).T
    print("Score Results: ")
    score_metrics = {}

    # Sub-score and Overall Score 
    subscore_names = ["[14]总评", "[1]准确性", "[2]连贯性", "[3]语种一致性", "[4]信息含量", "[5]知识新颖性", "[6]专注度", "[7]创意", 
                      "[8]专业性", "[9]风格一致性", "[10]语法多样性", "[11]结构规范性", "[12]内容原创性", "[13]敏感度"]
    for idx, subscore in enumerate(subscore_names):
        y_true, y_pred = true_scores[idx], pred_scores[idx]
        # Acc
        acc_five = accuracy_score(y_true, y_pred)
        true_two, pred_two = np.where(y_true < 3, 0, 1), np.where(y_pred < 3, 0, 1)
        acc_two = accuracy_score(true_two, pred_two)      
        # F1 scores
        f1_five = f1_score(y_true, y_pred, average='macro')
        f1_two = f1_score(true_two, pred_two, average='macro')
        if idx == 0:
            # Acc of Positive Sample, Negative Sample
            cm = confusion_matrix(true_two, pred_two)
            if cm.shape == (2, 2):
                denominator_0 = cm[0, 0] + cm[0, 1]
                denominator_1 = cm[1, 0] + cm[1, 1]
                # Avoid division by zero by checking if denominator is not zero
                acc_neg = cm[0, 0] / denominator_0 if denominator_0 > 0 else -1000
                acc_pos = cm[1, 1] / denominator_1 if denominator_1 > 0 else -1000
                acc_diff = -1000 if (denominator_0 <= 0 or denominator_1 <= 0) else acc_neg - acc_pos
            else:
                acc_neg = acc_pos = acc_diff = np.nan  # Using np.nan for undefined cases
                print("Confusion matrix is not 2x2. Please check your data.")
            # Miss Rate of Negative Sample 
            prob_neg_false1 = np.sum((y_true < 2) & (y_pred >= 3)) / np.sum(y_true < 3)
            prob_neg_false2 = np.sum((y_true >= 2) & (y_true < 3) & (y_pred > 3)) / np.sum(y_true < 3)
            prob_neg_false3 = np.sum((y_true >= 2) & (y_true < 3) & (y_pred == 3)) / np.sum(y_true < 3)
            # Misjudge Rate of Positive Sample 
            prob_pos_false1 = np.sum((y_true >= 4) & (y_pred < 3)) / np.sum(y_true >= 3)
            prob_pos_false2 = np.sum((y_true >= 3) & (y_true < 4) & (y_pred < 3)) / np.sum(y_true >= 3)
            # Corr
            pearson_corr, _ = pearsonr(y_true, y_pred)
            spearman_corr, _ = spearmanr(y_true, y_pred)
            score_metrics.update({subscore+'Five-class Accuracy': f"{acc_five:.3f}",
                                subscore+'Two-class Accuracy': f"{acc_two:.3f}",
                                subscore+'Positive Sample Accuracy': f"{acc_pos:.4f}",
                                subscore+'Negative Sample Accuracy': f"{acc_neg:.4f}", 
                                subscore+'Negative Sample Miss Rate1': f"{prob_neg_false1:.4f}", 
                                subscore+'Negative Sample Miss Rate2': f"{prob_neg_false2:.4f}", 
                                subscore+'Negative Sample Miss Rate3': f"{prob_neg_false3:.4f}", 
                                subscore+'Positive Sample Misjudge Rate1': f"{prob_pos_false1:.4f}",
                                subscore+'Positive Sample Misjudge Rate2': f"{prob_pos_false2:.4f}",
                                subscore+'Pearson\'s Correlation': f"{pearson_corr:.3f}", 
                                subscore+'Spearman\'s Correlation': f"{spearman_corr:.3f}",
                                subscore+'Five-class F1 Score': f"{f1_five:.3f}",
                                subscore+'Two-class F1 Score': f"{f1_two:.3f}",
                                })
        else:
            # Corr
            pearson_corr, _ = pearsonr(y_true, true_scores[-1])
            spearman_corr, _ = spearmanr(y_true, true_scores[-1])
            score_metrics.update({subscore+'Five-class Accuracy': f"{acc_five:.3f}",
                                subscore+'Two-class Accuracy': f"{acc_two:.3f}",
                                subscore+'Pearson\'s Correlation': f"{pearson_corr:.3f}", 
                                subscore+'Spearman\'s Correlation': f"{spearman_corr:.3f}",
                                subscore+'Five-class F1 Score': f"{f1_five:.3f}",
                                subscore+'Two-class F1 Score': f"{f1_two:.3f}",
                                })

    # Average Accuracy, Average Accuracy Variance
    acc_values_five = [float(value) for key, value in score_metrics.items() if key.endswith("Five-class Accuracy")]
    avg_acc_five, avg_var_five = np.mean(acc_values_five), np.var(acc_values_five)
    score_metrics['Average Five-class Accuracy'], score_metrics['Average Five-class Variance'] = round(avg_acc_five, 3), round(avg_var_five, 3)

    acc_values_two = [float(value) for key, value in score_metrics.items() if key.endswith("Two-class Accuracy")]
    avg_acc_two, avg_var_two = np.mean(acc_values_two), np.var(acc_values_two)
    score_metrics['Average Two-class Accuracy'], score_metrics['Average Two-class Variance'] = round(avg_acc_two, 3), round(avg_var_two, 3)

    # Average F1
    f1_values_five = [float(value) for key, value in score_metrics.items() if key.endswith("Five-class F1 Score")]
    avg_f1_five = np.mean(f1_values_five)
    score_metrics['Avearge Five-class F1 Score'] = round(avg_f1_five, 3)
    
    f1_values_two = [float(value) for key, value in score_metrics.items() if key.endswith("Two-class F1 Score")]
    avg_f1_two = np.mean(f1_values_two)
    score_metrics['Avearge Two-class F1 Score'] = round(avg_f1_two, 3)

    for k,v in score_metrics.items():
        print(f"{k} {v}")

    true_domains = [response.split("\n")[0] for response in true_responses]
    pred_domains = [response.split("\n")[0] for response in pred_responses]
    print("Domain Results: ")
    # Domain Acc and Domain Variance
    domain_metrics = compute_domain_metrics(true_domains, pred_domains)
    for k,v in list(domain_metrics.items()):
        print(f"{k} {v}")
    print(" ")


def report_score_only_result(prompts, true_responses, pred_responses, sources, print_false_case=False, print_each_domain_score=False):
    true_scores = np.array(true_responses).astype(np.float16).T
    pred_scores = np.array(pred_responses).astype(np.float16).T
    print("Score results: ")
    score_metrics = compute_score_metrics(prompts, true_scores, pred_scores, print_false_case=print_false_case)
    for k,v in score_metrics.items():
        print(f"{k} {v}")

    if print_each_domain_score:
        metrics_list = []
        for source in set(sources):
            source_indices = np.where(np.array(sources) == source) 
            source_true_scores, source_pred_scores = true_scores[source_indices],pred_scores[source_indices]
            source_score_metrics = compute_score_metrics(source_true_scores, source_pred_scores)
            metrics_list.append(source_score_metrics)
        sorted_metrics_list = sorted(metrics_list, 
                                    key=lambda x: x['Accuracy Diff between Positive & Negative Samples'], 
                                    reverse=True)
        for source_score_metrics in sorted_metrics_list:
            print(f"{source} Score results: ")
            for k, v in source_score_metrics.items():
                print(f"{k}: {v}")  
    print(" ")


def report_domain_only_result(true_responses, pred_responses):
    print("Domain Results: ")
    domain_metrics = compute_domain_metrics(true_responses, pred_responses)
    for k,v in domain_metrics.items():
        print(f"{k} {v}")
    print(" ")


def main(args):
    fix_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left", use_fast=True)
    # max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(temperature=args.temperature, seed=args.seed, stop_token_ids=[151643, 151645], max_tokens=args.max_tokens)
    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model=args.model_name_or_path, tokenizer=args.model_name_or_path, tensor_parallel_size=torch.cuda.device_count()) 
    sources, messages, responses = load_data(args.input_path)
    texts, pred_responses, true_responses = [], [], []
    for msg in messages:
        text = tokenizer.apply_chat_template(
            msg, 
            tokenize=False,
            add_generation_prompt=True)
        texts.append(text)
    outputs = llm.generate(texts, sampling_params)
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        pred_responses.append(generated_text)
        true_responses.append(responses[i]['content'])

    if args.model_type == "all_rating":
        report_all_rating_result(true_responses, pred_responses)
    elif args.model_type == "score_only":
        report_score_only_result(texts, true_responses, pred_responses, sources, args.print_false_case, args.print_each_domain_score)
    elif args.model_type == "domain_only":
        report_domain_only_result(true_responses, pred_responses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True, help='model file.')
    parser.add_argument('--input_path', type=str, required=True, help='input test file.')
    parser.add_argument('--output_path', type=str, default=None,  help='(optional) output result file.')
    parser.add_argument('--max_tokens', default=512, type=int, help='Maximum number of output tokens')
    parser.add_argument('--seed', default=1024, type=int, help='seed size')
    parser.add_argument('--temperature', default=0.0, type=float, help='generation temperature')
    parser.add_argument('--print_false_case', default=False, action="store_true", help='Printing false case: missing negative and misjuding positive texts')
    parser.add_argument('--print_each_domain_score', default=False, action="store_true", help='View classification imbalances in each domain')
    parser.add_argument('--model_type', default='score_only', type=str, help='use which model to inference', choices=['all_rating', 'score_only', 'domain_only'])
    args = parser.parse_args()
    if args.output_path is None:
        main(args)
    else:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        sys.stdout = open(args.output_path, 'a')
        main(args)
        sys.stdout.close()
        sys.stdout = sys.__stdout__
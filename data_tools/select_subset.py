import os
import gc
import time
import math
import random
import argparse
import numpy as np
import multiprocessing
from math import ceil
from tqdm import tqdm
import numpy.typing as npt
from functools import partial, reduce
from typing import List, Optional, Tuple, Iterable
from datasets import load_from_disk, load_dataset, concatenate_datasets, Dataset


def sharder(dataset_iterator: Iterable[Tuple[Dataset, int]], tokens_per_shard):
    """将数据集进行分片处理，每个分片允许的token数量由tokens_per_shard指定。"""
    tokens_in_shard, current_shard = 0, []                   # 初始化当前shard的token数和shard列表
    for dataset, tokens in dataset_iterator:                 # 遍历selector生成器，返回选择出dataset中的部分行所组成的新datasets对象，以及它的tokens总数
        tokens_in_shard += tokens                            # 累加当前shard的token数
        current_shard.append(dataset)                        # 将dataset加入当前shard
        if tokens_in_shard >= tokens_per_shard:              # 如果当前shard的token数超过指定阈值
            num_rows = sum(len(ds) for ds in current_shard)  # 计算当前shard的总行数

            # Assume that the number of tokens per row is roughly the same
            # 将t的相对位置（从0到tokens_in_shard的比例）转换为num_rows的行索引, 也即将token数的范围比例转换为行的比例，创建一个新的行索引的列表
            row_splits = [round(t / tokens_in_shard * num_rows) for t in range(0, tokens_in_shard, tokens_per_shard)]  
            
            # 遍历两个相邻索引的对，其中row_splits是将token 数的范围比例转换为行的比例，所创建的一个新的行索引的列表
            for a, b in zip(row_splits[:-1], row_splits[1:]):  
                yield datasets_select(current_shard, a, b)  # 返回指定范围内的dataset
                tokens_in_shard -= tokens_per_shard         # 更新剩余token数
            current_shard = [datasets_select(current_shard, row_splits[-1], num_rows)]   # 处理最后残余的tokens_in_shard数据
            tokens_in_shard = round(len(current_shard[0]) / num_rows * tokens_in_shard)  # 更新剩余token数
            gc.collect()  # 垃圾回收
    if len(current_shard) > 0 and tokens_in_shard > 0:  # 如果最后还有数据，则返回剩余数据
        yield concatenate_datasets(current_shard)


def datasets_select(datasets: List[Dataset], start: int, end: int):
    """这段代码遍历多个 dataset 对象，收集符合 [start, end) 索引范围的数据子集。
       它使用 output 列表临时存储满足条件的子集，并更新累加索引 i 和跳过索引 skipped。
       通过 maybe_concatenate() 和 select() 返回合并后的数据子集，只保留索引范围为 [start - skipped, end - skipped) 的数据。"""
    i = 0               # 当前遍历过程中已经处理的行数累计。
    skipped = 0         # skipped: 累计跳过的行数。
    output = []         # 用于存储满足条件的数据子集的列表。

    # start和end源于row_splits，也即将token 数的范围比例转换为行的比例，所创建的一个新的行索引的列表
    for dataset in datasets:   # 遍历当前所添加的所有datasets对象
        # 判定开始索引是否超出当前 dataset 范围
        if start > i + len(dataset):   # 如果 start 大于当前遍历位置 i 加当前 dataset 的长度，说明 start 索引在当前 dataset 之后。
            i += len(dataset)          # 更新 i，累加当前 dataset 的长度，表示我们跳过了这个 dataset。
            skipped += len(dataset)    # 更新 skipped，累加跳过的行数。
            continue                   # 使用 continue 跳过当前循环，继续检查下一个 dataset。
        # 判定结束索引是否在当前 dataset 之前
        elif end < i:  # 如果 end 小于当前遍历位置 i，说明所有需要的数据已经全部在之前的 datasets 中找到。
            break      # 使用 break 退出循环，因为后续的 datasets 全部无需处理。
        # 如果当前 dataset 部分或全部处于 [start, end) 范围内，添加它到 output 列表，并更新 i（累加当前 dataset 的行数）
        output.append(dataset)
        i += len(dataset)
    return concatenate_datasets(output).select(range(start - skipped, end - skipped))  # 对合并后的 Dataset 对象进行选择，从 start - skipped 到 end - skipped 范围的行。


def selector(dataset_paths: List[str],
             selected_rows: Optional[npt.NDArray[np.float32]] = None,
             sequence_lengths: Optional[npt.NDArray[np.int32]] = None,
             seq_len_field: Optional[str] = None,
             json: bool = False) -> Iterable[Tuple[Dataset, int]]:
    """从给定路径加载数据集，并根据选择的行或序列长度进行筛选。"""
    rows_processed = 0  # 定义已处理行的索引
    for path in dataset_paths:  # 遍历数据集路径
        print(f"Loading {path}")  # 打印加载信息
        if json:  # 如果是json格式
            try:
                dataset = load_dataset("json", data_files=[path], split="train")  # 加载json数据集
            except Exception as e:  # 处理加载异常
                print("Encountered exception")
                print(e)
                wait_time = random.random() * 10  # 随机等待时间
                print("Waiting for {} seconds".format(wait_time))
                time.sleep(wait_time)  # 等待后重试加载
                dataset = load_dataset("json", data_files=[path], split="train", download_mode='force_redownload')
        else:
            dataset = load_from_disk(path, keep_in_memory=False)  # 从磁盘加载数据集
        
        num_rows = len(dataset)  # 拿到加载好的数据集的条数
        if selected_rows is not None: # 若要选择的行索引>已处理完的数据行的索引，但<已处理完的数据行的索引+新加载进的数据集的条数
            subset_mask = (selected_rows >= rows_processed) & (selected_rows < rows_processed + num_rows)  # 创建子集掩码
            subset_selected_rows = selected_rows[subset_mask]  # 取出上述成功选择出来的行索引
            subset_selected_rows.sort()  # 按照索引大小，从小到大排序
            dataset = dataset.select(subset_selected_rows - rows_processed)  # 根据该索引列表选择dataset中的部分行，创建一个新datasets对象
        else:
            subset_mask = slice(rows_processed, rows_processed + num_rows)  # 更新掩码
        rows_processed += num_rows  # 更新已处理完的数据行的索引
        if sequence_lengths is not None:
            num_tokens = sequence_lengths[subset_mask].sum()  # 更新新选择datasets中的部分行的tokens总数
        else:
            num_tokens = sum(dataset[seq_len_field])  # 获取token长度以计算总token数
        yield dataset, num_tokens  # 返回dataset及其token数


def percentile_indices(metrics, num_tokens, total_num_tokens, tokens_to_select, margin):
    """计算在给定参数下的百分位索引, 默认会采样"""
    ## 若此处想将metrics全取出来，可注释掉Sorting这两行
    if len(np.unique(metrics)) == 1:                    # 若metrics所有元素都相等,也即提前用score筛选过，可先随机打乱
        print(f"Shuffling...")
        indices = np.random.permutation(len(metrics))
    else:
        print(f"Sorting...")
        indices = np.argsort(metrics)
    
    # (要选择的tokens数 / 当前数据中的tokens数+ 余量比率) * 当前数据条数 = 要在当前数据中选择的行数上限
    upper_limit = ceil(len(metrics) * (tokens_to_select / total_num_tokens + margin)) 
    indices = indices[:upper_limit]                     # 根据上限进行索引切片
    selected_num_tokens = num_tokens[indices]           # 获取选择的token数
    cum_tokens = np.cumsum(selected_num_tokens)         # 计算累积token数
    cutoff = np.argmax(cum_tokens >= tokens_to_select)  # 找到满足条件的cutoff位置

    # 确保cutoff是有效的，若无有效cutoff则处理
    if cum_tokens.size == 0 or (cutoff == 0 and cum_tokens[0] < tokens_to_select):
        cutoff = -1  # 表示未找到有效cutoff

    ## Qurating论文实现：若已选择的数据token数 < 要选择的数据token数，通过margin比例来采样足够数据以处理可变序列长度！
    ## 但只有现有数据总tokens数（total_num_tokens）超过（tokens_to_select），这种方式并不是上采样，不会凭空造出数据来，故会报错
    # if cum_tokens[cutoff] < tokens_to_select:
    #     print(f"Margin insufficient: {cum_tokens[cutoff]}/{tokens_to_select}")
    #     return percentile_indices(metrics, num_tokens, total_num_tokens, tokens_to_select, 2 * margin)  # 递归提高margin
  
    # 我们论文实现：若已选择的数据token数 < 要选择的数据token数，直接开始做上采样！
    # 循环处理以确保累积token数满足选择条件
    while cum_tokens[cutoff] < tokens_to_select:       
        print(f"Margin insufficient: {cum_tokens[cutoff]}/{tokens_to_select}")

        # 1.上采样逻辑，以确保满足tokens_to_select的条件
        # 计算还缺少多少tokens
        tokens_shortfall = tokens_to_select - cum_tokens[cutoff]
        
        # 2.根据原始的num_tokens进行上采样
        # 选择的样本数量 - 根据 shortfall 和原始的 num_tokens 分布
        num_to_sample = max(0, math.ceil(tokens_shortfall / np.mean(num_tokens[indices])))

        # 3. 随机选择需要的样本进行上采样
        upsampled_indices = np.random.choice(indices, size=num_to_sample, replace=True) 
        # 将上采样的token数添加到原来的选择中
        indices = np.concatenate([indices, upsampled_indices])
        
        # 4. 重新计算selected_num_tokens
        selected_num_tokens = num_tokens[indices]
        cum_tokens = np.cumsum(selected_num_tokens)         # 重新计算累积token数
        cutoff = np.argmax(cum_tokens >= tokens_to_select)  # 找到新的cutoff

        # 5. 处理防止 cuttoff 失效的情况
        if cutoff == 0 and cum_tokens[0] < tokens_to_select:
            cutoff = -1  # 表示未找到有效cutoff

    return indices[:cutoff + 1], selected_num_tokens[:cutoff + 1]  # 返回满足条件的index和token数


def domain_percentile_indices(metrics, num_tokens, total_num_tokens, domains, tokens_to_select, margin, strategy):
    """从tokens_to_select中均匀采样每个领域的tokens数，也即获取每个领域要选择索引"""
    unique_domains = np.unique(domains)                      # 获取唯一领域
    indices = []                                             # 存储索引
    selected_num_tokens = []                                 # 存储选中的token数

    for domain in tqdm(unique_domains):                      # 遍历每个领域
        domain_mask = (domains == domain)                    # 创建领域掩码，指示哪些样本（或数据点）属于当前领域
        domain_metrics = metrics[domain_mask]                # 获取当前领域的metrics数组
        domain_num_tokens = num_tokens[domain_mask]          # 获取当前领域的token数组
        total_domain_num_tokens = np.sum(domain_num_tokens)  # 计算当前领域的总token数
        if strategy == "org_domain":                         # 当前领域的要选择token数 = (当前领域的总tokens数 / 所有领域的总tokens数) * 总共要选取的tokens数
            tokens_to_select_in_domain = int(total_domain_num_tokens / total_num_tokens * tokens_to_select)     
        elif strategy == "equi_domain":                     # 计算当前领域的要选择token数 = (总共要选取的tokens数 / 领域数)
            tokens_to_select_in_domain = int(tokens_to_select / len(unique_domains))   
        print("Domain index:", domain, "Domain size:", len(domain_metrics), "Domain tokens:", total_domain_num_tokens, "Select:", tokens_to_select_in_domain)
        
        domain_indices, domain_num_tokens = percentile_indices(
            domain_metrics,
            domain_num_tokens,
            total_domain_num_tokens,
            tokens_to_select_in_domain,
            margin)                                               # 根据metrics选择相应的token
        if total_domain_num_tokens > np.sum(domain_num_tokens):
            print("Only sampling, Domain index:", domain, "Domain size:", len(domain_indices), "Domain tokens:", np.sum(domain_num_tokens), "Select:", tokens_to_select_in_domain)
        else:         
            print("Need upsampling, Domain index:", domain, "Domain size:", len(domain_indices), "Domain tokens:", np.sum(domain_num_tokens), "Select:", tokens_to_select_in_domain)
        indices.append(np.where(domain_mask)[0][domain_indices])  # 将当前领域选择的相对索引domain_indice转换回原数据集的全局索引
        selected_num_tokens.append(domain_num_tokens)             # 添加选中的token数

    return (
        np.concatenate(indices),             # 合并并返回所有领域选择的索引
        np.concatenate(selected_num_tokens)  # 合并并返回所有领域选中的token数
    )

def filter_score_entries(metrics, num_tokens, domains, org_domains, target_scores):
    """ 筛选总分在target_scores内的样本 """
    target_scores = np.array(target_scores)                       # 将target_scores转为NumPy 数组
    score_indices = np.where(np.isin(metrics, target_scores))[0]  # 找到所有总分在target_scores内的索引
    if domains is None:
        return score_indices, metrics[score_indices], num_tokens[score_indices], np.sum(num_tokens[score_indices]), None, None
    else:
        if domains.ndim == 1:  # 如果 domains 是一维数组
            return score_indices, metrics[score_indices], num_tokens[score_indices], np.sum(num_tokens[score_indices]), domains[score_indices], org_domains[score_indices]
        elif domains.ndim == 2:  # 如果 domains 是二维数组
            return score_indices, metrics[score_indices], num_tokens[score_indices], np.sum(num_tokens[score_indices]), domains[:, score_indices], org_domains[:, score_indices]

def filter_domain_entries(global_indices, metrics, num_tokens, domains, org_domains, target_domains):
    """ 筛选总分在target_domains内的样本 """
    target_domains = np.array(target_domains)                            # 将target_domains转为NumPy 数组
    domain_indices = np.where(np.isin(org_domains[0], target_domains))[0]  # 找到所有领域在target_domains内的索引，注意这里要拿到dataset的原始domain数组
    if domains is None:
        return global_indices[domain_indices], metrics[domain_indices], num_tokens[domain_indices], np.sum(num_tokens[domain_indices]), None, None
    else:
        if domains.ndim == 1:  # 如果 domains 是一维数组
            return global_indices[domain_indices], metrics[domain_indices], num_tokens[domain_indices], np.sum(num_tokens[domain_indices]), domains[domain_indices], org_domains[domain_indices]
        elif domains.ndim == 2:  # 如果 domains 是二维数组
            return global_indices[domain_indices], metrics[domain_indices], num_tokens[domain_indices], np.sum(num_tokens[domain_indices]), domains[:, domain_indices], org_domains[:, domain_indices]


# def filter_score_entries(metrics, num_tokens, domains, target_scores):
#     """ 筛选总分在target_scores内的样本 """
#     for target_scores in [1, 2, 3, 4, 5]:
#         target_scores = np.array(target_scores)                       # 将target_scores转为NumPy 数组
#         score_indices = np.where(np.isin(metrics, target_scores))[0]  # 找到所有总分在target_scores内的索引
#         if domains is None:
#             A, B, C, D, E = score_indices, metrics[score_indices], num_tokens[score_indices], np.sum(num_tokens[score_indices]), None
#             print(f"Score {target_scores}")
#             print(f"Example Count: {len(score_indices)}")
#             print(f"Token Count: {np.sum(num_tokens[score_indices])}")
#         else:
#             A, B, C, D, E = score_indices, metrics[score_indices], num_tokens[score_indices], np.sum(num_tokens[score_indices]), domains[score_indices]
#             print(f"Score {target_scores}")
#             print(f"Example Count: {len(score_indices)}")
#             print(f"Token Count: {np.sum(num_tokens[score_indices])}")

def load_attributes_for_dataset(path, seq_len_field, metric_field, domain_field, seed=42):
    """加载单个数据集的属性"""
    try:
        print(f"Loading {path}")
        dataset = load_from_disk(path)
        if metric_field is None:
            metrics = np.ones(len(dataset))                 # 如果没有metrics，返回全1数组
        else:  
            metrics = sum(
                np.array(dataset[field], dtype=np.float32)  # 遍历metric_field，累加各metric值
                for field in metric_field
            )
        if domain_field is not None:                # 如果有领域字段
            domains = []
            org_domains = []
            for field in domain_field:
                if field == "single_label":     # 只对ChineseWebText2.0的'domain'下的'single_label'专门处理
                    org_domains.append(np.array([x['single_label'] for x in dataset['domain']]))
                    domains.append(np.array([hash(x['single_label']) % 2**32 for x in dataset['domain']]))
                else:
                    if "." in field:
                        field, dict_fields = field.split(".")
                        get_field = lambda x: reduce(dict.get, [x, *dict_fields])  # 跳过第一个元素，它是字段名
                    else:
                        get_field = lambda x: x
                    org_domains.append(np.array([get_field(x) for x in dataset[field]]))     # <修改代码>
                    domains.append(np.array([hash(get_field(x)) % 2**32 for x in dataset[field]], dtype=np.uint32)) # 计算领域散列值，为每个字段生成一列
        else:
            domains = None
        seq_len = np.array(dataset[seq_len_field], dtype=np.int32)  # 获取token长度信息
        return metrics, seq_len, domains, org_domains  # 返回metrics、序列token长度和领域
    except Exception as e:
        print("*****"* 10)
        print(f"PROBLEM WITH LOADING ATTRIBUTES FOR PATH '{path}'")
        print("*****" * 10)
        raise e


def load_attributes_for_all_datasets(args):
    """加载多个数据集的属性"""
    with multiprocessing.Pool(args.num_workers) as pool:  
        attributes = list(tqdm(pool.imap(
            partial(
                load_attributes_for_dataset,
                seq_len_field=args.seq_len_field,
                metric_field=args.metric_field,
                domain_field=args.domain_field,
                seed=args.seed),
            args.inputs), total=len(args.inputs)))
    metrics = np.concatenate([m[0] for m in attributes])                                 # 合并metrics
    num_tokens = np.concatenate([m[1] for m in attributes])                              # 合并序列token长度
    domains = np.vstack([np.concatenate([m[2][i] for m in attributes]) 
                         for i in range(len(args.domain_field))]) if args.domain_field else None            # 将所有领域字段的散列值组合成一个多维数组
    org_domains = np.vstack([np.concatenate([m[3][i] for m in attributes]) 
                         for i in range(len(args.domain_field))]) if args.domain_field else None            # 将所有领域字段的散列值组合成一个多维数组

    return metrics, num_tokens, domains, org_domains                                                  # 返回合并后的metrics、序列token长度和领域


def main(args):
    if args.tokens > 0:           # 若要求选择的tokens数大于0
        metrics, num_tokens, domains, org_domains = load_attributes_for_all_datasets(args)   # 加载所有数据集属性

        if args.target_scores:    # 若指定target_score，则基于它过滤数据, 然后更新metrics, num_tokens, total_num_tokens, domains
            global_indices, score_metrics, score_num_tokens, score_total_num_tokens, score_domains, score_org_domains = filter_score_entries(metrics, num_tokens, domains, org_domains, args.target_scores) 
            metrics, num_tokens, total_num_tokens, domains, org_domains = score_metrics, score_num_tokens, score_total_num_tokens, score_domains, score_org_domains
            print(f"Counting tokens...")
            print(f"Score {','.join(map(str, args.target_scores))} has {total_num_tokens} tokens.")
        else:
            print(f"Counting tokens...")
            total_num_tokens = np.sum(num_tokens)
            print(f"{total_num_tokens} tokens")

        if args.target_domains:    # 若指定target_domain，则基于它过滤数据, 然后更新metrics, num_tokens, total_num_tokens, domains
            global_indices, domain_metrics, domain_num_tokens, domain_total_num_tokens, domain_domains, domain_org_domains = filter_domain_entries(global_indices, metrics, num_tokens, domains, org_domains, args.target_domains) 
            metrics, num_tokens, total_num_tokens, domains, org_domains = domain_metrics, domain_num_tokens, domain_total_num_tokens, domain_domains, domain_org_domains
            print(f"Counting tokens...")
            print(f"Domain {','.join(map(str, args.target_domains))} has {total_num_tokens} tokens.")
        else:
            print(f"Counting tokens...")
            total_num_tokens = np.sum(num_tokens)
            print(f"{total_num_tokens} tokens")

        # 因为np.argsort是从小到大, select_bottom=True就是metrics从小到大, select_bottom=False就是-metrics从大到小 
        if args.select_bottom:
            metrics = metrics   # 设置为bottom-k采样
        else:
            metrics = -metrics  # 设置为top-k采样

        if args.strategy == "multi_domain":
            source_domains, application_domains = domains[0, :], domains[1, :]  # 分离source_domain和application_domain

            # 基于source_domain进行初次采样
            unique_domains = np.unique(source_domains)               
            indices = []                                             # 存储索引
            selected_num_tokens = []                                 # 存储选中的token数

            for domain in tqdm(unique_domains):                      # 遍历每个源
                domain_mask = (source_domains == domain)             # 创建源掩码，指示哪些样本（或数据点）属于当前源
                domain_metrics = metrics[domain_mask]                # 获取当前源的metrics数组
                domain_num_tokens = num_tokens[domain_mask]          # 获取当前源的token长度数组
                total_domain_num_tokens = np.sum(domain_num_tokens)  # 计算当前源的总token数
                tokens_to_select_in_domain = int(total_domain_num_tokens / total_num_tokens * args.tokens)     # 当前源的要选择token数 = (当前源的总tokens数 / 所有源的总tokens数) * 总共要选取的tokens数
        
                # 特别注意，要给到基于application_domain进行二次采样的领域数组，用源掩码为true的位置的application_domains
                selected_application_domains = application_domains[domain_mask]

                # 遍历当前源，拿到它的metrics数组、token长度数组、总token数、要从该源选择出多少tokens、margin、采样策略后，基于application_domain进行二次采样
                application_domain_indices, application_domain_num_tokens = domain_percentile_indices(domain_metrics, domain_num_tokens, total_domain_num_tokens, \
                                                                            selected_application_domains, tokens_to_select_in_domain, args.margin, "org_domain")
                indices.append(np.where(domain_mask)[0][application_domain_indices])  # 从当前领域在所有数据中的索引中确定要选择出的样本索引
                selected_num_tokens.append(application_domain_num_tokens)             # 添加选中的token数
            indices, num_tokens  = np.concatenate(indices),  np.concatenate(selected_num_tokens)  # 合并并返回所有领域选择的索引和选中的token数           
        elif args.strategy == "uniform":                        # 进行正态分布采样
            indices, num_tokens = percentile_indices(metrics, num_tokens, total_num_tokens, args.tokens, args.margin)
        elif args.strategy in ["equi_domain", "org_domain"]:  # 进行领域采样: 原始领域比例（更标准的正态分布采样，具体到每个领域内）和均匀领域比例
            indices, num_tokens = domain_percentile_indices(metrics, num_tokens, total_num_tokens, domains, args.tokens, args.margin, args.strategy)
        elif args.strategy == "None":
            if len(np.unique(metrics)) == 1:                    # 若metrics所有元素都相等,也即提前用score筛选过，可先随机打乱
                print(f"Shuffling...")
                indices = np.random.permutation(len(metrics))
            else:
                print(f"Sorting...")
                indices = np.argsort(metrics)               

        is_permutation = np.array_equal(np.sort(indices), np.arange(len(indices)))

        if args.target_scores:    # 这里indices实际上是按target_score筛选后的 metrics、num_tokens 和 domains 中从0开始选择的，因此需要根据 score_indices 进行筛选，以返回最初的全局数据索引。
            indices = global_indices[indices]   
            # # 检查 indices 是否是从 0 到 len(score_indices) - 1 的乱序数组
            # is_permutation = np.array_equal(np.sort(indices), np.arange(len(indices)))

            # # 输出结果
            # print("Filtered Indices:", indices)
            # print("Is a permutation of [0, ..., len(score_indices)-1]:", is_permutation)
        elif args.target_domains:    # 这里indices实际上是按target_score筛选后的 metrics、num_tokens 和 domains 中从0开始选择的，因此需要根据 score_indices 进行筛选，以返回最初的全局数据索引。
            indices = global_indices[indices]  
            # # 检查 indices 是否是从 0 到 len(score_indices) - 1 的乱序数组
            # is_permutation = np.array_equal(np.sort(indices), np.arange(len(indices)))

            # # 输出结果
            # print("Filtered Indices:", indices)
            # print("Is a permutation of [0, ..., len(score_indices)-1]:", is_permutation)                                             
        dataset_generator = selector(args.inputs, indices, num_tokens, json=args.json)              # 创建数据生成器, 生成器创建后不会立马调用，只有for循环后才会调用
    else:
        dataset_generator = selector(args.inputs, seq_len_field=args.seq_len_field, json=args.json) # 创建数据生成器, 生成器创建后不会立马调用，只有for循环后才会调用
    
    for shard, dataset in enumerate(sharder(dataset_generator, args.tokens_per_shard)):             # 遍历生成的shard, 若状态文件不存在, 保存数据集到磁盘
        print(f"Saving shard {shard}")
        if not os.path.exists(args.output + f"/{shard}/state.json"):
            dataset.save_to_disk(args.output + f"/{shard}", num_proc=args.num_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for selecting percentile from a dataset.")
    parser.add_argument("--inputs", type=str, nargs="+", help="Path to the input datasets.")
    parser.add_argument("--output", type=str, help="Path to the output dataset.")
    parser.add_argument("--json", action="store_true", help="Save as json instead of arrow.")
    parser.add_argument("--metric_field", type=str, nargs="+", default=None, help="Field for metric. Leave empty for random selection")
    parser.add_argument("--seq_len_field", type=str, default="input_len", help="Num token field.")
    parser.add_argument("--domain_field", type=str, nargs="+", default=None, help="Domain field for equi-proprotional selection")
    parser.add_argument("--tokens", type=int, default=5_000_000_000, help="Tokens to select")
    parser.add_argument("--tokens_per_shard", type=int, default=500_000_000, help="Tokens per shard")
    parser.add_argument("--margin", type=float, default=0.1, help="Extra proportion for sampling enough data to deal with variable sequence lengths.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random selection. NOTE: seed will also depend on folder name of dataset.")
    parser.add_argument("--num_workers", type=int, default=32, help="Workers for saving.")
    parser.add_argument("--target_scores", type=int, nargs='+', default=None, help="Filter the data based on the target scores.")
    parser.add_argument("--target_domains", type=str, nargs='+', default=None, help="Filter the data based on the target domains.")
    parser.add_argument("--strategy", type=str, choices=['uniform', "equi_domain", "org_domain", "multi_domain", "None"], default='uniform', help="Data selection strategy")
    parser.add_argument("--select_bottom", action="store_true", help="Select bottom scores.")
    args = parser.parse_args() 
    print("Arguments:", args)
    main(args)
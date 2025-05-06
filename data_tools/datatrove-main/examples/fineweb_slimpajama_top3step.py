"""
This file contains the code used to process and create the
FineWeb dataset (https://huggingface.co/datasets/HuggingFaceFW/fineweb)
"""
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter, MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers import JsonlReader, WarcReader, HuggingFaceDatasetReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig
import argparse
import gc

def main(args):
    """
        we first ran the following pipeline for each dump
    """
    DUMP_TO_PROCESS = "out"  # example
    MAIN_OUTPUT_PATH = args.output
    FILTERING_OUTPUT_PATH = f"{MAIN_OUTPUT_PATH}/base_processing"

    main_processing_executor = LocalPipelineExecutor(
        pipeline=[
            HuggingFaceDatasetReader(
            # 指定使用 huggingface Dataset 数据集格式
            dataset="arrow",  
            dataset_options={
                "data_files": args.input,
            },
            streaming=False,   # 根据需要启用流式加载
            limit=-1,          # 设置处理的文档数量限制
            skip=0,            # 设置跳过的文档数量
            batch_size=10000,   # 设置批处理大小
            doc_progress=True,  # 启用文档处理进度条
            text_key="text",    # 指定文本数据的 key
            id_key="id",        # 指定 ID 数据的 key
            default_metadata={"dump": DUMP_TO_PROCESS},  # 设置默认的元数据
            shuffle_files=False,  # 启用文件顺序打乱（根据需要）
            ),
            ## 第一步是过滤掉wrac的html网页在blokclist列表中，slimpajama不需要
            # URLFilter(exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/1_url/{DUMP_TO_PROCESS}")), 
            ## 第二步是从wrac的html网页爬虫，直接转text
            # Trafilatura(favour_precision=True),
            ## 第三步是利用nltk_data的语言包，用FastLanguageDeteor过滤非英语
            # LanguageFilter(
            #     exclusion_writer=JsonlWriter(
            #         f"{FILTERING_OUTPUT_PATH}/2_non_english/",
            #         output_filename="${language}/" + DUMP_TO_PROCESS + "/${rank}.jsonl.gz",
            #         # folder structure: language/dump/file
            #     )
            # ),
            GopherRepetitionFilter(
                exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/3_gopher_rep/{DUMP_TO_PROCESS}")
            ),
            GopherQualityFilter(
                exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/4_gopher_qual/{DUMP_TO_PROCESS}")
            ),
            C4QualityFilter(
                filter_no_terminal_punct=False,
                exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/5_c4/{DUMP_TO_PROCESS}"),
            ),
            FineWebQualityFilter(
                exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/6_fineweb_qual/{DUMP_TO_PROCESS}")
            ),
            JsonlWriter(f"{FILTERING_OUTPUT_PATH}/output/{DUMP_TO_PROCESS}"),
        ],
        tasks=128,
        logging_dir=f"{MAIN_OUTPUT_PATH}/logs/base_processing/{DUMP_TO_PROCESS}",
        randomize_start_duration=180,  # 不同任务的启动延迟，防止瞬时高并发
        workers=128,  # 根据需要设置并行任务数
    )
    main_processing_executor.run()

    """
        we then applied minhash deduplication to each individual dump,
    """

    # you can also change ngrams or the number of buckets and their size here
    minhash_config = MinhashConfig(
        hash_config=HashConfig(
            hash_fc="sha1",  # better precision -> fewer false positives (collisions)
            precision=64,
        ),
        num_buckets=14,
        hashes_per_bucket=8,
        n_grams=5,
    )

    S3_MINHASH_BASE_PATH = f"{MAIN_OUTPUT_PATH}/minhash"

    S3_LOGS_FOLDER = f"{MAIN_OUTPUT_PATH}/logs/minhash"
    LOCAL_LOGS_FOLDER = "logs/minhash"

    TOTAL_TASKS = 128

    # this is the original data that we want to deduplicate
    INPUT_READER = JsonlReader(
        f"{FILTERING_OUTPUT_PATH}/output/{DUMP_TO_PROCESS}"
    )  # this is the output from the first part

    # stage 1 computes minhash signatures for each task (each task gets a set of files)
    stage1 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            MinhashDedupSignature(
                output_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/signatures", config=minhash_config
            ),
        ],
        tasks=TOTAL_TASKS,
        logging_dir=f"{S3_LOGS_FOLDER}/signatures",
        randomize_start_duration=180,
        depends=main_processing_executor,  # only start after the first one completes
        workers=128,  # 根据需要设置并行任务数
    )

    # stage 2 buckets the minhash signatures
    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/signatures",
                output_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/buckets",
                config=MinhashConfig(hash_config=minhash_config.hash_config),
            ),
        ],
        tasks=minhash_config.num_buckets * 2,  # the code supports parallelizing each bucket. here we run 50
        # workers per bucket
        randomize_start_duration=180,
        logging_dir=f"{S3_LOGS_FOLDER}/buckets",
        depends=stage1,
        workers=128,  # 根据需要设置并行任务数
    )

    # stage 3 clusters the deduplicated buckets
    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/buckets",
                output_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/remove_ids",
                config=minhash_config,
            ),
        ],
        tasks=1,  # this step runs on a single task
        logging_dir=f"{S3_LOGS_FOLDER}/clustering",
        depends=stage2,
        workers=-1,  # 根据需要设置并行任务数
    )

    # stage 4 filters out the deduplicated data and formats the final output
    stage4 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            TokensCounter(tokenizer_name_or_path="/mnt/nas/pengru.pr/checkpoints/princeton-nlp/Sheared-LLaMA-1.3B/tokenizer.json"),  # you can remove this one, it's just a nice way to know how many tokens we have
            # before and after dedup
            MinhashDedupFilter(input_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/remove_ids"),
            # run the PII removal
            PIIFormatter(),
            JsonlWriter(f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/deduped_output"),
        ],
        tasks=TOTAL_TASKS,
        logging_dir=f"{S3_LOGS_FOLDER}/filtering",
        depends=stage3,
        workers=128,  # 根据需要设置并行任务数
    )

    # launch dedup pipelines
    stage4.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+", help="Path to the input datasets.")
    parser.add_argument("--output", type=str, help="Path to the output dataset.")
    args = parser.parse_args()
    main(args)
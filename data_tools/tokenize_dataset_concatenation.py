import argparse
from datasets import load_from_disk, load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import time
from itertools import islice
import numpy as np
from tqdm import tqdm
from multiprocessing import Manager

def chunk(iterable, chunk_size, overlap=0):
    it = iter(iterable)
    cache = tuple(islice(it, overlap))
    stride = chunk_size - overlap
    while batch := tuple(islice(it, stride)):
        yield cache + batch
        cache = batch[stride - overlap:]

def chunk_with_position(iterable, chunk_size, overlap=0):
    position = 0
    for seg in chunk(iterable, chunk_size, overlap):
        yield seg, position
        position += len(seg) - overlap

def min_char_split(input_string, split_string, min_char, include_separator):
    if not split_string:
        yield input_string
        return

    start = 0

    while True:
        # Find the next occurrence of the split string
        index = input_string.find(split_string, start + min_char)

        # Break the loop if no more split strings are found
        if index == -1:
            yield input_string[start:]
            return

        # Add the substring to the results
        if include_separator:
            text = input_string[start:index + len(split_string)]
        else:
            text = input_string[start:index]

        if text:
            yield text
        start = index + len(split_string)


class TokenizeMapper:
    def __init__(self, args, tokenizer, previous_chunk):
        self.args = args
        self.tokenizer = tokenizer
        self.previous_chunk = previous_chunk  # 共享的previous_chunk, 用于存储上一块的数据

    def random_chunk(self, seq, index):
        np.random.seed(index + self.args.seed)
        document_position = np.random.randint(0, max(len(seq) - self.args.max_length + 1, 1))
        seq_chunk = seq[document_position:document_position + self.args.max_length]
        yield seq_chunk, document_position

    def __call__(self, examples, indices=None):
        """Tokenizes the text from the input_field in the dataset.
           Ensures that every output chunk is of size max_length (1024 tokens)."""
        if self.args.input_tokens_field in examples:
            ids = examples[self.args.input_tokens_field]
        else:
            texts = examples[self.args.input_field]

            if self.args.min_num_chars_for_split >= 0 and any(len(text) > self.args.min_num_chars_for_split for text in texts):
                ids = []
                for text in texts:
                    if len(text) <= self.args.min_num_chars_for_split:
                        chunks = [text]
                    else:
                        chunks = min_char_split(text, self.args.min_num_chars_split_separator, self.args.min_num_chars_for_split, self.args.min_num_chars_include_separator)
                        
                    ids.append([
                        token
                        for chunk in chunks
                        for token in self.tokenizer(chunk, truncation=False, add_special_tokens=False).input_ids
                    ])
            else:
                ids = self.tokenizer(texts, truncation=False, add_special_tokens=False).input_ids

        # 若example中已有"indice_field"(如"id")或"document_index"，则不用dataset.map返回的当前exmample在dataset的索引
        if (self.args.indices_field in examples) or ("document_index" in examples):
            indices = examples[self.args.indices_field]

        output = {}
        if self.args.remove_column_names:
            keep_columns = [field for field in examples if field not in set(self.args.remove_column_names)]
        else:
            keep_columns = [field for field in examples]
        output.update({field: [] for field in keep_columns})

        # Tokenize和chunking
        if self.args.no_chunk:
            additional_fields = {self.args.tokens_field, self.args.length_field, "document_index"}   # self.args.input_field: dataset原始自带，不需要额外添加; "document_position": 没有分chunk，故不需要
            output.update({field: [] for field in additional_fields})

            for i, seq in enumerate(ids):
                output[self.args.tokens_field].append(seq)
                output[self.args.length_field].append(len(seq))
                output["document_index"].append(indices[i])          
        else:
            additional_fields = {self.args.input_field, self.args.tokens_field, self.args.length_field, "document_position", "document_index"}
            output.update({field: [] for field in additional_fields})

            current_chunk = self.previous_chunk[:]   # 使用共享的previous_chunk的副本作为当前chunk
            current_length = len(current_chunk)      # 当前chunk的大小

            for i, seq in enumerate(ids):
                chunk_generator = self.random_chunk(seq, indices[i]) if self.args.random_chunk else chunk_with_position(seq, self.args.max_length, self.args.overlap)
                for seq_chunk, document_position in chunk_generator:
                    if len(seq_chunk) < self.args.min_length:
                        break

                    while current_length + len(seq_chunk) >= self.args.max_length:
                        space_left = self.args.max_length - current_length
                        current_chunk.extend(seq_chunk[:space_left])        #  Fill the current chunk up to max_length
                        
                        output[self.args.tokens_field].append(list(current_chunk))
                        output[self.args.length_field].append(len(current_chunk))
                        output["document_position"].append(document_position)
                        output["document_index"].append(indices[i])
                        
                        # 更新keep_columns中的内容, keep_column哪怕为空也不会报错
                        for field in keep_columns:              
                            if field == self.args.input_field:        ## 更新example的input_field（"text"），即chunk过后的文本
                                output[self.args.input_field].append(self.tokenizer.decode(current_chunk)) 
                            elif field not in additional_fields:      ## 防止在保留example原始field时，更新"input_ids","length","document_position","document_index"这4项 
                                output[field].append(examples[field][i])

                        current_chunk.clear()  # 清空当前chunk
                        current_length = 0
                        seq_chunk = seq_chunk[space_left:]  # 删除seq_chunk中添加的部分 
                
                    if seq_chunk:  # 将短句拼接进chunk
                        current_chunk.extend(seq_chunk)
                        current_length += len(seq_chunk)

            # 处理任何剩余tokens，准备保留用于下一个batch
            if current_chunk:
                self.previous_chunk[:] = current_chunk  # 更新共享列表的内容
            else:
                self.previous_chunk[:] = []  # 清空共享列表

            # 处理所有输入中最后一个残留chunk，则将其添加到输出中
            if current_chunk and len(current_chunk) == self.args.max_length:
                output[self.args.tokens_field].append(current_chunk)
                output[self.args.length_field].append(len(current_chunk))
                output["document_position"].append(document_position+len(current_chunk))    # document_position（最后残留的chunk的前一个chunk位置）+ len(current_chunk)（最后残留的chunk长度）
                output["document_index"].append(indices[-1])                                # 最后残留的chunk，肯定属于最后一个seq的
                
                # 更新keep_columns中的内容, keep_column哪怕为空也不会报错
                for field in keep_columns:              
                    if field == self.args.input_field:        ## 更新example的input_field（"text"），即chunk过后的文本
                        output[self.args.input_field].append(self.tokenizer.decode(current_chunk)) 
                    elif field not in additional_fields:      ## 防止在保留example原始field时，更新"input_ids","length","document_position","document_index"这4项 
                        output[field].append(examples[field][i])
        return output


def main(args):
    """Main function to perform tokenization."""

    print(f"Loading '{args.input}'...")
    if args.json:
        dataset = concatenate_datasets([
            load_dataset("json", data_files=path, split="train") for path in tqdm(args.input)
        ])
    else:
        dataset = concatenate_datasets([
            load_from_disk(path)
            for path in tqdm(args.input)
        ])
    print(f"Loaded '{args.input}'")

    dataset = dataset.shard(args.shard[0], args.shard[1], contiguous=True)

    required_fields = {args.tokens_field, args.length_field}
    args.remove_column_names = [col for col in args.remove_column_names if col not in required_fields] if args.remove_column_names else None
    start_time = time.time()

    with Manager() as manager:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=args.use_fast)
        previous_chunk = manager.list()  
        tokenized_dataset = dataset.map(
            TokenizeMapper(args, tokenizer, previous_chunk),
            batched=True,
            batch_size=args.batch_size,
            remove_columns=args.remove_column_names,
            num_proc=args.num_workers,
            with_indices=True
        )
        print(f"Finished mapping in {time.time() - start_time:.2f}s")

    print(f"Saving to '{args.output}'...")
    tokenized_dataset.save_to_disk(args.output, num_proc=args.num_workers)
    print(f"Saved to '{args.output}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for tokenizing a dataset.")
    parser.add_argument("--input", type=str, nargs="+", help="Path to the input datasets.")
    parser.add_argument("--output", type=str, help="Path to the output tokenized dataset.")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer.")
    parser.add_argument("--use_fast", action="store_true", help="Use fast tokenizer.")
    parser.add_argument("--input_field", type=str, default="text", help="Field in the dataset to tokenize.")
    parser.add_argument("--input_tokens_field", type=str, default="input_ids", help="Maybe data is already tokenized?")
    parser.add_argument("--indices_field", type=str, default="id", help="Maybe indices is already exist?")
    parser.add_argument("--tokens_field", type=str, default="input_ids", help="Store tokenized data in this field")
    parser.add_argument("--length_field", type=str, default="length", help="Store number of tokens in this field")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum length of tokenized chunks.")
    parser.add_argument("--min_length", type=int, default=128, help="Minimum length of tokenized chunks.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers.")
    parser.add_argument("--remove_column_names", type=str, nargs="+", default=None, help="Columns to remove.")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for the map function.")
    parser.add_argument("--shard", type=int, nargs=2, default=[1, 0], help="Pick a shard of the dataset")
    parser.add_argument("--overlap", type=int, default=1, help="Overlap between contexts.")
    parser.add_argument("--random_chunk", default=False, action="store_true", help="One segment per document.")
    parser.add_argument("--no_chunk", default=False, action="store_true", help="Chunking according to the maximum token length.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--json", action="store_true", help="Input is json dataset.")
    parser.add_argument("--min_num_chars_for_split", type=int, help="Split input string if above this length", default=-1)
    parser.add_argument("--min_num_chars_split_separator", type=str, help="Split separator", default=" ")
    parser.add_argument("--min_num_chars_include_separator", action="store_true", help="Include split separator", default=False)
    args = parser.parse_args()
    main(args)

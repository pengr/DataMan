# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import pickle
import re
import string
import time
from collections import defaultdict
from glob import glob

from lm_dataformat import Reader
import jsonlines
import json

def clean(s):
    # lower cased
    s = s.lower()
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    s = re.sub(r"\s+", " ", s.strip())
    return s


# def get_short_documents(input_dir, threshold, dataset_name):
#     short_documents = defaultdict(set)
    
#     # for file_path in files:
#     file_path = input_dir
#     reader = Reader(file_path)
#     for doc_id, doc in enumerate(reader._stream_data(jsonl_key="text")):
#         if len(clean(doc)) < threshold:
#             short_documents[file_path.replace(input_dir + "/", "")].add(doc_id)

#     return short_documents


def filter_dataset(args):
    start_time = time.time()

    # short_documents = get_short_documents(args.input_dir, args.threshold, args.dataset_name)
    # print("Finished processing, writing to disk!")
    # with open(args.output_file, "wb") as fout:
    #     pickle.dump(short_documents, fout)    

    with jsonlines.open(args.input_dir) as rdr:
        with open(args.filter_output_file, "w") as ff, open(args.keep_output_file, "w") as kf:
            for ob in rdr:
                doc = ob["text"]
                if len(clean(doc)) < args.threshold:
                    ff.write(json.dumps(ob) + "\n")
                else:
                    kf.write(json.dumps(ob) + "\n")

    print(f"Total time: {time.time() - start_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="Dataset input directory.")
    parser.add_argument("--filter_output_file", help="File to output short docs to filter.")
    parser.add_argument("--keep_output_file", help="File to output non-short docs to keep.")
    parser.add_argument("--dataset_name")
    parser.add_argument(
        "--threshold", type=int, help="Minimum length of the document to keep."
    )
    args = parser.parse_args()
    filter_dataset(args)

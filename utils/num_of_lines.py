from argparse import ArgumentParser
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Tuple
from functools import partial
from math import ceil
import os
import re
#python script.num_of_lines.py --input_file clean_tiku_v1/dt-xueku-prof/dt-xueku-prof.jsonl.res --num_of_processes 16 --buffer_size_in_bytes 1000000000
parser = ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--num_of_processes', type=int, default=32)
parser.add_argument('--buffer_size_in_bytes', type=int, default=1000000000)

args = parser.parse_args()


def count_function(seek_size: int, seek_size_end: int):
    if seek_size != 0:
        with open(args.input_file, mode='rb') as f:
            f.seek(seek_size)
            x = f.read(1)
            while x != b'\n' and f.tell() < seek_size_end:
                x = f.read(1)
                continue
            new_seek_size = f.tell()
    else:
        new_seek_size = seek_size

    count = 0
    with open(args.input_file, mode='r', encoding='utf-8') as f:
        f.seek(new_seek_size)
        while f.tell() < seek_size_end:
            text = f.readline()
            count += 1
    return count


def main():
    size_in_bytes = os.stat(args.input_file).st_size
    print(f"File size: {size_in_bytes}")
    count = 0

    with Pool(args.num_of_processes) as pool:
        num_of_buffer_rounds = ceil(size_in_bytes / args.buffer_size_in_bytes)
        for buffer_round in range(0, num_of_buffer_rounds):
            if buffer_round == num_of_buffer_rounds - 1:
                size_in_bytes_group = ceil((size_in_bytes - buffer_round * args.buffer_size_in_bytes) / args.num_of_processes)
            else:
                size_in_bytes_group = ceil(args.buffer_size_in_bytes / args.num_of_processes)

            if buffer_round == 0:
                size_boundaries_in_bytes = [(
                    0,
                    buffer_round * args.buffer_size_in_bytes + size_in_bytes_group
                )]
            else:
                size_boundaries_in_bytes = [(
                buffer_round * args.buffer_size_in_bytes - 1,
                buffer_round * args.buffer_size_in_bytes + size_in_bytes_group
            )]

            for i in range(1, args.num_of_processes - 1):
                size_boundaries_in_bytes.append((
                    buffer_round * args.buffer_size_in_bytes + size_in_bytes_group * i - 1,
                    buffer_round * args.buffer_size_in_bytes + size_in_bytes_group * (i + 1)
            ))

            if args.num_of_processes > 1:
                if buffer_round == num_of_buffer_rounds - 1:
                    size_boundaries_in_bytes.append((
                        buffer_round * args.buffer_size_in_bytes + size_in_bytes_group * (args.num_of_processes - 1) - 1,
                        size_in_bytes
                ))
                else:
                    size_boundaries_in_bytes.append((
                        buffer_round * args.buffer_size_in_bytes + size_in_bytes_group * (args.num_of_processes - 1) - 1,
                        buffer_round * args.buffer_size_in_bytes + size_in_bytes_group * args.num_of_processes
                ))

            print(f'Round {buffer_round} for buffering texts, start at {size_boundaries_in_bytes[0][0]}, end at {size_boundaries_in_bytes[-1][-1]} ...')

            count_groups = pool.starmap(count_function, size_boundaries_in_bytes)
            count += sum(count_groups)

    print(f'Number of lines: {count}')

    return count


if __name__ == '__main__':
    main()
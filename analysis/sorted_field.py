
import json

def sort_jsonl(input_filename, output_filename, sort_key):
    # 读取JSONL文件并解析每一行
    with open(input_filename, 'r', encoding='utf-8') as file:
        lines = [json.loads(line) for line in file]

    # 根据指定的key进行排序
    sorted_lines = sorted(lines, key=lambda x: x[sort_key], reverse=True)

    # 将排好序的结果写回到新的JSONL文件中
    with open(output_filename, 'w', encoding='utf-8') as file:
        for line in sorted_lines:
            json.dump(line, file)
            file.write('\n')

# 文件名
input_filename = '/mnt/nas/pengru.pr/data/princeton-nlp/QuRatedPajama-1B_tokens_for_analysis/data/train-00000-of-00017.annotated.jsonl'

# 排序并输出结果
sort_jsonl(input_filename, 'sorted_by_writing_style.jsonl', 'writing_style_average')
sort_jsonl(input_filename, 'sorted_by_facts_and_trivia.jsonl', 'facts_and_trivia_average')
sort_jsonl(input_filename, 'sorted_by_educational_value.jsonl', 'educational_value_average')
sort_jsonl(input_filename, 'sorted_by_required_expertise.jsonl', 'required_expertise_average')


def sort_jsonl(input_filename, output_filename, fields_to_sum):
    # 读取JSONL文件并解析每一行
    with open(input_filename, 'r', encoding='utf-8') as file:
        lines = [json.loads(line) for line in file]

    # 过滤出所有 'overall_score' 为 5 的数据
    # filtered_lines = [line for line in lines if line.get("overall_score") == 5]

    # 根据多个字段的值累加进行排序
    def sum_fields(item):
        return sum(float(item.get(field, 0)) for field in fields_to_sum)

    sorted_lines = sorted(lines, key=sum_fields, reverse=True)

    # 将排好序的结果写回到新的JSONL文件中
    with open(output_filename, 'w', encoding='utf-8') as file:
        for line in sorted_lines:
            json.dump(line, file)
            file.write('\n')

# 文件名
input_filename = '/mnt/nas/pengru.pr/data/princeton-nlp/QuRatedPajama-1B_tokens_for_analysis/data/train-00000-of-00017.annotated.jsonl'

# 需要累加的字段名
fields_to_sum = [
    "accuracy", "coherence", "language_consistency", "semantic_density", 
    "knowledge_novelty", "topic_focus", "creativity", 
    "professionalism", "style_consistency", "grammatical_diversity", 
    "structural_standardization", "originality", "sensitivity", "overall_score"
]

# 调用排序函数
sort_jsonl(input_filename, 'sorted_by_overall_score.jsonl', fields_to_sum)

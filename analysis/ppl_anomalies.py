########## 分析PPL异常数据
import json

# 请将这个路径替换为你的文件路径
## ZH PPL Anomalies Example
file_path = "/mnt/nas/pengru.pr/data/xxxx/all_zh.csv"  
## EN PPL Anomalies Example
file_path = "/mnt/nas/pengru.pr/data/xxxx/all_en.csv"

# 读取文件并解析每行为JSON
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    data = [json.loads(line) for line in lines]

# 按'id'字段的source分组
grouped_data = {}
for item in data:
    source, lineID = item['id'].rsplit("_", 1)
    if source not in grouped_data:
        grouped_data[source] = []
    grouped_data[source].append(item)

# 对每个组内的数据按ppl从大到小排序，并打印前百分之一的ppl对应的文本
for source in grouped_data:
    # 按照'ppl'字段降序排序
    grouped_data[source].sort(key=lambda x: float(x['ppl']), reverse=True)

    total_items = len(grouped_data[source])
    # 计算前后百分之一的位置
    one_percent_count = max(int(total_items * 0.01), 1)

    # 获取前百分之一的条目
    top_one_percent_entries = grouped_data[source][:one_percent_count]
    print(f"Top 1% PPLs for source {source}:")
    for entry in top_one_percent_entries:
        print(json.dumps({"score": entry['score'], "ppl": entry['ppl'], "id": entry['id'], "text": entry['text']}, ensure_ascii=False))
        # print(f"Score: {entry['score']}, PPL: {entry['ppl']}, Id: {entry['id']}, Text: {entry['text']}")

    # 获取后百分之一的条目
    bottom_one_percent_entries = grouped_data[source][-one_percent_count:]
    print(f"Bottom 1% PPLs for source {source}:")
    for entry in bottom_one_percent_entries:
        print(json.dumps({"score": entry['score'], "ppl": entry['ppl'], "id": entry['id'], "text": entry['text']}, ensure_ascii=False))
        # print(f"Score: {entry['score']}, PPL: {entry['ppl']}, Id: {entry['id']}, Text: {entry['text']}")
    print("\n")  # 添加额外的换行分隔不同的source组
      
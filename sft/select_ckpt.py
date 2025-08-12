import json
import argparse

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Extract steps with lowest eval_loss.')
parser.add_argument('--json_file_path', type=str, required=True, help='Path to the JSON file')
parser.add_argument('--top_k', type=int, default=None, help='Only select k ckpts with the lowest eval loss')

# 获取命令行参数
args = parser.parse_args()
json_file_path = args.json_file_path
top_k = args.top_k

# 加载JSON数据
with open(json_file_path, "r") as json_file:
    data = json.load(json_file)
    log_history = data["log_history"]

# 筛选带有 eval_loss 字段的记录，并按 eval_loss 排序, 挑选前24个ckpt
log_history_with_eval = [entry for entry in log_history if "eval_loss" in entry]
if top_k is not None:
    best_steps = sorted(log_history_with_eval, key=lambda x: x["eval_loss"])[:top_k]
else:
    best_steps = sorted(log_history_with_eval, key=lambda x: x["eval_loss"])

# 提取最佳步骤并打印它们
best_steps_values = [entry["step"] for entry in best_steps]
for step in best_steps_values:
    print(step)
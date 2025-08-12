
import numpy as np

import alpaca_eval.annotators as eval_annotators
import alpaca_eval.utils as ae_utils
import alpaca_eval.main as ae_main
import json
import requests
import copy
import time
import traceback
import os
import sys

__all__ = ["alpaca_leaderboard", "PairwiseAutoAnnotator"]

#! Important:
# The leaderboard is different from teh paper because Davinci003 is depreciated. We now use AlpacaEval1 to
# evaluate the models.AlpacaEval2 is cheaper and has will have more evaluated models, but the baseline is too stong
# => models from AlpacaFarm will have very low scores.
def alpaca_leaderboard(
        *args,
        **kwargs,
):
    return ae_main.evaluate(*args,
                           leaderboard_mode_to_print=["alpaca-farm-ppo-human", "alpaca-7b", "text_davinci_001",
                                                      "gpt35_turbo_instruct", "alpaca-farm-ppo-sim-gpt4-20k"],
                           **kwargs)



class PairwiseAutoAnnotator(eval_annotators.PairwiseAnnotator):
    def __init__(self, *args, input_keys=("input", "instruction"), **kwargs):
        self.user_template = """I want you to create a leaderboard of different of large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

Here is the prompt:
{instruction_dict}

Here are the outputs of the models:
[
{output_1_dict},
{output_2_dict}
]

Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
[
    {{"model": "<model-name>", "rank": "<model-rank>"}},
    {{"model": "<model-name>", "rank": "<model-rank>"}}
]

Your response must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give.

        """
        super().__init__(*args, input_keys=input_keys, **kwargs)

    def __call__(self, to_annotate, **decoding_kwargs):
        df_to_annotate = ae_utils.convert_to_dataframe(to_annotate)
        # merge input and instruction column into one. but only if input is not empty
        merged_col = df_to_annotate["instruction"] + "\n\n" + df_to_annotate["input"]
        df_to_annotate["instruction"] = np.where(df_to_annotate["input"] != "",
                                                merged_col,
                                                df_to_annotate["instruction"])
        # <修改代码>, 因为df_to_annotate["instruction"]为None，反正他也要和input拼接，但又因为没有instruct，干脆直接引入df_to_annotate["prompt"]
        # df_to_annotate["instruction"] = df_to_annotate["prompt"]
        # df_to_annotate["instruction"] = df_to_annotate["input"]

        results = []
        for index, row in df_to_annotate.iterrows():
            print(f"\n\n第{index}个Input的输出质量对比:")
            instruction_dict = json.dumps({"instruction": row["instruction"]}, indent=4)
            output_1_dict = json.dumps({"model": "model_1", "answer": row["output_1"]}, indent=4)
            output_2_dict = json.dumps({"model": "model_2", "answer": row["output_2"]}, indent=4)

            decoding_kwargs={'max_tokens': 100, 
                            'top_p': 1.0, 
                            'temperature': 0}
            kwargs = dict(model="gpt-4o-mini", **decoding_kwargs)
            prompt_batch = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant, that ranks models by the quality of their answers."
                        },
                        {
                            "role": "user",
                            "content": self.user_template.format(instruction_dict=instruction_dict, output_1_dict=output_1_dict, output_2_dict=output_2_dict)
                        },
                    ]
            curr_kwargs = copy.deepcopy(kwargs)
            completion_batch = self.dash_call(messages=prompt_batch, stream=True, **curr_kwargs)
            results.append(completion_batch)

        return results

    def dash_call(self, **kwargs):
        # print("dash_call")
        # print(kwargs)
        os.environ['DASHSCOPE_API_KEY'] = 'sk-996d1049591c486494f073c868e92ad5'
        CALL_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
        HEADERS = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}"
        }

        payload = copy.deepcopy(kwargs)
        assert 'model' in payload

        # 设置 stream 参数为 True
        payload['stream'] = True

        max_try = 10
        for i in range(max_try):
            try:
                with requests.post(CALL_URL, json=payload, headers=HEADERS, timeout=180, stream=True) as response:
                    if response.status_code != 200:
                        raise Exception(f"http status_code: {response.status_code}\n{response.content}")

                    # 缓存流式数据
                    full_content = []

                    # 处理流式数据
                    for line in response.iter_lines(decode_unicode=True):
                        if line.startswith("data: "):  # 检查是否以 "data: " 开头
                            try:
                                # 去掉 "data: " 前缀并解析 JSON
                                data = json.loads(line[6:])
                                if 'choices' in data and len(data['choices']) > 0:
                                    choice = data['choices'][0]
                                    if 'delta' in choice and 'content' in choice['delta']:
                                        # 缓存流式内容
                                        full_content.append(choice['delta']['content'])
                            except json.JSONDecodeError:
                                print(f"Failed to decode JSON from line: {line}")
                    
                    # 打印完整内容
                    result = ''.join(full_content)
                    print(result)
                    sys.stdout.flush()  # 在关键的print语句后添加
                    return result  # 返回完整内容
            except Exception as e:
                print(traceback.format_exc())
                time.sleep(10)
        raise Exception('Max Retry!!!')
import argparse
from alpaca_farm.utils import jload
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator
from alpaca_eval.metrics import pairwise_to_winrate
import json

def main(outputs_A_path, outputs_B_path, annotators_config):
    # 自定义评估代码
    outputs_A = jload(outputs_A_path)
    outputs_B = jload(outputs_B_path)
    
    annotator = PairwiseAutoAnnotator(annotators_config=annotators_config)

    annotated_sft = annotator.annotate_head2head(outputs_1=outputs_A, outputs_2=outputs_B)

    # 注释器更倾向于outputs_B，而非outputs_A的概率
    # for a in annotated_sft:
    #     print(a["preference"])
    
    # result = pairwise_to_winrate(preferences=[a["preference"] for a in annotated_sft])
    # print(annotated_sft)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some inputs for Pairwise Auto Annotation.')
    
    parser.add_argument('--outputs_A', type=str, required=True, 
                        help='Path to the outputs for A.')
    parser.add_argument('--outputs_B', type=str, required=True, 
                        help='Path to the outputs for B.')
    parser.add_argument('--annotators_config', type=str, default="alpaca_eval_gpt4",
                        help='Configuration for the annotators.')

    args = parser.parse_args()
    
    main(args.outputs_A, args.outputs_B, args.annotators_config)

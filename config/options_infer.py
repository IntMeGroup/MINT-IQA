import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--tokenizer', type=str, default='ckpt/bert-base-uncased')
parser.add_argument('--llm', type=str, default='ckpt/vicuna-7b-v1.1')
opts = parser.parse_args()
current_path = os.path.abspath(__file__)
grandfather_path = os.path.abspath(os.path.dirname(os.path.dirname(current_path)) + os.path.sep + ".")
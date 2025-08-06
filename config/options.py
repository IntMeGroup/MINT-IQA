import os
import argparse
import yaml

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# basic settings
parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the config file.')
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--savepath', type=str, default=None, help='')
parser.add_argument('--preload_path', type=str, default='')
parser.add_argument('--rank_pair', default = True, dest='rank_pair', action='store_true')
parser.add_argument('--train_file',type=str, default='/DATA/DATA3/wjr/AIGI_2025/data/train_AIGI_mos2_fold_1.jsonl')
parser.add_argument('--test_file',type=str, default='/DATA/DATA3/wjr/AIGI_2025/data/test_AIGI_mos2_fold_1.jsonl')
parser.add_argument('--imgpath',type=str, default='/DATA/DATA3/wjr/AIGI2025')
# parser.add_argument('--logfile_name', type=str, default='overall_csa', help='Name of the log file.')
parser.add_argument('--logfile_name', type=str, default='AIGI2025mos2')
parser.add_argument('--tokenizer', type=str, default='/DATA/DATA3/wjr/weight/bert-base-uncased')
parser.add_argument('--llm', type=str, default='/DATA/DATA3/wjr/weight/vicuna-7b-v1.1')
# parser.add_argument('--logfile_name', type=str, default='lgvq_mos3_1e-05_fix0.6_bs8r2', help='Name of the log file.')
# parser.add_argument('--logfile_name', type=str, default='train2', help='Name of the log file.')

# training settings
parser.add_argument('--batch_size', type=int, default=8, help='')
parser.add_argument('--accumulation_steps', type=int, default=1, help='')
parser.add_argument('--epochs', type=int, default=500, help='')
parser.add_argument('--train-iters', type=int, default=None,
                    help='total number of iterations to train over all training runs')

# device settings
parser.add_argument('--distributed', default=False, type=bool)
parser.add_argument('--gpu_num', type=int, default=1)
parser.add_argument('--gpu_id', type=str, default='0,1,2,3,4,5,6,7')

# training options
parser.add_argument("--load_emb", dest='load_emb', action='store_true')
parser.add_argument("--load_pair_store", dest='load_pair_store', action='store_true')
parser.add_argument("--fix_base", dest='fix_base', action='store_true')

# param loose/fix settings
parser.add_argument("--fix_rate", type=float, default=0.6)

# Learning rate scheduling.
parser.add_argument('--lr', type=float, default=1e-05,
                    help='initial learning rate')
parser.add_argument('--lr-decay-iters', type=int, default=None,
                    help='number of iterations to decay LR over,'
                        ' If None defaults to `--train-iters`*`--epochs`')
parser.add_argument('--lr-decay-style', type=str, default='cosine',
                    choices=['constant', 'linear', 'cosine', 'exponential', 'inverse_square_root'],
                    help='learning rate decay function')
parser.add_argument('--lr-decay-ratio', type=float, default=0.0)
parser.add_argument('--warmup', type=float, default=0.01,
                    help='percentage of data to warmup on (.01 = 1% of all '
                        'training iters). Default 0.01')
parser.add_argument('--adam-beta1', type=float, default=0.9)
parser.add_argument('--adam-beta2', type=float, default=0.999)
parser.add_argument('--adam-eps', type=float, default=1e-8)

# save options
parser.add_argument('--clear_visualizer', dest='clear_visualizer', action='store_true')
parser.add_argument('--std_log', dest='std_log', action='store_true')
parser.add_argument('--valid_per_epoch', type=int, default=1)

# test settings
parser.add_argument('--test_ckpt', type=str, default=None, help='ckpt absolute path')

opts = parser.parse_args()

# additional parameters
current_path = os.path.abspath(__file__)
grandfather_path = os.path.abspath(os.path.dirname(os.path.dirname(current_path)) + os.path.sep + ".")
with open(os.path.join(grandfather_path, opts.config), 'r') as stream:
    config = yaml.full_load(stream)
import time
import argparse

parser = argparse.ArgumentParser(description='parameters for model.')


#########################
# model parameters
#########################
parser.add_argument('--ghid-size', type=int, default=96, help='RNN hidden size')
parser.add_argument('--glayer', type=int, default=2, help='layer number of RNN')
parser.add_argument('--auxiliary-labels', type=int, default=3, help='dim of auxiliary labels')
parser.add_argument('--label-dim', type=int, default=6, help='dim of final labels')
parser.add_argument('--auxi-weight', type=float, default=0.2, help='loss weight of auxiliary_labels')
parser.add_argument('--final-weight', type=float, default=1, help='loss weight final_labels')
parser.add_argument('--max-margin', type=float, default=0.9, help='max probs of margin')
parser.add_argument('--min-margin', type=float, default=0.1, help='min probs of margin')
parser.add_argument('--loss-alpha', type=float, default=1e-2, help='discount of auxi_margin_loss')
parser.add_argument('--sswe-alpha', type=float, default=0.6, help='trade-off factor by syn_loss of sswe model')


#########################
#  based parameters for training
#########################
parser.add_argument('--n-fold', type=int, default=10, help='number of fold to split data for validate')
parser.add_argument('--batch-size', '-b', type=int, default=64, help='batch size')
parser.add_argument('--embed-dim', type=int, default=100, help='embedding dimension')
parser.add_argument('--lr', type=float, default=5e-3, help='initial learning rate')
parser.add_argument('--clip', type=float, default=0, help='clip learning rate')
parser.add_argument('--epochs', type=int, default=10, help='iter number of epochs for training')
parser.add_argument('--global-step', type=int, default=0, help='global step for batch training')
parser.add_argument('--init-embed', type=str, default='sswe', help='rand|w2v|sswe')
parser.add_argument('--init-sswe-embed', type=str, default='w2v', help='rand|w2v to init SSWE Model.')
parser.add_argument('--attention', type=str, default='pos', help='word|pos|null')
parser.add_argument('--num-workers', type=int, default=4, help='number of workers to load data for training')
parser.add_argument('--log-interval', type=int, default=10, help='report interval')

#########################
# file path based
#########################

parser.add_argument('--data', type=str, default='../docs/data/', help='dataset dirname')
parser.add_argument('--train-file', type=str, default='HML_JD_ALL.new.dat', help='file name of train dataset')
parser.add_argument('--eval-file', type=str, default='test_2000.tsv', help='file name of eval dataset')
parser.add_argument('--embed-path', type=str, default='lookup_alpha0.5_06-03-22:46',help='embedding file from w2v|sswe')
parser.add_argument('--voc-size', type=int, default=23757, help='word vocab size')
parser.add_argument('--pos-size', type=int, default=57, help='pos vocab size')
parser.add_argument('--save_name', type=str, default=time.strftime('%m-%d_%H:%M.pth'), help='name of saved model ')
parser.add_argument('--max-length', type=int, default=494,help='max length of senlltences by word')

#########################
# gpu parameters
#########################
parser.add_argument('--gpu', type=str, default="1", help='use which gpu to run training, optional 0, 1')
parser.add_argument('--seed', type=int, default=10, help='random seed (default: 10)')
parser.add_argument('--cuda', action='store_true',help='use CUDA')

# parser args
params = parser.parse_args()

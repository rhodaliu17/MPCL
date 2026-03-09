import torch
import logging
from medpy import metric
import time
from tqdm import tqdm
import logging
import argparse
import re
import sys
import os
from pathlib import Path

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from networks.vnet_AMC import VNet_AMC

from dataloader.LeftAtrium import LAHeart
from dataloader.pancreas import Pancreas
from dataloader.brats2019 import BraTS2019

from utils.train_util import *
from utils.test_util import test_calculate_metric_LA_AMC, test_calculate_metric_Pancreas_AMC, test_calculate_metric_BraTS_AMC

def get_arguments():

    parser = argparse.ArgumentParser(description='Embracing Intra-Class Heterogeneity for Semi-Supervised Medical Image Segmentation: From Diversity to Precision')

    # Model
    parser.add_argument('--num_classes', type=int, default=2,
                        help='output channel of network')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default='../results')

    # dataset
    parser.add_argument("--data_dir", type=str, default='../../../Datasets/LA_dataset',
                        help="Path to the dataset.")
    parser.add_argument("--list_dir", type=str, default='../datalist/LA',
                        help="Paths to cross-validated datasets, list of test sets and all training sets (including all labeled and unlabeled samples)")
    parser.add_argument("--save_path", type=str, default='../results',
                        help="Path to save.")

    # Optimization options
    parser.add_argument('--lr', type=float,  default=0.001, help='maximum epoch number to train')
    parser.add_argument('--beta1', type=float,  default=0.5, help='params of optimizer Adam')
    parser.add_argument('--beta2', type=float,  default=0.999, help='params of optimizer Adam')
    
    # Miscs
    parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
    parser.add_argument('--seed', type=int, default=1337, help='set the seed of random initialization')
    
    return parser.parse_args()


def create_model(args, ema=False):
    net = nn.DataParallel(VNet_AMC(n_channels=1, n_classes=args.num_classes, n_branches=4))
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

@torch.no_grad()
def test_LA_Pancreas(net, val_loader, args, maxdice=0, print_result=False,save_path =None):
    time_start = time.time()
    if 'LA' in args.data_dir:
        avg_metrics, std_metrics = test_calculate_metric_LA_AMC(net, val_loader.dataset, print_result=print_result,test_save_path=save_path)
    elif 'BraTS' in args.data_dir:
        avg_metrics, std_metrics = test_calculate_metric_BraTS_AMC(net, val_loader.dataset, print_result=print_result,test_save_path=save_path)
    else:
        avg_metrics, std_metrics = test_calculate_metric_Pancreas_AMC(net, val_loader.dataset, print_result=print_result,test_save_path=save_path)
    time_end = time.time()
    val_dice = avg_metrics[0]

    if val_dice > maxdice:
        maxdice = val_dice
        max_flag = True
    else:
        max_flag = False

    logging.info('Evaluation : val_dice: %.4f, val_maxdice: %.4f\n' % (val_dice, maxdice))
    
    logging.info('\nDice:')
    logging.info('Mean :%.2f(%.2f)' % (avg_metrics[0], std_metrics[0]))

    logging.info('\nJaccard:')
    logging.info('Mean :%.2f(%.2f)' % (avg_metrics[1], std_metrics[1]))

    logging.info('\nHD95:')
    logging.info('Mean :%.2f(%.2f)' % (avg_metrics[2], std_metrics[2]))

    logging.info('\nASSD:')
    logging.info('Mean :%.2f(%.2f)' % (avg_metrics[3], std_metrics[3]))

    logging.info('Inference time: %.2f' % ((time_end-time_start)/len(val_loader)))

    return val_dice, maxdice, max_flag



def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred,gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred,gt)
        return np.array([dice, jc, hd95, asd])
    else:
        return np.zeros(4)

def main():
    args = get_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # create logger
    save_path = os.path.join(os.path.dirname(args.load_path),'resultlog')
    os.makedirs(save_path,exist_ok = True)

    # record
    logging.basicConfig(filename=save_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('Save at: {}'.format(save_path))

    set_random_seed(args.seed)

    net = create_model(args)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    load_net_opt(net, optimizer, Path(args.load_path) / 'best.pth')

    if 'LA' in args.data_dir:
        testset = LAHeart(args.data_dir,args.list_dir,split='test')   
        test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
        test_LA_Pancreas(net, test_loader, args, print_result=True, save_path =save_path)
    elif 'BraTS' in args.data_dir:
        testset = BraTS2019(args.data_dir,args.list_dir,split='test') 
        test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
        test_LA_Pancreas(net, test_loader, args, print_result=True,save_path=save_path)
    else:
        testset = Pancreas(args.data_dir,args.list_dir,split='test')   
        test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
        test_LA_Pancreas(net, test_loader, args, print_result=True,save_path=save_path)


if __name__ == '__main__':
    main()
    # pass

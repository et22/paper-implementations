import os
import torch
import wandb
import torchvision.utils as vutils
import torch.nn as nn

from torchvision import transforms
from torch.utils.data import DataLoader

from PIL import Image
from model import ImageTNet
from loss import PerceptualLoss
from dataset import StyleDataset
from multiprocessing import freeze_support
from argparse import ArgumentParser

class EvalOptions:
    """
    EvalOptions defines arguments for eval.py. 
    """
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--test_img_path', type=str, default='./data/', help='Path to test image')
        self.parser.add_argument('--test_img_name', type=str, default='', help='Name of test image')
        self.parser.add_argument('--checkpoint_net_path', type=str, default=None, help='Path to model checkpoint if resuming training')
    def parse(self):
        opts = self.parser.parse_args()
        return opts

def eval(args):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    #### parameters ####
    test_img_path = args.test_img_path
    image_size = 256
    checkpoint_net_path = args.checkpoint_net_path
    save_dir = './results/'
    ####################
    transform = transforms.Compose([
                                transforms.Resize((image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])


    net = ImageTNet().to(device)

    input = transform(Image.open(test_img_path)).to(device)[None, :, :, :]
    
    net.load_state_dict(torch.load(checkpoint_net_path, map_location='cpu'))
    output = net(input.to(device))
    vutils.save_image(input.add(1).mul(0.5), save_dir + args.test_img_name + '_input.jpg', nrow=1)
    vutils.save_image(output.add(1).mul(0.5),  save_dir + args.test_img_name + '_out.jpg', nrow=1)
    

if __name__ == '__main__':
    freeze_support()
    args = EvalOptions().parse()
    eval(args)
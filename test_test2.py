import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from val_data_functions import ValData_map
# from utils import validation, validation_val
import os
import numpy as np
import random
import sys

from models.GUPDM import GUPDM
from utility.metrics_calculation import calculate_UIQM, calculate_metrics_ssim_psnr_all
from PIL import Image, ImageFilter

from torchvision.utils import save_image

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-val_data_dir', help='test image path', default='/root/autodl-tmp/underwater/data/UFO/test/',
                    type=str)
parser.add_argument('-val_filename',
                    help='### The following files should be placed inside the directory "./data/test/"',
                    default='input.txt', type=str)
parser.add_argument('-category', help='output image path', default='v35_UFO', type=str)
parser.add_argument('-pretrain_dir', help='pretrain model path', default='/root/autodl-tmp/underwater/code/hyper_net/weight/v35_LUSI/epoch_30.pth', type=str)

parser.add_argument('-train_batch_size', help='Set the training batch size', default=1, type=int)

parser.add_argument('--scale_factor', type=int, default=4)
# sr_net
parser.add_argument('--input_channels', type=int, default=3, help='the number of input channels for sr net')
parser.add_argument('--channels', type=int, default=64, help='the number of hidden channels for sr net')
parser.add_argument('--residual_lr', type=float, default=1.0, help='the lr coefficient of residual connection')
parser.add_argument('--kernel_size', type=int, default=3, help='the kernel_size of conv')
parser.add_argument('--n_block', type=int, default=9, help='the number of res-block')
parser.add_argument('--n_block1', type=int, default=9, help='the number of res-block')
parser.add_argument('--n_block2', type=int, default=1, help='the number of res-block')
parser.add_argument('--n_conv_each_block', type=int, default=2, help='the number of conv for each res-block')
# condition_net
parser.add_argument('--conv_index', type=str, default='22', help='VGG 22|54')
parser.add_argument('--group', type=int, default=64, help='the number of group conv')
parser.add_argument('--task_size', type=int, default=1)
parser.add_argument('--support_size', type=int, default=1)
parser.add_argument('--use_pretrained_sr_net', type=bool, default=False)

args = parser.parse_args()

val_batch_size = args.val_batch_size
exp_name = args.exp_name
val_data_dir = args.val_data_dir
val_filename = args.val_filename
category = args.category
pretrain_dir = args.pretrain_dir
# set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))


# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# --- Validation data loader --- #


val_data_loader = DataLoader(ValData_map(val_data_dir, val_filename), batch_size=val_batch_size, shuffle=False,
                             num_workers=8)

# --- Define the network --- #

net = GUPDM(args)

net = nn.DataParallel(net, device_ids=device_ids)

# --- Load the network weight --- #
net.load_state_dict(torch.load(pretrain_dir))

print('--- Testing starts! ---')


# --- Use the evaluation model in testing --- #
net.eval()

if os.path.exists('./results/{}/'.format(category)) == False:
    os.makedirs('./results/{}/'.format(category))
output_images_path = './results/{}/'.format(category)

for batch_id, train_data in enumerate(val_data_loader):
    with torch.no_grad():
        input_image, map, gt, imgid = train_data
        # input_image = input_image.cuda()
        # map = map.cuda()
        # gt = gt.cuda()
        im_out = net(input_image, map)

        print(imgid)

        save_image(im_out, './results/{}/{}.png'.format(category, imgid[0][:-4]), normalize=True)
        # my_save_image(imgid, im_out, './results/{}/'.format(category))

SSIM_BGR, PSNR_BGR, MSE = calculate_metrics_ssim_psnr_all(output_images_path, val_data_dir + 'gt/')
sys.stdout.write(
            "\r[SSIM %f] , [PSNR: %f] , [MSE: %f]"
            % (
                float(SSIM_BGR),
                float(PSNR_BGR),
                float(MSE)
            )
        )

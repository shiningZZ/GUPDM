import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from train_data_functions import TrainData_map
from val_data_functions import ValData_map
# from utils import to_psnr, print_log, validation, adjust_learning_rate
from torchvision.models import vgg16
import sys
from utility.metrics_calculation import calculate_UIQM, calculate_metrics_ssim_psnr_all
import torchvision

from perceptual import LossNetwork
import os
import numpy as np
import random
import pytorch_ssim
# from mymodel import UNet
from models.my_CMDSR_v33 import CMDSR

# from transweather_model_v0 import Transweather

plt.switch_backend('agg')

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-learning_rate2', help='Set the learning rate', default=1e-6, type=float)
parser.add_argument('-learning_rate3', help='Set the learning rate', default=1e-6, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=1, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-num_epochs', help='number of epochs', default=200, type=int)
parser.add_argument('-category', help='output image path', default='v33_8811', type=str)
parser.add_argument('-weight_out', help='output weight path', default='v33_8811', type=str)
parser.add_argument('-train_data_dir', help='train image path', default='/root/autodl-tmp/underwater/data/LUSI2/train/',
                    type=str)
parser.add_argument('-val_data_dir', help='test image path', default='/root/autodl-tmp/underwater/data/LUSI2/test/',
                    type=str)
parser.add_argument('-labeled_name', help='The following file should be placed inside the directory "./data/train/',
                    default='input.txt', type=str)
parser.add_argument('-val_filename1',
                    help='### The following files should be placed inside the directory "./data/test/"',
                    default='input.txt', type=str)

parser.add_argument('--scale_factor', type=int, default=4)
# sr_net
parser.add_argument('--input_channels', type=int, default=3, help='the number of input channels for sr net')
parser.add_argument('--channels', type=int, default=64, help='the number of hidden channels for sr net')
parser.add_argument('--residual_lr', type=float, default=1.0, help='the lr coefficient of residual connection')
parser.add_argument('--kernel_size', type=int, default=3, help='the kernel_size of conv')
parser.add_argument('--n_block', type=int, default=9, help='the number of res-block')
parser.add_argument('--n_block1', type=int, default=9, help='the number of res-block')
parser.add_argument('--n_block2', type=int, default=4, help='the number of res-block')
parser.add_argument('--n_conv_each_block', type=int, default=2, help='the number of conv for each res-block')
# condition_net
parser.add_argument('--conv_index', type=str, default='22', help='VGG 22|54')
parser.add_argument('--group', type=int, default=64, help='the number of group conv')
parser.add_argument('--task_size', type=int, default=1)
parser.add_argument('--support_size', type=int, default=1)
parser.add_argument('--use_pretrained_sr_net', type=bool, default=False)
parser.add_argument('--lr_gamma', type=float, default=0.1)

parser.add_argument('--milestones', nargs='+', type=int, default=[5000000, 9000000])
parser.add_argument('--lr_gamma_condition', type=float, default=0.1)

args = parser.parse_args()

learning_rate = args.learning_rate
learning_rate2 = args.learning_rate2
learning_rate3 = args.learning_rate3
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs
category = args.category
weight_out = args.weight_out
train_data_dir = args.train_data_dir
val_data_dir = args.val_data_dir
labeled_name = args.labeled_name
val_filename1 = args.val_filename1

# set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for training ---')
print(
    'learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nlambda_loss: {}\nweight_out:{}'.format(learning_rate,
                                                                                                         crop_size,
                                                                                                         train_batch_size,
                                                                                                         val_batch_size,
                                                                                                         lambda_loss,
                                                                                                         weight_out))

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
net = CMDSR(args)

# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                                       gamma=args.lr_gamma)

conditionnet_mix_optimizer1 = torch.optim.Adam(net.condition_net1.parameters(), lr=learning_rate2)
# conditionnet_mix_scheduler = torch.optim.lr_scheduler.MultiStepLR(conditionnet_mix_optimizer)
conditionnet_mix_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(conditionnet_mix_optimizer1,
                                                                  milestones=args.milestones,
                                                                  gamma=args.lr_gamma_condition)


conditionnet_mix_optimizer2 = torch.optim.Adam(net.condition_net2.parameters(), lr=learning_rate3)
# conditionnet_mix_scheduler = torch.optim.lr_scheduler.MultiStepLR(conditionnet_mix_optimizer)
conditionnet_mix_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(conditionnet_mix_optimizer2,
                                                                  milestones=args.milestones,
                                                                  gamma=args.lr_gamma_condition)
# --- Multi-GPU --- #
net = net.to(device)
# net = nn.DataParallel(net, device_ids=device_ids)

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

# --- Load the network weight --- #
if os.path.exists('./{}/'.format(exp_name)) == False:
    os.mkdir('./{}/'.format(exp_name))
try:
    net.load_state_dict(torch.load('/root/autodl-tmp/underwater/code/hyper_net/weight/v33_2/epoch_10.pth'))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')

# pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
# print("Total_params: {}".format(pytorch_total_params))
loss_network = LossNetwork(vgg_model)
loss_network.eval()

# 均方误差
MSEloss = nn.MSELoss(size_average=False)
MSEloss.to(device)

loss_reconstruct = torch.nn.MSELoss().cuda()

# ssim loss
loss_ssim = pytorch_ssim.SSIM().to(device)

lambda_reconstruct = 0.1
lambda_perceptual = 0.04
lambda_ssim = 0.02
lambda_sa = 0.0000001
# --- Load training data and validation/test data--- #


lbl_train_data_loader = DataLoader(TrainData_map(crop_size, train_data_dir, labeled_name),
                                   batch_size=train_batch_size,
                                   shuffle=True, num_workers=8)

val_data_loader = DataLoader(ValData_map(val_data_dir, val_filename1), batch_size=val_batch_size, shuffle=False,
                             num_workers=8)

max_PSNR = 0


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    learning_rate = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


if os.path.exists('./weight/{}/'.format(weight_out)) == False:
    os.makedirs('./weight/{}/'.format(weight_out))


def checkpoint(epoch):
    model_out_path = "weight/" + weight_out + "/epoch_{}.pth".format(epoch)
    torch.save(net.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def eval(val_data_loader, test_model):
    if os.path.exists('./results/{}/'.format(category)) == False:
        os.makedirs('./results/{}/'.format(category))
    output_images_path = './results/{}/'.format(category)
    test_model.eval()
    for batch_id, train_data in enumerate(val_data_loader):
        with torch.no_grad():
            input_image, map, gt, imgid = train_data
            input_image = input_image.cuda()
            map = map.cuda()
            gt = gt.cuda()
            # net.cuda()
            # print(input_image.device)
            im_out = net(input_image, map)

            save_image(im_out, './results/{}/{}.png'.format(category, imgid[0][:-4]), normalize=True)
            # my_save_image(imgid, im_out, './results/{}/'.format(category))
    SSIM_BGR, PSNR_BGR, MSE = calculate_metrics_ssim_psnr_all(output_images_path, val_data_dir + 'gt/')
    return SSIM_BGR, PSNR_BGR, MSE


def transform_rgb(input_image, map):
    ret = input_image.repeat(3, 1, 1, 1)
    for i in range(ret.shape[0]):
        random_b, random_g, random_r = torch.rand(3) * 0.3 + 0.3
        img_b = torch.ones(1, 3, 256, 256).to(device)
        img_b[:, 0, :, :] = img_b[:, 0, :, :] * random_b  # 降低偏绿
        img_b[:, 1, :, :] = img_b[:, 1, :, :] * random_g  # 降低偏蓝  #提高偏绿    b和g一起降低 变蓝
        img_b[:, 2, :, :] = img_b[:, 2, :, :] * random_r  # 降低偏绿

        ret[i, :, :, :] = map * input_image + (1 - map) * img_b
        # ret = torch.clamp(ret, 0, 255)
    return ret

def transform_map(input_image, map):
    ret = input_image.repeat(3, 1, 1, 1)
    for i in range(ret.shape[0]):
        # random_t= torch.rand(256) * 0.6 + 0.5
        random_t = torch.rand(1) * 0.6 + 0.5
        map = map * random_t.to(device)  # 降低偏绿

        random_b, random_g, random_r = torch.rand(3) * 0.3 + 0.3
        img_b = torch.ones(1, 3, 256, 256).to(device)
        img_b[:, 0, :, :] = img_b[:, 0, :, :] * random_b  # 降低偏绿
        img_b[:, 1, :, :] = img_b[:, 1, :, :] * random_g  # 降低偏蓝  #提高偏绿    b和g一起降低 变蓝
        img_b[:, 2, :, :] = img_b[:, 2, :, :] * random_r  # 降低偏绿


        ret[i, :, :, :] = map * input_image + (1 - map) * img_b
        # ret = torch.clamp(ret, 0, 255)
    return ret

condition_frequence = 10
condition_frequence2 = 11
# val=False
net.train()
for epoch in range(epoch_start, num_epochs):

    start_time = time.time()
    adjust_learning_rate(optimizer, epoch)
    for batch_id, train_data in enumerate(lbl_train_data_loader):

        input_image, map, gt, imgid = train_data
        input_image = input_image.to(device)

        gt = gt.to(device)
        map = map.to(device)

        # start main network training
        optimizer.zero_grad()

        net.sr_net.train()
        net.sr_net.requires_grad_(True)
        pred_image = net(input_image, map)

        smooth_loss = F.smooth_l1_loss(pred_image, gt)
        perceptual_loss = loss_network(pred_image, gt)
        ssim_loss = - loss_ssim(pred_image, gt)

        loss = smooth_loss + lambda_loss * perceptual_loss + lambda_ssim * ssim_loss

        loss.backward()
        optimizer.step()

        # start condition net training
        if batch_id % condition_frequence == 0:
            conditionnet_mix_optimizer1.zero_grad()
            net.sr_net.eval()
            # net.reconstruct.eval()
            net.sr_net.requires_grad_(False)
            input = transform_rgb(input_image, map)
            pred_image = net(input , map)

            gt = gt.repeat(3, 1, 1, 1)
            # 算condition net的loss

            smooth_loss = F.smooth_l1_loss(pred_image, gt)
            perceptual_loss = loss_network(pred_image, gt)
            ssim_loss = - loss_ssim(pred_image, gt)

            # reconstruct_loss = loss_reconstruct(pred_image, gt)
            condition_loss = smooth_loss + lambda_loss * perceptual_loss + lambda_ssim * ssim_loss

            # condition_loss = loss * lambda_reconstruct

            condition_loss.backward()
            conditionnet_mix_optimizer1.step()
            conditionnet_mix_scheduler1.step()


        if batch_id % condition_frequence2 == 0:
            conditionnet_mix_optimizer2.zero_grad()
            net.sr_net.eval()
            # net.reconstruct.eval()
            net.sr_net.requires_grad_(False)
            input = transform_map(input_image, map)
            pred_image = net(input, map)

            if gt.shape[0]!=3:
                gt = gt.repeat(3, 1, 1, 1)
            # 算condition net的loss


            smooth_loss = F.smooth_l1_loss(pred_image, gt)
            perceptual_loss = loss_network(pred_image, gt)
            ssim_loss = - loss_ssim(pred_image, gt)

            # reconstruct_loss = loss_reconstruct(pred_image, gt)
            condition_loss = smooth_loss + lambda_loss * perceptual_loss + lambda_ssim * ssim_loss

            # condition_loss = loss * lambda_reconstruct

            condition_loss.backward()
            conditionnet_mix_optimizer2.step()
            conditionnet_mix_scheduler2.step()
            # torch.cuda.empty_cache()

        if not (batch_id % 10):
            sys.stdout.write(
                "\r[Epoch %d/%d] , [batch %d],[smooth_loss: %f],[perceptual_loss: %f],[ssim_loss : %f],[total_loss :%f]"
                % (
                    epoch,
                    num_epochs,
                    batch_id,
                    smooth_loss,
                    perceptual_loss * lambda_loss,
                    ssim_loss * lambda_ssim,
                    loss
                )
            )
            # print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))

    # --- Save the network parameters --- #
    torch.save(net.state_dict(), './{}/latest'.format(exp_name))
    net.eval()

    one_epoch_time = time.time() - start_time
    # if epoch % 10 == 0:
    #     checkpoint(epoch)
    # print(
    #     'Epoch:['+str(epoch + 1)+'/'+str(num_epochs)+']  one_epoch_time: '+ str(one_epoch_time)+'loss: '+str(loss))

    if epoch % 10 == 0:
        SSIM_BGR, PSNR_BGR, MSE = eval(val_data_loader, net)
        sys.stdout.write(
            "\r[Epoch %d/%d] , [SSIM %f] , [PSNR: %f] , [MSE: %f] , [ one_epoch_time: %f]"
            % (
                epoch,
                num_epochs,
                float(SSIM_BGR),
                float(PSNR_BGR),
                float(MSE),
                float(one_epoch_time)
            )
        )
        if float(PSNR_BGR) > max_PSNR:
            checkpoint(epoch)
            max_PSNR = float(PSNR_BGR)
        print()

import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import re
from PIL import ImageFile
from os import path
import numpy as np
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
# from utility.ColorCorrection_util111 import color_correction

# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir,train_filename):
        super().__init__()
        train_list = train_data_dir +train_filename
        with open(train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('input','gt') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]

        img_id = re.split('/',input_name)[-1][:-4]

        input_img = Image.open(self.train_data_dir +'input/'+ input_name)

        #########################################
        try:
            gt_img = Image.open(self.train_data_dir + 'gt/'+gt_name)

        except:
            gt_img = Image.open(self.train_data_dir + 'gt/'+ gt_name).convert('RGB')

        width, height = input_img.size

        if width < crop_width and height < crop_height :
            input_img = input_img.resize((crop_width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width :
            input_img = input_img.resize((crop_width,height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width,height), Image.ANTIALIAS)
        elif height < crop_height :
            input_img = input_img.resize((width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)

        width, height = input_img.size

        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)

        # --- Check the channel is 3 or not --- #
        if list(input_im.shape)[0] != 3 or list(gt.shape)[0] != 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return input_im, gt, img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)




# --- Training dataset --- #
class TrainData_sa(data.Dataset):
    def __init__(self, crop_size, train_data_dir,train_filename):
        super().__init__()
        train_list = train_data_dir +train_filename
        with open(train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('input','gt') for i in input_names]
            sa_names = [i.strip().replace('input','input_sa') for i in input_names]
        self.input_names = input_names
        self.gt_names = gt_names
        self.sa_names=sa_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        sa_name=self.sa_names[index]

        img_id = re.split('/',input_name)[-1][:-4]

        input_img = Image.open(self.train_data_dir +'input/'+ input_name)

        #########################################
        try:
            gt_img = Image.open(self.train_data_dir + 'gt/'+gt_name)
            sa_img = Image.open(self.train_data_dir + 'input_sa/' + sa_name)
        except:
            gt_img = Image.open(self.train_data_dir + 'gt/'+ gt_name).convert('RGB')
            sa_img = Image.open(self.train_data_dir + 'input_sa/' + sa_name[:-5]+'.jpg').convert('RGB')

        width, height = input_img.size

        if width < crop_width and height < crop_height :
            input_img = input_img.resize((crop_width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
            sa_img = sa_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width :
            input_img = input_img.resize((crop_width,height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width,height), Image.ANTIALIAS)
            sa_img = sa_img.resize((crop_width, height), Image.ANTIALIAS)
        elif height < crop_height :
            input_img = input_img.resize((width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)
            sa_img = sa_img.resize((width, crop_height), Image.ANTIALIAS)

        width, height = input_img.size

        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        sa_crop_img = sa_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        transform_sa1 = Compose([ToTensor()])
        transform_sa2 = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)
        sa_im = transform_sa1(sa_crop_img)
        # print(sa_im.shape)
        if sa_im.shape[0] != 3:
            sa_im = sa_im.repeat(3, 1, 1)
            # print("sa img shape", sa_im.shape)
        sa_im = transform_sa2(sa_im)
        # --- Check the channel is 3 or not --- #
        if list(input_im.shape)[0] != 3 or list(gt.shape)[0] != 3 or  list(sa_im.shape)[0] != 3:
            raise Exception('Bad image channel: {}'.format(gt_name))


        return input_im, sa_im,gt, img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)


# --- Training dataset --- #
class TrainData_map(data.Dataset):
    def __init__(self, crop_size, train_data_dir,train_filename):
        super().__init__()
        train_list = train_data_dir +train_filename
        with open(train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('input','gt') for i in input_names]
            map_names = [i.strip().replace('input', 'input_map') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.map_names = map_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        map_name = self.map_names[index]

        img_id = re.split('/',input_name)[-1][:-4]

        input_img = Image.open(self.train_data_dir +'input/'+ input_name)

        #########################################
        try:
            gt_img = Image.open(self.train_data_dir + 'gt/'+gt_name)
            map_img = Image.open(self.train_data_dir + 'input_map/' + map_name)
        except:
            gt_img = Image.open(self.train_data_dir + 'gt/'+ gt_name).convert('RGB')
            map_img = Image.open(self.train_data_dir + 'input_map/' + map_name).convert('RGB')

        width, height = input_img.size

        if width < crop_width and height < crop_height :
            input_img = input_img.resize((crop_width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
            map_img = map_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width :
            input_img = input_img.resize((crop_width,height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width,height), Image.ANTIALIAS)
            map_img = map_img.resize((crop_width, height), Image.ANTIALIAS)
        elif height < crop_height :
            input_img = input_img.resize((width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)
            map_img = map_img.resize((width, crop_height), Image.ANTIALIAS)

        width, height = input_img.size

        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        map_crop_img = map_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        transform_map = Compose([ToTensor()])
        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)
        map = transform_map(map_crop_img)

        # # --- Check the channel is 3 or not --- #
        # if list(input_im.shape)[0] != 3 or list(gt.shape)[0] != 3 or list(map.shape)[0] != 3:
        #     raise Exception('Bad image channel: {}'.format(map_name))
        if map.shape[0] != 3:
            map = map.repeat(3, 1, 1)

        return input_im,map, gt, img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)


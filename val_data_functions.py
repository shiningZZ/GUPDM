import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np


# --- Validation/test dataset --- #
class ValData(data.Dataset):
    def __init__(self, val_data_dir, val_filename):
        super().__init__()
        val_list = val_data_dir + val_filename
        with open(val_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('input', 'gt') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        input_name = self.input_names[index]
        # 改过
        gt_name = self.gt_names[index]
        # gt_name = self.gt_names[index][:-4]+'.png'
        # print(gt_name)
        input_img = Image.open(self.val_data_dir + 'input/' + input_name).convert('RGB')
        gt_img = Image.open(self.val_data_dir + 'gt/' + gt_name).convert('RGB')

        # Resizing image in the multiple of 16"
        wd_new, ht_new = input_img.size
        if ht_new > wd_new and ht_new > 1024:
            wd_new = int(np.ceil(wd_new * 1024 / ht_new))
            ht_new = 1024
        elif ht_new <= wd_new and wd_new > 1024:
            ht_new = int(np.ceil(ht_new * 1024 / wd_new))
            wd_new = 1024
        wd_new = int(16 * np.ceil(wd_new / 16.0))
        ht_new = int(16 * np.ceil(ht_new / 16.0))
        # wd_new = int(32 * np.ceil(wd_new / 32.0))
        # ht_new = int(32 * np.ceil(ht_new / 32.0))
        # input_img = input_img.resize((wd_new, ht_new), Image.ANTIALIAS)
        # gt_img = gt_img.resize((wd_new, ht_new), Image.ANTIALIAS)
        input_img = input_img.resize((256, 256), Image.ANTIALIAS)
        gt_img = gt_img.resize((256, 256), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_img)
        gt = transform_gt(gt_img)

        return input_im, gt, input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)


# --- Validation/test dataset --- #
class ValData_map(data.Dataset):
    def __init__(self, val_data_dir, val_filename):
        super().__init__()
        val_list = val_data_dir + val_filename
        with open(val_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('input', 'gt') for i in input_names]
            map_names = [i.strip().replace('input', 'input_map') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.map_names = map_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        input_name = self.input_names[index]
        # 改过
        gt_name = self.gt_names[index]
        # gt_name = self.gt_names[index][:-4]+'.png'
        # print(gt_name)
        map_name = self.map_names[index]
        input_img = Image.open(self.val_data_dir + 'input/' + input_name).convert('RGB')
        gt_img = Image.open(self.val_data_dir + 'gt/' + gt_name).convert('RGB')
        map_img = Image.open(self.val_data_dir + 'input_map/' + map_name).convert('RGB')
        # Resizing image in the multiple of 16"
        wd_new, ht_new = input_img.size
        if ht_new > wd_new and ht_new > 1024:
            wd_new = int(np.ceil(wd_new * 1024 / ht_new))
            ht_new = 1024
        elif ht_new <= wd_new and wd_new > 1024:
            ht_new = int(np.ceil(ht_new * 1024 / wd_new))
            wd_new = 1024
        wd_new = int(16 * np.ceil(wd_new / 16.0))
        ht_new = int(16 * np.ceil(ht_new / 16.0))
        # wd_new = int(32 * np.ceil(wd_new / 32.0))
        # ht_new = int(32 * np.ceil(ht_new / 32.0))
        # input_img = input_img.resize((wd_new, ht_new), Image.ANTIALIAS)
        # gt_img = gt_img.resize((wd_new, ht_new), Image.ANTIALIAS)

        input_img = input_img.resize((256, 256), Image.ANTIALIAS)
        gt_img = gt_img.resize((256, 256), Image.ANTIALIAS)
        map_img = map_img.resize((256, 256), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        transform_map = Compose([ToTensor()])
        input_im = transform_input(input_img)
        gt = transform_gt(gt_img)
        map = transform_map(map_img)

        if map.shape[0] != 3:
            map = map.repeat(3, 1, 1)

        return input_im, map, gt, input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)




# --- Validation/test dataset --- #
class ValData_map_un(data.Dataset):
    def __init__(self, val_data_dir, val_filename):
        super().__init__()
        val_list = val_data_dir + val_filename
        with open(val_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('input', 'gt') for i in input_names]
            map_names = [i.strip().replace('input', 'input_map') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.map_names = map_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        input_name = self.input_names[index]
        # 改过
        gt_name = self.gt_names[index]
        # gt_name = self.gt_names[index][:-4]+'.png'
        # print(gt_name)
        map_name = self.map_names[index]
        input_img = Image.open(self.val_data_dir + 'input/' + input_name).convert('RGB')
        gt_img = Image.open(self.val_data_dir + 'input/' + gt_name).convert('RGB')
        map_img = Image.open(self.val_data_dir + 'input_map/' + map_name).convert('RGB')
        # Resizing image in the multiple of 16"
        wd_new, ht_new = input_img.size
        if ht_new > wd_new and ht_new > 1024:
            wd_new = int(np.ceil(wd_new * 1024 / ht_new))
            ht_new = 1024
        elif ht_new <= wd_new and wd_new > 1024:
            ht_new = int(np.ceil(ht_new * 1024 / wd_new))
            wd_new = 1024
        wd_new = int(16 * np.ceil(wd_new / 16.0))
        ht_new = int(16 * np.ceil(ht_new / 16.0))
        # wd_new = int(32 * np.ceil(wd_new / 32.0))
        # ht_new = int(32 * np.ceil(ht_new / 32.0))
        # input_img = input_img.resize((wd_new, ht_new), Image.ANTIALIAS)
        # gt_img = gt_img.resize((wd_new, ht_new), Image.ANTIALIAS)
        input_img = input_img.resize((1024, 1024), Image.ANTIALIAS)
        gt_img = gt_img.resize((1024, 1024), Image.ANTIALIAS)
        map_img = map_img.resize((1024, 1024), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        transform_map = Compose([ToTensor()])
        input_im = transform_input(input_img)
        gt = transform_gt(gt_img)
        map = transform_map(map_img)

        if map.shape[0] != 3:
            map = map.repeat(3, 1, 1)

        return input_im, map, gt, input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
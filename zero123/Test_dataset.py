import os
import json
import random
import math
import torch
import torchvision.transforms as transforms
import webdataset as wds
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from einops import rearrange
from pathlib import Path
import matplotlib.pyplot as plt

def custom_transform(x):
    return rearrange(x * 2. - 1., 'c h w -> h w c')

def custom_collate(batch, transform):
    """
    Custom collate function to handle tuples in batch samples and apply image transformations.
    """
    batch_out = {'png': [], 'npy': []}
    for sample in batch:
        if len(sample) == 2:
            batch_out['png'].append(transform(sample[0]))
            batch_out['npy'].append(torch.tensor(sample[1]))
    batch_out['png'] = torch.stack(batch_out['png'])
    batch_out['npy'] = torch.stack(batch_out['npy'])
    return batch_out

class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        if train is not None:
            dataset_config = train
        elif validation is not None:
            dataset_config = validation
        else:
            dataset_config = test

        if 'image_transforms' in dataset_config:
            image_transforms = [transforms.Resize(dataset_config['image_transforms']['size'])]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(), custom_transform])
        self.image_transforms = transforms.Compose(image_transforms)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self.create_dataset(validation=False)
            self.val_dataset = self.create_dataset(validation=True)
        if stage == 'test' or stage == 'predict':
            self.test_dataset = self.create_dataset(validation=self.validation)

    def create_dataset(self, validation=False):
        tar_path = self.root_dir
        dataset = (wds.WebDataset(tar_path)
                   .shuffle(1000)
                   .decode("pil")
                   .select(lambda x: "png" in x and "npy" in x)  # Filter to include only valid samples
                   .to_tuple("png", "npy"))
        return dataset

    def train_dataloader(self):
        dataset = self.train_dataset
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=lambda batch: custom_collate(batch, self.image_transforms))
        return loader

    def val_dataloader(self):
        dataset = self.val_dataset
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=lambda batch: custom_collate(batch, self.image_transforms))
        return loader

    def test_dataloader(self):
        dataset = self.test_dataset
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=lambda batch: custom_collate(batch, self.image_transforms))
        return loader

class ObjaverseData(Dataset):
    def __init__(self, root_dir='.objaverse/hf-objaverse-v1/views', image_transforms=None,
                 ext="png", default_trans=torch.zeros(3), postprocess=None, return_paths=False,
                 total_view=12, validation=False):
        """
        Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view
        self.validation = validation

        if not isinstance(ext, (tuple, list)):
            ext = [ext]

        with open(os.path.join(root_dir, 'valid_paths.json')) as f:
            self.paths = json.load(f)
            
        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training
        print(f'============= length of dataset {len(self.paths)} =============')
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T

    def load_im(self, path, color):
        '''
        Replace background pixel with random color in rendering.
        '''
        try:
            img = plt.imread(path)
        except:
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def __getitem__(self, index):
        data = {}
        total_view = self.total_view
        index_target, index_cond = random.sample(range(total_view), 2) # without replacement
        filename = os.path.join(self.root_dir, self.paths[index])

        if self.return_paths:
            data["path"] = str(filename)
        
        color = [1., 1., 1., 1.]

        try:
            target_im = self.process_im(self.load_im(os.path.join(filename, f'{index_target:03d}.png'), color))
            cond_im = self.process_im(self.load_im(os.path.join(filename, f'{index_cond:03d}.png'), color))
            target_RT = np.load(os.path.join(filename, f'{index_target:03d}.npy'))
            cond_RT = np.load(os.path.join(filename, f'{index_cond:03d}.npy'))
        except:
            filename = os.path.join(self.root_dir, '692db5f2d3a04bb286cb977a7dba903e_1') # this one we know is valid
            target_im = self.process_im(self.load_im(os.path.join(filename, f'{index_target:03d}.png'), color))
            cond_im = self.process_im(self.load_im(os.path.join(filename, f'{index_cond:03d}.png'), color))
            target_RT = np.load(os.path.join(filename, f'{index_target:03d}.npy'))
            cond_RT = np.load(os.path.join(filename, f'{index_cond:03d}.npy'))
            target_im = torch.zeros_like(target_im)
            cond_im = torch.zeros_like(cond_im)

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = self.get_T(target_RT, cond_RT)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)

# Usage
root_dir = '/export/compvis-nfs/group/datasets/views_release.tar.gz'
batch_size = 4
total_view = 12
train_config = {'image_transforms': {'size': 256}}
validation_config = {'image_transforms': {'size': 256}}

data_module = ObjaverseDataModuleFromConfig(root_dir, batch_size, total_view, train=train_config, validation=validation_config)
data_module.setup('fit')

train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Now you can use train_loader and val_loader for training and validation

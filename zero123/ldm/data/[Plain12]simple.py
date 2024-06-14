from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
from ldm.util import instantiate_from_config
from datasets import load_dataset
import pytorch_lightning as pl
import copy
import csv
import cv2
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import webdataset as wds
import math
from torch.utils.data.distributed import DistributedSampler
import gzip

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Some hacky things to make experimentation easier
def make_transform_multi_folder_data(paths, caption_files=None, **kwargs):
    ds = make_multi_folder_data(paths, caption_files, **kwargs)
    return TransformDataset(ds)

def make_nfp_data(base_path):
    dirs = list(Path(base_path).glob("*/"))
    print(f"Found {len(dirs)} folders")
    print(dirs)
    tforms = [transforms.Resize(512), transforms.CenterCrop(512)]
    datasets = [NfpDataset(x, image_transforms=copy.copy(tforms), default_caption="A view from a train window") for x in dirs]
    return torch.utils.data.ConcatDataset(datasets)


class VideoDataset(Dataset):
    def __init__(self, root_dir, image_transforms, caption_file, offset=8, n=2):
        self.root_dir = Path(root_dir)
        self.caption_file = caption_file
        self.n = n
        ext = "mp4"
        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.offset = offset

        if isinstance(image_transforms, ListConfig):
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms
        with open(self.caption_file) as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        self.captions = dict(rows)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        for i in range(10):
            try:
                return self._load_sample(index)
            except Exception:
                # Not really good enough but...
                print("uh oh")

    def _load_sample(self, index):
        n = self.n
        filename = self.paths[index]
        min_frame = 2*self.offset + 2
        vid = cv2.VideoCapture(str(filename))
        max_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        curr_frame_n = random.randint(min_frame, max_frames)
        vid.set(cv2.CAP_PROP_POS_FRAMES,curr_frame_n)
        _, curr_frame = vid.read()

        prev_frames = []
        for i in range(n):
            prev_frame_n = curr_frame_n - (i+1)*self.offset
            vid.set(cv2.CAP_PROP_POS_FRAMES,prev_frame_n)
            _, prev_frame = vid.read()
            prev_frame = self.tform(Image.fromarray(prev_frame[...,::-1]))
            prev_frames.append(prev_frame)

        vid.release()
        caption = self.captions[filename.name]
        data = {
            "image": self.tform(Image.fromarray(curr_frame[...,::-1])),
            "prev": torch.cat(prev_frames, dim=-1),
            "txt": caption
        }
        return data

# end hacky things


def make_tranforms(image_transforms):
    # if isinstance(image_transforms, ListConfig):
    #     image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    image_transforms = []
    image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
    image_transforms = transforms.Compose(image_transforms)
    return image_transforms


def make_multi_folder_data(paths, caption_files=None, **kwargs):
    """Make a concat dataset from multiple folders
    Don't suport captions yet

    If paths is a list, that's ok, if it's a Dict interpret it as:
    k=folder v=n_times to repeat that
    """
    list_of_paths = []
    if isinstance(paths, (Dict, DictConfig)):
        assert caption_files is None, \
            "Caption files not yet supported for repeats"
        for folder_path, repeats in paths.items():
            list_of_paths.extend([folder_path]*repeats)
        paths = list_of_paths

    if caption_files is not None:
        datasets = [FolderData(p, caption_file=c, **kwargs) for (p, c) in zip(paths, caption_files)]
    else:
        datasets = [FolderData(p, **kwargs) for p in paths]
    return torch.utils.data.ConcatDataset(datasets)



class NfpDataset(Dataset):
    def __init__(self,
        root_dir,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        ) -> None:
        """assume sequential frames and a deterministic transform"""

        self.root_dir = Path(root_dir)
        self.default_caption = default_caption

        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        return len(self.paths) - 1


    def __getitem__(self, index):
        prev = self.paths[index]
        curr = self.paths[index+1]
        data = {}
        data["image"] = self._load_im(curr)
        data["prev"] = self._load_im(prev)
        data["txt"] = self.default_caption
        return data

    def _load_im(self, filename):
        im = Image.open(filename).convert("RGB")
        return self.tform(im)
        
def custom_transform(x):
    return rearrange(x * 2. - 1., 'c h w -> h w c')
        
class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation
            
        if 'image_transforms' in dataset_config:
            image_transforms = [torchvision.transforms.Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []
        #image_transforms.extend([transforms.ToTensor(),
                                #transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms.extend([torchvision.transforms.ToTensor(), custom_transform])
                                
        self.image_transforms = torchvision.transforms.Compose(image_transforms)


    def train_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=False, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=True, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=self.validation),\
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='.objaverse/hf-objaverse-v1/views',
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=12,
        validation=False
        ) -> None:
        """Create a dataset from a folder of images.
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

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        with open(os.path.join(root_dir, 'valid_paths.json')) as f:
            self.paths = json.load(f)
            
        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
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
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            #print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def __getitem__(self, index):

        data = {}
        total_view = self.total_view
        index_target, index_cond = random.sample(range(total_view), 2) # without replacement
        filename = os.path.join(self.root_dir, self.paths[index])

        # print(self.paths[index])

        if self.return_paths:
            data["path"] = str(filename)
        
        color = [1., 1., 1., 1.]

        try:
            print(index_target)
            print(index_cond)
            print(os.path.join(filename, '%03d.png' % index_target))
            print(os.path.join(filename, '%03d.png' % index_cond))
            target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
            cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
        except:
            # very hacky solution, sorry about this
            filename = os.path.join(self.root_dir, '692db5f2d3a04bb286cb977a7dba903e_1') # this one we know is valid
            target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
            cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
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
    
class Co3dDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        # torchvision.transforms.Resize(dataset_config.image_transforms.size)
        if 'image_transforms' in dataset_config:
            image_transforms = [torchvision.transforms.Resize((dataset_config.image_transforms.size, dataset_config.image_transforms.size))]
            print("RESIZE")
            print(dataset_config.image_transforms.size)
        else:
            image_transforms = []
        #image_transforms.extend([transforms.ToTensor(),
                                #transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms.extend([torchvision.transforms.ToTensor(), custom_transform])
                                
        self.image_transforms = torchvision.transforms.Compose(image_transforms)
            
    def train_dataloader(self):
        dataset = Co3dData(root_dir=self.root_dir, total_view=self.total_view, validation=False, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = Co3dData(root_dir=self.root_dir, total_view=self.total_view, validation=True, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(Co3dData(root_dir=self.root_dir, total_view=self.total_view, validation=self.validation),\
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class Co3dData(Dataset):
    def __init__(self,
        root_dir='CO3DV2_DATASET_ROOT/',
        image_transforms=[],
        ext="jpg",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=12,
        validation=False
        ) -> None:
        """Create a dataset from a folder of images.
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
        self.annotations = self.open_gzipped_json('co3d/CO3DV2_DATASET_ROOT/hydrant/frame_annotations.jgz')
        self.samples = self._load_samples()
        #self.subset_file = 'hydrant/set_lists.json'
        self.tform = image_transforms

    # ['429_60517_117787', 100, 'co3d\\CO3DV2_DATASET_ROOT\\hydrant/429_60517_117787/images/frame000101.jpg'], 
    # ['429_60517_117787', 101, 'co3d\\CO3DV2_DATASET_ROOT\\hydrant/429_60517_117787/images/frame000102.jpg']]
    #def _load_samples(self):
        #subset_path = os.path.join(self.root_dir, 'hydrant/set_lists.json')
        #with open(subset_path, 'r') as f:
            #subsets = json.load(f)

        #samples = subsets.get("train_known", [])
        #for i, (sequence_name, frame_number, _) in enumerate(samples):
            #samples[i][2] = os.path.join(self.root_dir, samples[i][2])  # Adjust image path
            
        #print(len(samples))
        #print(samples)
        #return samples
        
    def _load_samples(self):
        subset_path = os.path.join(self.root_dir, 'hydrant/set_lists.json')
        with open(subset_path, 'r') as f:
            subsets = json.load(f)

        all_samples = subsets.get("train_known", [])
        samples_limited = []  # This will store the modified list with limited frames
        sequence_frame_count = {}
        
        # Define the sequences to exclude
        exclude_sequences = ['417_57803_111610', '427_59907_115784', '194_20931_42673', '344_35905_66057', '411_56031_108316', '216_22866_49900', '304_31888_60537', '194_20878_39742', '427_59883_115498', '216_22826_47834', '106_12648_23157', '106_12660_22718', '106_12669_24034', '106_12677_24990', '106_12686_26118']

        for i, (sequence_name, frame_number, image_path) in enumerate(all_samples):
            # Skip the sequence if it's in the list of sequences to exclude
            if sequence_name in exclude_sequences:
                continue

            # Limit the number of frames to 12 per sequence
            if sequence_name not in sequence_frame_count:
                sequence_frame_count[sequence_name] = 0

            if sequence_frame_count[sequence_name] < 12:
                adjusted_image_path = os.path.join(self.root_dir, image_path)  # Adjust image path
                # Append the modified sample to a new list
                samples_limited.append([sequence_name, frame_number, adjusted_image_path])
                sequence_frame_count[sequence_name] += 1

        print(f"Total samples included: {len(samples_limited)}")
        return samples_limited

        
    @staticmethod
    def open_gzipped_json(filepath):
        with gzip.open(filepath, 'rt', encoding='utf-8') as gzipped_file:
            return json.load(gzipped_file)

    def __len__(self):
        return len(self.samples)
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        #print("GET TTTTTTTTTTTTTT")
        #print(f"Calculating relative transformation. Target RT: {target_RT}, Cond RT: {cond_RT}")
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
        #print(f"Relative transformation: {d_T}")
        return d_T

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            #print("load_im")
            img = plt.imread(path)
            #print("Does work")
        except:
            print("FAil Load")
            #print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        #print("DID IT")
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        #print("DID IT 2")
        return img
        
    @staticmethod
    def open_gzipped_json(filepath):
        with gzip.open(filepath, 'rt', encoding='utf-8') as gzipped_file:
            return json.load(gzipped_file)

    def __getitem__(self, index):

        data = {}
        total_view = self.total_view
        #index_target, index_cond = random.sample(range(101), 2) # without replacement
        #filename = os.path.join(self.root_dir, self.samples[index])
        filename = self.samples[index][0]
        
        #print(filename)
        #print(len(filename))

        #print(self.samples[index][0])

        if self.return_paths:
            data["path"] = str(filename)
        
        color = [1., 1., 1., 1.]
        color_2 = np.array([255, 255, 255, 255])
        
        #print("HERE RIGHT NOW")
        
        exclude_sequences = ['417_57803_111610', '427_59907_115784', '194_20931_42673', '344_35905_66057', '411_56031_108316', '216_22866_49900', '304_31888_60537', '194_20878_39742', '427_59883_115498', '216_22826_47834', '106_12648_23157', '106_12660_22718', '106_12669_24034', '106_12677_24990', '106_12686_26118']
        
        image_loaded = False
        
        try:
            while not image_loaded:
                index_target, index_cond = random.sample(range(101), 2) # without replacement
                index_target += 1
                index_cond += 1
                target_path = os.path.join("co3d", "CO3DV2_DATASET_ROOT", "hydrant", filename, "images", f"frame000{index_target:03d}.jpg")
                cond_path = os.path.join("co3d", "CO3DV2_DATASET_ROOT", "hydrant", filename, "images", f"frame000{index_cond:03d}.jpg")
                
                #target_mask = os.path.join("co3d", "CO3DV2_DATASET_ROOT", "hydrant", filename, "masks", f"frame000{index_target:03d}.png")
                #cond_mask = os.path.join("co3d", "CO3DV2_DATASET_ROOT", "hydrant", filename, "masks", f"frame000{index_cond:03d}.png")
                
                #sys.exit()
                #target_im
                
                if filename in exclude_sequences:
                    print(filename)
                    sys.exit()
                    continue
                
                if not os.path.exists(target_path) or not os.path.exists(cond_path):
                    continue

                #print(target_path)
                #print(cond_path)
                #print(target_mask)
                #print(cond_mask)
                
                #target_im = cv2.imread(target_path)
                #cond_im = cv2.imread(cond_path)
                
                target_im = plt.imread(target_path)
                cond_im = plt.imread(cond_path) 
                
                if target_im.shape[2] < 4:
                    # Add an alpha channel, set all alpha values to 255 (fully opaque)
                    alpha_channel = np.ones((target_im.shape[0], target_im.shape[1], 1)) * 255
                    #img_with_alpha = np.concatenate((img, alpha_channel), axis=-1)
                    target_im = np.concatenate((target_im, alpha_channel), axis=-1)
                    #print("DONE")
                    
                if cond_im.shape[2] < 4:
                    # Add an alpha channel, set all alpha values to 255 (fully opaque)
                    alpha_channel = np.ones((cond_im.shape[0], cond_im.shape[1], 1)) * 255
                    #img_with_alpha = np.concatenate((img, alpha_channel), axis=-1)
                    cond_im = np.concatenate((cond_im, alpha_channel), axis=-1)
                    #print("DONE")
                    
                target_im = np.where(target_im[:, :, 3:] == 0, color, target_im)
                cond_im = np.where(cond_im[:, :, 3:] == 0, color, cond_im)
                
                target_im = Image.fromarray(target_im.astype('uint8'), 'RGBA')
                cond_im = Image.fromarray(cond_im.astype('uint8'), 'RGBA')
                
                target_im = target_im.convert("RGB")
                cond_im = cond_im.convert("RGB")
                
                # Display the images
                #plt.figure(figsize=(10, 5))
                #plt.subplot(1, 2, 1)
                #plt.imshow(target_im)
                #plt.title('Target Image')
                #plt.axis('off')

                #plt.subplot(1, 2, 2)
                #plt.imshow(cond_im)
                #plt.title('Conditional Image')
                #plt.axis('off')

                #plt.show()
                
                target_im = self.tform(target_im)
                cond_im = self.tform(cond_im)
            
                ###################################################################################################

                annotations = self.open_gzipped_json(os.path.join("co3d", "CO3DV2_DATASET_ROOT", "hydrant", 'frame_annotations.jgz'))
                #print(annotations['image']['path'])
                #print(target_path)    
                
                target = next((item for item in annotations if item['sequence_name'] == filename and item['frame_number'] == (index_target - 1 )), None)
                cond = next((item for item in annotations if item['sequence_name'] == filename and item['frame_number'] == (index_cond - 1)), None)
                
                relative_target_path = target_path.replace("\\", "/").split("CO3DV2_DATASET_ROOT/", 1)[-1]
                relative_cond_path = cond_path.replace("\\", "/").split("CO3DV2_DATASET_ROOT/", 1)[-1]
                
                if target is None or cond is None or target['image']['path'] != relative_target_path or cond['image']['path'] != relative_cond_path:
                    continue
                    
                #print(target_path)
                #print(cond_path)
                #print(target['image'])
                #print(cond['image'])
                image_loaded = True
            
            target_viewpoint = target['viewpoint']
            cond_viewpoint = cond['viewpoint']
            
            target_RT = np.hstack((np.array(target_viewpoint['R']), np.array(target_viewpoint['T']).reshape(-1, 1)))
            cond_RT = np.hstack((np.array(cond_viewpoint['R']), np.array(cond_viewpoint['T']).reshape(-1, 1)))  

        except Exception as e:
            print("Except Clause")
            print(f"Error processing index {index_target}: {e}")
            print(f"Error processing index {index_cond}: {e}")
            print(f"Target Path {target_path}")
            print(f"Cond Path {cond_path}")
            
            sys.exit()
            # very hacky solution, sorry about this
            #filename = os.path.join(self.root_dir, '692db5f2d3a04bb286cb977a7dba903e_1') # this one we know is valid
            #target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
            #cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
            #target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            #cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
            #target_im = torch.zeros_like(target_im)
            #cond_im = torch.zeros_like(cond_im)

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = self.get_T(target_RT, cond_RT)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)
        
class FolderData(Dataset):
    def __init__(self,
        root_dir,
        caption_file=None,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        postprocess=None,
        return_paths=False,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_caption = default_caption
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        if caption_file is not None:
            with open(caption_file, "rt") as f:
                ext = Path(caption_file).suffix.lower()
                if ext == ".json":
                    captions = json.load(f)
                elif ext == ".jsonl":
                    lines = f.readlines()
                    lines = [json.loads(x) for x in lines]
                    captions = {x["file_name"]: x["text"].strip("\n") for x in lines}
                else:
                    raise ValueError(f"Unrecognised format: {ext}")
            self.captions = captions
        else:
            self.captions = None

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        # Only used if there is no caption file
        self.paths = []
        for e in ext:
            self.paths.extend(sorted(list(self.root_dir.rglob(f"*.{e}"))))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        if self.captions is not None:
            return len(self.captions.keys())
        else:
            return len(self.paths)

    def __getitem__(self, index):
        data = {}
        if self.captions is not None:
            chosen = list(self.captions.keys())[index]
            caption = self.captions.get(chosen, None)
            if caption is None:
                caption = self.default_caption
            filename = self.root_dir/chosen
        else:
            filename = self.paths[index]

        if self.return_paths:
            data["path"] = str(filename)

        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        data["image"] = im

        if self.captions is not None:
            data["txt"] = caption
        else:
            data["txt"] = self.default_caption

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)
        
import random

class TransformDataset():
    def __init__(self, ds, extra_label="sksbspic"):
        self.ds = ds
        self.extra_label = extra_label
        self.transforms = {
            "align": transforms.Resize(768),
            "centerzoom": transforms.CenterCrop(768),
            "randzoom": transforms.RandomCrop(768),
        }


    def __getitem__(self, index):
        data = self.ds[index]

        im = data['image']
        im = im.permute(2,0,1)
        # In case data is smaller than expected
        im = transforms.Resize(1024)(im)

        tform_name = random.choice(list(self.transforms.keys()))
        im = self.transforms[tform_name](im)

        im = im.permute(1,2,0)

        data['image'] = im
        data['txt'] = data['txt'] + f" {self.extra_label} {tform_name}"

        return data

    def __len__(self):
        return len(self.ds)

def hf_dataset(
    name,
    image_transforms=[],
    image_column="image",
    text_column="text",
    split='train',
    image_key='image',
    caption_key='txt',
    ):
    """Make huggingface dataset with appropriate list of transforms applied
    """
    ds = load_dataset(name, split=split)
    tform = make_tranforms(image_transforms)

    assert image_column in ds.column_names, f"Didn't find column {image_column} in {ds.column_names}"
    assert text_column in ds.column_names, f"Didn't find column {text_column} in {ds.column_names}"

    def pre_process(examples):
        processed = {}
        processed[image_key] = [tform(im) for im in examples[image_column]]
        processed[caption_key] = examples[text_column]
        return processed

    ds.set_transform(pre_process)
    return ds

class TextOnly(Dataset):
    def __init__(self, captions, output_size, image_key="image", caption_key="txt", n_gpus=1):
        """Returns only captions with dummy images"""
        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus*[x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = rearrange(dummy_im * 2. - 1., 'c h w -> h w c')
        return {self.image_key: dummy_im, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, 'rt') as f:
            captions = f.readlines()
        return [x.strip('\n') for x in captions]



import random
import json
class IdRetreivalDataset(FolderData):
    def __init__(self, ret_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(ret_file, "rt") as f:
            self.ret = json.load(f)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        key = self.paths[index].name
        matches = self.ret[key]
        if len(matches) > 0:
            retreived = random.choice(matches)
        else:
            retreived = key
        filename = self.root_dir/retreived
        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        # data["match"] = im
        data["match"] = torch.cat((data["image"], im), dim=-1)
        return data

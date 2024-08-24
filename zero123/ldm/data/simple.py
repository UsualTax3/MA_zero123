from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
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

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import webdataset as wds
import ast
import imageio.v2 as imageio
import io
import numpy as np
import os
import random
import torchvision
from einops import rearrange
from PIL import Image
import torch
import matplotlib.pyplot as plt

def custom_transform(x):
    return rearrange(x, 'c h w -> h w c')

def npy_decoder(value):
    assert isinstance(value, bytes)
    return np.frombuffer(value, dtype=np.float64)

def png_decoder(value):
    assert isinstance(value, bytes)
    return imageio.imread(io.BytesIO(value)) 

def load_im_from_array(img_array, color):
    if img_array.shape[2] == 4:  
        alpha_channel = img_array[:, :, 3]
        rgb_channels = img_array[:, :, :3]
        mask = alpha_channel == 0.0
        rgb_channels[mask] = color[:3]
        img_array = rgb_channels  
    img = Image.fromarray(np.uint8(img_array * 255.0))
    return img

# Function to assemble frames
def assemble_frames(sample):
    data = {}
    total_view = 12
    color = [1.0, 1.0, 1.0]

    index_target, index_cond = random.sample(range(total_view), 2)
    try:
        target_im_array = png_decoder(sample[f"{index_target:03d}.png"])
        cond_im_array = png_decoder(sample[f"{index_cond:03d}.png"])

        if target_im_array.max() > 1.0:
            target_im_array = target_im_array.astype(np.float64) / 255.0
        if cond_im_array.max() > 1.0:
            cond_im_array = cond_im_array.astype(np.float64) / 255.0

        target_im = load_im_from_array(target_im_array, color).convert('RGB')
        cond_im = load_im_from_array(cond_im_array, color).convert('RGB')

        target_RT_raw = npy_decoder(sample[f"{index_target:03d}.npy"])
        cond_RT_raw = npy_decoder(sample[f"{index_cond:03d}.npy"])

        target_RT = target_RT_raw.reshape(3, 4)
        cond_RT = cond_RT_raw.reshape(3, 4)

        if target_im is None or cond_im is None:
            raise ValueError("Image loading failed")

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.ToTensor(),
        ])

        target_im = transform(target_im)
        cond_im = transform(cond_im)

        # Apply custom transformations (if needed)
        target_im = custom_transform(target_im)
        cond_im = custom_transform(cond_im)
        
        target_im = target_im * 2 - 1  # Scale to [-1, 1]
        cond_im = cond_im * 2 - 1  # Scale to [-1, 1]
        
    except Exception as e:
        print(f"Error assembling frames: {e}")
        return None

    data["image_target"] = target_im
    data["image_cond"] = cond_im
    data["T"] = get_T(target_RT, cond_RT)

    return data

# Function to calculate transformation T
def get_T(target_RT, cond_RT):
    R, T = target_RT[:3, :3], target_RT[:, -1]
    T_target = -R.T @ T

    R, T = cond_RT[:3, :3], cond_RT[:, -1]
    T_cond = -R.T @ T

    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(T_cond[None, :])
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target[None, :])

    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * np.pi)
    d_z = z_target - z_cond

    d_T = torch.tensor([d_theta.item(), np.sin(d_azimuth.item()), np.cos(d_azimuth.item()), d_z.item()])
    return d_T

# Function to convert Cartesian coordinates to spherical coordinates
def cartesian_to_spherical(xyz):
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    z = np.sqrt(xy + xyz[:, 2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # Elevation angle from Z-axis down
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.array([theta, azimuth, z])

# Data module class for configuration
class WebDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, tar_base, batch_size, train=None, validation=None, test=None, num_workers=4, multinode=True, val_batch_size=None, assemble_frames=True, **kwargs) -> None:
        super().__init__()
        print(f'Setting tar base to {tar_base}')
        self.tar_base = tar_base
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.validation = validation
        self.test = test
        self.multinode = multinode
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.assemble_frames = assemble_frames

    def make_loader(self, dataset_config, train=True):
        shuffle = dataset_config.get('shuffle', 0)
        shardshuffle = shuffle > 0

        nodesplitter = wds.shardlists.split_by_node if self.multinode else wds.shardlists.single_node_only

        tars = os.path.join(self.tar_base, dataset_config["shards"])
        
        print(f"Creating WebDataset loader for {tars}")

        dset = wds.WebDataset(
            tars,
            nodesplitter=nodesplitter,
            shardshuffle=shardshuffle,
            handler=wds.warn_and_continue).repeat(100)

        if self.assemble_frames:
            dset = (dset
                    .map(assemble_frames)
                    )
        else:
            dset = (dset
                    .decode("pil", handler=wds.handle_extension("png", png_decoder))
                    .decode("npy", handler=wds.handle_extension("npy", npy_decoder))
                    .map(self.map)
                    )

        bs = self.batch_size if train else self.val_batch_size

        loader = wds.WebLoader(dset, batch_size=bs, shuffle=False, num_workers=self.num_workers)

        return loader

    def setup(self, stage=None):
        pass

    def prepare_data(self) -> None:
        pass

    def train_dataloader(self):
        print("Creating train dataloader")
        return self.make_loader(self.train)

    def val_dataloader(self):
        if self.validation is None:
            return None
        print("Creating val dataloader")
        return self.make_loader(self.validation, train=False)

    def test_dataloader(self):
        if self.test is None:
            return None
        return self.make_loader(self.test, train=False)

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
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

# Custom transform function
def custom_transform(x):
    return rearrange(x * 2. - 1., 'c h w -> h w c')

# NPY decoder function
def npy_decoder(value):
    assert isinstance(value, bytes)
    return np.frombuffer(value, dtype=np.float64)

# PNG decoder function
def png_decoder(value):
    assert isinstance(value, bytes)
    return imageio.imread(io.BytesIO(value))

# Function to load image from numpy array
def load_im_from_array(img_array, color):
    img_array[img_array[:, :, -1] == 0.] = color
    img = Image.fromarray(np.uint8(img_array[:, :, :3] * 255.))
    return img

# Function to assemble frames
def assemble_frames(sample):
    data = {}
    total_view = 12
    color = [1., 1., 1., 1.]

    index_target, index_cond = random.sample(range(total_view), 2)
    try:
        target_im_array = png_decoder(sample[f"{index_target:03d}.png"])
        cond_im_array = png_decoder(sample[f"{index_cond:03d}.png"])
        
        target_im = load_im_from_array(target_im_array, color)
        cond_im = load_im_from_array(cond_im_array, color)
        
        target_RT_raw = npy_decoder(sample[f"{index_target:03d}.npy"])
        cond_RT_raw = npy_decoder(sample[f"{index_cond:03d}.npy"])
        
        #print(f"Raw target_RT array: {target_RT_raw}, Shape: {target_RT_raw.shape}")
        #print(f"Raw cond_RT array: {cond_RT_raw}, Shape: {cond_RT_raw.shape}")

        target_RT = target_RT_raw.reshape(3, 4)
        cond_RT = cond_RT_raw.reshape(3, 4)
        
        #print(f"Selected indices - target: {index_target}, cond: {index_cond}")
        #print(f"target_RT contents: {target_RT}")
        #print(f"cond_RT contents: {cond_RT}")

        if target_im is None or cond_im is None:
            raise ValueError("Image loading failed")

        print(f"Original target image shape: {target_im_array.shape}")
        print(f"Original cond image shape: {cond_im_array.shape}")

        # Apply custom transformations
        target_im = torchvision.transforms.ToTensor()(target_im)
        target_im = custom_transform(target_im)

        cond_im = torchvision.transforms.ToTensor()(cond_im)
        cond_im = custom_transform(cond_im)
        
        print(f"Transformed target image shape: {target_im.shape}")
        print(f"Transformed cond image shape: {cond_im.shape}")
        
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

        dset = wds.WebDataset(
            tars,
            nodesplitter=nodesplitter,
            shardshuffle=shardshuffle,
            handler=wds.warn_and_continue).repeat().shuffle(shuffle)

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
        return self.make_loader(self.train)

    def val_dataloader(self):
        return self.make_loader(self.validation, train=False)

    def test_dataloader(self):
        return self.make_loader(self.test, train=False)
        
# Example usage to test the data loader
if __name__ == "__main__":
    # Configuration dictionary for the dataset
    train_config = {
        "shards": "train-{000000..000001}.tar",
        "shuffle": 1000
    }

    # Initialize the data module
    tar_base = "/export/compvis-nfs/group/datasets/obja_shards/shards"
    batch_size = 64
    data_module = WebDataModuleFromConfig(tar_base, batch_size, train=train_config)

    # Prepare data and setup the data module
    data_module.prepare_data()
    data_module.setup()

    # Get train and validation data loaders
    train_loader = data_module.train_dataloader()

    # Iterate through one batch to test
    for batch in train_loader:
        print(batch)
        break

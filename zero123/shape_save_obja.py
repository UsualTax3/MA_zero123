import sys
import os
import argparse
import numpy as np
import webdataset as wds
from tqdm import tqdm
from typing import List
import json
from PIL import Image
import io

DATASET_ROOT = "/export/compvis-nfs/group/datasets/views_release"
path_data = "/export/compvis-nfs/group/datasets/obja_shards"

def check_files_exist(folder, file_stems, extensions):
    for stem in file_stems:
        for ext in extensions:
            if not os.path.isfile(os.path.join(folder, f"{stem}.{ext}")):
                return False
    return True

def load_image_from_bytes(image_bytes):
    return Image.open(io.BytesIO(image_bytes))

def save_with_shape(array):
    return np.lib.format.open_memmap(
        io.BytesIO(), dtype=array.dtype, mode='w+', shape=array.shape
    ).tobytes() + np.lib.format.dtype_to_descr(array.dtype).encode() + np.array(array.shape).tobytes()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default=DATASET_ROOT, help="path to dataset")
    parser.add_argument("-p2", "--path2", type=str, default=path_data, help="path to output dataset")
    parser.add_argument("-s", "--split", type=str, default="train", help="split to convert")
    parser.add_argument("--max_size", type=float, default=2.0, help="gb per shard")
    opt = parser.parse_args()

    output_shard_path = os.path.join(opt.path2, "shards")
    os.makedirs(output_shard_path, exist_ok=True)

    json_path = os.path.join('/export/compvis-nfs/user/rbarlog/zero123/zero123/', 'valid_paths.json')
    if not os.path.isfile(json_path):
        print(f"Error: {json_path} not found.")
        sys.exit(1)

    with open(json_path, 'r') as f:
        folders = json.load(f)

    writer = wds.ShardWriter(os.path.join(output_shard_path, f"{opt.split}-%06d.tar"), maxcount=1e6, maxsize=opt.max_size*1e9)

    for folder in tqdm(folders, desc="Processing folders"):
        folder_path = os.path.join(opt.path, folder)

        if not check_files_exist(folder_path, [f"{i:03}" for i in range(12)], ["png", "npy"]):
            continue

        sample = {}
        sample["__key__"] = folder

        for i in range(12):
            png_path = os.path.join(folder_path, f"{i:03}.png")
            npy_path = os.path.join(folder_path, f"{i:03}.npy")

            if os.path.isfile(png_path) and os.path.isfile(npy_path):
                with open(png_path, "rb") as png_file:
                    png_bytes = png_file.read()
                    image = load_image_from_bytes(png_bytes)
                    # print(f"Image {i:03} shape: {image.size}")
                    sample[f"{i:03}.png"] = png_bytes

                np_data = np.load(npy_path).astype(np.float32)
                # print(f"Numpy {i:03} shape: {np_data.shape}")
                sample[f"{i:03}.npy"] = save_with_shape(np_data)

        writer.write(sample)

    writer.close()

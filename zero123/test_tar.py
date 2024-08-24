import webdataset as wds
from PIL import Image
from io import BytesIO
import numpy as np

# Path to your .tar.gz file
dataset_path = "/export/compvis-nfs/group/datasets/views_release.tar.gz"

# Create the WebDataset object
dataset = wds.WebDataset(dataset_path)

# Iterate through the samples in the dataset
for sample in dataset:
    # Print the keys in the sample
    print("Keys in sample:", sample.keys())

    for key, value in sample.items():
        if key.endswith(".png"):
            # Process and display the image
            img = Image.open(BytesIO(value))
            print(f"Key: {key}")
            print(f"  Image size: {img.size}")
            #img.show()

        elif key.endswith(".npy"):
            # Process and print the numpy array
            np_array = np.load(BytesIO(value))
            print(f"Key: {key}")
            print(f"  Numpy array shape: {np_array.shape}")
            print(np_array)

        elif key in ['__key__', '__url__']:
            # Print metadata
            print(f"Key: {key}")
            print(f"  Metadata: {value}")

    # Optionally, break after first sample for brevity
    break

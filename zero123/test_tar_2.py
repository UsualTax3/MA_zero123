import webdataset as wds
from PIL import Image
from io import BytesIO
import numpy as np

# Define the key you want to access
desired_key = "views_release/37a69b32c7024876831b1fd79673c06e/006"

# Define a function to filter and process each sample
def process_and_filter_sample(sample):
    if sample['__key__'] == desired_key:
        data = {}
        for key, value in sample.items():
            if key == 'png':
                # Process image
                img = Image.open(BytesIO(value))
                img_array = np.array(img)  # Convert to numpy array if needed
                data['image'] = img_array

            elif key == 'npy':
                # Process numpy array
                np_array = np.load(BytesIO(value))
                data['array'] = np_array

            elif key == '__key__':
                # Store the key for reference
                data['key'] = value

            elif key == '__url__':
                # Store the URL for reference
                data['url'] = value

        return data
    return None

# Path to your .tar.gz file
dataset_path = "/export/compvis-nfs/group/datasets/views_release.tar.gz"

# Create the WebDataset object and filter the desired sample
dataset = wds.WebDataset(dataset_path).map(process_and_filter_sample)

# Retrieve the specific sample
for sample in dataset:
    if sample is not None:
        print("Sample key:", sample['key'])
        print("Image shape:", sample['image'].shape)
        if 'array' in sample:
            print("Numpy array shape:", sample['array'].shape)
        print("URL:", sample['url'])
        # Add your processing logic here
        break  # Stop after retrieving the desired sample
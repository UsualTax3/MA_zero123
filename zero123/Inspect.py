import webdataset as wds
from PIL import Image

# Define a function to sample and inspect a small subset of the dataset
def inspect_webdataset(root_dir, num_samples=5):
    """
    Inspect a small subset of the WebDataset given by root_dir.
    
    Args:
        root_dir (str): The root directory containing the .tar.gz file.
        num_samples (int): The number of samples to inspect.
    """
    tar_path = root_dir
    
    # Create a WebDataset object
    dataset = wds.WebDataset(tar_path).shuffle(1000).decode("pil")
    
    # Iterate over a few samples and print the keys and types
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        print(f"Sample {i+1}:")
        for key, value in sample.items():
            print(f"  {key}: {type(value)}")
            if isinstance(value, Image.Image):  # If the value is an image, print its size
                print(f"    Image size: {value.size}")
            elif key.endswith("npy"):  # If the value is a numpy array, print its shape
                print(f"    Numpy array shape: {value.shape}")
            else:
                print(f"    Value: {value}")
        print("\n")

# Example usage
root_dir = '/export/compvis-nfs/group/datasets/views_release.tar.gz'
inspect_webdataset(root_dir)

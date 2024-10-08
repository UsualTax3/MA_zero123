import os
import math
import fire
import numpy as np
import time
import json
from torch.utils.data import Dataset, DataLoader
import gzip
import cv2

import matplotlib.pyplot as plt
import torch
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from transformers import AutoFeatureExtractor
from torch import autocast
from torchvision import transforms
from torchvision.transforms import functional as TF
from ldm.modules.evaluate.ssim import ssim
import lpips
from torchvision.models import inception_v3
from scipy.linalg import sqrtm

# Global Variables
_GPU_INDEX = 0
device = f"cuda:{_GPU_INDEX}"
models = {}

# Function to load and initialize models
def load_and_initialize_models():
    global models
    ckpt_path = "/export/compvis-nfs/user/rbarlog/logs/2024-10-01T02-17-24_sd-objaverse-finetune-c_concat-256/checkpoints/epoch=000000-v1.ckpt"
    #ckpt_path = "/export/scratch/ru89tew/105000.ckpt"
    config_path = "configs/sd-objaverse-finetune-c_concat-256.yaml"
    config = OmegaConf.load(config_path)

    print('Loading and initializing models...')
    models['turncam'] = load_model_from_config(config, ckpt_path, device)
    models['carvekit'] = create_carvekit_interface()
    models['nsfw'] = StableDiffusionSafetyChecker.from_pretrained('CompVis/stable-diffusion-safety-checker').to(device)
    models['clip_fe'] = AutoFeatureExtractor.from_pretrained('CompVis/stable-diffusion-safety-checker')

    # Initialize LPIPS model separately as it's already a global variable
    global lpips_model
    lpips_model = lpips.LPIPS(net='alex').to(device)

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model

def initialize_inception_v3(pretrained=True, device='cuda'):
    model = inception_v3(pretrained=pretrained, init_weights=False)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    return model
    
def preprocess_image_for_inception_v3(image):
    print("Preprocessing image for InceptionV3...")
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image)
    print(f"Image shape after preprocessing: {image_tensor.shape}")
    return image_tensor

def extract_features(model, image_tensor):
    print("Extracting features...")
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        features = model(image_tensor.unsqueeze(0))
    print(f"Features shape: {features.shape}")
    return features.squeeze(0)
    

def calculate_fid(real_features, fake_features):
    print("Calculating FID score...")
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    print(f"FID components - ssdiff: {ssdiff}, trace: {np.trace(sigma1 + sigma2 - 2.0 * covmean)}")
    return fid
    
@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w,
                 ddim_steps, n_samples, scale, ddim_eta, elevation_first, azimuth_first, radius_first,
                 elevation, azimuth, radius):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    
    # Convert both sets of azimuth and elevation to radians
    azimuth_rad_first = np.deg2rad(azimuth_first % 360)
    elevation_rad_first = np.deg2rad(np.clip(elevation_first, -5, 5))
    azimuth_rad = np.deg2rad(azimuth % 360)
    elevation_rad = np.deg2rad(np.clip(elevation, -5, 5))

    # Calculate relative transformations between the two viewpoints
    delta_azimuth = (azimuth_rad_first - azimuth_rad + np.pi) % (2 * np.pi) - np.pi
    delta_elevation = elevation_rad_first - elevation_rad
    #delta_elevation = np.clip(delta_elevation, -np.pi/2, np.pi/2)
    
    print(f"Delta azimuth (rad): {delta_azimuth}, Delta elevation (rad): {delta_elevation}")

    # Normalize the radius as per the interval [1.5, 2.2] and calculate the difference
    normalized_radius_first = (radius_first - 1.5) / (2.2 - 1.5)
    normalized_radius = (radius - 1.5) / (2.2 - 1.5)
    delta_radius = normalized_radius_first - normalized_radius

    print(f"Normalized initial radius: {normalized_radius_first}, Normalized current radius: {normalized_radius}, Delta radius: {delta_radius}")

    # Calculate and adjust the normalized delta radius
    normalized_delta_radius = delta_radius / (2.2 - 1.5)
    adjusted_normalized_radius = np.clip(normalized_radius_first + normalized_delta_radius, 0, 1)
    adjusted_radius = adjusted_normalized_radius * (2.2 - 1.5) + 1.5
    adjusted_radius = np.clip(adjusted_radius, 1.5, 2.2)
    
    print(delta_elevation)
    print(delta_azimuth)
    print(delta_radius)
    print(adjusted_radius)
    
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([delta_elevation,
                              math.sin(delta_azimuth), math.cos(delta_azimuth),
                              (adjusted_radius/100)])
                    
            print("SHAPEEEEE")        
            print(T.shape)
            print(c.shape)
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1).float()
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            # print(samples_ddim.shape)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def preprocess_image(models, input_im, input_mask, preprocess=True):
    # If input_im is a string (path), read the image using cv2.imread
    if isinstance(input_im, str):
        input_im = cv2.imread(input_im)
    # If input_im is a PIL.Image object, convert it to a NumPy array
    elif isinstance(input_im, Image.Image):
        input_im = np.array(input_im)
        # Assuming OpenCV will be used for further processing, convert RGB (PIL) to BGR (OpenCV)
        input_im = cv2.cvtColor(input_im, cv2.COLOR_RGB2BGR)
    
    if preprocess:
        # Load the mask and resize it to match the image size
        if isinstance(input_mask, str):
            mask = cv2.imread(input_mask, cv2.IMREAD_GRAYSCALE)
            
            plt.imshow(mask)
            plt.axis('off')  # Hides the axis
            plt.show()
            
            if mask is not None:
                # Resize the mask to match the input image's dimensions
                #mask_resized = cv2.resize(mask, (input_im.shape[1], input_im.shape[0]))
                # Create a white background of the same size as input_im
                
                #mask_bool_target = mask_target > 128
                #white_background_target = np.ones_like(image_target) * 255
                #result_image_target = np.where(mask_bool_target[:, :, np.newaxis], image_target, white_background_target)
                #target_im = cv2.cvtColor(result_image_target, cv2.COLOR_BGR2RGB)
                
                mask_bool_target = mask > 128
                background = np.ones_like(input_im) * 255
                # Apply the mask: keep input_im where mask is not zero, else background
                input_im_masked = np.where(mask_bool_target[:, :, None] > 0, input_im, background)
                
                # Optional: Display the processed image
                #plt.imshow(cv2.cvtColor(input_im_masked, cv2.COLOR_BGR2RGB))
                #plt.axis('off')  # Hides the axis
                #plt.show()
                
        # Resize the image after applying the mask to the target size
        target_size = (256, 256)  # Desired output size
        input_im_resized = cv2.resize(input_im_masked, target_size)

    else:
        # No preprocessing is to be done, input_im is returned as is
        input_im_resized = input_im  # This line is modified to use the potentially unmodified input_im

    # Convert the NumPy array back to PIL.Image for consistency
    #input_im_resized = cv2.cvtColor(input_im_resized, cv2.COLOR_BGR2RGB)  
    #input_im_resized = Image.fromarray(input_im_resized.astype('uint8'))
    
    input_im_resized = cv2.cvtColor(input_im_resized, cv2.COLOR_BGR2RGB)  
    input_im_resized = Image.fromarray(input_im_resized.astype('uint8'))

    return input_im_resized


def main_run(raw_im,
             models, device, mask, elevation_first=0.0, azimuth_first=0.0, radius_first=0.0,
             elevation=0.0, azimuth=0.0, radius=0.0,
             preprocess=True,
             scale=3.0, n_samples=1, ddim_steps=50, ddim_eta=1.0,
             precision='fp32', h=256, w=256):
    '''
    :param raw_im (PIL Image).
    '''
   
    #raw_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
    #safety_checker_input = models['clip_fe'](raw_im, return_tensors='pt').to(device)
    #(image, has_nsfw_concept) = models['nsfw'](
        #images=np.ones((1, 3)), clip_input=safety_checker_input.pixel_values)
    #print('has_nsfw_concept:', has_nsfw_concept)
    #if np.any(has_nsfw_concept):
        #print('NSFW content detected.')
        #to_return = [None] * 10
        #description = ('###  <span style="color:red"> Unfortunately, '
                       #'potential NSFW content was detected, '
                       #'which is not supported by our model. '
                       #'Please try again with a different image. </span>')
        #to_return[0] = description
        #return to_return
    #else:
        #print('Safety check passed.')

    input_im = preprocess_image(models, raw_im, mask, preprocess)
    input_tensor = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_tensor = input_tensor * 2 - 1
    input_tensor = transforms.functional.resize(input_tensor, [h, w])

    sampler = DDIMSampler(models['turncam'])
    x_samples_ddim = sample_model(input_tensor, models['turncam'], sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta, elevation_first, azimuth_first, radius_first, elevation, azimuth, radius)
    
    lpips_model = lpips.LPIPS(net='alex').to(device)

    output_ims = []
    #ssim_values = []  # List to store SSIM values
    #lpips_values = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_image = Image.fromarray(x_sample.astype(np.uint8))
        output_ims.append(output_image)

        # Convert the output image to tensor for SSIM calculation
        output_tensor = transforms.ToTensor()(output_image).unsqueeze(0).to(device)
        
        # Calculate SSIM and store the value
        #current_ssim = ssim(input_tensor, output_tensor)
        #ssim_values.append(current_ssim.item())  # Convert to Python float and store
        #current_lpips = lpips_model(input_tensor, output_tensor)
        #lpips_values.append(current_lpips.item()) 

    # Optionally, print or return the SSIM values
    #print("SSIM values for each generated image:", ssim_values)
    #print("LPIPS values for each generated image:", lpips_values)

    return output_ims


def predict(cond_image_path: str = "cond.png",
            elevation_first_degree: float = 0.0,
            azimuth_first_degree: float = 0.0,
            radius_first: float = 0.0,
            elevation_in_degree: float = 0.0,
            azimuth_in_degree: float = 0.0,
            radius: float = 0.0,
            output_image_path: str = "output_azimuth_90.png",
            mask: str = "mask.png"):
    #device = f"cuda:{device_idx}"
    #config = OmegaConf.load(config)

    #assert os.path.exists(ckpt)
    #assert os.path.exists(cond_image_path)

    # Instantiate all models beforehand for efficiency.
    #models = dict()
    #print('Instantiating LatentDiffusion...')
    #models['turncam'] = load_model_from_config(config, ckpt, device=device)
    #print('Instantiating Carvekit HiInterface...')
    #models['carvekit'] = create_carvekit_interface()
    #print('Instantiating StableDiffusionSafetyChecker...')
    #models['nsfw'] = StableDiffusionSafetyChecker.from_pretrained(
        #'CompVis/stable-diffusion-safety-checker').to(device)
    #print('Instantiating AutoFeatureExtractor...')
    #models['clip_fe'] = AutoFeatureExtractor.from_pretrained(
        #'CompVis/stable-diffusion-safety-checker')

    #cond_image = Image.open(cond_image_path)

    preds_images = main_run(raw_im=cond_image_path,
                            models=models, device=device,
                            elevation_first=elevation_first_degree,
                            azimuth_first=azimuth_first_degree,
                            radius_first=radius_first,
                            elevation=elevation_in_degree,
                            azimuth=azimuth_in_degree,
                            radius=radius,
                            mask=mask)

    pred_image = preds_images[-1]
    pred_image.save(output_image_path)

# Define the function to calculate azimuth and elevation from the rotation matrix
def calculate_azimuth_elevation(R):
    R = np.array([[r.item() for r in row] for row in R])  # Convert R from tensors to numpy array
    azimuth = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    elevation = np.degrees(np.arcsin(-R[2, 0]))
    return azimuth, elevation
    
def calculate_radius(T):
    """
    Calculate the radius from the translation vector.
    :param T: Translation vector (list or tensor with 3 elements [Tx, Ty, Tz])
    :return: Radius (float)
    """
    # Ensure T is a numpy array for calculation
    T = np.array([t.item() for t in T])
    radius = np.sqrt(np.sum(T**2))
    return radius

# Define the function to print details for each sequence
def print_sequence_details(sequence_name, details):
    if details:
        for idx, detail in enumerate(details):
            azimuth, elevation = calculate_azimuth_elevation(detail['viewpoint']['R'])
            if idx == 0:
                print(f"First frame of sequence {sequence_name}: {detail['path']}")
            else:
                print(f" - Path: {detail['path']}")
            print(f"   Viewpoint: Azimuth={azimuth}°, Elevation={elevation}°")
        print()
        
def calculate_metrics(generated_img_path, preprocessed_corresponding_img):
    # Load the generated image
    generated_img = Image.open(generated_img_path).convert('RGB')

    # Convert both images to tensors
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    generated_tensor = transform(generated_img).unsqueeze(0).to(device)
    corresponding_tensor = transform(preprocessed_corresponding_img).unsqueeze(0).to(device)
    
    # Calculate SSIM and LPIPS
    ssim_value = ssim(generated_tensor, corresponding_tensor).item()
    lpips_value = lpips_model(generated_tensor, corresponding_tensor).item()
    
    print(f"SSIM: {ssim_value}, LPIPS: {lpips_value}")
    return ssim_value, lpips_value

        
class ChairDataset(Dataset):
    def __init__(self, dataset_root, annotations_file, subset_file='hydrant/set_lists.json', transform=None):
        """
        Args:
            dataset_root (str): Root directory of the chair category dataset.
            annotations_file (str): Path to the gzipped JSON file containing frame annotations.
            subset_file (str): Filename of the subset file. Defaults to 'set_lists.json'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset_root = dataset_root
        self.annotations = self.open_gzipped_json(os.path.join(dataset_root, annotations_file))
        self.subset_file = subset_file
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        subset_path = os.path.join(self.dataset_root, self.subset_file)
        with open(subset_path, 'r') as f:
            subsets = json.load(f)

        samples = subsets.get("train_known", [])
        for i, (sequence_name, frame_number, _) in enumerate(samples):
            samples[i][2] = os.path.join(self.dataset_root, samples[i][2])  # Adjust image path
        return samples

    @staticmethod
    def open_gzipped_json(filepath):
        with gzip.open(filepath, 'rt', encoding='utf-8') as gzipped_file:
            return json.load(gzipped_file)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence_name, frame_number, image_path = self.samples[idx]
        image = Image.open(image_path).convert('RGB')

        # Find the corresponding annotation by sequence name and frame number
        annotation = next((item for item in self.annotations if item['sequence_name'] == sequence_name and item['frame_number'] == frame_number), None)

        if self.transform:
            image = self.transform(image)
        
        # Printing the frame-specific information for demonstration
        #print(f"Sequence: {sequence_name}, Frame: {frame_number}, Annotation: {annotation}")
        #print(annotation['viewpoint'])

        return {
            'image': image,
            'sequence_name': sequence_name,
            'frame_number': frame_number,
            'annotation_image': annotation['image'],
            'annotation_view': annotation['viewpoint'],
            'annotation_mask': annotation['mask']
        }
        
if __name__ == "__main__":
    load_and_initialize_models()
    device = 'cuda'  # or 'cpu'
    inception_model = initialize_inception_v3(device=device)
    dataset_root = 'co3d/CO3DV2_DATASET_ROOT/'
    annotations_file = 'hydrant/frame_annotations.jgz'
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    chair_dataset = ChairDataset(dataset_root, annotations_file, transform=transform)
    chair_dataloader = DataLoader(chair_dataset, batch_size=1, shuffle=False)

    processed_sequences = set()
    current_sequence = None
    
    real_features_list = []
    generated_features_list = []
    
    ssim_values = []
    lpips_values = []

    while len(processed_sequences) < 11:
        first_frame_details = None
        subsequent_frames = []

        for batch in chair_dataloader:
            sequence_name = batch['sequence_name'][0]
            if sequence_name in processed_sequences:
                print(f"Sequence {sequence_name} already processed, skipping.")
                continue

            frame_number = batch['frame_number'][0]
            image_path = os.path.join(dataset_root, batch['annotation_image']['path'][0])
            mask_path = os.path.join(dataset_root, batch['annotation_mask']['path'][0])
            viewpoint = batch['annotation_view']

            if current_sequence is None:
                current_sequence = sequence_name
                print(f"Starting new sequence processing: {sequence_name}")

            if sequence_name != current_sequence:
                print(f"Encountered a different sequence {sequence_name}, currently processing {current_sequence}. Skipping this batch.")
                break

            print(f"Processing sequence: {sequence_name}, Frame: {frame_number}")

            if first_frame_details is None:
                first_frame_details = {
                    'frame_number': frame_number,
                    'image_path': image_path,
                    'viewpoint': viewpoint,
                    'sequence_name': sequence_name
                }
                print(f"Set as first frame for sequence {sequence_name}")
            else:
                subsequent_frames.append({
                    'frame_number': frame_number,
                    'image_path': image_path,
                    'viewpoint': viewpoint,
                    'mask_path': mask_path
                })
                #real_features_list.append({
                    #'image_path': image_path,
                #})
                print(f"Added frame {frame_number} to subsequent frames for sequence {sequence_name}")
                #print(f"Added frame {frame_number} to REAL frames for sequence {sequence_name}")

        processed_sequences.add(current_sequence)
        print(f"Completed processing all frames for sequence: {current_sequence}.")
        current_sequence = None  # Reset for the next sequence
        print(first_frame_details['image_path'])

        # Evaluate the processed sequence
        for frame_detail in subsequent_frames:
            #print(frame_detail['image_path'])
            print(f"First Frame {first_frame_details['image_path']}")
            print(f"Current Frame {frame_detail['image_path']}")
            
            azimuth_first, elevation_first = calculate_azimuth_elevation(first_frame_details['viewpoint']['R'])
            radius_first = calculate_radius(first_frame_details['viewpoint']['T'])
            
            azimuth, elevation = calculate_azimuth_elevation(frame_detail['viewpoint']['R'])
            radius = calculate_radius(frame_detail['viewpoint']['T'])

            print(f"Sequence: {sequence_name}, Conditional Frame: {first_frame_details['frame_number']}, Current Frame: {frame_detail['frame_number']}")
            print(f"Azimuth: {azimuth}°, Elevation: {elevation}°, Radius: {radius} meters")
            #print("frame_number:", first_frame_details.get('frame_number'))
            
            hardcode = first_frame_details['sequence_name']
            #target_mask = os.path.join("co3d", "CO3DV2_DATASET_ROOT", "hydrant", hardcode, "masks", f"frame000{:03d}.png")
            target_mask = os.path.join("co3d", "CO3DV2_DATASET_ROOT", "hydrant", hardcode, "masks", f"frame000001.png")
            print("TARGET MASK")
            print(target_mask)
            print("--------------------------------")
            print(first_frame_details['image_path'])
            print(frame_detail['image_path'])
            print("--------------------------------")
            
            output_image_path = f"output_2/output_{sequence_name}_{frame_detail['frame_number']}.png"
            predict(
                cond_image_path=first_frame_details['image_path'],
                elevation_first_degree = elevation_first,
                azimuth_first_degree = azimuth_first,
                radius_first =radius_first,
                elevation_in_degree = elevation,
                azimuth_in_degree= azimuth,
                radius=radius,
                output_image_path=output_image_path,
                mask=target_mask
            )

            print(f"Generated prediction for frame {frame_detail['frame_number']}. Calculating metrics...")
            
            generated_img_path = output_image_path
            corresponding_img_path = frame_detail['image_path']
            
            generated_img = Image.open(generated_img_path).convert('RGB')
            corresponding_img = Image.open(corresponding_img_path).convert('RGB')
            
            #models = dict()
            #models['carvekit'] = create_carvekit_interface()
            
            corresponding_mask = frame_detail['mask_path']
            print("CORRES MASK")
            print(corresponding_mask)
            
            preprocessed_corresponding_img_array = preprocess_image(models, corresponding_img, corresponding_mask, preprocess=True)
            #preprocessed_corresponding_img_2 = Image.fromarray((preprocessed_corresponding_img_array * 255).astype('uint8')).convert('RGB')
            
            preprocessed_image_path = f"GT_2/preprocessed_{sequence_name}_{frame_detail['frame_number']}.png"
            preprocessed_corresponding_img_array.save(preprocessed_image_path)
            
            real_image_tensor = preprocess_image_for_inception_v3(corresponding_img)
            real_features = extract_features(inception_model, real_image_tensor)
            real_features_list.append(real_features.cpu().numpy())
            
            generated_image_tensor = preprocess_image_for_inception_v3(preprocessed_corresponding_img_array)
            generated_features = extract_features(inception_model, generated_image_tensor)
            generated_features_list.append(generated_features.cpu().numpy())
            #image = Image.open(cm)
            #image.show()
            #preprocessed_corresponding_img.show()
            
            ssim_value, lpips_value = calculate_metrics(generated_img_path, preprocessed_corresponding_img_array)

            ssim_values.append(ssim_value)
            lpips_values.append(lpips_value)

            print(f"Frame {frame_detail['frame_number']} - SSIM: {ssim_value}, LPIPS: {lpips_value}")

        ssim_mean = sum(ssim_values) / len(ssim_values) if ssim_values else 0
        lpips_mean = sum(lpips_values) / len(lpips_values) if lpips_values else 0
        
        print(f"LENGTH SSIM ARRAY: {len(ssim_values)}")
        print(f"LENGTH LPIPS ARRAY: {len(lpips_values)}")

        print(f"Finished evaluation for sequence: {sequence_name}.")
        print(f"Mean SSIM for {sequence_name}: {ssim_mean}")
        print(f"Mean LPIPS for {sequence_name}: {lpips_mean}\n\n")
        # Example for processing a batch of real and generated images
        
        real_features_array = np.vstack(real_features_list)
        generated_features_array = np.vstack(generated_features_list)
        print(f"Real features array shape: {real_features_array.shape}")
        print(f"Generated features array shape: {generated_features_array.shape}")
        
        print(f"LENGTH Real features array shape: {len(real_features_array)}")
        print(f"LENGTH Generated features array shape: {len(generated_features_array)}")
        
        fid_score = calculate_fid(real_features_array, generated_features_array)
        print(f"FID Score: {fid_score}")
        print(f"FID Score: {(fid_score / 1000)}")
        
        # Print the first few features of the real and generated arrays to inspect
        print("First few features of real images:")
        print(real_features_array[:5])

        print("First few features of generated images:")
        print(generated_features_array[:5])

        # Additionally, print basic statistics of the arrays to get an idea of their distribution
        print("Statistics of real features array:")
        print(f"Mean: {np.mean(real_features_array, axis=0)[:5]}")  # Mean of the first few features
        print(f"Std Dev: {np.std(real_features_array, axis=0)[:5]}")  # Std deviation of the first few features

        print("Statistics of generated features array:")
        print(f"Mean: {np.mean(generated_features_array, axis=0)[:5]}")  # Mean of the first few features
        print(f"Std Dev: {np.std(generated_features_array, axis=0)[:5]}")  # Std deviation of the first few features

        # To further understand the distribution, consider printing min and max values
        print("Min and Max of real features array:")
        print(f"Min: {np.min(real_features_array, axis=0)[:5]}")
        print(f"Max: {np.max(real_features_array, axis=0)[:5]}")

        print("Min and Max of generated features array:")
        print(f"Min: {np.min(generated_features_array, axis=0)[:5]}")
        print(f"Max: {np.max(generated_features_array, axis=0)[:5]}")
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {}

# Function to load and initialize models
def load_and_initialize_models():
    global models
    ckpt_path = "/export/compvis-nfs/user/rbarlog/logs/1_]Test[_Rank_4_1e-4_100_warmupsteps/checkpoints/epoch=000003.ckpt"
    config_path = "configs/sd-objaverse-finetune-c_concat-256.yaml"
    config = OmegaConf.load(config_path)

    print('Loading and initializing models...')
    models['turncam'] = load_model_from_config(config, ckpt_path)
    models['carvekit'] = create_carvekit_interface()
    models['nsfw'] = StableDiffusionSafetyChecker.from_pretrained('CompVis/stable-diffusion-safety-checker').to(device)
    models['clip_fe'] = AutoFeatureExtractor.from_pretrained('CompVis/stable-diffusion-safety-checker')

    # Initialize LPIPS model separately as it's already a global variable
    global lpips_model
    lpips_model = lpips.LPIPS(net='alex').to(device)

def load_model_from_config(config, ckpt, verbose=False):
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

    # Use DataParallel to use multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

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

    with precision_scope('cuda'):
        # Check if model is DataParallel, access underlying model correctly
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([delta_elevation,
                              math.sin(delta_azimuth), math.cos(delta_azimuth),
                              (adjusted_radius / 100)]).to(c.device)

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
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()

def preprocess_image(models, input_im, preprocess):
    start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0

        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]

    print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    return input_im

def main_run(raw_im,
             models, device, elevation_first=0.0, azimuth_first=0.0, radius_first=0.0,
             elevation=0.0, azimuth=0.0, radius=0.0,
             preprocess=True,
             scale=3.0, n_samples=1, ddim_steps=50, ddim_eta=1.0,
             precision='fp32', h=256, w=256):
    input_im = preprocess_image(models, raw_im, preprocess)
    input_tensor = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_tensor = input_tensor * 2 - 1
    input_tensor = transforms.functional.resize(input_tensor, [h, w])

    # Check if model is DataParallel, access underlying model correctly
    turncam_model = models['turncam']
    if isinstance(turncam_model, torch.nn.DataParallel):
        turncam_model = turncam_model.module

    sampler = DDIMSampler(turncam_model)
    x_samples_ddim = sample_model(input_tensor, models['turncam'], sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta, elevation_first, azimuth_first, radius_first, elevation, azimuth, radius)

    lpips_model = lpips.LPIPS(net='alex').to(device)

    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_image = Image.fromarray(x_sample.astype(np.uint8))
        output_ims.append(output_image)

    return output_ims

def predict(cond_image_path: str = "cond.png",
            elevation_first_degree: float = 0.0,
            azimuth_first_degree: float = 0.0,
            radius_first: float = 0.0,
            elevation_in_degree: float = 0.0,
            azimuth_in_degree: float = 0.0,
            radius: float = 0.0,
            output_image_path: str = "output_azimuth_90.png"):
    cond_image = Image.open(cond_image_path)

    preds_images = main_run(raw_im=cond_image,
                            models=models, device=device,
                            elevation_first=elevation_first_degree,
                            azimuth_first=azimuth_first_degree,
                            radius_first=radius_first,
                            elevation=elevation_in_degree,
                            azimuth=azimuth_in_degree,
                            radius=radius)

    pred_image = preds_images[-1]
    pred_image.save(output_image_path)

def calculate_azimuth_elevation(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
    azimuth = np.degrees(y)
    elevation = np.degrees(x)
    return azimuth, elevation

def calculate_radius(T):
    radius = np.linalg.norm(T)
    return radius

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

def calculate_metrics(generated_img_path, preprocessed_corresponding_img_path):
    generated_img = Image.open(generated_img_path).convert('RGB')
    corresponding_img = Image.open(preprocessed_corresponding_img_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    generated_tensor = transform(generated_img).unsqueeze(0).to(device)
    corresponding_tensor = transform(corresponding_img).unsqueeze(0).to(device)
    
    ssim_value = ssim(generated_tensor, corresponding_tensor).item()
    lpips_value = lpips_model(generated_tensor, corresponding_tensor).item()
    
    print(f"SSIM: {ssim_value}, LPIPS: {lpips_value}")
    return ssim_value, lpips_value

class ChairDataset(Dataset):
    def __init__(self, dataset_root, annotations_file, subset_file='hydrant/set_lists.json', transform=None):
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
            samples[i][2] = os.path.join(self.dataset_root, samples[i][2])
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

        annotation = next((item for item in self.annotations if item['sequence_name'] == sequence_name and item['frame_number'] == frame_number), None)

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'sequence_name': sequence_name,
            'frame_number': frame_number,
            'annotation_image': annotation['image'],
            'annotation_view': annotation['viewpoint'],
            'annotation_mask': annotation['mask']
        }

def load_images_and_npy(folder_path):
    images = []
    npy_data = []

    # Iterate over the files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            # Load and process the image
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)
            images.append(image)

            # Assuming the corresponding .npy file has the same base name
            npy_path = os.path.join(folder_path, os.path.splitext(file_name)[0] + '.npy')
            if os.path.exists(npy_path):
                data = np.load(npy_path)
                npy_data.append(data)
    
    return images, npy_data

if __name__ == "__main__":
    load_and_initialize_models()
    inception_model = initialize_inception_v3(device=device)
    dataset_root = 'co3d/CO3DV2_DATASET_ROOT/'
    annotations_file = 'hydrant/frame_annotations.jgz'
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # chair_dataset = ChairDataset(dataset_root, annotations_file, transform=transform)
    # chair_dataloader = DataLoader(chair_dataset, batch_size=1, shuffle=False)

    folder_path = './syn_data'
    images, npy_data = load_images_and_npy(folder_path)

    cond_img = images[0]
    cond_ext = npy_data[0]

    print(cond_ext)

    real_features_list = []
    generated_features_list = []

    ssim_values = []
    lpips_values = []

    images.pop(0)
    npy_data.pop(0)

    # Evaluate the processed sequence
    for i, image_path in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}")

        RT_1 = cond_ext
        R_1 = RT_1[:, :3]  # Rotation matrix
        T_1 = RT_1[:, 3]   # Translation vector

        azimuth_first, elevation_first = calculate_azimuth_elevation(R_1)
        radius_first = calculate_radius(T_1)

        # Load the corresponding RT matrix
        target_img = images[i]
        RT = npy_data[i]
        R = RT[:, :3]  # Rotation matrix
        T = RT[:, 3]   # Translation vector

        azimuth, elevation = calculate_azimuth_elevation(R)
        radius = calculate_radius(T)

        print(f"Image: {image_path}")
        print(f"Azimuth: {azimuth}, Elevation: {elevation}, Radius: {radius}")

        cond_img_array = np.array(cond_img)
        target_img_array = np.array(target_img)

        output_image_path = f"output_2/output_{i+1}.png"
        predict(
            cond_image_path='./syn_data/000.png',
            elevation_first_degree=elevation_first,
            azimuth_first_degree=azimuth_first,
            radius_first=radius_first,
            elevation_in_degree=elevation,
            azimuth_in_degree=azimuth,
            radius=radius,
            output_image_path=output_image_path,
        )

        generated_img = Image.open(output_image_path).convert('RGB')
        corresponding_img_path = f'./syn_data/0{i:02}.png'
        corresponding_img = Image.open(corresponding_img_path).convert('RGB')

        preprocessed_corresponding_img = preprocess_image(models, corresponding_img, preprocess=True)
        preprocessed_corresponding_img_2 = Image.fromarray((preprocessed_corresponding_img * 255).astype('uint8')).convert('RGB')

        preprocessed_image_path = f"GT_2/preprocessed_{i+1}.png"
        preprocessed_corresponding_img_2.save(preprocessed_image_path)

        real_image_tensor = preprocess_image_for_inception_v3(corresponding_img)
        real_features = extract_features(inception_model, real_image_tensor)
        real_features_list.append(real_features.cpu().numpy())

        generated_image_tensor = preprocess_image_for_inception_v3(preprocessed_corresponding_img_2)
        generated_features = extract_features(inception_model, generated_image_tensor)
        generated_features_list.append(generated_features.cpu().numpy())

        ssim_value, lpips_value = calculate_metrics(output_image_path, preprocessed_image_path)
        ssim_values.append(ssim_value)
        lpips_values.append(lpips_value)

    ssim_mean = sum(ssim_values) / len(ssim_values) if ssim_values else 0
    lpips_mean = sum(lpips_values) / len(lpips_values) if lpips_values else 0
    print(f"Mean SSIM: {ssim_mean}, Mean LPIPS: {lpips_mean}")

    real_features_array = np.vstack(real_features_list)
    generated_features_array = np.vstack(generated_features_list)
    print(f"Real features array shape: {real_features_array.shape}")
    print(f"Generated features array shape: {generated_features_array.shape}")

    print(f"LENGTH Real features array shape: {len(real_features_array)}")
    print(f"LENGTH Generated features array shape: {len(generated_features_array)}")

    fid_score = calculate_fid(real_features_array, generated_features_array)
    print(f"FID Score: {fid_score}")
# Fine-Tuning Diffusion Models for Novel View Synthesis

**Computer Vision & Learning Group (CompVis)** <br />
Faculty of Mathematics, Computer Science, and Statistics <br />
Ludwig Maximilian University of Munich (LMU) <br />

## Overview
This repository contains the code and instructions associated with the Master's Thesis based on the original GitHub repository of [Zero-1-to-3](https://github.com/cvlab-columbia/zero123).  <br />

## Training Instructions

To train the model using the provided configuration:
1. Download the Stable Diffuion checkpoint:
   
```bash wget https://cv.cs.columbia.edu/zero123/assets/sd-image-conditioned-v2.ckpt ```

2.Download the Objaverse dataset:
```bash wget https://tri-ml-public.s3.amazonaws.com/datasets/views_release.tar.gz ```

3. Create a sharded dataset:
   ```bash
   python shard_creation.py
   ```

4. Download and unzip valid_paths.json.zip which is contained in [Zero-1-to-3](https://github.com/cvlab-columbia/zero123). Then place the the valid_paths.json file in the view_release folder.
  
5. Run the following command:

```bash
python main.py -t \\
    --base configs/sd-objaverse-finetune-c_concat-256.yaml \\
    --gpus 0,1,2,3 \\
    --scale_lr False \\
    --num_nodes 1 \\
    --seed 42 \\
    --check_val_every_n_epoch 10 \\
    --finetune_from sd-image-conditioned-v2.ckpt
```

### Explanation of Key Arguments:

- \`-t\`: Triggers the training mode.
- \`--base\`: Specifies the configuration file for the model (\`sd-objaverse-finetune-c_concat-256.yaml\`).
- \`--gpus\`: Specifies which GPUs to use (e.g., \`0,1,2,3\`).
- \`--scale_lr\`: Indicates whether to scale the learning rate by the number of GPUs (set to \`False\`).
- \`--num_nodes\`: Number of nodes to use (set to \`1\`).
- \`--seed\`: Random seed for reproducibility (set to \`42\`).
- \`--check_val_every_n_epoch\`: Specifies validation frequency (set to every \`10\` epochs).
- \`--finetune_from\`: Path to the pre-trained checkpoint (\`sd-image-conditioned-v2.ckpt\`).

## Evaluation on GSO Dataset

To evaluate the model on the **Google Scanned Objects (GSO)** dataset, follow these steps:

1. **Run the prediction pipeline:**

   ```bash
   python SYN_Object_Predict_Pipeline.py
   ```

2. **Evaluate using perceptual metrics:**

   ```bash
   python evaluate_perceptualism.py
   ```

3. **Obtain the FID score:**

   ```bash
   python -m pytorch_fid original generated
   ```

## Evaluation on CO3D Dataset

To evaluate the model on the **CO3D** dataset, proceed with the following steps:

1. **Run the object prediction pipeline:**

   ```bash
   python Predict_Object.py
   ```

2. **Evaluate using perceptual metrics:**

   ```bash
   python CO3D_evaluate_perceptualism.py
   ```

3. **Obtain the FID score:**

   ```bash
   python -m pytorch_fid original generated
   ```

## Acknowledgments

This project is based on the work from the [Zero-1-to-3 GitHub repository](https://github.com/cvlab-columbia/zero123).

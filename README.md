Fine-Tuning Diffusion Models for Novel View Synthesis
This repository contains the code and instructions associated with the Master's Thesis submitted to the Computer Vision & Learning Group (CompVis), Faculty of Mathematics, Computer Science, and Statistics, Ludwig Maximilian University of Munich (LMU).

Overview
This repository is based on the original GitHub repository of Zero-1-to-3.

Training Instructions
To train the model using the provided configuration, run the following command:

bash
Code kopieren
python main.py -t \
  --base configs/sd-objaverse-finetune-c_concat-256.yaml \
  --gpus 0,1,2,3 \
  --scale_lr False \
  --num_nodes 1 \
  --seed 42 \
  --check_val_every_n_epoch 10 \
  --finetune_from sd-image-conditioned-v2.ckpt
Explanation of Key Arguments:
-t: Triggers the training mode.
--base: Specifies the configuration file for the model (sd-objaverse-finetune-c_concat-256.yaml).
--gpus: Specifies which GPUs to use (e.g., 0,1,2,3).
--scale_lr: Indicates whether to scale the learning rate by the number of GPUs (set to False).
--num_nodes: Number of nodes to use (set to 1).
--seed: Random seed for reproducibility (set to 42).
--check_val_every_n_epoch: Specifies validation frequency (set to every 10 epochs).
--finetune_from: Path to the pre-trained checkpoint (sd-image-conditioned-v2.ckpt).
Evaluation on GSO Dataset
To evaluate the model on the Google Scanned Objects (GSO) dataset, follow these steps:

Run the prediction pipeline:

bash
Code kopieren
python SYN_Object_Predict_Pipeline.py
Evaluate using perceptual metrics:

bash
Code kopieren
python evaluate_perceptualism.py
Obtain the FID score:

bash
Code kopieren
python -m pytorch_fid original generated
Evaluation on CO3D Dataset
To evaluate the model on the CO3D dataset, proceed with the following steps:

Run the object prediction pipeline:

bash
Code kopieren
python Predict_Object.py
Evaluate using perceptual metrics:

bash
Code kopieren
python CO3D_evaluate_perceptualism.py
Obtain the FID score:

bash
Code kopieren
python -m pytorch_fid original generated
Acknowledgments
This project is based on the work from the Zero-1-to-3 GitHub repository.

Fine-Tuning Diffusion Models for Novel View Synthesis

This is the repository to the Master's Thesis.

Computer Vision & Learning Group (CompVis)
Faculty of Mathematics, Computer Science, and Statistics
Ludwig Maximilian University M¨unchen

This repository is based on the original Github of Zero-1-to-3: https://github.com/cvlab-columbia/zero123.

To train a model:

python main.py \
    -t \
    --base configs/sd-objaverse-finetune-c_concat-256.yaml \
    --gpus 0,1,2,3 \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --finetune_from sd-image-conditioned-v2.ckpt

To evaluate the trained model on the GSO dataset:
python SYN_Object_Predict_Pipeline.py

and then

python evaluate_perceptualism.py

and to obtain the FID score:

python -m pytorch_fid original generated


To evaluate the trained model on the CO3D dataset:
python Predict_Object.py

and then

python CO3D_evaluate_perceptualism.py

and to obtain the FID score:

python -m pytorch_fid original generated

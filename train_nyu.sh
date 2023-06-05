CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python ./ovo/scripts/train_ovo.py \
dataset=NYU \
NYU_root=/path/to/NYU_dataset/depthbin/ \
NYU_preprocess_root=/path/to/nyu_preprocess_ov \
NYU_prepare_total=/path/to/nyu_preprocess_total \
logdir=./outputs \
n_gpus=8 batch_size=8
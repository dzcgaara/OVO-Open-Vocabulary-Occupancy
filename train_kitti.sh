CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python ./ovo/scripts/train_ovo.py \
dataset=kitti \
kitti_root=/path/to/kitti_dataset/ \
kitti_preprocess_root=/path/to/kitti_preprocess_ov/ \
kitti_prepare_total=/path/to/kitti_preprocess_total \
logdir=./outputs \
n_gpus=8 batch_size=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python ovo/scripts/infer_ovo.py \
dataset=kitti \
kitti_root=/path/to/kitti_dataset/ \
kitti_preprocess_root=/path/to/kitti_preprocess_ori/ \
+word_path=ovo/prompt_embedding/kitti_prompt_embedding.json \
+model_path=/path/to/model_file/last.ckpt \
+output_path=/path/to/visualization_file/ \
+novel_class_lbl=[1,9,13] \
+target_lbl=9 \
n_gpus=1 batch_size=1 \
vis=True miou=True
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python ovo/scripts/infer_ovo.py \
dataset=NYU \
NYU_root=/path/to/NYU_dataset/depthbin/ \
NYU_preprocess_root=/path/to/nyu_preprocess_ori \
+word_path=ovo/prompt_embedding/nyu_prompt_embedding.json \
+model_path=/path/to/model_file/last.ckpt \
+output_path=/data/visualization_file/ \
+novel_class_lbl=[6,8,11] \
+target_lbl=11 \
n_gpus=1 batch_size=1 \
vis=True miou=True
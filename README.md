# Open-Vocabulary Occupancy

This repo is the official implementation of: [OVO: Open-Vocabulary Occupancy](https://arxiv.org/abs/2305.16133)

> OVO: Open-Vocabulary Occupancy
> 
> Zhiyu Tan*, Zichao Dong*, Cheng Zhang, Weikun Zhang, Hang Ji, Hao Li $\dagger$ 

## Introduction

Open Vocabulary Occupancy (OVO) is a novel approach that enables semantic occupancy prediction for arbitrary classes without the need for 3D annotations during training. The key components of our approach are: (1) knowledge distillation from a pre-trained 2D open-vocabulary segmentation model to the 3D occupancy network, and (2) pixel-voxel filtering for generating high-quality training data. The resulting framework is simple, compact, and compatible with most state-of-the-art semantic occupancy prediction models. On the NYUv2 and SemanticKITTI datasets, OVO achieves competitive performance compared to supervised semantic occupancy prediction approaches. Additionally, extensive analyses and ablation studies are conducted to provide insights into the design of the proposed framework.

OVO enables knowledge distillation from a pre-trained 2D open-vocabulary segmentation model to the 3D occupancy network. We also propose a simple yet effective voxel filtering mechanism for high-quality training data selection. The whole pipeline is trained end-to-end and only the parameters of the 3D occupancy network will be updated. Red dashed arrows indicate the backward pass for the three feature alignment losses. During inference, text embeddings (bottom row) of both base and novel categories can be used to predict the semantic label for each voxel.

![](assets/model.png)

## Preparing OVO

**Installation**

1. Create conda environment:
   
   ```
   $ conda create -y -n ovo python=3.7
   $ conda activate ovo
   ```

2. Install pytorch:
   
   ```
   $ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
   ```

3. Install the additional dependencies:
   
   ```
   $ pip install -r requirements.txt
   ```

4. Install tbb:
   
   ```
   $ conda install -c bioconda tbb=2020.2
   ```

5. Downgrade torchmetrics to 0.6.0:
   
   ```
   $ pip install torchmetrics==0.6.0
   ```

6. Finally, install OVO:
   
   ```
   $ pip install -e .
   ```

**Data Preprocess**

1. Generate LSeg embedding.

   Refer to [CLIP](https://github.com/openai/CLIP) and [get_img_embedding.py](tools/get_img_embedding.py).

2. Generate prompt embedding.

   Refer to [CLIP](https://github.com/openai/CLIP) and [get_prompt_embedding.py](tools/get_prompt_embedding.py). 
   
   Or directly use [prompt_embeddings](tools/prompt_embedding) offered in this repository.
3. Label preprocess.

   **NYUv2 ov labels** (Used for training):
      
      Change `seg_class_map` in ovo/data/NYU/preprocess_ov.py

      In this repository we offer an example  of merging 'bed', 'table' and '
      other' into 'other'.
      ```shell
      python ovo/data/NYU/preprocess_ov.py NYU_root=/path/to/NYU_dataset/depthbin/ NYU_preprocess_root=/path/to/nyu_preprocess_ov
      ```
   **SemanticKITTI ov labels** (Used for training):
      
      Change `learning_map_inv` in ovo/data/semantic_kitti/semantic-kitti.yaml

      In this repository we offer an example  of merging 'car', 'road' and '
      building' into 'road'.

      ```shell
      python ovo/data/semantic_kitti/preprocess_ov.py kitti_root=/path/to/kitti_dataset/ kitti_preprocess_root=/path/to/kitti_preprocess_ov
      ```
   
   **NYUv2 ori labels** (Used for inference):
      ```shell
      python ovo/data/NYU/preprocess_ori.py NYU_root=/path/to/NYU_dataset/depthbin/ NYU_preprocess_root=/path/to/nyu_preprocess_ori
      ```
   **SemanticKITTI ori labels** (Used for inference):
      ```shell
      python ovo/data/semantic_kitti/preprocess_ov.py kitti_root=/path/to/kitti_dataset/ kitti_preprocess_root=/path/to/kitti_preprocess_ori
      ```

4. Occlusion preprocess.

   ```shell
   python ovo/occlusion_preprocess/find_occ_pairs_kitti.py /path/to/kitti_preprocess_ov
   ```

   ```shell
   python ovo/occlusion_preprocess/find_occ_pairs_nyu.py /path/to/nyu_preprocess_ov/base/NYUtrain/
   ```

5. Voxel selection.

   Filling the path parameters in ovo/data/NYU/nyu_valid_pairs.py
   ```shell
   python ovo/data/NYU/nyu_valid_pairs.py
   ```

   Filling the path parameters in ovo/data/NYU/nyu_valid_pairs.py
   ```shell
   python ovo/data/semantic_kitti/kitti_valid_pairs.py
   ```

6. Integrate all pre-processed data.
      
      Filling the path parameters in ovo/data/NYU/prepare_total.py
      ```shell
      python ovo/data/NYU/prepare_total.py
      ```

      Filling the path parameters in ovo/data/semantic_kitti/prepare_total.py
      ```shell
      python ovo/data/semantic_kitti/prepare_total.py
      ```

## Training OVO

**NYUv2**

```shell
# train_nyu.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python ./ovo/scripts/train_ovo.py \
dataset=NYU \
NYU_root=/path/to/NYU_dataset/depthbin/ \
NYU_preprocess_root=/path/to/nyu_preprocess_ov \
NYU_prepare_total=/path/to/nyu_preprocess_total \
logdir=./outputs \
n_gpus=8 batch_size=8
```

**SemanticKITTI**

```shell
# train_kitti.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python ./ovo/scripts/train_ovo.py \
dataset=kitti \
kitti_root=/path/to/kitti_dataset/ \
kitti_preprocess_root=/path/to/kitti_preprocess_ov/ \
kitti_prepare_total=/path/to/kitti_preprocess_total \
logdir=./outputs \
n_gpus=8 batch_size=8
```

## Inference

**NYUv2**

```shell
# infer_nyu.sh
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
```

**SemanticKITTI**

```shell
# infer_kitti.sh
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
```

## Visualization

Refer to [MonoScene visualization](https://github.com/astra-vision/MonoScene#visualization)

Filling the path parameters in ovo/scripts/visualization/nyu_vis.py

```shell
python ovo/scripts/visualization/nyu_vis.py
```

Filling the path parameters in ovo/scripts/visualization/kitti_vis.py

```shell
python ovo/scripts/visualization/kitti_vis.py
```

## Main results

**NYUv2**

| Method               | Input | bed   | table | other | mean  | ceiling | floor | wall  | window | chair | sofa  | tvs   | furniture | mean  |
| -------------------- | ----- | ----- | ----- | ----- | ----- | ------- | ----- | ----- | ------ | ----- | ----- | ----- | --------- | ----- |
| **Fully-supervised** |       |       |       |       |       |         |       |       |        |       |       |       |           |       |
| AICNet               | C, D  | 35.87 | 11.11 | 6.45  | 17.81 | 7.58    | 82.97 | 9.15  | 0.05   | 6.93  | 22.92 | 0.71  | 15.90     | 18.28 |
| SSCNet               | C, D  | 32.10 | 13.00 | 10.10 | 18.40 | 15.10   | 94.70 | 24.40 | 0.00   | 12.60 | 35.0  | 7.80  | 27.10     | 27.10 |
| 3DSketch             | C     | 42.29 | 13.88 | 8.19  | 21.45 | 8.53    | 90.45 | 9.94  | 5.67   | 10.64 | 29.21 | 9.38  | 23.83     | 23.46 |
| MonoScene            | C     | 48.19 | 15.13 | 12.94 | 25.42 | 8.89    | 93.50 | 12.06 | 12.57  | 13.72 | 36.11 | 15.22 | 27.96     | 27.50 |
| **Zero-shot**        |       |       |       |       |       |         |       |       |        |       |       |       |           |       |
| MonoScene*           | C     | --    | --    | --    | --    | 8.10    | 93.49 | 9.94  | 10.32  | 13.24 | 34.47 | 11.75 | 26.41     | 25.96 |
| ours                 | C     | 41.61 | 10.45 | 8.39  | 20.15 | 7.77    | 93.16 | 7.77  | 6.95   | 10.01 | 33.83 | 8.22  | 25.64     | 24.17 |

**SemanticKITTI**

| Method               | Input      | car  | road | building | mean | sidewalk | parking | other ground | truck | bicycle | motorcycle | other vehicle | vegetation | trunk | terrain | person | bicyclist | motorcyclist | fence | pole | traffic sign | mean |
| -------------------- | ---------- | ---- | ---- | -------- | ---- | -------- | ------- | ------------ | ----- | ------- | ---------- | ------------- | ---------- | ----- | ------- | ------ | --------- | ------------ | ----- | ---- | ------------ | ---- |
| **Fully-supervised** |            |      |      |          |      |          |         |              |       |         |            |               |            |       |         |        |           |              |       |      |              |      |
| AICNet               | C, D       | 15.3 | 39.3 | 9.6      | 21.4 | 18.3     | 19.8    | 1.6          | 0.7   | 0.0     | 0.0        | 0.0           | 9.6        | 1.9   | 13.5    | 0.0    | 0.0       | 0.0          | 5.0   | 0.1  | 0.0          | 4.4  |
| 3DSketch             | C $\dagger$ | 17.1 | 37.7 | 12.1     | 22.3 | 19.8     | 0.0     | 0.0          | 0.0   | 0.0     | 0.0        | 0.0           | 12.1       | 0.0   | 16.1    | 0.0    | 0.0       | 0.0          | 3.4   | 0.0  | 0.0          | 3.2  |
| MonoScene            | C          | 18.8 | 54.7 | 14.4     | 29.3 | 27.1     | 24.8    | 5.7          | 3.3   | 0.5     | 0.7        | 4.4           | 14.9       | 2.4   | 19.5    | 1.0    | 1.4       | 0.4          | 11.1  | 3.3  | 2.1          | 7.7  |
| TPVFormer            | C×6        | 23.8 | 56.5 | 13.9     | 31.4 | 25.9     | 20.6    | 0.9          | 8.1   | 0.4     | 0.1        | 4.4           | 16.9       | 2.3   | 30.4    | 0.5    | 0.9       | 0.0          | 5.9   | 3.1  | 1.5          | 7.6  |
| **Zero-shot**        |            |      |      |          |      |          |         |              |       |         |            |               |            |       |         |        |           |              |       |      |              |      |
| ours                 | C          | 13.3 | 53.9 | 9.7      | 25.7 | 26.5     | 14.4    | 0.1          | 0.7   | 0.4     | 0.3        | 2.5           | 17.2       | 2.3   | 29.0    | 0.6    | 0.7       | 0.0          | 5.4   | 3.0  | 1.7          | 6.6  |


## Related projects

Our code is based on [MonoScene](https://github.com/astra-vision/MonoScene). Many thanks to the authors for their great work.


## Citation

If you find this project helpful, please consider citing the following paper:

```
@misc{tan2023ovo,
      title={OVO: Open-Vocabulary Occupancy}, 
      author={Zhiyu Tan and Zichao Dong and Cheng Zhang and Weikun Zhang and Hang Ji and Hao Li},
      year={2023},
      eprint={2305.16133},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import pickle
from PIL import Image
from torchvision import transforms
from monoscene.data.utils.helpers import (
    vox2pix,
    compute_local_frustums,
    compute_CP_mega_matrix,
)
import json
from tqdm import tqdm


class NYUDataset(Dataset):
    def __init__(
        self,
        split,
        root,
        preprocess_root,
        occ_root,
        n_relations=4,
        color_jitter=None,
        frustum_size=4,
        fliplr=0.0,
    ):
        self.n_relations = n_relations
        self.frustum_size = frustum_size
        self.n_classes = 12
        self.root = os.path.join(root, "depthbin", "NYU" + split)
        self.preprocess_root = preprocess_root
        self.occ_root = occ_root
        self.base_dir = os.path.join(preprocess_root, "base", "NYU" + split)
        self.mask_dir = os.path.join(
            preprocess_root, "nyu_masked", "NYU" + split)

        self.clip_gt_dir = "/data/lseg_embedding_nyu"

        self.fliplr = 0.0

        self.voxel_size = 0.08  # 0.08m
        self.scene_size = (4.8, 4.8, 2.88)  # (4.8m, 4.8m, 2.88m)
        self.num_max_instances = 50
        self.img_W = 640
        self.img_H = 480
        self.cam_k = np.array(
            [[518.8579, 0, 320], [0, 518.8579, 240], [0, 0, 1]])

        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )

        self.scan_names = glob.glob(os.path.join(self.root, "*.bin"))

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, index):
        file_path = self.scan_names[index]
        filename = os.path.basename(file_path)
        name = filename[:-4]

        os.makedirs(self.base_dir, exist_ok=True)
        filepath = os.path.join(self.base_dir, name + ".pkl")
        mask_filepath = os.path.join(self.mask_dir, name + ".pkl")
        occ_filepath = os.path.join(self.occ_root, name + ".pkl")
        with open(filepath, "rb") as handle:
            data = pickle.load(handle)
        with open(occ_filepath, "rb") as handle:
            occ = pickle.load(handle)
        data["occ"] = occ["occ_vis"]
        # with open(mask_filepath, "rb") as handle:
        #     mask_data = pickle.load(handle)

        clip_gt_path = os.path.join(self.clip_gt_dir, name + ".pkl")
        with open(clip_gt_path, "rb") as handle:
            clip_gt = pickle.load(handle)

        cam_pose = data["cam_pose"]
        T_world_2_cam = np.linalg.inv(cam_pose)
        vox_origin = data["voxel_origin"]
        data["cam_k"] = self.cam_k
        target = data[
            "target_1_4"
        ]  # Following SSC literature, the output resolution on NYUv2 is set to 1:4
        data["target"] = target

        target_1_4 = data["target_1_16"]

        CP_mega_matrix = compute_CP_mega_matrix(
            target_1_4, is_binary=self.n_relations == 2
        )
        data["CP_mega_matrix"] = CP_mega_matrix

        # compute the 3D-2D mapping
        projected_pix, fov_mask, pix_z = vox2pix(
            T_world_2_cam,
            self.cam_k,
            vox_origin,
            self.voxel_size,
            self.img_W,
            self.img_H,
            self.scene_size,
        )

        data["projected_pix_1"] = projected_pix
        data["fov_mask_1"] = fov_mask

        # compute the masks, each indicates voxels inside a frustum
        frustums_masks, frustums_class_dists = compute_local_frustums(
            projected_pix,
            pix_z,
            target,
            self.img_W,
            self.img_H,
            dataset="NYU",
            n_classes=12,
            size=self.frustum_size,
        )
        data["frustums_masks"] = frustums_masks
        data["frustums_class_dists"] = frustums_class_dists

        rgb_path = os.path.join(self.root, name + "_color.jpg")
        img = Image.open(rgb_path).convert("RGB")

        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)

        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0

        # randomly fliplr the image
        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            data["projected_pix_1"][:, 0] = (
                img.shape[1] - 1 - data["projected_pix_1"][:, 0]
            )

        data["img"] = self.normalize_rgb(img)  # (3, img_H, img_W)
        data["lseg_feat"] = clip_gt["feat"]

        return data

    def __len__(self):
        return len(self.scan_names)

    @staticmethod
    def read_calib(calib_path):
        """
        Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out


def preprocess_valid_pairs(naive):

    if not naive:
        with open(clip_text_gt_feat_path, "r") as f:
            info = json.load(f)
        k_word_tokens = []
        for k in info:
            k_word_tokens.append(torch.Tensor(info[k]).unsqueeze(0))
        k_word_tokens = torch.cat(k_word_tokens)

    train_ds = NYUDataset(
        split="train",
        root=nyu_root,
        preprocess_root=nyu_preprocess_root,
        occ_root=nyu_occ_root,
        frustum_size=8,
        fliplr=0.0,
        color_jitter=(0.4, 0.4, 0.4),
    )

    total_pair_info = {}

    for data in tqdm(train_ds):
        valid_pairs = []
        voxel2pixel = data["projected_pix_1"]
        gt = data["target"]
        occ = data["occ"]
        img_shape = [480, 640]

        idx = 0

        if not naive:
            lseg_feat = torch.tensor(data['lseg_feat'][0])
            lseg_clip_similirty = torch.einsum(
                "kd,dwh->kwh", k_word_tokens, lseg_feat)
            lseg_lbl = torch.argmax(lseg_clip_similirty, dim=0)

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(gt.shape[2]):
                    v, u = voxel2pixel[idx]
                    idx += 1
                    if gt[i, j, k] == 0 or gt[i, j, k] == 255:
                        continue

                    if u < img_shape[0] and u > 0 and v < img_shape[1] and v > 0:
                        if naive:
                            tmp_pair = [(int(u), int(v)), (int(
                                i), int(j), int(k)), int(gt[i, j, k])]
                            valid_pairs.append(tmp_pair)
                        else:
                            un_occlusion = occ[i, j, k] == 2
                            eq = lseg_lbl[u//6, v//8] == gt[i, j, k]
                            approx_eq = (
                                lseg_lbl[u//6, v//8] in novel_class) and (gt[i, j, k] in novel_class)
                            if un_occlusion and (eq or approx_eq):
                                tmp_pair = [(int(u), int(v)), (int(
                                    i), int(j), int(k)), int(gt[i, j, k])]
                                valid_pairs.append(tmp_pair)

        key_name = data["name"]

        total_pair_info[key_name] = valid_pairs
    if naive:
        with open("./nyu_naive_valid_pairs.json", "w") as f:
            json.dump(total_pair_info, f)
    else:
        with open("./nyu_valid_pairs.json", "w") as f:
            json.dump(total_pair_info, f)


if __name__ == "__main__":
    novel_class = []  # [6, 8, 11]
    nyu_root = ""  # /path/to/NYU_dataset
    nyu_preprocess_root = ""  # /path/to/nyu_preprocess_ori
    nyu_occ_root = ""  # /path/to/nyu_occ_reslut
    clip_text_gt_feat_path = ""  # /path/to/nyu_prompt_embedding.json
    preprocess_valid_pairs(naive=False)

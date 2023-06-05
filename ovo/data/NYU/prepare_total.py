import os
from torch.utils.data import Dataset
import numpy as np
import pickle
import json
import tqdm


def read_json(file_path="./nyu_valid_pairs.json"):
    with open(file_path, "r") as f:
        info = json.load(f)
    return info


def get_clip_word_embedding(file_path):
    with open(file_path, "r") as f:
        info = json.load(f)

    clsidx2emd = {}
    idx = 0
    helper = []
    for k in info.keys():
        clsidx2emd[idx] = np.array(info[k])
        idx += 1
    return clsidx2emd


def get_cos_similar(v1: list, v2: list):
    num = float(np.dot(v1, v2))
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


def convert_idx(i, j, k):
    return i*36*60 + j * 60 + k


def process_one_frame(file_name, valid_info):
    valid_img_idx = []
    valid_img_feat = []
    valid_img_weight = []

    valid_word_idx = []
    valid_word_feat = []
    valid_word_lbl = []

    total_ret = {}

    for i in valid_info:
        uv, xyz, lbl = i
        x, y, z = xyz
        u, v = uv

        # get lseg embedding from file
        path = os.path.join(lseg_root, file_name + ".pkl")
        with open(path, "rb") as f:
            lseg_emd = pickle.load(f)

        # process image first
        # first fill img_idx
        valid_img_idx.append(convert_idx(x, y, z))
        # second fill img_feat
        valid_img_feat.append(lseg_emd["feat"][:, :, u // 6, v // 8])
        # then fill img_weight
        weight = get_cos_similar(
            clip_word_embedding[lbl], lseg_emd["feat"][0, :, u // 6, v // 8])
        valid_img_weight.append(weight)

        # process word now
        # do not mimic novel class
        if lbl in [6, 8, 11]:
            continue

        valid_word_idx.append(convert_idx(x, y, z))
        valid_word_feat.append(clip_word_embedding[lbl][np.newaxis, :])
        valid_word_lbl.append(lbl)

    if len(valid_word_idx) == 0:
        # hack logic
        valid_word_idx.append(convert_idx(0, 0, 0))
        valid_word_feat.append(clip_word_embedding[0][np.newaxis, :])
        valid_word_lbl.append(0)

    if len(valid_img_idx) == 0:
        # hack logic
        valid_img_idx.append(convert_idx(0, 0, 0))
        valid_img_feat.append(lseg_emd["feat"][:, :, 0, 0])
        valid_img_weight.append(0)

    total_ret["lseg_2d"] = lseg_emd["feat"]
    total_ret["valid_img_idx"] = np.array(valid_img_idx)
    total_ret["valid_img_weight"] = np.array(valid_img_weight)
    total_ret["valid_img_feat"] = np.concatenate(valid_img_feat, axis=0)

    total_ret["valid_word_idx"] = np.array(valid_word_idx)
    total_ret["valid_word_lbl"] = np.array(valid_word_lbl)
    total_ret["valid_word_feat"] = np.concatenate(valid_word_feat, axis=0)

    return total_ret


if __name__ == "__main__":

    # NOTE naive: w/o occlusion judge & label consistency
    naive = False
    ######

    # step 1. feed in input
    if not naive:
        info = read_json("./nyu_valid_pairs.json")
    else:
        info = read_json("./nyu_naive_valid_pairs.json")
    # feed in clip word embedding
    clip_word_embedding = get_clip_word_embedding(
        "")  # /path/to/nyu_prompt_embedding.json
    # set lseg root path
    lseg_root = ""  # /path/to/lseg_embedding_nyu

    # step2. traverse and process each frame
    if not naive:
        save_root = ""  # /path/to/save_path
    else:
        save_root = ""  # /path/to/naive_save_path

    for k in tqdm.tqdm(info.keys()):
        total_ret = process_one_frame(k, info[k])
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        with open(os.path.join(save_root, k + ".pkl"), "wb") as handle:
            pickle.dump(total_ret, handle)
            print("wrote ", k)

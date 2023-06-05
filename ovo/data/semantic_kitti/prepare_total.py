import os
import numpy as np
import pickle
import json
import tqdm
import random


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
    # return i*36*60 + j * 60 + k
    return i*256*32 + j * 32 + k


def process_one_frame(sequence, frame_id, valid_info):
    valid_img_idx = []
    valid_img_feat = []
    valid_img_weight = []
    valid_img_class_balance_weight = []

    valid_word_idx = []
    valid_word_feat = []
    valid_word_lbl = []

    total_ret = {}

    balance_weight = [1.0 for i in range(20)]
    balance_weight[1] = 10
    balance_weight[9] = 1.7
    balance_weight[13] = 4.25

    # get lseg embedding from file
    path = os.path.join(lseg_root, sequence, frame_id + ".pkl")
    # print(path)
    with open(path, "rb") as f:
        lseg_emd = pickle.load(f)

    for i in tqdm.tqdm(valid_info):
        uv, xyz, lbl = i
        if mini:
            if lbl not in [1, 9, 13]:
                # continue
                seed = random.randint(1, 10)
                if seed > 3:
                    continue

        x, y, z = xyz
        u, v = uv

        # process image first
        # first fill img_idx
        valid_img_idx.append(convert_idx(x, y, z))
        # second fill img_feat
        valid_img_feat.append(lseg_emd["feat"][:, :, int(
            u//(370/60)), int(v//(1220/180))].astype("float32"))

        # then fill img_weight
        weight = get_cos_similar(clip_word_embedding[lbl], lseg_emd["feat"][0, :, int(
            u//(370/60)), int(v//(1220/180))])
        valid_img_weight.append(weight)

        class_balance_weight = balance_weight[lbl]
        valid_img_class_balance_weight.append(class_balance_weight)

        # process word now
        # do not mimic novel class
        if lbl in [1, 9, 13]:
            continue

        valid_word_idx.append(convert_idx(x, y, z))
        valid_word_feat.append(
            clip_word_embedding[lbl][np.newaxis, :].astype("float32"))
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
        valid_img_class_balance_weight.append(0)

    # for end2end only:
    # total_ret["lseg_2d"] = lseg_emd["feat"]

    total_ret["valid_img_idx"] = np.array(valid_img_idx)
    total_ret["valid_img_weight"] = np.array(valid_img_weight)
    total_ret["valid_img_feat"] = np.concatenate(valid_img_feat, axis=0)
    # print(valid_img_weight)
    # print(valid_img_class_balance_weight)
    # exit()
    total_ret["valid_img_class_balance_weight"] = np.array(
        valid_img_class_balance_weight)
    print(total_ret["valid_img_class_balance_weight"].shape)

    # for voxel-word align only:
    total_ret["valid_word_idx"] = np.array(valid_word_idx)
    total_ret["valid_word_lbl"] = np.array(valid_word_lbl)
    total_ret["valid_word_feat"] = np.concatenate(valid_word_feat, axis=0)

    return total_ret


if __name__ == "__main__":
    # NOTE naive: w/o occlusion judge & label consistency
    naive = False
    # NOTE mini: filter out 70% valid pairs (not in novel class)
    mini = True

    # step 1. feed in input
    if not naive:
        json_path = ""  # path to naive small json file
        info = read_json(json_path)
        print(json_path)
    else:
        json_path = ""  # path to small json file
        info = read_json(json_path)
        print(json_path)
    # feed in clip word embedding
    clip_word_embedding = get_clip_word_embedding(
        "")  # /path/to/kitti_prompt_embedding.json
    # set lseg root path
    lseg_root = ""  # /path/to/lseg_embedding_kitti

    # step2. traverse and process each frame
    if not naive:
        save_root = ""  # /path/to/kitti_preprocess_total_occ
        print(save_root)
    else:
        save_root = ""  # /path/to/naive_save_path

    if mini:
        save_root = save_root+"_mini_30"

    for k in tqdm.tqdm(info.keys()):
        sequence = k.split('_')[0]
        frame_id = k.split('_')[1]

        total_ret = process_one_frame(sequence, frame_id, info[k])
        if not os.path.exists(os.path.join(save_root, sequence)):
            os.makedirs(os.path.join(save_root, sequence))
        with open(os.path.join(save_root, sequence, frame_id + ".pkl"), "wb") as handle:
            pickle.dump(total_ret, handle)
            print("wrote ", k)

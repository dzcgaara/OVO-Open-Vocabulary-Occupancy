import torch


def collate_fn_train(batch):
    data = {}
    imgs = []
    targets = []

    names = []
    cam_poses = []

    vox_origins = []
    cam_ks = []

    # NOTE that below attributes are new added in dataset, distribute and send them here
    # data["valid_img_idx"] = all_prep_data["valid_img_idx"]
    # data["valid_img_weight"] = all_prep_data["valid_img_weight"]
    # data["valid_img_feat"] = all_prep_data["valid_img_feat"]

    # data["valid_word_idx"] = all_prep_data["valid_word_idx"]
    # data["valid_word_lbl"] = all_prep_data["valid_word_lbl"]
    # data["valid_word_feat"] = all_prep_data["valid_word_feat"]

    valid_img_idxs = []
    valid_img_weights = []
    valid_img_feats = []
    valid_word_idxs = []
    valid_word_lbls = []
    valid_word_feats = []
    lseg_2ds = []

    CP_mega_matrices = []

    data["projected_pix_1"] = []
    data["fov_mask_1"] = []
    data["frustums_masks"] = []
    data["frustums_class_dists"] = []

    for idx, input_dict in enumerate(batch):
        CP_mega_matrices.append(torch.from_numpy(input_dict["CP_mega_matrix"]))
        for key in data:
            if key in input_dict:
                data[key].append(torch.from_numpy(input_dict[key]))

        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).double())
        cam_poses.append(torch.from_numpy(input_dict["cam_pose"]).float())
        vox_origins.append(torch.from_numpy(
            input_dict["voxel_origin"]).double())

        names.append(input_dict["name"])

        img = input_dict["img"]
        imgs.append(img)

        target = torch.from_numpy(input_dict["target"])
        targets.append(target)

        # NOTE only for train
        valid_img_idx = torch.from_numpy(input_dict["valid_img_idx"])
        valid_img_idxs.append(valid_img_idx)

        valid_img_weight = torch.from_numpy(input_dict["valid_img_weight"])
        valid_img_weights.append(valid_img_weight)

        valid_img_feat = torch.from_numpy(input_dict["valid_img_feat"])
        valid_img_feats.append(valid_img_feat)

        valid_word_idx = torch.from_numpy(input_dict["valid_word_idx"])
        valid_word_idxs.append(valid_word_idx)

        valid_word_lbl = torch.from_numpy(input_dict["valid_word_lbl"])
        valid_word_lbls.append(valid_word_lbl)

        valid_word_feat = torch.from_numpy(input_dict["valid_word_feat"])
        valid_word_feats.append(valid_word_feat)

        lseg_2d = torch.from_numpy(input_dict["lseg_2d"])
        lseg_2ds.append(lseg_2d)

    ret_data = {
        "CP_mega_matrices": CP_mega_matrices,
        "cam_pose": torch.stack(cam_poses),
        "cam_k": torch.stack(cam_ks),
        "vox_origin": torch.stack(vox_origins),
        "name": names,
        "img": torch.stack(imgs),
        "target": torch.stack(targets),
        "lseg_2d": torch.stack(lseg_2ds),
        "valid_img_idx": torch.stack(valid_img_idxs),
        "valid_img_weight": torch.stack(valid_img_weights),
        "valid_img_feat": torch.stack(valid_img_feats),
        "valid_word_idx": torch.stack(valid_word_idxs),
        "valid_word_lbl": torch.stack(valid_word_lbls),
        "valid_word_feat": torch.stack(valid_word_feats)
    }
    for key in data:
        ret_data[key] = data[key]
    return ret_data


def collate_fn_test(batch):
    data = {}
    imgs = []
    targets = []

    names = []
    cam_poses = []

    vox_origins = []
    cam_ks = []

    CP_mega_matrices = []

    data["projected_pix_1"] = []
    data["fov_mask_1"] = []
    data["frustums_masks"] = []
    data["frustums_class_dists"] = []

    for idx, input_dict in enumerate(batch):
        CP_mega_matrices.append(torch.from_numpy(input_dict["CP_mega_matrix"]))
        for key in data:
            if key in input_dict:
                data[key].append(torch.from_numpy(input_dict[key]))

        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).double())
        cam_poses.append(torch.from_numpy(input_dict["cam_pose"]).float())
        vox_origins.append(torch.from_numpy(
            input_dict["voxel_origin"]).double())

        names.append(input_dict["name"])

        img = input_dict["img"]
        imgs.append(img)

        target = torch.from_numpy(input_dict["target"])
        targets.append(target)

    ret_data = {
        "CP_mega_matrices": CP_mega_matrices,
        "cam_pose": torch.stack(cam_poses),
        "cam_k": torch.stack(cam_ks),
        "vox_origin": torch.stack(vox_origins),
        "name": names,
        "img": torch.stack(imgs),
        "target": torch.stack(targets),
    }
    for key in data:
        ret_data[key] = data[key]
    return ret_data

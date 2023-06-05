import torch


def collate_fn_train(batch):
    data = {}
    imgs = []
    CP_mega_matrices = []
    targets = []
    frame_ids = []
    sequences = []

    cam_ks = []
    T_velo_2_cams = []
    frustums_masks = []
    frustums_class_dists = []

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
    # lseg_2ds = []

    scale_3ds = batch[0]["scale_3ds"]
    for scale_3d in scale_3ds:
        data["projected_pix_{}".format(scale_3d)] = []
        data["fov_mask_{}".format(scale_3d)] = []

    for idx, input_dict in enumerate(batch):
        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).double())
        T_velo_2_cams.append(torch.from_numpy(
            input_dict["T_velo_2_cam"]).float())

        if "frustums_masks" in input_dict:
            frustums_masks.append(torch.from_numpy(
                input_dict["frustums_masks"]))
            frustums_class_dists.append(
                torch.from_numpy(input_dict["frustums_class_dists"]).float()
            )

        for key in data:
            data[key].append(torch.from_numpy(input_dict[key]))

        img = input_dict["img"]
        imgs.append(img)

        frame_ids.append(input_dict["frame_id"])
        sequences.append(input_dict["sequence"])

        target = torch.from_numpy(input_dict["target"])
        targets.append(target)
        CP_mega_matrices.append(torch.from_numpy(input_dict["CP_mega_matrix"]))

        # NOTE only for train
        valid_img_idx = torch.from_numpy(input_dict["valid_img_idx"])
        valid_img_idxs.append(valid_img_idx)

        valid_img_weight = torch.from_numpy(input_dict["valid_img_weight"])
        valid_img_weights.append(valid_img_weight)

        valid_img_feat = torch.from_numpy(input_dict["valid_img_feat"])
        valid_img_feats.append(valid_img_feat)

        # for voxel-word align only
        valid_word_idx = torch.from_numpy(input_dict["valid_word_idx"])
        valid_word_idxs.append(valid_word_idx)

        valid_word_lbl = torch.from_numpy(input_dict["valid_word_lbl"])
        valid_word_lbls.append(valid_word_lbl)

        valid_word_feat = torch.from_numpy(input_dict["valid_word_feat"])
        valid_word_feats.append(valid_word_feat)

        # for end2end only (2d align)
        # lseg_2d = torch.from_numpy(input_dict["lseg_2d"])
        # lseg_2ds.append(lseg_2d)

    ret_data = {
        "frame_id": frame_ids,
        "sequence": sequences,
        "frustums_class_dists": frustums_class_dists,
        "frustums_masks": frustums_masks,
        "cam_k": cam_ks,
        "T_velo_2_cam": T_velo_2_cams,
        "img": torch.stack(imgs),
        "CP_mega_matrices": CP_mega_matrices,
        "target": torch.stack(targets),
        # for end2end only (2d align)
        # "lseg_2d": torch.stack(lseg_2ds),

        "valid_img_idx": torch.stack(valid_img_idxs),
        "valid_img_weight": torch.stack(valid_img_weights),
        "valid_img_feat": torch.stack(valid_img_feats),
        # for voxel-word align only
        "valid_word_idx": torch.stack(valid_word_idxs),
        "valid_word_lbl": torch.stack(valid_word_lbls),
        "valid_word_feat": torch.stack(valid_word_feats),
    }

    for key in data:
        ret_data[key] = data[key]
    return ret_data


def collate_fn_test(batch):
    data = {}
    imgs = []
    CP_mega_matrices = []
    targets = []
    frame_ids = []
    sequences = []

    cam_ks = []
    T_velo_2_cams = []
    frustums_masks = []
    frustums_class_dists = []

    scale_3ds = batch[0]["scale_3ds"]
    for scale_3d in scale_3ds:
        data["projected_pix_{}".format(scale_3d)] = []
        data["fov_mask_{}".format(scale_3d)] = []

    for idx, input_dict in enumerate(batch):
        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).double())
        T_velo_2_cams.append(torch.from_numpy(
            input_dict["T_velo_2_cam"]).float())

        if "frustums_masks" in input_dict:
            frustums_masks.append(torch.from_numpy(
                input_dict["frustums_masks"]))
            frustums_class_dists.append(
                torch.from_numpy(input_dict["frustums_class_dists"]).float()
            )

        for key in data:
            data[key].append(torch.from_numpy(input_dict[key]))

        img = input_dict["img"]
        imgs.append(img)

        frame_ids.append(input_dict["frame_id"])
        sequences.append(input_dict["sequence"])

        target = torch.from_numpy(input_dict["target"])
        targets.append(target)
        CP_mega_matrices.append(torch.from_numpy(input_dict["CP_mega_matrix"]))

    ret_data = {
        "frame_id": frame_ids,
        "sequence": sequences,
        "frustums_class_dists": frustums_class_dists,
        "frustums_masks": frustums_masks,
        "cam_k": cam_ks,
        "T_velo_2_cam": T_velo_2_cams,
        "img": torch.stack(imgs),
        "CP_mega_matrices": CP_mega_matrices,
        "target": torch.stack(targets),
    }

    for key in data:
        ret_data[key] = data[key]
    return ret_data

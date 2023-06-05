from pytorch_lightning import Trainer
from ovo.models.ovo import OVO
from ovo.data.NYU.nyu_dm import NYUDataModule
from ovo.data.semantic_kitti.kitti_dm import KittiDataModule
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import os
from hydra.utils import get_original_cwd
from tqdm import tqdm
import pickle
import json 
import torch.nn as nn
import torch.nn.functional as F


@hydra.main(config_name="../config/ovo.yaml")
def main(config: DictConfig):
    
    novel_class_lbl = config.novel_class_lbl
    target_lbl = config.target_lbl
    
    torch.set_grad_enabled(False)
    # Setup dataloader
    if config.dataset == "kitti":
        feature = 64
        project_scale = 2
        full_scene_size = (256, 256, 32)
        class_lbl = ["empty", "ceiling", "floor", "wall", "window", "chair", "bed", "sofa", "table", "tvs", "furniture", "other"]
        data_module = KittiDataModule(
            root=config.kitti_root,
            preprocess_root=config.kitti_preprocess_root,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
        )
        data_module.setup()
        data_loader = data_module.test_dataloader() # test_dataloader: sequence 08

    elif config.dataset == "NYU":
        project_scale = 1
        feature = 200
        full_scene_size = (60, 36, 60)
        class_lbl = ["empty", "ceiling", "floor", "wall", "window", "chair", "bed", "sofa", "table", "tvs", "furniture", "other"]
        data_module = NYUDataModule(
            root=config.NYU_root,
            preprocess_root=config.NYU_preprocess_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
        )
        data_module.setup()
        data_loader = data_module.test_dataloader()
    else:
        print("dataset not support")

    # Load pretrained models
    model_path = config.model_path
    print("loading from ", model_path)
    
    model = OVO.load_from_checkpoint(
        model_path,
        feature=feature,
        project_scale=project_scale,
        fp_loss=config.fp_loss,
        full_scene_size=full_scene_size,
        strict = False
    )
    model.cuda()
    model.eval()

    # Save prediction and additional data
    # to draw the viewing frustum and remove scene outside the room for NYUv2
    write_root = config.output_path
    write_path = os.path.join(write_root,config.dataset)
    
    # =========================start init for clip loss===============================
    clip_text_gt_feat_path = config.word_path
    with open(clip_text_gt_feat_path, "r") as f:
        info = json.load(f)
    
    clip_dict = {}
    k_word_tokens = []
    
    cnt=0
    for k in info:
        k_word_tokens.append(torch.Tensor(info[k]).unsqueeze(0))
        clip_dict[k] = torch.Tensor(info[k])
        if cnt in novel_class_lbl:
            print(k)
        cnt+=1    
        
    k_word_tokens = torch.cat(k_word_tokens)
    # =========================ends init for clip loss===============================
    base_class_lbl = [label for label in range(len(class_lbl)) if label not in novel_class_lbl]
    print("base_class_lbl: ", base_class_lbl)
    print("novel_class_lbl: ", novel_class_lbl)
    
    right_vox_num = {}
    recalled_vox_num = {}
    all_vox_num = {}
    for i in range(len(class_lbl)):
        right_vox_num[i] = 0
        recalled_vox_num[i] = 0
        all_vox_num[i] = 0

    scene = {}
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            gt = batch["target"][0]
            
            batch["img"] = batch["img"].cuda()
            pred = model(batch)
            
            ssc_logit = pred["ssc_logit"][0]
            ssc_logit = torch.argmax(ssc_logit, dim = 0)
            
            vox_feat = pred["aligned_vox_feat"][0]
            
            # Just reserve novel class and others 
            novel_tokens = torch.stack([k_word_tokens[i] for i in novel_class_lbl]).cuda()
            novel_probo = torch.einsum("cwhd,nc->nwhd", vox_feat, novel_tokens)
            novel_prob = torch.argmax(novel_probo, dim = 0)
            
            # flatten ssc_logit and gt 
            pred_cls = ssc_logit.flatten(0).cpu()
            novel_cls = novel_prob.flatten(0).cpu()
            gt_cls = gt.flatten(0).cpu()
 
            if config.vis:
                scene["cam_pose"] = np.array(batch["cam_pose"].cpu())
                scene["vox_origin"] = np.array(batch["vox_origin"].cpu())
                scene["normal"] = np.array(ssc_logit.cpu())
                scene["novel"] = np.array(novel_prob.cpu())
                if not os.path.exists(write_path):
                    os.makedirs(write_path)
                with open(os.path.join(write_path, batch['name'][0] + ".pkl"), "wb") as handle:
                    pickle.dump(scene, handle)
                    print("wrote ", batch['name'][0])
                    
            if config.miou:
                masked_index = torch.where(gt_cls==255)
                novel_cls[masked_index] = -1
                pred_cls[masked_index] = -1
                
                # =========================start compute miou for each class===============================
                if len(base_class_lbl)!=0: # (novel+base)
                    target_idx = torch.where(pred_cls==target_lbl)
                    novel_pred_lbl = 0
                    for i in range(len(class_lbl)):
                        all_vox_num[i] += torch.sum(gt_cls==i)
                        if i in novel_class_lbl:
                            recalled_vox_num[i] += torch.sum(novel_cls[target_idx]==novel_pred_lbl)
                            right_vox_num[i] += torch.sum((novel_cls[target_idx]==novel_pred_lbl) & (gt_cls[target_idx]==i))
                            novel_pred_lbl += 1
                        elif i in base_class_lbl:
                            recalled_vox_num[i]+=torch.sum(pred_cls==i)
                            right_vox_num[i]+=(torch.sum((pred_cls==i) & (gt_cls==i)))
                
                else: # all novel
                    novel_pred_lbl = 0
                    for i in range(len(class_lbl)):
                        all_vox_num[i] += torch.sum(gt_cls==i)
                        if i in novel_class_lbl:
                            recalled_vox_num[i] += torch.sum(novel_cls==novel_pred_lbl)
                            right_vox_num[i] += torch.sum((novel_cls==novel_pred_lbl) & (gt_cls==i))
                            novel_pred_lbl += 1
                        elif i in base_class_lbl:
                            recalled_vox_num[i]+=torch.sum(pred_cls==i)
                            right_vox_num[i]+=(torch.sum((pred_cls==i) & (gt_cls==i)))
                    # =========================ends compute miou for each class===============================

    if config.miou: 
        print("=====================================")
        all_novel_miou = 0
        for i in range(len(class_lbl)):
            if i in base_class_lbl:
                continue
            novel_miou = right_vox_num[i] / (all_vox_num[i] + recalled_vox_num[i] - right_vox_num[i])
            all_novel_miou += novel_miou
            print(class_lbl[i], " miou is ", novel_miou.item())
        print("all novel miou is ", all_novel_miou.item() / len(novel_class_lbl))
        
        print("=====================================")
        all_base_miou = 0    
        for i in range(len(class_lbl)):
            if i in novel_class_lbl:
                continue
            base_miou = right_vox_num[i] / (all_vox_num[i] + recalled_vox_num[i] - right_vox_num[i])
            all_base_miou += base_miou
            print(class_lbl[i], " miou is ", base_miou.item())
        print("all base miou is ", all_base_miou.item() / len(base_class_lbl))
        print("=====================================")
            
if __name__ == "__main__":
    main()

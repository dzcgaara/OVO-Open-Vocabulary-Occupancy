import torch
import torch.nn as nn
from scipy import optimize
import numpy as np
import torch.nn.functional as F
import json

class ClipAlignmentLoss:
    def __init__(self):
        self.clip_text_gt_feat_path = "/dzc_workspace/clip/CLIP-main/nyu_prompt_embedding.json"
        self.cls_idx_mapping = {0:"empty", 1:"ceiling",2:"floor",3:"wall",4:"window", 5:"chair", 
                      6:"bed", 7:"sofa", 8:"table", 9:"tv", 10:"furniture", 11:"object"}
        self.clip_dict = {}
        self.load_clip_gt()
        self.num_classes = 11
        self.align_loss = nn.L1Loss()
    
    def load_clip_gt(self):
        with open(self.clip_text_gt_feat_path, "r") as f:
            info = json.load(f)
        for k in info:
            self.clip_dict[k] = torch.Tensor(info[k]).cuda()
    
    def get_clip_loss(self, instance_masks, clip_feat, voxel_seg_gt):
        clip_loss = 0
        num_query = instance_masks.shape[0]
        for i in range(num_query):
            # we should get gt first 
            masked_gt = instance_masks[i] * voxel_seg_gt[0]
            num_vox = []
            for j in range(1, self.num_classes + 1):
                tmp_cnt = 60*36*60 - torch.count_nonzero(masked_gt - j)
                num_vox.append(tmp_cnt.cpu().item())
            major_idx = np.argmax(np.array(num_vox))
            mapped_cls = self.cls_idx_mapping[major_idx + 1]
            mapped_embedding = self.clip_dict[mapped_cls]
            
            mapped_embedding = F.normalize(mapped_embedding.unsqueeze(1), dim = 0).squeeze()
            clip_feat = F.normalize(clip_feat.unsqueeze(1), dim = 0).squeeze()
            
            clip_loss += self.align_loss(mapped_embedding, clip_feat)
        return clip_loss

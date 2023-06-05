import torch
import torch.nn as nn
from scipy import optimize
import numpy as np
import torch.nn.functional as F
import json

class Align23dLoss:
    def __init__(self):
        self.get_similarity = nn.CosineSimilarity(dim = 1)
        self.num_classes = 11
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def get_align_23d_loss(self, aligned_feat, dataset, valid_img_idx, valid_img_weight, valid_img_feat, valid_word_idx, valid_word_feat, confidence_weight=True, voxel_pixel_align=True, voxel_word_align=True):
        if dataset == "NYU":
            aligned_feat = aligned_feat.flatten(2).squeeze().permute(1,0)

        if voxel_pixel_align:
            ###### for full feat
            if dataset == "NYU":
                aligned_feat_img = aligned_feat.index_select(0, valid_img_idx[0]) 
            ###### for selected feat
            elif dataset == "kitti":
                aligned_feat_img = aligned_feat

            similarity = self.get_similarity(aligned_feat_img, valid_img_feat[0])
            img_align_loss = 1 - similarity
            ###### ablation study: w/ or w/o confidence
            if confidence_weight:
                img_align_loss *= valid_img_weight[0]
        else:
            img_align_loss = torch.tensor([0.])
        
        if voxel_word_align and dataset != "kitti":
            if aligned_feat.shape[0] != len(valid_word_idx[0]):
                aligned_feat_word = aligned_feat.index_select(0, valid_word_idx[0])       
            similarity = self.get_similarity(aligned_feat_word, valid_word_feat[0])
            word_align_loss = 1 - similarity
        else:
            word_align_loss = torch.tensor([0.])
        return img_align_loss.mean(), word_align_loss.mean()
import torch
import torch.nn as nn
from scipy import optimize
import numpy as np
import torch.nn.functional as F
import json

class Align2dLoss:
    def __init__(self):
        self.get_similarity = nn.CosineSimilarity(dim = 1)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
    def get_align_2d_loss(self, aligned_feat, clip_gt):
        aligned_feat = F.interpolate(
            aligned_feat,
            size=(80, 80),
            mode="bilinear",
            align_corners=True,
        )
        
        aligned_feat = aligned_feat.flatten(2).squeeze().permute(1,0)
        clip_gt = clip_gt.flatten(2).squeeze().permute(1,0)
        
        aligned_feat = F.normalize(aligned_feat, dim = 1)
        clip_gt = F.normalize(clip_gt, dim = 1)
        
        similarity = self.get_similarity(aligned_feat, clip_gt)
        align_loss = 1 - similarity
        
        return align_loss.sum() / (80 * 80)
        
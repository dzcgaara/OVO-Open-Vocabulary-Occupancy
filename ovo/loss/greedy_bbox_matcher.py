import torch
import torch.nn as nn
from scipy import optimize
import numpy as np
import torch.nn.functional as F

class GreedyBboxMatcher:
    def __init__(self, metric = "giou"):
        self.metric = metric
        self.cost_matrix = None
        self.reg_loss = nn.L1Loss()
        # self.cls_loss = F.binary_cross_entropy()
        
    def get_3d_iou_giou(self, box1, box2):
        x_max = box1[0] + box1[3] / 2
        y_max = box1[1] + box1[4] / 2
        z_max = box1[2] + box1[5] / 2
        x_min = box1[0] - box1[3] / 2
        y_min = box1[1] - box1[4] / 2
        z_min = box1[2] - box1[5] / 2	

        l_x_max = box2[0] + box2[3] / 2
        l_y_max = box2[1] + box2[4] / 2
        l_z_max = box2[2] + box2[5] / 2
        l_x_min = box2[0] - box2[3] / 2
        l_y_min = box2[1] - box2[4] / 2
        l_z_min = box2[2] - box2[5] / 2

        inter_x_max = min(x_max, l_x_max)
        inter_x_min = max(x_min, l_x_min)
        inter_y_max = min(y_max, l_y_max)
        inter_y_min = max(y_min, l_y_min)
        inter_z_max = min(z_max,l_z_max)
        inter_z_min = max(z_min,l_z_min)

        inter_w = inter_x_max - inter_x_min
        inter_l = inter_y_max - inter_y_min
        inter_h = inter_z_max - inter_z_min

        inter = 0 if inter_w < 0 or inter_l < 0 or inter_h  < 0 else inter_w * inter_l * inter_h
        area1 = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        area2 = (l_x_max - l_x_min) * (l_y_max - l_y_min) * (l_z_max - l_z_min)
        union = area1 + area2 - inter
        iou = inter / union
        
        area_all = (max(x_max, l_x_max) - min(x_min, l_x_min)) * (max(y_max, l_y_max) - min(y_min, l_y_min)) * (max(z_max, l_z_max) - min(z_min, l_z_min))
        giou = iou - ((area_all - (area1 + area2 - inter)) / area_all)
        return iou, giou 
        
    def get_3d_giou_loss(self, tmp_pred, tmp_gt):
        iou_3d, giou = self.get_3d_iou_giou(tmp_pred, tmp_gt)
        return 1 - giou
        
    def get_reg_loss(self, assignments, pred_reg, gt_reg):
        reg_loss = 0
        giou_loss = 0
        for idx in range(assignments[1].shape[0]):
            tmp_pred = pred_reg[0][assignments[0][idx]]
            tmp_gt = gt_reg[0][assignments[1][idx]]
            tmp_loss = self.reg_loss(tmp_pred, tmp_gt)
            tmp_giou_loss = self.get_3d_giou_loss(tmp_pred, tmp_gt)
            if idx == 0:
                print("The first matching pair is ", tmp_pred.cpu().detach().numpy().tolist(), tmp_gt.cpu().detach().numpy().tolist())
            reg_loss += tmp_loss
            giou_loss += tmp_giou_loss
        return reg_loss, giou_loss
    
    def get_cls_loss(self, assignments, pred_cls):
        cls_loss = 0
        pos_cls_idx = assignments[0].tolist()
        cls_gt = np.zeros([pred_cls.shape[1], 1])
        for i in pos_cls_idx:
            cls_gt[i, 0] = 1
        cls_gt = torch.tensor(cls_gt).float().cuda()
        cls_loss = F.binary_cross_entropy(pred_cls[0], cls_gt)
        return cls_loss
        
    def get_center_distance(self, pt1, pt2):
        dist = (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 + (pt1[2] - pt2[2])**2
        return dist
        
    def get_cost_matrix(self, pred_center, gt_center):
        self.cost_matrix = np.zeros([pred_center.shape[0], gt_center.shape[0]])
        for i in range(pred_center.shape[0]):
            for j in range(gt_center.shape[0]):
                self.cost_matrix[i][j] = self.get_center_distance(pred_center[i], gt_center[j])
        return self.cost_matrix
    
    def get_cost_matrix_giou(self, pred, gt):
        self.cost_matrix = np.zeros([pred.shape[0], gt.shape[0]])
        for i in range(pred.shape[0]):
            for j in range(gt.shape[0]):
                self.cost_matrix[i][j] = self.get_3d_giou_loss(pred[i], gt[j])
        return self.cost_matrix
        
    def get_assignments(self, pred_bbox, gt_bbox):
        # The inputs including pred_bbox([num_queries, 6]) and gt_bbox([num_gt, 6])
        # Note that max num_gt in NYU is 25 and num_query is 30 for now
        # The output is [pred_idx, ... ] ~ [gt_idx, ...]
        
        if self.metric == "centroid":
            # step 1. Construct cost matrix 
            # The cost matrix should be have shape [num_queries, num_gt]
            cost_matrix = self.get_cost_matrix(pred_bbox[0][:,:3], gt_bbox[0][:,:3])
            
            # step2. HM matching
            assignments = optimize.linear_sum_assignment(cost_matrix)
        elif self.metric == "giou":
            # step 1. Construct cost matrix 
            # The cost matrix should be have shape [num_queries, num_gt]
            cost_matrix = self.get_cost_matrix_giou(pred_bbox[0], gt_bbox[0])
            
            # step2. HM matching
            assignments = optimize.linear_sum_assignment(cost_matrix)
        return assignments
    
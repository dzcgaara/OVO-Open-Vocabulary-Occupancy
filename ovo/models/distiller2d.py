import torch
import torch.nn as nn
import torch.nn.functional as F


class Distiller2D(nn.Module):
    def __init__(self, embedding_dim=512):
        super(Distiller2D, self).__init__()
        self.conv_2d_1 = nn.Conv2d(
            200, 128, kernel_size=3, stride=1, padding=1
        )
        self.conv_2d_2 = nn.Conv2d(
            200, 128, kernel_size=3, stride=1, padding=1
        )
        self.conv_2d_4 = nn.Conv2d(
            200, 128, kernel_size=3, stride=1, padding=1
        )
        self.conv_2d_8 = nn.Conv2d(
            200, 128, kernel_size=3, stride=1, padding=1
        )
        self.conv_2d_16 = nn.Conv2d(
            200, 128, kernel_size=3, stride=1, padding=1
        )
        self.relu = nn.ReLU()
        self.projection3d = nn.Conv2d(
            128*5, 256, kernel_size=3, stride=1, padding=1
        )
        self.projection = nn.Conv2d(
            256, 512, kernel_size=1, stride=1, padding=0
        )

    def forward(self, img_feature):
        feat1_1 = F.interpolate(self.relu(self.conv_2d_1(img_feature["1_1"])), [
                                80, 80], mode='bilinear', align_corners=True)  # 200 -> 128
        feat1_2 = F.interpolate(self.relu(self.conv_2d_2(img_feature["1_2"])), [
                                80, 80], mode='bilinear', align_corners=True)  # 200 -> 128
        feat1_4 = F.interpolate(self.relu(self.conv_2d_4(img_feature["1_4"])), [
                                80, 80], mode='bilinear', align_corners=True)  # 200 -> 128
        feat1_8 = F.interpolate(self.relu(self.conv_2d_8(img_feature["1_8"])), [
                                80, 80], mode='bilinear', align_corners=True)  # 200 -> 128
        feat1_16 = F.interpolate(self.relu(self.conv_2d_16(img_feature["1_16"])), [
                                 80, 80], mode='bilinear', align_corners=True)  # 200 -> 128

        merged_feat = torch.cat(
            [feat1_1, feat1_2, feat1_4, feat1_8, feat1_16], dim=1)
        merged_feat = self.relu(self.projection3d(merged_feat))  # 640 -> 256
        merged_feat = self.projection(merged_feat)  # 256 -> 512
        return merged_feat

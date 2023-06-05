import torch.nn as nn


class Distiller3D(nn.Module):
    def __init__(self, embedding_dim=512):
        super(Distiller3D, self).__init__()
        self.num_ref_points = 100

        self.relu = nn.ReLU()

        # for NYU
        self.conv_3d = nn.Conv3d(
            200, 256, kernel_size=3, stride=1, padding=1
        )
        self.projection_0 = nn.Conv3d(
            256, 512, kernel_size=1, stride=1, padding=0
        )
        self.projection = nn.Conv3d(
            512, 512, kernel_size=1, stride=1, padding=0
        )

        # for kitti
        self.conv_3d_kitti = nn.Conv3d(
            32, 64, kernel_size=3, stride=1, padding=1
        )
        self.projection_kitti = nn.Conv3d(
            64, 512, kernel_size=1, stride=1, padding=0
        )
        self.projection_kitti_2 = nn.Conv3d(
            512, 512, kernel_size=1, stride=1, padding=0
        )
        self.fc_1 = nn.Linear(32, 64)
        self.fc_2 = nn.Linear(64, 128)
        self.fc_3 = nn.Linear(128, 256)
        self.fc_4 = nn.Linear(256, 512)
        self.fc_5 = nn.Linear(512, 512)

    def forward(self, bev_feature, is_kitti=False):
        if is_kitti:
            bev_feature = self.relu(self.fc_1(bev_feature.cuda()))  # 32 -> 64
            bev_feature = self.relu(self.fc_2(bev_feature))  # 64 -> 128
            bev_feature = self.relu(self.fc_3(bev_feature))  # 128 -> 256
            bev_feature = self.relu(self.fc_4(bev_feature))  # 256 -> 512
            bev_feature = self.relu(self.fc_5(bev_feature))  # 512 -> 512
            aligned_feat = bev_feature[0, :, :]
        else:
            bev_feature = self.conv_3d(bev_feature)  # 200 -> 256
            bev_feature = self.relu(bev_feature)  # 256 -> 256
            bev_feature = self.relu(
                self.projection_0(bev_feature))  # 256 -> 512
            aligned_feat = self.projection(bev_feature)  # 512 -> 512
        return aligned_feat

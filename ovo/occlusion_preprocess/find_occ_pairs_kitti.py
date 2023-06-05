import numpy as np
import os
import pickle
import json
from tqdm import tqdm
import sys


class SdfSolverKitti:
    def __init__(self, file_name=None, observation_pt=None, segment_id=None) -> None:
        self.observation_pt = observation_pt
        self.voxels = None
        self.invalid_lbl = [0, 255]
        self.save_root = f"./kitti_occ_reslut/{segment_id}/"
        if not os.path.exists(self.save_root):
            os.mkdir(self.save_root)
        self.file_name = file_name
        self.get_input(file_name)
        self.voxel_size = 0.2
        # coord is like
        #     z
        #     |   x
        #     |  /
        #     | /
        # - - | y
        self.x_max_idx = self.voxels.shape[0]
        self.y_max_idx = self.voxels.shape[1]
        self.z_max_idx = self.voxels.shape[2]
        self.distance_queue = []
        self.unocc = []
        self.occ = []
        self.occ_helper = np.zeros(self.voxels.shape)
        self.line_deltas = [(0, 0, 0)]

    def get_input(self, file_name=None):
        pkl_path = file_name
        data = np.load(pkl_path)
        self.voxels = data

    def get_two_pt_distance(self, pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + (pt1[2] - pt2[2]) ** 2)

    def voxel_center_2_coord(self, voxel_center):
        x_idx, y_idx, z_idx = voxel_center
        ret = [x_idx * self.voxel_size + self.voxel_size / 2, y_idx * self.voxel_size +
               self.voxel_size / 2, z_idx * self.voxel_size + self.voxel_size / 2]
        return ret

    def get_distance_priority_queue(self):
        # ret like [(distance, [x_idx, y_idx, z_idx]), ...]
        self.distance_queue = []
        for x in range(self.x_max_idx):
            for y in range(self.y_max_idx):
                for z in range(self.z_max_idx):
                    if self.voxels[x, y, z] in self.invalid_lbl:
                        continue
                    center = [x, y, z]
                    coord = self.voxel_center_2_coord(center)
                    distance = self.get_two_pt_distance(
                        self.observation_pt, coord)
                    self.distance_queue.append((distance, center))
        self.distance_queue.sort(key=lambda x: x[0])

    def get_x_y(self, pt1, pt2, z):
        x1, y1, z1 = pt1[0], pt1[1], pt1[2]
        x2, y2, z2 = pt2[0], pt2[1], pt2[2]
        x = (z - z1) / (z2 - z1) * (x2 - x1) + x1
        y = (z - z1) / (z2 - z1) * (y2 - y1) + y1
        return x, y

    def get_x_z(self, pt1, pt2, y):
        x1, y1, z1 = pt1[0], pt1[1], pt1[2]
        x2, y2, z2 = pt2[0], pt2[1], pt2[2]
        x = (y - y1) / (y2 - y1) * (x2 - x1) + x1
        z = (y - y1) / (y2 - y1) * (z2 - z1) + z1
        return x, z

    def get_y_z(self, pt1, pt2, x):
        x1, y1, z1 = pt1[0], pt1[1], pt1[2]
        x2, y2, z2 = pt2[0], pt2[1], pt2[2]
        y = (x - x1) / (x2 - x1) * (y2 - y1) + y1
        z = (x - x1) / (x2 - x1) * (z2 - z1) + z1
        return y, z

    def scan_one_line(self, start_pt_coord, end_pt_coord, pt_coord):
        coord_x, coord_z = self.get_x_z(
            start_pt_coord, end_pt_coord, pt_coord[1] - self.voxel_size / 2)
        flag = False
        if coord_x <= pt_coord[0] + self.voxel_size / 2 and coord_x >= pt_coord[0] - self.voxel_size / 2 and coord_z <= pt_coord[2] + self.voxel_size / 2 and coord_z >= pt_coord[2] - self.voxel_size / 2:
            flag = True
        return flag

    def coord_to_voxel_ind(self, coord):
        x_idx = int((coord[0] - self.voxel_size / 2) / self.voxel_size)
        y_idx = int((coord[1] - self.voxel_size / 2) / self.voxel_size)
        z_idx = int((coord[2] - self.voxel_size / 2) / self.voxel_size)
        return [x_idx, y_idx, z_idx]

    def scan_shot_debug(self, cur_voxel_pt):
        # shot one scan and paint all voxel on the line
        start_pt_coord = self.observation_pt
        end_pt = cur_voxel_pt
        end_pt_coord = self.voxel_center_2_coord(end_pt)
        cur_x = cur_voxel_pt[0]
        for ext_x in range(cur_x + 1, self.x_max_idx):
            ext_x = ext_x * self.voxel_size + self.voxel_size / 2
            ext_y, ext_z = self.get_y_z(start_pt_coord, end_pt_coord, ext_x)
            ext_x_idx, ext_y_idx, ext_z_idx = self.coord_to_voxel_ind(
                [ext_x, ext_y, ext_z])
            if ext_x_idx >= self.x_max_idx or ext_y_idx >= self.y_max_idx or ext_z_idx >= self.z_max_idx or ext_x_idx < 0 or ext_y_idx < 0 or ext_z_idx < 0:
                continue
            else:
                if self.occ_helper[ext_x_idx, ext_y_idx, ext_z_idx] == 0:
                    self.occ_helper[ext_x_idx, ext_y_idx, ext_z_idx] = 1

    def traverse(self):
        while len(self.distance_queue) > 0:
            # print(len(self.distance_queue))
            tmp_pt = self.distance_queue.pop(0)
            # if tmp_pt[1] in self.occ:
            if self.occ_helper[tmp_pt[1][0], tmp_pt[1][1], tmp_pt[1][2]] == 1:
                continue
            self.unocc.append(tmp_pt[1])
            self.scan_shot_debug(tmp_pt[1])

    def dump_result(self):
        ret = np.zeros((self.x_max_idx, self.y_max_idx, self.z_max_idx))
        # for i in self.occ:
        #     if self.voxels[i[0], i[1], i[2]] != 0 and self.voxels[i[0], i[1], i[2]] != 255:
        #         ret[i[0], i[1], i[2]] = 1
        for j in self.unocc:
            ret[j[0], j[1], j[2]] = 2
        tmp_result = {"occ_vis": ret, "gt": self.voxels}
        # print("num occ is ", len(self.occ))
        print("num unocc is ", len(self.unocc))
        with open(os.path.join(self.save_root, self.file_name.split("/")[-1] + ".pkl"), "wb") as handle:
            pickle.dump(tmp_result, handle)

    def process(self):
        # step1. Sort all valid voxel by distance to observation point
        self.get_distance_priority_queue()
        # step2. Traverse all valid voxel from the closest one to the farthest one
        self.traverse()
        # step3. Dump result array
        self.dump_result()


if __name__ == "__main__":
    observation_pt_file = "ovo/data/semantic_kitti/camera_position_kitti_all.json"
    with open(observation_pt_file, "r") as f:
        info = json.load(f)
        print(observation_pt_file)
        print(info.keys())

    pkl_root = sys.argv[1]  # /path/to/kitti_preprocess_ov

    for segment_id in tqdm(info.keys()):
        frames = os.listdir(os.path.join(pkl_root, segment_id))
        for frame in tqdm(frames):
            sdf_solver = SdfSolverKitti(os.path.join(os.path.join(
                pkl_root, segment_id), frame), info[segment_id], segment_id)
            sdf_solver.process()

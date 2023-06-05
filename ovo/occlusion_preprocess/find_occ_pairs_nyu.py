import numpy as np
import os
import pickle
import json
from tqdm import tqdm
import sys


class SdfSolver:
    def __init__(self, file_name=None, observation_pt=None) -> None:
        self.observation_pt = observation_pt
        self.voxels = None
        self.invalid_lbl = [0, 255]
        self.pkl_root = "/data/nyu_preprocess_cbt/base/NYUtrain/"
        self.save_root = "./nyu_occ_reslut"
        self.file_name = file_name
        self.get_input(file_name)
        self.voxel_size = 0.08
        # coord is like
        #    x
        #    |   z
        #    |  /
        #    | /
        #    |------ y
        self.x_max_idx = self.voxels.shape[0]
        self.y_max_idx = self.voxels.shape[1]
        self.z_max_idx = self.voxels.shape[2]
        self.distance_queue = []
        self.unocc = []
        self.occ = []
        self.line_deltas = [(0, 0, 0)]

    def get_input(self, file_name=None):
        # self.observation_pt = [3.1458745, -0.8880716, 1.36446755]
        pkl_path = os.path.join(self.pkl_root, file_name + ".pkl")
        with open(pkl_path, "rb") as handle:
            data = pickle.load(handle)
        self.voxels = data["target_1_4"]
        self.voxels = self.voxels.transpose(0, 2, 1)

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
        cur_y = cur_voxel_pt[1]
        for ext_y in range(cur_y + 1, self.y_max_idx):
            ext_y = ext_y * self.voxel_size + self.voxel_size / 2
            ext_x, ext_z = self.get_x_z(start_pt_coord, end_pt_coord, ext_y)
            ext_x_idx, ext_y_idx, ext_z_idx = self.coord_to_voxel_ind(
                [ext_x, ext_y, ext_z])
            if ext_x_idx >= self.x_max_idx or ext_y_idx >= self.y_max_idx or ext_z_idx >= self.z_max_idx or ext_x_idx < 0 or ext_y_idx < 0 or ext_z_idx < 0:
                continue
            else:
                if [ext_x_idx, ext_y_idx, ext_z_idx] not in self.occ:
                    self.occ.append([ext_x_idx, ext_y_idx, ext_z_idx])

    def scan_shot(self, cur_voxel_pt):
        # shot one scan and paint all voxel on the line
        start_pt_coord = self.observation_pt
        end_pt = cur_voxel_pt
        cur_y = cur_voxel_pt[1]
        end_pt_coord = self.voxel_center_2_coord(end_pt)
        # one end pt coord stand for a scan line
        for delta in self.line_deltas:
            end_pt_coord[0], end_pt_coord[1], end_pt_coord[2] = end_pt_coord[0] + \
                delta[0], end_pt_coord[1] + \
                delta[1], end_pt_coord[2] + delta[2]
            for ext_y in range(cur_y + 1, self.y_max_idx):
                ext_y = ext_y * self.voxel_size + self.voxel_size / 2
                ext_x, ext_z = self.get_x_z(
                    start_pt_coord, end_pt_coord, ext_y)
                ext_x_idx, ext_y_idx, ext_z_idx = self.coord_to_voxel_ind(
                    [ext_x, ext_y, ext_z])
                if ext_x_idx >= self.x_max_idx or ext_y_idx >= self.y_max_idx or ext_z_idx >= self.z_max_idx or ext_x_idx < 0 or ext_y_idx < 0 or ext_z_idx < 0:
                    continue
                else:
                    if [ext_x_idx, ext_y_idx, ext_z_idx] not in self.occ:
                        self.occ.append([ext_x_idx, ext_y_idx, ext_z_idx])

    def traverse(self):
        while len(self.distance_queue) > 0:
            tmp_pt = self.distance_queue.pop(0)
            if tmp_pt[1] in self.occ:
                continue
            self.unocc.append(tmp_pt[1])
            self.scan_shot_debug(tmp_pt[1])

    def dump_result(self):
        ret = np.zeros((self.x_max_idx, self.y_max_idx, self.z_max_idx))
        for i in self.occ:
            if self.voxels[i[0], i[1], i[2]] != 0 and self.voxels[i[0], i[1], i[2]] != 255:
                ret[i[0], i[1], i[2]] = 1
        for j in self.unocc:
            ret[j[0], j[1], j[2]] = 2
        ret = ret.transpose(0, 2, 1)
        tmp_result = {"occ_vis": ret, "gt": self.voxels}
        print("num occ is ", len(self.occ))
        print("num unocc is ", len(self.unocc))
        with open(os.path.join(self.save_root, self.file_name + ".pkl"), "wb") as handle:
            pickle.dump(tmp_result, handle)

    def process(self):
        # step1. Sort all valid voxel by distance to observation point
        self.get_distance_priority_queue()
        # step2. Traverse all valid voxel from the closest one to the farthest one
        self.traverse()
        # step3. Dump result array
        self.dump_result()


if __name__ == "__main__":
    observation_pt_file = "ovo/data/NYU/camera_position_nyu.json"
    with open(observation_pt_file, "r") as f:
        info = json.load(f)

    pkl_root = sys.argv[1]  # /path/to/nyu_preprocess_ov/base/NYUtrain/

    for k in tqdm(info.keys()):
        sdf_solver = SdfSolver(k, info[k])
        sdf_solver.process()

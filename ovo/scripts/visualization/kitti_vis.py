import pickle
import pandas as pd
import numpy as np
import ovo.data.utils.fusion as fusion
from mayavi import mlab


def vox2pix(cam_E, cam_k,
            vox_origin, voxel_size,
            img_W, img_H,
            scene_size):
    """
    compute the 2D projection of voxels centroids

    Parameters:
    ----------
    cam_E: 4x4
       =camera pose in case of NYUv2 dataset
       =Transformation from camera to lidar coordinate in case of SemKITTI
    cam_k: 3x3
        camera intrinsics
    vox_origin: (3,)
        world(NYU)/lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
    img_W: int
        image width
    img_H: int
        image height
    scene_size: (3,)
        scene size in meter: (51.2, 51.2, 6.4) for SemKITTI and (4.8, 4.8, 2.88) for NYUv2

    Returns
    -------
    projected_pix: (N, 2)
        Projected 2D positions of voxels
    fov_mask: (N,)
        Voxels mask indice voxels inside image's FOV 
    pix_z: (N,)
        Voxels'distance to the sensor in meter
    """
    # Compute the x, y, z bounding of the scene in meter
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = vox_origin
    vol_bnds[:, 1] = vox_origin + np.array(scene_size)

    # Compute the voxels centroids in lidar cooridnates
    vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) /
                      voxel_size).copy(order='C').astype(int)
    xv, yv, zv = np.meshgrid(
        range(vol_dim[0]),
        range(vol_dim[1]),
        range(vol_dim[2]),
        indexing='ij'
    )
    vox_coords = np.concatenate([
        xv.reshape(1, -1),
        yv.reshape(1, -1),
        zv.reshape(1, -1)
    ], axis=0).astype(int).T

    # Project voxels'centroid from lidar coordinates to camera coordinates
    cam_pts = fusion.TSDFVolume.vox2world(vox_origin, vox_coords, voxel_size)
    cam_pts = fusion.rigid_transform(cam_pts, cam_E)

    # Project camera coordinates to pixel positions
    projected_pix = fusion.TSDFVolume.cam2pix(cam_pts, cam_k)
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

    # Eliminate pixels outside view frustum
    pix_z = cam_pts[:, 2]
    fov_mask = np.logical_and(pix_x >= 0,
                              np.logical_and(pix_x < img_W,
                                             np.logical_and(pix_y >= 0,
                                                            np.logical_and(pix_y < img_H,
                                                                           pix_z > 0))))

    return projected_pix, fov_mask, pix_z


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    sensor_pose = 10
    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float)

    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid


def draw(
    voxels,
    T_velo_2_cam,
    vox_origin,
    fov_mask,
    img_size,
    f,
    voxel_size=0.2,
    d=7,  # 7m - determine the size of the mesh representing the camera
):
    # Compute the coordinates of the mesh representing camera
    x = d * img_size[0] / (2 * f)
    y = d * img_size[1] / (2 * f)
    tri_points = np.array(
        [
            [0, 0, 0],
            [x, y, d],
            [-x, y, d],
            [-x, -y, d],
            [x, -y, d],
        ]
    )
    tri_points = np.hstack([tri_points, np.ones((5, 1))])
    tri_points = (np.linalg.inv(T_velo_2_cam) @ tri_points.T).T
    x = tri_points[:, 0] - vox_origin[0]
    y = tri_points[:, 1] - vox_origin[1]
    z = tri_points[:, 2] - vox_origin[2]
    triangles = [
        (0, 1, 2),
        (0, 1, 4),
        (0, 3, 4),
        (0, 2, 3),
    ]

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords[fov_mask, :]

    # Get the voxels outside FOV
    outfov_grid_coords = grid_coords[~fov_mask, :]

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 255)
    ]
    outfov_voxels = outfov_grid_coords[
        (outfov_grid_coords[:, 3] > 0) & (outfov_grid_coords[:, 3] < 255)
    ]

    figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))

    # Draw the camera
    print("camera_position: ", x[0], y[0], z[0])
    mlab.triangular_mesh(
        x, y, z, triangles, representation="wireframe", color=(0, 0, 0), line_width=5
    )

    # axis_length = 50.0
    # axis_color = (1, 1, 1)
    # mlab.axes(
    #     xlabel='X',
    #     ylabel='Y',
    #     zlabel='Z',
    #     nb_labels=6,
    #     color=axis_color,
    #     extent=[0, axis_length, 0, axis_length, 0, axis_length],
    # )

    # Draw occupied inside FOV voxels
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=20,
    )

    # Draw occupied outside FOV voxels
    plt_plot_outfov = mlab.points3d(
        outfov_voxels[:, 0],
        outfov_voxels[:, 1],
        outfov_voxels[:, 2],
        outfov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=18,
    )

    colors = np.array(
        [
            [100, 150, 245, 255],  # car
            [100, 230, 245, 255],  # bicycle
            [30, 60, 150, 255],  # motorcycle
            [80, 30, 180, 255],
            [100, 80, 250, 255],
            [255, 30, 30, 255],
            [255, 40, 200, 255],
            [150, 30, 90, 255],
            [255, 0, 255, 255],
            [255, 150, 255, 255],
            [75, 0, 75, 255],
            [175, 0, 75, 255],
            [255, 200, 0, 255],
            [255, 120, 50, 255],
            [0, 175, 0, 255],
            [135, 60, 0, 255],
            [150, 240, 80, 255],
            [255, 240, 150, 255],
            [255, 0, 0, 255],
            [153, 153, 153, 255],  # 20:novel
        ]
    ).astype(np.uint8)

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_outfov.glyph.scale_mode = "scale_by_vector"

    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    outfov_colors = colors
    outfov_colors[:, :3] = outfov_colors[:, :3] // 3 * 2
    plt_plot_outfov.module_manager.scalar_lut_manager.lut.table = outfov_colors
    mlab.view(azimuth=220, elevation=None, distance=None,
              focalpoint=None, roll=None, reset_roll=True, figure=None)


def main():

    gt_file = ''  # /path/to/gt.npy
    scene_file = ''  # /path/to/scene.pkl

    label = "08"
    gt_scene = np.load(gt_file)  # [256, 256, 32]
    with open(scene_file, "rb") as handle:
        scene_data = pickle.load(handle)

    scene_normal = scene_data["normal"]
    scene_novel = scene_data["novel"]
    scene_normal[(gt_scene == 255)] = 255
    scene_novel[(gt_scene == 255)] = 255

    params = pd.read_pickle('kitti_params_all.pkl')
    param = params[label]
    fov_mask_1 = np.array(param["fov_mask_1"])
    T_velo_2_cam = param["T_velo_2_cam"]
    print('T_velo_2_cam: ', T_velo_2_cam)
    vox_origin = np.array([0, -25.6, -2])

    # gt
    draw(
        gt_scene,
        T_velo_2_cam,
        vox_origin,
        fov_mask_1,
        img_size=(1220, 370),
        f=707.0912,
        voxel_size=0.2,
        d=7,
    )

    # mono
    scene_normal = np.where(scene_normal == 9, 20, scene_normal)
    draw(
        scene_normal,
        T_velo_2_cam,
        vox_origin,
        fov_mask_1,
        img_size=(1220, 370),
        f=707.0912,
        voxel_size=0.2,
        d=7,
    )

    # merge
    scene = scene_normal
    class_map = {0: 1, 1: 9, 2: 13}
    for i in range(256):
        for j in range(256):
            for k in range(32):
                if scene[i, j, k] == 20:
                    scene[i, j, k] = class_map[scene_novel[i, j, k]]

    draw(
        scene,
        T_velo_2_cam,
        vox_origin,
        fov_mask_1,
        img_size=(1220, 370),
        f=707.0912,
        voxel_size=0.2,
        d=7,
    )

    mlab.show()


if __name__ == "__main__":
    main()

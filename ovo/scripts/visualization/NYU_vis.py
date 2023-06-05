import pickle
import numpy as np
from mayavi import mlab
import pickle


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)

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
    cam_pose,
    vox_origin,
    voxel_size=0.08,
    d=0.75,  # 0.75m - determine the size of the mesh representing the camera
):
    # Compute the coordinates of the mesh representing camera
    y = d * 480 / (2 * 518.8579)
    x = d * 640 / (2 * 518.8579)
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

    tri_points = (cam_pose @ tri_points.T).T
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
        [voxels.shape[0], voxels.shape[2], voxels.shape[1]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack(
        (grid_coords.T, np.moveaxis(voxels, [0, 1, 2], [0, 2, 1]).reshape(-1))
    ).T

    # Remove empty and unknown voxels
    occupied_voxels = grid_coords[(
        grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)]
    figure = mlab.figure(size=(1600, 900), bgcolor=(1, 1, 1))

    # Draw the camera
    print("camera_position: ", x[0], y[0], z[0])
    mlab.triangular_mesh(
        x,
        y,
        z,
        triangles,
        representation="wireframe",
        color=(0, 0, 0),
        line_width=5,
    )

    # Draw axis
    # axis_length = 5.0
    # axis_color = (0, 0.5, 1)
    # mlab.axes(
    #     xlabel='X',
    #     ylabel='Y',
    #     zlabel='Z',
    #     nb_labels=6,
    #     color=axis_color,
    #     extent=[0, axis_length, 0, axis_length, 0, axis_length],
    # )

    # Draw occupied voxels
    plt_plot = mlab.points3d(
        occupied_voxels[:, 0],
        occupied_voxels[:, 1],
        occupied_voxels[:, 2],
        occupied_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.1 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=12,
    )

    colors = np.array(
        [
            [22, 191, 206, 255],  # free
            [214, 38, 40, 255],  # ceiling
            [43, 160, 43, 255],  # floor
            [158, 216, 229, 255],  # wall
            [114, 158, 206, 255],  # window
            [204, 204, 91, 255],  # chair
            [255, 186, 119, 255],  # bed
            [147, 102, 188, 255],  # sofa
            [30, 119, 181, 255],  # table
            [188, 188, 33, 255],  # tvs
            [255, 127, 12, 255],  # furniture
            [196, 175, 214, 255],  # objects/other
            [153, 153, 153, 255],  # novel
        ]
    )

    plt_plot.glyph.scale_mode = "scale_by_vector"

    plt_plot.module_manager.scalar_lut_manager.lut.table = colors
    mlab.view(azimuth=240, elevation=None, distance=None,
              focalpoint=None, roll=None, reset_roll=True, figure=None)


def main():

    gt_file = ''  # /path/to/gt.pkl
    scene_file = ''  # /path/to/scene.pkl

    with open(scene_file, "rb") as handle:
        scene_data = pickle.load(handle)

    with open(gt_file, "rb") as handle:
        # gt_data = pickle.load(handle)
        gt_scene = pickle.load(handle)["target_1_4"]

    cam_pose = scene_data["cam_pose"]
    print("cam_pose: ", cam_pose)
    vox_origin = scene_data['vox_origin'][0]
    print("vox_origin: ", vox_origin)

    scene_normal = scene_data["normal"]
    scene_novel = scene_data["novel"]
    scene_normal[(gt_scene == 255)] = 255
    scene_novel[(gt_scene == 255)] = 255
    draw(
        gt_scene,
        cam_pose,
        vox_origin,
        voxel_size=0.08,
        d=0.75,
    )

    scene_normal = np.where(scene_normal == 11, 12, scene_normal)
    draw(
        scene_normal,
        cam_pose,
        vox_origin,
        voxel_size=0.08,
        d=0.75,
    )

    # merge
    scene = scene_normal
    class_map = {0: 6, 1: 8, 2: 11}
    for i in range(60):
        for j in range(36):
            for k in range(60):
                if scene[i, j, k] == 12:
                    scene[i, j, k] = class_map[scene_novel[i, j, k]]

    draw(
        scene,
        cam_pose,
        vox_origin,
        voxel_size=0.08,
        d=0.75,
    )
    mlab.show()


if __name__ == "__main__":
    main()

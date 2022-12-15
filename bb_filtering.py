import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from render import util

POINT_CLOUD_FILE = "data/colmap/aloy/points3D.txt"
CAMERAS_FILE = "data/colmap/aloy/images.txt"
BOUNDING_BOX = np.array(
    [
        [(1920-470)/2,       (1342-1200)/2       ],
        [(1920-470)/2 + 470, (1342-1200)/2 + 1200]
    ], dtype=np.uint) 
RESOLUTION = [1342, 1920]

def load_points(filename):
    file = open(filename)
    points = np.array([])
    mapping = {}
    i = 0
    for line in file:
        if not '#' in line:
            info = line.split(' ')
            id = int(info[0])
            x = float(info[1])
            y = float(info[2])
            z = float(info[3])
            red = float(info[4]) / 255.0
            green = float(info[5]) / 255.0
            blue = float(info[6]) / 255.0
            points = np.append(points, np.array([x, -y, -z, red, green, blue]))
            mapping[id] = i
            i += 1
    file.close()
    return (points.reshape((-1, 6)), mapping)

def load_cameras(filename):
    description_file = open(filename)
    nex_is_list = False
    view_mtxs = {}
    view_points = {}
    last_key = ""
    for line in description_file:
        if not '#' in line and not nex_is_list:
            units = line.split(" ")
            qw = float(units[1])
            qx = float(units[2])
            qy = float(units[3])
            qz = float(units[4])
            tx = float(units[5])
            ty = float(units[6])
            tz = float(units[7])
            rm = np.array(Rotation.from_quat([qx, -qy, -qz, qw]).as_matrix())
            tm = np.array([[tx], [-ty], [-tz]])
            mtx = np.hstack((rm, tm))
            mtx = np.vstack((mtx, np.zeros((1, 4))))
            mtx[3,3] = 1
            view_mtxs[units[9][:-1]] = mtx.astype(np.float32)
            last_key = units[9][:-1]
            nex_is_list = True
        elif not '#' in line and nex_is_list:
            units = line.split(" ")
            for i in range(0, len(units), 3):
                px = float(units[i])
                py = RESOLUTION[0] - float(units[i + 1])
                pid = int(units[i + 2])
                if pid >= 0:
                    if not last_key in view_points.keys():
                        view_points[last_key] = np.array([])
                    view_points[last_key] = np.append(view_points[last_key], np.array([float(pid), px, py]))
            if last_key in view_points.keys():
                view_points[last_key] = np.reshape(view_points[last_key], (-1, 3))
            nex_is_list = False
    description_file.close()
    return (view_mtxs, view_points)

def apply_bounding_box(bb, points, array_mapping, points_in_views, top=100, plot=True, save_interval=300):
    """
    proj_mtx = util.perspective(np.deg2rad(45), RESOLUTION[1] / RESOLUTION[0], 0.1, 1000.0).numpy()
    window_mtx = np.array([ 
        [RESOLUTION[0]/2, 0,               0, (RESOLUTION[0])/2],
        [0,               RESOLUTION[1]/2, 0, (RESOLUTION[1])/2],
        [0,               0,               1, 0],
        [0,               0,               0, 1]
    ], dtype=np.float32)
    filtered_points = points
    iters = 0
    filter_values = np.zeros(filtered_points.shape[0], dtype=np.float32)
    for key in views:
        mvp = proj_mtx @ views[key]
        tpoints = np.transpose(np.pad(filtered_points[:, 0:3], ((0,0), (0,1)), 'constant', constant_values=1))
        projected_points = np.einsum("nn,ni->ni", mvp, tpoints).transpose()
        projected_points = projected_points / np.tile(projected_points[:, 3], (1, 4)).reshape((-1, 4))
        #projected_points = (window_mtx @ projected_points.transpose()).transpose()

        if plot and iters % save_interval == 0:
            fig = plt.figure()
            fig.set_size_inches(2 * RESOLUTION[1]/100, RESOLUTION[0]/100)
            fig.set_dpi(100)

            ax1 = fig.add_subplot(1, 2, 1)
            ax1.scatter(projected_points[:, 0], projected_points[:, 1], c=filtered_points[:, 3:6])
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_xlim([-1, 1])
            ax1.set_ylim([-1, 1])
        
        hx = projected_points[:, 0] > np.tile(2 * bb[0, 0] / RESOLUTION[1] - 1, (projected_points.shape[0]))
        lx = projected_points[:, 0] < np.tile(2 * bb[1, 0] / RESOLUTION[1] - 1, (projected_points.shape[0]))
        inside_x = np.logical_and(hx, lx)
        hy = projected_points[:, 1] > np.tile(2 * bb[0, 1] / RESOLUTION[0] - 1, (projected_points.shape[0]))
        ly = projected_points[:, 1] < np.tile(2 * bb[1, 1] / RESOLUTION[0] - 1, (projected_points.shape[0]))
        inside_y = np.logical_and(hy, ly)
        filter_values += np.logical_and(inside_x, inside_y).astype(np.float32)
        projected_points = projected_points[np.logical_and(inside_x, inside_y)]
        
        if plot and iters % save_interval == 0:
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.scatter(projected_points[:, 0], projected_points[:, 1], c=filtered_points[np.logical_and(inside_x, inside_y), 3:6])
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_xlim([-1, 1])
            ax2.set_ylim([-1, 1])

            plt.savefig("filtering_{:s}".format(key))
            plt.close()
        iters += 1
    filter_values /= len(views.keys())
    filtered_points = filtered_points[filter_values >= (np.ones(filter_values.shape) * threshold)]
    """
    filtered_points = points
    filter_values = np.zeros(filtered_points.shape[0], dtype=np.float32)
    iters = 0
    for key in points_in_views:
        keypoints = points_in_views[key]
        hx = keypoints[:, 1] > np.tile(bb[0, 0], (keypoints.shape[0]))
        lx = keypoints[:, 1] < np.tile(bb[1, 0], (keypoints.shape[0]))
        inside_x = np.logical_and(hx, lx)
        hy = keypoints[:, 2] > np.tile(bb[0, 1], (keypoints.shape[0]))
        ly = keypoints[:, 2] < np.tile(bb[1, 1], (keypoints.shape[0]))
        inside_y = np.logical_and(hy, ly)
        keypoints = keypoints[np.logical_and(inside_x, inside_y)]

        visible_rows = np.array([], dtype=int)
        for i in range(0, keypoints.shape[0]):
            row = array_mapping[ int(keypoints[i, 0]) ]
            visible_rows = np.append(visible_rows, [row])
        filter_values[visible_rows] += 1

        if plot and iters % save_interval == 0:
            fig = plt.figure()
            fig.set_size_inches(RESOLUTION[1]/100, RESOLUTION[0]/100)
            fig.set_dpi(100)

            ax1 = fig.add_subplot()
            ax1.scatter(keypoints[:, 1], keypoints[:, 2], c=filtered_points[visible_rows, 3:6])
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_xlim([0, RESOLUTION[1]])
            ax1.set_ylim([0, RESOLUTION[0]])

            plt.savefig("filtering_{:s}".format(key))
            plt.close()
        
        iters += 1
    filtered_inds = np.argsort(-filter_values)[:min(top, filter_values.shape[0])]
    filter_values = filter_values[filtered_inds]
    filtered_points = filtered_points[filtered_inds]
    return (filtered_points, filter_values)

if __name__ == "__main__":
    print("Loading points...")
    points, array_mapping = load_points(POINT_CLOUD_FILE)
    print("{} points loaded.".format(points.shape[0]))
    print("Loading views...")
    views, points_in_views = load_cameras(CAMERAS_FILE)
    print("{} cameras loaded".format(len(views.keys())))
    print("Filtering points using bounding box {}...".format(BOUNDING_BOX))
    fpoints, weights = apply_bounding_box(BOUNDING_BOX, points, array_mapping, points_in_views)
    print("{} points have been removed".format(points.shape[0] - fpoints.shape[0]))
    object_center = np.average(fpoints, axis=0, weights=weights)
    print("Found object center at {}".format(object_center))
    print("Plotting points...")
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(fpoints[:, 0], fpoints[:, 2], fpoints[:, 1], c=fpoints[:, 3:6])
    ax.scatter(object_center[0], object_center[2], object_center[1], c=np.array([[0, 1, 0]]), marker='^')
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    #ax.set_xlim([-4, 4])
    #ax.set_zlim([-4, 4])
    #ax.set_ylim([-4, 4])
    plt.title("View distribution and mapping")
    plt.show()
    print("Finished.")
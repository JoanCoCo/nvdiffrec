import os
import glob
import json
from re import I
from scipy.spatial.transform import Rotation

import torch
import numpy as np

from render import util

from .dataset import Dataset
import matplotlib.pyplot as plt

def _load_img(path):
    assert os.path.exists(path), "DatasetColmap: Tried to find image file for: %s, but found 0 files" % (path)
    img = util.load_image_raw(path)
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    if img.shape[2] < 4:
        img = torch.tensor(np.concatenate((img, np.ones((img.shape[0], img.shape[1], 1))), axis=-1), dtype=torch.float32)
    return img

class DatasetColmap(Dataset):
    def __init__(self, base_path, FLAGS, validation=False, show_estimation=True):
        self.FLAGS = FLAGS
        self.is_val = validation
        self.base_dir = base_path
        self.view_mtxs = {}
        camera_centers = {}
        look_ats = {}

        # Load config / transforms
        assert os.path.exists(os.path.join(self.base_dir, "images.txt")), "DatasetColmap: images.txt file is missing."
        description_file = open(os.path.join(self.base_dir, "images.txt"), "r")
        skip_next = False
        for line in description_file:
            if not '#' in line and not skip_next:
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
                self.view_mtxs[units[9][:-1]] = torch.tensor(mtx.astype(np.float32), dtype=torch.float32)
                camera_centers[units[9][:-1]] = torch.tensor(-(np.transpose(rm)) @ tm)
                look_ats[units[9][:-1]] = -torch.tensor(np.transpose(rm) @ np.array([0, 0, 1]).reshape(3, 1))
                skip_next = True
            else:
                skip_next = False
        description_file.close()

        points = np.zeros((len(camera_centers), 3))
        directions = np.zeros((len(camera_centers), 3))
        i = 0
        for key in camera_centers:
            points[i] = camera_centers[key].reshape(3)
            directions[i] = look_ats[key].reshape(3)
            i += 1
        average_center = self.sphere_search(points, directions)

        for key in self.view_mtxs:
            rotM = np.eye(4,4)
            rotM[0:3, 0:3] = self.view_mtxs[key][0:3, 0:3]
            trasM = np.eye(4,4)
            trasM[0:3, 3] = -1 * (camera_centers[key].reshape(3) - average_center)
            self.view_mtxs[key] = torch.tensor(rotM @ trasM, dtype=torch.float32)
        
        print("DatasetColmap: Estimated center is {}".format(average_center))

        if show_estimation:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

            ax.scatter(points[:, 0], points[:, 2], points[:, 1], marker='o')
            ax.scatter(points[:, 0] - average_center[0], points[:, 2]  - average_center[2], points[:, 1] - average_center[1], marker='x')
            ax.scatter([average_center[0]], [average_center[2]], [average_center[1]], marker='^')
            ax.scatter([0], [0], [0], marker='+')
            for i in range(0, points.shape[0]):
                ax.plot([points[i, 0], points[i, 0] + directions[i, 0]], [points[i, 2], points[i, 2] + directions[i, 2]], [points[i, 1], points[i, 1] + directions[i, 1]])
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_zlabel("Y")
            ax.set_xlim([-4, 4])
            ax.set_zlim([-4, 4])
            ax.set_ylim([-4, 4])
            plt.title("View distribution and mapping")
            plt.show()

        self.base_dir = os.path.join(self.base_dir, "test" if self.is_val else "train")
        self.image_files = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jepg')]
        self.n_images = len(self.image_files)
        assert self.n_images > 0, "DatasetColmap: The dataset is empty"

        # Determine resolution & aspect ratio
        self.resolution = _load_img(self.image_files[0]).shape[0:2]
        self.aspect = self.resolution[1] / self.resolution[0]

        if self.FLAGS.local_rank == 0:
            print("DatasetColmap: %d images with shape [%d, %d]" % (self.n_images, self.resolution[0], self.resolution[1]))

        # Pre-load from disc to avoid slow png parsing
        if self.FLAGS.pre_load:
            self.preloaded_data = []
            for i in range(self.n_images):
                self.preloaded_data += [self._parse_frame(i)]

    def find_sphere(self, P1, P2, P3, P4):
        K1 = np.power(P1[0], 2) + np.power(P1[1], 2) + np.power(P1[2], 2)
        K2 = np.power(P2[0], 2) + np.power(P2[1], 2) + np.power(P2[2], 2)
        K3 = np.power(P3[0], 2) + np.power(P3[1], 2) + np.power(P3[2], 2)
        K4 = np.power(P4[0], 2) + np.power(P4[1], 2) + np.power(P4[2], 2)

        X1 = -2 * P1[0]
        Y1 = -2 * P1[1]
        Z1 = -2 * P1[2]

        X2 = -2 * P2[0]
        Y2 = -2 * P2[1]
        Z2 = -2 * P2[2]

        X3 = -2 * P3[0]
        Y3 = -2 * P3[1]
        Z3 = -2 * P3[2]

        X4 = -2 * P4[0]
        Y4 = -2 * P4[1]
        Z4 = -2 * P4[2]

        F = (Y1 - Y2)/(Y1 - Y3) if (Y1 - Y3) != 0.0 else 0.0
        G = (Y1 - Y2)/(Y1 - Y4) if (Y1 - Y4) != 0.0 else 0.0
        H = (Z1 - Z2 - F * Z1 + F * Z3)/(Z1 - Z2 - G * Z1 + G * Z4) if (Z1 - Z2 - G * Z1 + G * Z4) != 0.0 else 0.0

        cA = K1 - K2 - F * K1 + F * K3 - H * K1 + H * K2 + H * G * K1 - H * G * K4
        dA = X1 - X2 - F * X1 + F * X3 - H * X1 + H * X2 + H * G * X1 - H * G * X4
        a = - cA / dA if dA != 0 else 0.0

        cC = K1 - K2 - F * K1 + F * K3 + (X1 - X2 - F * X1 + F * X3) * a
        dC = Z1 - Z2 - F * Z1 + F * Z3
        c = - cC / dC if dC != 0 else 0.0

        cB = (K1 - K2) + (Z1 - Z2) * c + (X1 - X2) * a
        dB = Y1 - Y2
        b = - cB / dB if dB != 0 else 0.0

        d = - (K1 + Z1 * c + X1 * a + Y1 * b)
        r = np.sqrt(np.power(c, 2) + np.power(b, 2) + np.power(a, 2) - d)

        return (np.array([a, b, c]), r)

    def sphere_search(self, points, directions, iterations=1000, generations=20, max_depth=50, alpha=0.6, exhaustive_search=False):
        np.random.seed(237)

        mods = np.linalg.norm(points, axis=-1)
        avr_mod = np.mean(mods)
        outliers = points[mods > 2 * avr_mod]
        if outliers.shape[0] > 0:
            print("DatasetColmap: Outliers found and removed: \n{}".format(outliers))
        else:
            print("DatasetColmap: No outliers have been found.")
        points = points[mods <= 2 * avr_mod]
        directions = directions[mods <= 2 * avr_mod]
        directions = directions / np.tile(np.linalg.norm(directions, axis=-1), (1, directions.shape[1])).reshape(directions.shape)

        n_points = points.shape[0]
        dims = points.shape[1]
        gridA = np.reshape(np.tile(points, (1, n_points)), (n_points, n_points, dims))
        gridB = np.transpose(gridA, axes=(1, 0, 2))
        distances = np.linalg.norm(gridA - gridB, axis=-1)
        
        for i in np.random.randint(0, n_points, 3):
            for j in np.random.randint(0, n_points, 3):
                assert distances[i, j] == np.linalg.norm(points[i] - points[j])

        max_radius = 2.0 * np.max(distances)

        best_sol = np.zeros(dims)
        best_sol_inds = np.zeros(4)
        best_cost = np.Infinity
        indices = np.array(range(0, n_points))
        for _ in range(0, iterations):
            # -------------- Constructive phase -------------- #
            #print("Constructing...")
            for _ in range(0, generations):
                built_sol = np.random.randint(0, n_points, 1)
                for _ in range(1, 4):
                    candidates = indices[np.logical_not(np.isin(indices, built_sol))]
                    costs = np.zeros(candidates.shape[0])
                    for i in built_sol:
                        for inx in range(0, candidates.shape[0]):
                            costs[inx] += 1/distances[i, candidates[inx]]
                    sorted_c = np.argsort(costs)
                    candidates = candidates[sorted_c]       
                    costs = costs[sorted_c]
                    c_min = np.min(costs)
                    c_max = np.max(costs)
                    filtered_c = costs <= (c_min + alpha * (c_max - c_min))
                    costs = costs[filtered_c]
                    candidates = candidates[filtered_c]
                    choosen = candidates[np.random.randint(0, candidates.shape[0], 1)]
                    built_sol = np.append(built_sol, choosen)
                assert built_sol.shape[0] == 4 and np.unique(built_sol).shape[0] == built_sol.shape[0]

                center, radius = self.find_sphere(points[built_sol[0]], points[built_sol[1]], points[built_sol[2]], points[built_sol[3]])
                mapped_directions = np.reshape(np.tile(center, (1, n_points)), (n_points, dims)) - points
                dists = np.linalg.norm(mapped_directions, axis=-1)
                mapped_directions = mapped_directions / np.tile(dists, (1, mapped_directions.shape[1])).reshape(mapped_directions.shape)
                cosines = np.ones(directions.shape[0]) - np.einsum('ij,ij->i', directions, mapped_directions)
                ca = np.sum(np.abs(dists - radius))
                cb = np.sum(np.abs(cosines))
                current_cost = 0.40 * ca + 1.0 * cb
                if current_cost < best_cost and radius <= max_radius:
                    best_cost = current_cost
                    best_sol = center
                    best_sol_inds = built_sol
                    print("DatasetColmap: [CONSTR] New best solution found with cost {:.4f}: {}".format(best_cost, best_sol))
            
            # -------------- Local improvement phase -------------- #
            #print("Exploring local neighborhood...")
            for _ in range(0, max_depth):
                shifts = np.unique(np.random.randint(-1, 1, (60, 4)), axis=0)
                neighs = ((np.reshape(np.tile(best_sol_inds, (1, shifts.shape[0])), (shifts.shape[0], 4)) + shifts) + n_points) % n_points
                n_costs = np.zeros(neighs.shape[0])
                for i in range(0, neighs.shape[0]):
                    center, radius = self.find_sphere(points[neighs[i][0]], points[neighs[i][1]], points[neighs[i][2]], points[neighs[i][3]])
                    mapped_directions = np.reshape(np.tile(center, (1, n_points)), (n_points, dims)) - points
                    dists = np.linalg.norm(mapped_directions, axis=-1)
                    mapped_directions = mapped_directions / np.tile(dists, (1, mapped_directions.shape[1])).reshape(mapped_directions.shape)
                    cosines = np.ones(directions.shape[0]) - np.einsum('ij,ij->i', directions, mapped_directions)
                    ca = np.sum(np.abs(dists - radius))
                    cb = np.sum(np.abs(cosines))
                    n_costs[i] = 0.40 * ca + 1.0 * cb
                n_sort = np.argsort(n_costs)
                neighs = neighs[n_sort]
                n_costs = n_costs[n_sort]
                if n_costs[0] < best_cost and radius <= max_radius:
                    best_cost = n_costs[0]
                    best_sol, _ = self.find_sphere(points[neighs[0][0]], points[neighs[0][1]], points[neighs[0][2]], points[neighs[0][3]])
                    best_sol_inds = neighs[0]
                    print("DatasetColmap: [SEARCH] New best solution found with cost {:.4f}: {}".format(best_cost, best_sol))
                else:
                    #print("No improvements found in local neighborhood")
                    if not exhaustive_search:
                        break

        return best_sol

    def _parse_frame(self, idx):
        # Config projection matrix (static, so could be precomputed)
        fovy = util.fovx_to_fovy(np.pi/2, self.aspect)
        proj = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
        #proj = util.ortographic(self.aspect, n=self.FLAGS.cam_near_far[0], f=self.FLAGS.cam_near_far[1])
        # Load image data and modelview matrix
        while self.image_files[idx].split('/')[-1] not in self.view_mtxs.keys():
            idx = (idx + 1) % self.n_images
        img = _load_img(self.image_files[idx])
        mv = self.view_mtxs[self.image_files[idx].split('/')[-1]]
        campos = torch.linalg.inv(mv)[:3, 3]
        mvp    = proj @ mv

        return img[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...] # Add batch dimension

    def __len__(self):
        return self.n_images if self.is_val else (self.FLAGS.iter+1)*self.FLAGS.batch

    def __getitem__(self, itr):
        iter_res = self.FLAGS.train_res
        
        img = []
        fovy = util.fovx_to_fovy(np.pi/2, self.aspect)

        if self.FLAGS.pre_load:
            img, mv, mvp, campos = self.preloaded_data[itr % self.n_images]
        else:
            img, mv, mvp, campos = self._parse_frame(self.cfg, itr % self.n_images)

        return {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : iter_res,
            'spp' : self.FLAGS.spp,
            'img' : img
        }
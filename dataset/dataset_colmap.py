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
from hvrl import colmap_utils

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

        parent_base = self.base_dir
        self.base_dir = os.path.join(self.base_dir, "test" if self.is_val else "train")
        self.image_files = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jepg')]
        self.n_images = len(self.image_files)
        assert self.n_images > 0, "DatasetColmap: The dataset is empty"

        # Determine resolution & aspect ratio
        self.resolution = _load_img(self.image_files[0]).shape[0:2]
        self.aspect = self.resolution[1] / self.resolution[0]

        assert os.path.exists(os.path.join(parent_base, "images.txt")), "DatasetColmap: images.txt file is missing."
        assert os.path.exists(os.path.join(parent_base, "points3D.txt")), "DatasetColmap: points3D.txt file is missing."
        print("DatasetColmap: loading camera information.")
        self.view_mtxs, view_points, camera_centers, look_ats = colmap_utils.load_cameras(os.path.join(parent_base, "images.txt"), FLAGS.colmap_res)

        camera_points = np.zeros((len(camera_centers), 3))
        camera_directions = np.zeros((len(camera_centers), 3))
        i = 0
        for key in camera_centers:
            camera_points[i] = camera_centers[key].reshape(3)
            camera_directions[i] = look_ats[key].reshape(3)
            i += 1

        average_center = np.zeros(3)
        suggested_center = np.zeros(3)
        found_center = np.zeros(3)
        filtered_samples = []

        if FLAGS.use_bb:
            print("DatasetColmap: loading point cloud information.")
            bb = np.array(FLAGS.bounding_box, dtype=np.uint)
            print("DatasetColmap: Filtering with bounding box ({}, {}) - ({}, {}).".format(bb[0, 0], bb[0, 1], bb[1, 0], bb[1, 1]))
            cloud_points, array_mapping = colmap_utils.load_points(os.path.join(parent_base, "points3D.txt"))
            filtered_samples, weights = colmap_utils.apply_bounding_box(bb, cloud_points, array_mapping, view_points, FLAGS.colmap_res)
            suggested_center = np.average(filtered_samples[:, 0:3], axis=0, weights=weights)
            found_center = colmap_utils.sphere_search(camera_points, camera_directions)
            centers = np.array([suggested_center, found_center])
            scores = np.array([ colmap_utils.evaluate_sphere_no_radius(suggested_center, camera_points, camera_directions), 
                                colmap_utils.evaluate_sphere_no_radius(found_center, camera_points, camera_directions)])
            average_center = np.average(centers, axis=0, weights=scores)
        else:
            average_center = colmap_utils.sphere_search(camera_points, camera_directions)

        for key in self.view_mtxs:
            self.view_mtxs[key] = colmap_utils.relocate_view_matrix(average_center, self.view_mtxs[key])
        
        print("DatasetColmap: Estimated center is {}".format(average_center))

        if show_estimation:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

            ax.scatter(camera_points[:, 0], camera_points[:, 2], camera_points[:, 1], marker='o')
            ax.scatter(camera_points[:, 0] - average_center[0], camera_points[:, 2]  - average_center[2], camera_points[:, 1] - average_center[1], marker='x')
            ax.scatter([average_center[0]], [average_center[2]], [average_center[1]], marker='^')
            if FLAGS.use_bb:
                ax.scatter(filtered_samples[:, 0], filtered_samples[:, 2], filtered_samples[:, 1], c=filtered_samples[:, 3:6])
                ax.scatter(suggested_center[0], suggested_center[2], suggested_center[1], c=np.array([[1, 0, 1]]), marker='x')
                ax.scatter(found_center[0], found_center[2], found_center[1], c=np.array([[0, 0.7, 0.5]]), marker='+')
            ax.scatter([0], [0], [0], marker='+')
            for i in range(0, camera_points.shape[0]):
                ax.plot([camera_points[i, 0], camera_points[i, 0] + camera_directions[i, 0]], [camera_points[i, 2], camera_points[i, 2] + camera_directions[i, 2]], [camera_points[i, 1], camera_points[i, 1] + camera_directions[i, 1]])
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_zlabel("Y")
            ax.set_xlim([-4, 4])
            ax.set_zlim([-4, 4])
            ax.set_ylim([-4, 4])
            plt.title("View distribution and mapping")
            plt.show()

        if self.FLAGS.local_rank == 0:
            print("DatasetColmap: %d images with shape [%d, %d]" % (self.n_images, self.resolution[0], self.resolution[1]))

        # Pre-load from disc to avoid slow png parsing
        if self.FLAGS.pre_load:
            self.preloaded_data = []
            for i in range(self.n_images):
                self.preloaded_data += [self._parse_frame(i)]

    def _parse_frame(self, idx):
        # Config projection matrix (static, so could be precomputed)
        fovy = util.fovx_to_fovy(np.pi/2, self.aspect)
        proj = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
        #proj = util.ortographic(self.aspect, n=self.FLAGS.cam_near_far[0], f=self.FLAGS.cam_near_far[1])
        # Load image data and modelview matrix
        while self.image_files[idx].split('/')[-1] not in self.view_mtxs.keys():
            idx = (idx + 1) % self.n_images
        img = _load_img(self.image_files[idx])
        mv = torch.tensor(self.view_mtxs[self.image_files[idx].split('/')[-1]], dtype=torch.float32)
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
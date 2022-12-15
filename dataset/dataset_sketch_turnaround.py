import os
import glob
import json

import torch
import numpy as np

from render import util

from .dataset import Dataset
from PIL import Image

def _load_img(path, angle=0.0):
    files = glob.glob(path)
    assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
    #img = util.load_image_raw(files[0])
    img = Image.open(files[0])
    img = img.convert('RGBA').rotate(angle)
    img = np.array(img)
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

class DatasetSketchTurnAround(Dataset):
    def __init__(self, base_path, FLAGS, validation=False):
        self.FLAGS = FLAGS
        self.validation = 0.2
        self.is_val = validation
        self.base_dir = base_path

        # Load config / transforms
        self.cfg = json.load(open(os.path.join(base_path, "info.json"), 'r'))
        self.n_images = self.cfg['size']
        self.angle_frags = self.n_images
        self.name = self.cfg['name']
        self.filenames = [f for f in glob.glob(os.path.join(self.base_dir, "*")) if f.lower().endswith('png')]
        self.filenames = sorted(self.filenames, key=lambda x: float(x.split('/')[-1].split('_')[-1][:-4]) + (0.99 if len(x.split('/')[-1].split('_')[-1][:-4].split('.')) == 1 else 0.0))

        #for f in self.filenames: print(f)
        #exit()

        self.n_images = len(self.filenames)
        assert len(self.filenames) > 0, "DatasetSketchTurnaround: the dataset is empty"

        # Determine resolution & aspect ratio
        self.resolution = _load_img(self.filenames[0]).shape[0:2]
        self.aspect = self.resolution[1] / self.resolution[0]

        if self.FLAGS.local_rank == 0:
            print("DatasetSketchTurnAround: %d images with shape [%d, %d]" % (self.n_images, self.resolution[0], self.resolution[1]))

        # Pre-load from disc to avoid slow png parsing
        if self.FLAGS.pre_load:
            self.preloaded_data = []
            for i in range(self.n_images):
                self.preloaded_data += [self._parse_frame(self.cfg, i)]

    def _parse_frame(self, cfg, idx):
        # Config projection matrix (static, so could be precomputed)
        fovy = util.fovx_to_fovy(np.pi/2, self.aspect)
        #proj = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
        proj = util.ortographic(self.aspect, n=self.FLAGS.cam_near_far[0], f=self.FLAGS.cam_near_far[1])
        # Load image data and modelview matrix
        #angle = np.random.rand() * (2.0 * np.pi)
        #img = _load_img(self.filenames[idx], angle=angle * (-180.0 / np.pi))
        img = _load_img(self.filenames[idx])
        #mv = util.random_rotation()
        #lookat = -np.transpose(mv[0:3, 0:3]) @ np.array([0, 0, 1]).reshape(3, 1)
        #lookat = np.reshape(lookat, 3)
        #lookat = 2.0 * (lookat / np.linalg.norm(lookat))
        #mv = mv @ util.translate(lookat[0], lookat[1], lookat[2]) #@ util.rotate_y((-2 * np.pi / self.angle_frags) * idx)
        mv = util.translate(0, 0, -2.0) @ util.rotate_y((-2 * np.pi / self.angle_frags) * idx)
        #mv = util.rotate_y(-np.pi/2) @ util.rotate_x(angle) @ util.rotate_y(np.pi/2) @ util.translate(0, 0, -2.0) @ util.rotate_y((-2 * np.pi / self.angle_frags) * idx)
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
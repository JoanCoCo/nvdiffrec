import argparse

import numpy as np
import torch
import nvdiffrast.torch as dr

import render.renderutils as ru
from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import light
from render import render
import numpy as np

from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist, euclidean
from math import sqrt

from hvrl import colmap_utils

MODEL_FOLDER = "out/sketch_dog_orto_2/mesh"
MESH_FILE = MODEL_FOLDER + "/mesh.obj"
MATERIAL_FILE = MODEL_FOLDER + "/mesh.mtl"
RESULT_FILE = "render"
SPP = 1
RESOLUTION = [550, 550]
ORTO = False
BOUNDING_BOX = np.array(
    [
        [(1920-470)/2,       (1342-1200)/2       ],
        [(1920-470)/2 + 470, (1342-1200)/2 + 1200]
    ], dtype=np.uint) 

def initial_guess_material(geometry, FLAGS, init_mat):
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')
    
    kd_map_opt = texture.create_trainable(init_mat['kd'], FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])
    ks_map_opt = texture.create_trainable(init_mat['ks'], FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])

    # Setup normal map
    if 'normal' not in init_mat:
        normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])
    else:
        normal_map_opt = texture.create_trainable(init_mat['normal'], FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])

    mat = material.Material({
        'kd'     : kd_map_opt,
        'ks'     : ks_map_opt,
        'normal' : normal_map_opt
    })

    mat['bsdf'] = init_mat['bsdf']

    return mat

def generate_scene():
    proj_mtx = util.ortographic(RESOLUTION[1] / RESOLUTION[0], n=0.1, f=1000.0) if ORTO else util.perspective(np.deg2rad(45), RESOLUTION[1] / RESOLUTION[0], 0.1, 1000.0)
    # Random rotation/translation matrix for optimization.

    mv = torch.tensor(load_matrix(), dtype=torch.float32)
    
    #mv = util.translate(1, 0, -3) #@ util.rotate_x(-1 * np.pi / 4)
    print(mv)

    mvp = proj_mtx @ mv
    campos = torch.linalg.inv(mv)[:3, 3]

    return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda() # Add batch dimension

def load_matrix(source="data/colmap/aloy/"):
    """
    view_mtxs, view_points, camera_centers, look_ats = colmap_utils.load_cameras(source + "images.txt", RESOLUTION)
    cloud_points, array_mapping = colmap_utils.load_points(source + "points3D.txt")

    camera_points = np.zeros((len(camera_centers), 3))
    camera_directions = np.zeros((len(camera_centers), 3))
    i = 0
    for key in camera_centers:
        camera_points[i] = camera_centers[key].reshape(3)
        camera_directions[i] = look_ats[key].reshape(3)
        i += 1
    print(BOUNDING_BOX)
    filtered_samples, weights = colmap_utils.apply_bounding_box(BOUNDING_BOX, cloud_points, array_mapping, view_points, RESOLUTION)
    suggested_center = np.average(filtered_samples[:, 0:3], axis=0, weights=weights)

    found_center = colmap_utils.sphere_search(camera_points, camera_directions)

    centers = np.array([suggested_center, found_center])
    scores = np.array([ colmap_utils.evaluate_sphere_no_radius(suggested_center, camera_points, camera_directions), 
                        colmap_utils.evaluate_sphere_no_radius(found_center, camera_points, camera_directions)])
    average_center = np.average(centers, axis=0, weights=scores)
    
    for key in view_mtxs:
        view_mtxs[key] = colmap_utils.relocate_view_matrix(average_center, view_mtxs[key])
        #view_mtxs[key] = view_mtxs[key] @ util.translate(-average_center[0], -average_center[1], -average_center[2])
    
    print("Center: {}".format(average_center))

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    
    ax.scatter(camera_points[:, 0], camera_points[:, 2], camera_points[:, 1], marker='o')
    #ax.scatter(camera_points[:, 0] - average_center[0], camera_points[:, 2]  - average_center[2], camera_points[:, 1] - average_center[1], marker='x')
    ax.scatter([average_center[0]], [average_center[2]], [average_center[1]], marker='^')
    #ax.scatter([0], [0], [0], marker='+')
    ax.scatter(filtered_samples[:, 0], filtered_samples[:, 2], filtered_samples[:, 1], c=filtered_samples[:, 3:6])
    ax.scatter(suggested_center[0], suggested_center[2], suggested_center[1], c=np.array([[1, 0, 1]]), marker='x')
    ax.scatter(found_center[0], found_center[2], found_center[1], c=np.array([[0, 0.7, 0.5]]), marker='+')
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

    return view_mtxs["img_1.png"]
    """

    idx = 24
    angle_frags = 28
    mv = util.rotate_y(-np.pi/2) @ util.rotate_x(0*np.pi/2) @ util.rotate_y(np.pi/2) @ util.translate(0, 0, -2.0) @ util.rotate_y((-2 * np.pi / angle_frags) * idx)
    rm = np.copy(mv[0:3,0:3])
    #look_at = -np.transpose(rm) @ np.array([0, 0, 1]).reshape(3, 1)
    #print(look_at)
    return mv

def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)
    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)
        
        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/R
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y
        
        if euclidean(y, y1) < eps:
            return y1

        y = y1

#https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/
def findCircle(x1, y1, x2, y2, x3, y3) :
    x12 = x1 - x2
    x13 = x1 - x3
 
    y12 = y1 - y2
    y13 = y1 - y3
 
    y31 = y3 - y1
    y21 = y2 - y1
 
    x31 = x3 - x1
    x21 = x2 - x1
 
    # x1^2 - x3^2
    sx13 = pow(x1, 2) - pow(x3, 2)
 
    # y1^2 - y3^2
    sy13 = pow(y1, 2) - pow(y3, 2)
 
    sx21 = pow(x2, 2) - pow(x1, 2)
    sy21 = pow(y2, 2) - pow(y1, 2)
 
    f = (((sx13) * (x12) + (sy13) *
          (x12) + (sx21) * (x13) +
          (sy21) * (x13)) // (2 *
          ((y31) * (x12) - (y21) * (x13))))
             
    g = (((sx13) * (y12) + (sy13) * (y12) +
          (sx21) * (y13) + (sy21) * (y13)) //
          (2 * ((x31) * (y12) - (x21) * (y13))))
 
    c = (-pow(x1, 2) - pow(y1, 2) -
         2 * g * x1 - 2 * f * y1)
 
    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
    # where centre is (h = -g, k = -f) and
    # radius r as r^2 = h^2 + k^2 - c
    h = -g
    k = -f
    sqr_of_r = h * h + k * k - c
 
    # r is the radius
    r = round(sqrt(sqr_of_r), 5)
 
    return (h, k)

parser = argparse.ArgumentParser(description='nvdiffrec')

base_mesh = mesh.load_mesh(MESH_FILE, mtl_override=MATERIAL_FILE)
base_mesh = mesh.compute_tangents(base_mesh)

glctx = dr.RasterizeGLContext()
#lgt = light.load_env("data/irrmaps/aerodynamics_workshop_2k.hdr")
lgt = light.create_white_env(512, scale=1.0)
mv, mvp, campos = generate_scene()

img = render.render_mesh(glctx, base_mesh, mvp, campos, lgt, RESOLUTION, spp=SPP, 
                                num_layers=4, msaa=True, background=None)['shaded']

util.save_image(RESULT_FILE + ("_orto" if ORTO else "_persp") + ".png", img[0].cpu().detach().numpy())

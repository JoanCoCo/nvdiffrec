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

MODEL_FOLDER = "out/nerf_drums"
MESH_FILE = MODEL_FOLDER + "/mesh/mesh.obj"
MATERIAL_FILE = MODEL_FOLDER + "/mesh/mesh.mtl"
RESULT_FILE = "render"
SPP = 1
RESOLUTION = [1080, 1920]
ORTO = False

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
    #proj_mtx = util.perspective(np.deg2rad(45), RESOLUTION[1] / RESOLUTION[0], 0.1, 1000.0)
    proj_mtx = util.ortographic(RESOLUTION[1] / RESOLUTION[0], n=0.1, f=1000.0) if ORTO else util.perspective(np.deg2rad(45), RESOLUTION[1] / RESOLUTION[0], 0.1, 1000.0)
    # Random rotation/translation matrix for optimization.

    mv = util.translate(0, 0, -3.0) @ util.rotate_x(-1 * np.pi / 4)
    print(mv)

    mvp = proj_mtx @ mv
    campos = torch.linalg.inv(mv)[:3, 3]

    return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda() # Add batch dimension


parser = argparse.ArgumentParser(description='nvdiffrec')

base_mesh = mesh.load_mesh(MESH_FILE, mtl_override=MATERIAL_FILE)
base_mesh = mesh.compute_tangents(base_mesh)

glctx = dr.RasterizeGLContext()
#lgt = light.load_env("data/irrmaps/aerodynamics_workshop_2k.hdr")
lgt = light.create_white_env(512, scale=0.5)
mv, mvp, campos = generate_scene()

img = render.render_mesh(glctx, base_mesh, mvp, campos, lgt, RESOLUTION, spp=SPP, 
                                num_layers=4, msaa=True, background=None)['shaded']

util.save_image(RESULT_FILE + ("_orto" if ORTO else "_persp") + ".png", img[0].cpu().detach().numpy())
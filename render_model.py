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

MODEL_FOLDER = "out/nerf_drums/mesh"
MESH_FILE = MODEL_FOLDER + "/mesh.obj"
MATERIAL_FILE = MODEL_FOLDER + "/mesh.mtl"
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

    mv = load_matrix()
    
    #mv = util.translate(1, 0, -3) #@ util.rotate_x(-1 * np.pi / 4)
    print(mv)

    mvp = proj_mtx @ mv
    campos = torch.linalg.inv(mv)[:3, 3]

    return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda() # Add batch dimension

def load_matrix(source="/home/joancc/Documents/TFM/Totoro test/images.txt"):
    description_file = open(source, "r")
    skip_next = False
    view_mtxs = {}
    camera_centers = {}
    look_ats = {}
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

            #points = np.append(points, [[tx, ty, tz]])

            mtx = np.hstack((rm, tm))
            mtx = np.vstack((mtx, np.zeros((1, 4))))
            mtx[3,3] = 1
            view_mtxs[units[9][:-1]] = torch.tensor(mtx.astype(np.float32), dtype=torch.float32)

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

    """
    average_centers = []
    for i in range(0, points.shape[0]):
        #cx, cz = findCircle(points[i, 0], points[i, 2], points[j, 0], points[j, 2], points[w, 0], points[w, 2])
        #ac = np.array([cx, np.mean(np.array([points[i, 1], points[j, 1], points[w, 1]])), cz])
        ac = find_sphere(points[i], points[(i + 5) % points.shape[0]], points[(i + 10) % points.shape[0]], points[(i + 15) % points.shape[0]])
        average_centers.append(ac)
    
    for _ in range(0, 5000):
        p1 = np.random.randint(0, points.shape[0], (1))[0]
        p2 = np.random.randint(0, points.shape[0], (1))[0]
        while p2 == p1:
            p2 = (p2 + 1) % points.shape[0]
        p3 = np.random.randint(0, points.shape[0], (1))[0]
        while p3 == p2 or p3 == p1:
            p3 = (p3 + 1) % points.shape[0]
        cx, cz = findCircle(points[p1, 0], points[p1, 2], points[p2, 0], points[p2, 2], points[p3, 0], points[p3, 2])
        ac = np.array([cx, np.mean(np.array([points[p1, 1], points[p2, 1], points[p3, 1]])), cz])
        average_centers.append(ac)
    
    average_center = np.mean(np.array(average_centers), axis=0)
    average_center[1] = 0.0
    """
    average_center = sphere_search(points, directions)
    
    for key in view_mtxs:
        rotM = np.eye(4,4)
        rotM[0:3, 0:3] = view_mtxs[key][0:3, 0:3]
        trasM = np.eye(4,4)
        trasM[0:3, 3] = -1 * (camera_centers[key].reshape(3) - average_center)
        view_mtxs[key] = torch.tensor(rotM @ trasM, dtype=torch.float32)
        #view_mtxs[key] = view_mtxs[key] @ util.translate(-average_center[0], -average_center[1], -average_center[2])
    
    print("Center: {}".format(average_center))

    import matplotlib.pyplot as plt
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
    ax.set_xlim([-10, 10])
    ax.set_zlim([-10, 10])
    ax.set_ylim([-10, 10])
    plt.title("View distribution and mapping")
    plt.show()

    return view_mtxs["img_1.png"]

def find_sphere(P1, P2, P3, P4):
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

def sphere_search(points, directions, iterations=1000, generations=20, max_depth=50, alpha=0.6, exhaustive_search=False):
    np.random.seed(237)

    mods = np.linalg.norm(points, axis=-1)
    avr_mod = np.mean(mods)
    outliers = points[mods > 2 * avr_mod]
    if outliers.shape[0] > 0:
        print("Outliers found and removed: \n{}".format(outliers))
    else:
        print("No outliers have been found.")
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

    max_radius = np.inf #2.0 * np.max(distances)

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
                        costs[inx] += 1/distances[i, candidates[inx]] if distances[i, candidates[inx]] > 0 else 1000000
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

            center, radius = find_sphere(points[built_sol[0]], points[built_sol[1]], points[built_sol[2]], points[built_sol[3]])
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
                print("[CONSTR] New best solution found with cost {:.4f}: {}".format(best_cost, best_sol))
        
        # -------------- Local improvement phase -------------- #
        #print("Exploring local neighborhood...")
        for _ in range(0, max_depth):
            shifts = np.unique(np.random.randint(-1, 1, (60, 4)), axis=0)
            neighs = ((np.reshape(np.tile(best_sol_inds, (1, shifts.shape[0])), (shifts.shape[0], 4)) + shifts) + n_points) % n_points
            n_costs = np.zeros(neighs.shape[0])
            for i in range(0, neighs.shape[0]):
                center, radius = find_sphere(points[neighs[i][0]], points[neighs[i][1]], points[neighs[i][2]], points[neighs[i][3]])
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
                best_sol, _ = find_sphere(points[neighs[0][0]], points[neighs[0][1]], points[neighs[0][2]], points[neighs[0][3]])
                best_sol_inds = neighs[0]
                print("[SEARCH] New best solution found with cost {:.4f}: {}".format(best_cost, best_sol))
            else:
                #print("No improvements found in local neighborhood")
                if not exhaustive_search:
                    break

    return best_sol

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

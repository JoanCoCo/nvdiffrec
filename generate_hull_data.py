import os
import glob
import numpy as np
from PIL import Image
from render import util
from hvrl import colmap_utils

SOURCE_FOLDER = "Visual Hull/boy"
SIZE = 8
RESULT_FOLDER = "Visual Hull/boy_hull"

FROM_COLMAP = False

if __name__ == "__main__":
    if not os.path.exists(RESULT_FOLDER):
        os.mkdir(RESULT_FOLDER)
    with open(os.path.join(RESULT_FOLDER, "cameras.txt"), 'w') as camerafile:
        if not FROM_COLMAP:
            for i in range(0, SIZE):
                image = np.array(Image.open(os.path.join(SOURCE_FOLDER, "{:d}.png".format(i))).convert("RGBA"))
                silt = np.ones_like(image) * 255
                silt[image[:,:,3] == 0] = 0
                #silt[:,:,3] = 255
                Image.fromarray(silt).save(os.path.join(RESULT_FOLDER, "sil_{:d}.png").format(i+1))
                camerafile.write("cameraVec({:d}).efl = 15000;\n".format(i+1))
                camerafile.write("cameraVec({:d}).u0 = {:.2f};\n".format(i+1, image.shape[1] / 2.0 + 0.5))
                camerafile.write("cameraVec({:d}).v0 = {:.2f};\n".format(i+1, image.shape[0] / 2.0 + 0.5))
                mv = util.rotate_y(np.pi) @ util.translate(0, 0, -2.0) @ util.rotate_y((-2 * np.pi / SIZE) * i)
                camerafile.write("cameraVec({:d}).pose = [{:f} {:f} {:f} {:f}; {:f} {:f} {:f} {:f}; {:f} {:f} {:f} {:f}; {:f} {:f} {:f} {:f}];\n".format(
                    i+1,
                    mv[0,0], mv[0,1], mv[0,2], mv[0,3],
                    mv[1,0], mv[1,1], mv[1,2], mv[1,3],
                    mv[2,0], mv[2,1], mv[2,2], mv[2,3],
                    mv[3,0], mv[3,1], mv[3,2], mv[3,3]
                ))
        else:
            imagefiles = [f for f in sorted(glob.glob(os.path.join(SOURCE_FOLDER, "*"))) if f.lower().endswith('png')]
            mvs, _, _, _ = colmap_utils.load_cameras(os.path.join(SOURCE_FOLDER, "images.txt"), [1342, 1920])
            for i, file in enumerate(imagefiles):
                #i = int(file.split('/')[-1].split('_')[-1].split('.')[0])
                image = np.array(Image.open(file).convert("RGBA"))
                silt = np.ones_like(image) * 255
                silt[image[:,:,3] == 0] = 0
                silt[:,:,3] = 255
                Image.fromarray(silt).save(os.path.join(RESULT_FOLDER, "sil_{:d}.png").format(i+1))
                camerafile.write("cameraVec({:d}).efl = 1620.8664;\n".format(i+1))
                camerafile.write("cameraVec({:d}).u0 = 960;\n".format(i+1))
                camerafile.write("cameraVec({:d}).v0 = 671;\n".format(i+1))
                mv = util.rotate_y(np.pi) @ mvs[file.split('/')[-1]]
                camerafile.write("cameraVec({:d}).pose = [{:f} {:f} {:f} {:f}; {:f} {:f} {:f} {:f}; {:f} {:f} {:f} {:f}; {:f} {:f} {:f} {:f}];\n".format(
                    i+1,
                    mv[0,0], mv[0,1], mv[0,2], mv[0,3],
                    mv[1,0], mv[1,1], mv[1,2], mv[1,3],
                    mv[2,0], mv[2,1], mv[2,2], mv[2,3],
                    mv[3,0], mv[3,1], mv[3,2], mv[3,3]
                ))
        camerafile.close()

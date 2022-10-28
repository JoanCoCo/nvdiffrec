import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch_scatter import scatter_mean
import cv2
from PIL import Image

from interpolation.models.sgm_model.my_models import create_VGGFeatNet
from interpolation.models.sgm_model.gen_sgm import dline_of, trapped_ball_processed, squeeze_label_map, superpixel_pooling, superpixel_count, mutual_matching, get_guidance_flow
from interpolation import models

def generate_flows(frame_pairs, rankSumThr=0):
    ## make models 
    vggNet = create_VGGFeatNet()
    vggNet = vggNet.cuda()

    toTensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    flows = []

    for frames in frame_pairs:
        img1_rs, img3_rs = frames
        boundImg1 = dline_of(img1_rs, 1, 20, [30,40,30]).astype(np.uint8)
        boundImg3 = dline_of(img3_rs, 1, 20, [30,40,30]).astype(np.uint8)
        ret, binMap1 = cv2.threshold(boundImg1, 220, 255, cv2.THRESH_BINARY)
        ret, binMap3 = cv2.threshold(boundImg3, 220, 255, cv2.THRESH_BINARY)

        fillMap1 = trapped_ball_processed(binMap1, img1_rs)
        fillMap3 = trapped_ball_processed(binMap3, img3_rs)

        labelMap1 = squeeze_label_map(fillMap1)
        labelMap3 = squeeze_label_map(fillMap3)

        # VGG features
        img1_rgb = cv2.cvtColor(img1_rs, cv2.COLOR_BGR2RGB)
        img3_rgb = cv2.cvtColor(img3_rs, cv2.COLOR_BGR2RGB)

        img1_tensor = normalize(toTensor(img1_rgb/255.).float())
        img1_tensor = img1_tensor.unsqueeze(dim=0)
        img3_tensor = normalize(toTensor(img3_rgb/255.).float())
        img3_tensor = img3_tensor.unsqueeze(dim=0)
        img1_tensor = img1_tensor.cuda()
        img3_tensor = img3_tensor.cuda()

        # featx1_1 = vggNet.slice1(img1_tensor)
        # featx1_3 = vggNet.slice1(img3_tensor)
        featx1_1, featx2_1, featx4_1, featx8_1, featx16_1 = vggNet(img1_tensor)
        featx1_3, featx2_3, featx4_3, featx8_3, featx16_3 = vggNet(img3_tensor)

        # superpixel pooling
        labelMap1_x2 = labelMap1[1::2,1::2]
        labelMap1_x4 = labelMap1_x2[1::2,1::2]
        labelMap1_x8 = labelMap1_x4[1::2,1::2]
        # labelMap1_x16 = labelMap1_x8[1::2,1::2]
        labelMap3_x2 = labelMap3[1::2,1::2]
        labelMap3_x4 = labelMap3_x2[1::2,1::2]
        labelMap3_x8 = labelMap3_x4[1::2,1::2]
        # labelMap3_x16 = labelMap3_x8[1::2,1::2]

        featx1_pool_1 = superpixel_pooling(featx1_1[0], labelMap1, True)
        featx2_pool_1 = superpixel_pooling(featx2_1[0], labelMap1_x2, True)
        featx4_pool_1 = superpixel_pooling(featx4_1[0], labelMap1_x4, True)
        featx8_pool_1 = superpixel_pooling(featx8_1[0], labelMap1_x8, True)
        # featx16_pool_1 = superpixel_pooling(featx16_1[0], labelMap1_x16, use_gpu)
        featx1_pool_3 = superpixel_pooling(featx1_3[0], labelMap3, True)
        featx2_pool_3 = superpixel_pooling(featx2_3[0], labelMap3_x2, True)
        featx4_pool_3 = superpixel_pooling(featx4_3[0], labelMap3_x4, True)
        featx8_pool_3 = superpixel_pooling(featx8_3[0], labelMap3_x8, True)
        # featx16_pool_3 = superpixel_pooling(featx16_3[0], labelMap3_x16, use_gpu)
        
        feat_pool_1 = torch.cat([featx1_pool_1, featx2_pool_1, featx4_pool_1, featx8_pool_1], dim=0)
        feat_pool_3 = torch.cat([featx1_pool_3, featx2_pool_3, featx4_pool_3, featx8_pool_3], dim=0)

        # normalization
        feat_p1_tmp = feat_pool_1 - feat_pool_1.min(dim=0)[0]
        feat_p1_norm = feat_p1_tmp/feat_p1_tmp.sum(dim=0)
        feat_p3_tmp = feat_pool_3 - feat_pool_3.min(dim=0)[0]
        feat_p3_norm = feat_p3_tmp/feat_p3_tmp.sum(dim=0)

        # for pixel distance
        lH, lW = labelMap1.shape
        gridX, gridY = np.meshgrid(np.arange(lW), np.arange(lH))

        gridX_flat = torch.tensor(gridX.astype(np.float), requires_grad=False).reshape(lH*lW)
        gridY_flat = torch.tensor(gridY.astype(np.float), requires_grad=False).reshape(lH*lW)

        labelMap1_flat = torch.tensor(labelMap1.reshape(lH*lW)).long()
        labelMap3_flat = torch.tensor(labelMap3.reshape(lH*lW)).long()

        gridX_flat = gridX_flat.cuda()
        gridY_flat = gridY_flat.cuda()
        labelMap1_flat = labelMap1_flat.cuda()
        labelMap3_flat = labelMap3_flat.cuda()

        mean_X_1 = scatter_mean(gridX_flat, labelMap1_flat).cpu().numpy()
        mean_Y_1 = scatter_mean(gridY_flat, labelMap1_flat).cpu().numpy()
        mean_X_3 = scatter_mean(gridX_flat, labelMap3_flat).cpu().numpy()
        mean_Y_3 = scatter_mean(gridY_flat, labelMap3_flat).cpu().numpy()

        # pixel count in superpixel
        pixelCounts_1 = superpixel_count(labelMap1)
        pixelCounts_3 = superpixel_count(labelMap3)

        # some other distance
        labelNum_1 = len(np.unique(labelMap1))
        labelNum_3 = len(np.unique(labelMap3))
        print('label num: %d, %d'%(labelNum_1, labelNum_3))

        maxDist = np.linalg.norm([lH,lW])
        maxPixNum = lH*lW

        corrMap = torch.zeros(labelNum_1, labelNum_3)
        ctxSimMap = torch.zeros(labelNum_1, labelNum_3)

        for x in range(labelNum_1):
            for y in range(labelNum_3):
                corrMap[x,y] = torch.sum(torch.min(feat_p1_norm[:,x], feat_p3_norm[:,y]))

                # pixel number as similarity
                num_1 = float(pixelCounts_1[x])
                num_3 = float(pixelCounts_3[y])
                
                sizeDiff = max(num_1/num_3, num_3/num_1)
                if sizeDiff > 3:
                    corrMap[x,y] -= sizeDiff/20

                # spatial distance as similarity
                dist = np.linalg.norm([mean_X_1[x] - mean_X_3[y], mean_Y_1[x] - mean_Y_3[y]])/maxDist
                
                if dist > 0.14:
                    corrMap[x,y] -= dist/5


        matchingMetaData = mutual_matching(corrMap)
        rankSum_1to3, matching_1to3, sortedCorrMap_1 = matchingMetaData[:3]
        rankSum_3to1, matching_3to1, sortedCorrMap_3 = matchingMetaData[3:]

        # create flows
        guideflow_1to3, matching_color_1to3 = get_guidance_flow(labelMap1, labelMap3, img1_rs, img3_rs,
                                rankSum_1to3, matching_1to3, sortedCorrMap_1,
                                mean_X_1, mean_Y_1, mean_X_3, mean_Y_3, 
                                rank_sum_thr = rankSumThr, use_gpu = True, maxPixNum=maxPixNum)
        guideflow_3to1, matching_color_3to1 = get_guidance_flow(labelMap3, labelMap1, img3_rs, img1_rs,
                                rankSum_3to1, matching_3to1, sortedCorrMap_3.transpose(1,0), 
                                mean_X_3, mean_Y_3, mean_X_1, mean_Y_1, 
                                rank_sum_thr = rankSumThr, use_gpu = True, maxPixNum=maxPixNum)
        guideflow_1to3 = torch.tensor(guideflow_1to3)
        guideflow_3to1 = torch.tensor(guideflow_3to1)
        flows += [(guideflow_1to3[None, :], guideflow_3to1[None, :])]

    del vggNet
    torch.cuda.empty_cache()

    return flows

def build_parings(sequence):
    parings = []
    for i in range(1, len(sequence)):
        parings += [(sequence[i-1], sequence[i])]
    return parings

def expand_sequence(frames, intermediante_frames=2, mean=[0., 0., 0.], std=[1, 1, 1], model='AnimeInterp', checkpoint='interpolation/checkpoints/anime_interp_full.ckpt'):
    normalize1 = transforms.Normalize(mean, [1.0, 1.0, 1.0])
    normalize2 = transforms.Normalize([0, 0, 0], std)
    trans = transforms.Compose([transforms.ToTensor(), normalize1, normalize2, ])

    # prepare model
    model = getattr(models, model)(None).cuda()
    model = nn.DataParallel(model)
    to_img = transforms.ToPILImage()

    revmean = [-x for x in mean]
    revstd = [1.0 / x for x in std]
    revnormalize1 = transforms.Normalize([0.0, 0.0, 0.0], revstd)
    revnormalize2 = transforms.Normalize(revmean, [1.0, 1.0, 1.0])
    revNormalize = transforms.Compose([revnormalize1, revnormalize2])

    # load weights
    dict1 = torch.load(checkpoint)
    model.load_state_dict(dict1['model_state_dict'], strict=False)

    img_pairs = build_parings(frames)
    flow_pairs = generate_flows(img_pairs)

    extended_frames = [img_pairs[0][0]]

    #output_size = None

    for i in range(0, len(img_pairs)):
        frame1, frame2 = img_pairs[i]
        #if output_size is None:
        #    output_size = (frame1.size()[1], frame2.size()[2])
        frame1 = trans(frame1)
        frame2 = trans(frame2)
        frame1 = frame1[None, :]
        frame2 = frame2[None, :]

        # initial SGM flow
        F12i, F21i = flow_pairs[i]

        F12i = F12i.float().cuda() 
        F21i = F21i.float().cuda()

        I1 = frame1.cuda()
        I2 = frame2.cuda()

        for tt in range(intermediante_frames):
            x = intermediante_frames
            t = 1.0/(x+1) * (tt + 1)
                
            outputs = model(I1, I2, F12i, F21i, t)

            It_warp = outputs[0]

            res_img = to_img(revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0))
            #res_img = res_img.resize(output_size, Image.ANTIALIAS)
            extended_frames += [np.array(res_img)]

            del outputs
            del It_warp
            
        extended_frames += [img_pairs[i][1]]
        
        del F12i
        del F21i
        del I1
        del I2

        torch.cuda.empty_cache()
    
    return extended_frames 
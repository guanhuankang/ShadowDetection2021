import torchvision
import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image
import random

import config
from shadownet import ShadowNet
from misc import loadModel, check_mkdir, crf_refine

args = {
    "scale":(320,320),
    "calcBER": True
}

def test():
    global args

    net = ShadowNet().cuda()
    loadModel(net, config.model)
    net.eval()

    with torch.no_grad():
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        trans_img = torchvision.transforms.Normalize(mean, std)
        trans_back = torchvision.transforms.Normalize(-mean/std, 1.0/std)

        img_list = os.listdir(config.img_path)
        tot = len(img_list)
        cnt = 1
        for _ in img_list:
            if _[-4::] not in [".jpg", ".png", ".JPG", ".PNG"]:continue
            img1 = Image.open( os.path.join(config.img_path, _) )
            img2 = torchvision.transforms.ToTensor()(
                torchvision.transforms.Resize( args["scale"] )(
                    img1
                )
            )
            img3 = trans_img(img2).unsqueeze(0).cuda()
            ## Our Model
            output = net(img3)
            ## CRF
            size = tuple(reversed(img1.size) )
            _mask = torchvision.transforms.ToPILImage()( nn.Sigmoid()( output["prediction"][0][0] ).cpu() )
            _mask = torchvision.transforms.Resize(size)(_mask)
            mask = crf_refine(
                np.array(img1),
                np.array( _mask )
            )
            ## Save
            check_mkdir(config.out_path)
            mask = Image.fromarray(mask)
            mask.save(os.path.join(config.out_path, _[0:-4]+".png" ))

            print("%3d/%3d"%(cnt, tot))
            cnt += 1

def calcBER():
    gtL = os.listdir(config.gt_path)
    outL = os.listdir(config.out_path)
    gtL.sort()
    outL.sort()

    ## Check
    n = len(gtL)
    if len(outL)!=n:print("The numbers of GT and Output are not matched!");exit(0)
    for _ in range(n):
        if gtL[_][0:-4]!=outL[_][0:-4]:print("GT and Output aren't consistent!");exit(0)

    ## Calc TP,TN,FP,FN
    TP, TN, FP, FN = 0.0, 0.0, 0.0, 0.0
    for _ in range(n):
        gt = np.array( Image.open(os.path.join(config.gt_path, gtL[_])).convert("L"), dtype=float )
        out = np.array( Image.open(os.path.join(config.out_path, outL[_])).convert("L"), dtype=float )
        gt = gt > 127.5
        out = out > 127.5
        tp = np.sum( np.logical_and(gt, out) )
        tn = np.sum( np.logical_and(np.logical_not(gt), np.logical_not(out)) )
        fp = np.sum( np.logical_and(np.logical_not(gt), out) )
        fn = np.sum( np.logical_and(gt, np.logical_not(out)) )

        TP += tp
        TN += tn
        FP += fp
        FN += fn
    
    ber = 100.0*(1.0 - 0.5*(TP/(TP+FN)+TN/(TN+FP)))
    print("BER:", ber)

if __name__ == "__main__":
    test()
    if args["calcBER"]:
        calcBER()

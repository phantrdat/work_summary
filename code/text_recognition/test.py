"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
sys.path.insert(0, './recognition_module')
from craft import CRAFT
from recognition_module.regconization_infer import infer
from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.6, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/home/phantrdat/Desktop/train_doc_2', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
parser.add_argument('--output_type', default='image', type=str, help='Output format')
args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)
image_list = sorted(image_list)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def plot_one_box(img, pts, label=None, score=None, color=None, line_thickness=None, poly = False):
    tl = int(round(0.001 * max(img.shape[0:2])))  # line thickness
    # color = color
    c1 = (max(0,int(pts[0][0])), max(0,int(pts[0][1]))) 
    c2 = (max(0,int(pts[2][0])), max(0,int(pts[2][1]))) 
    if poly:
        cv2.polylines(img, [np.array(pts).reshape((-1,1,2)).astype(np.int32)], True, color=color, thickness=1)
    else:
        cv2.rectangle(img, c1, c2, color, thickness=1)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        # s_size = cv2.getTextSize(str('{:.0%}'.format(score)),0, fontScale=float(tl) / 3, thickness=tf)[0]
        text_sz = float(tl) / 3
        t_size = cv2.getTextSize(label, 0, fontScale=text_sz, thickness=tf)[0]
        c2 = c1[0] + t_size[0]+15, c1[1] - t_size[1] -3
        cv2.rectangle(img, c1, c2 , color, -1)  # filled
        cv2.putText(img, '{}'.format(label), (c1[0],c1[1] - 2), 0, text_sz, [0, 0, 0], thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None, is_rendered=False):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    ret_score_text = None
    if is_rendered !=False:
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()
  
    for k, image_path in enumerate(image_list):
        print(image_path)
        t1 = time.time()
        # print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        
        t_stge1 = time.time()
        image = imgproc.loadImage(image_path)
        bboxes, polys, _ = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
        # print("Detection Stage:", time.time() - t1)

        t_stagemid = time.time()

        raw_img = image[:,:,::-1]
        fname = os.path.basename(image_path)
        clone = raw_img.copy()
        all_text = {}
        coords = []
        for i  in range(len(polys)):
            try:
                pts = polys[i]
                rect = cv2.boundingRect(pts)
                x,y,w,h = rect
                croped = clone[y:y+h, x:x+w].copy()
                p1 = max(0,int(pts[0][0])) 
                p2 = max(0,int(pts[0][1]))
                p3 = max(0,int(pts[2][0]))
                p4 = max(0,int(pts[2][1]))
                # px = max(0,int(pts[1][0]))
                # py = max(0,int(pts[1][1]))
                # pz = max(0,int(pts[3][0]))
                # pt = max(0,int(pts[3][1]))
                
                # coords.append((p1, p2, p3, p4, px, py, pz, pt))
                pts = pts - pts.min(axis=0)
                mask = np.zeros(croped.shape[:2], np.uint8)
                ctr = np.array(pts).reshape((-1,1,2)).astype(np.int32)
                cv2.drawContours(mask, [ctr], -1, (255, 255, 255), -1, cv2.LINE_AA)

                ## (3) do bit-op
                dst = cv2.bitwise_and(croped, croped, mask=mask)
                ## (4) add the white background
                bg = np.ones_like(croped, np.uint8)*255
                cv2.bitwise_not(bg,bg, mask=mask)
                final_crop = bg + dst

                
                # cropped_im = clone[p2:p4, p1:p3]
                # cropped_im = cv2.cvtColor(cropped_im, cv2.COLOR_BGR2RGB)
                # cropped_im = Image.fromarray(cropped_im)
                # cbb = f'{c[0]}-{c[1]}_{c[2]}-{c[3]}_{c[4]}-{c[5]}_{c[6]}-{c[7]}'
                cbb = f'{fname[:-4]}_{p1}-{p2}_{p3}-{p4}'
                # cv2.imwrite(f'./test_im/text_cropped/{cbb}.jpg', final_crop)
                all_text[cbb] = Image.fromarray(final_crop)
            except Exception:
                pass
        # print("Transfer Stage", time.time()- t_stagemid)
        # t_reg = time.time()
        pred_str = infer(all_text)
        
        
        if args.output_type == 'image':
            raw_img = cv2.imread(image_path)
            for boxes, text in zip(polys, pred_str):
                plot_one_box(raw_img , boxes, text, color=(0, 0, 255), line_thickness=1, poly=True)
            filename, file_ext = os.path.splitext(os.path.basename(image_path))
            out_file = "result/"+ filename + ".jpg"
            
            cv2.imwrite(out_file, raw_img)
        if args.output_type == 'text':
            polys_with_text = []
            for boxes, text in zip(polys, pred_str):
                polys_with_text.append([text, boxes.tolist()])
            print(polys_with_text)
            polys_with_text = craft_utils.sortBoundingBox(polys_with_text)
            filename, file_ext = os.path.splitext(os.path.basename(image_path))
            out_file = "result/"+ filename + ".txt"
            f = open(out_file, 'w')
            for line in polys_with_text:
                f.write(' '.join(line)+'\n')
            f.close()


        
        
    print(time.time() - t)
    print("elapsed time : {}s".format(time.time() - t))

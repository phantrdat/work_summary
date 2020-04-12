from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
from data import cfg_mnet, cfg_slim, cfg_rfb
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from models.net_slim import Slim
from models.net_rfb import RFB
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import json
from torch2trt import torch2trt


def load_config():
    return json.load(open('detect_config.json','r'))

config = load_config()
trained_model = config['trained_model']
network = config['network']
long_side = int(config['long_side'])
origin_size = json.loads(config['origin_size'].lower())
confidence_threshold = float(config['confidence_threshold'])
top_k = int(config['top_k'])
nms_threshold = float(config['nms_threshold'])
keep_top_k = int(config['keep_top_k'])
save_image = json.loads(config['save_image'].lower())
vis_thres = float(config['vis_thres'])

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True
def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}
def load_model(model, pretrained_path):
    print('Loading pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

if network == "mobile0.25":
    cfg = cfg_mnet
    net = RetinaFace(cfg = cfg, phase = 'test')
elif network == "slim":
    cfg = cfg_slim
    net = Slim(cfg = cfg, phase = 'test')
elif network == "RFB":
    cfg = cfg_rfb
    net = RFB(cfg = cfg, phase = 'test')
else:
    print("Don't support network!")
    exit(0)
trained_model = "./weights/RBF_Final.pth"
device = torch.device("cuda")
model = load_model(net, trained_model).eval().to(device)

sample = torch.ones((1, 3, long_side, long_side)).cuda()
model_trt = torch2trt(model, [sample])
print('Finished loading model!')
# print(net)
cudnn.benchmark = True
def detect_all_images(input_path, output_path):
    image_names = os.listdir(input_path)
    sorted(image_names) 
    # begin inference
    all_bbox_res = {}
    for name in image_names:
        bbox_res = detect_one_image(name, input_path, output_path)
        all_bbox_res.update(bbox_res)
    return all_bbox_res
def detect_one_image(name, input_path, output_path):
        image_path = os.path.join(input_path, name)
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        sz = min(img_raw.shape[0], img_raw.shape[1])
        
        

        # Trace which axis is scaled 
        scale_x = sz/img_raw.shape[1]
        scale_y = sz/img_raw.shape[0]

        if (img_raw.shape[0]>img_raw.shape[1]):
            scale_by = 0 # Scale by y
            
        if img_raw.shape[0]<img_raw.shape[1]:
            scale_by = 1 # Scale by x
        
        img = np.float32(img_raw)
        img = cv2.resize(img, (sz, sz))

        # testing scale
        target_size = long_side
        im_shape = img.shape
        max_size = long_side
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if origin_size:
            resize = 1		
        if resize != 1:
            # img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            img = cv2.resize(img, (target_size, target_size))
        
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)		
        tic = time.time()
        
        loc, conf, landms = model_trt(img)  # forward pass
        print('Inferece {} take: {:.4f}'.format(name, time.time() - tic))		
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()		
        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]		
        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]		
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]		
        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]		
        dets = np.concatenate((dets, landms), axis=1)		
        # show image
        bbox_res = {}
        bbox_res[name] = []
        if save_image:
            for b in dets:
                if b[4] < vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                x,y, x_plus_w, y_plus_h = b[0], b[1], b[2],b[3]
                if scale_by == 0:
                    y = int(np.round(y / scale_y))
                    y_plus_h = int(np.round(y_plus_h / scale_y))
                else:
                    x = int(np.round(x / scale_x))
                    x_plus_w = int(np.round(x_plus_w / scale_x))
                bbox_res[name].append({'x':x, 'y':y, 'w':x_plus_w-x, 'h':y_plus_h -y})
                # sub_face = img_raw[y:y_plus_h, x:x_plus_w]
                # sub_face = cv2.GaussianBlur(sub_face, (81, 81), 75)
                # img_raw[y:y_plus_h, x:x_plus_w] = sub_face
                cv2.rectangle(img_raw, (x, y), (x_plus_w, y_plus_h), (0, 0, 255), 2)		

            # save image
            out_name = os.path.join(output_path, name)
            cv2.imwrite(out_name, img_raw)
        return bbox_res
if __name__=='__main__':
    bbox = detect_one_image("1.jpg","img/","img_out/")
    json.dump(bbox,open('tensorrt_det.json','w'))
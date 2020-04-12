import torch
from torch2trt import torch2trt
from models.net_rfb import RFB
import numpy as np
import cv2
import onnxruntime as rt
import onnx
import time












def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True 


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}\n'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model







#Prepare input
# img = cv2.imread('./img/lana_rhoades13-08.png', cv2.IMREAD_COLOR)
# img = np.float32(img)
# img = img.transpose(2, 0, 1)
# x = torch.from_numpy(img).unsqueeze(0)  
# x = torch.ones((1, 3, 640, 640))



#Run ONNX runtime inference
# onnx_time = []
# sess_options = rt.SessionOptions()
# sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
# sess_options.optimized_model_filepath = "./weights/RFB.onnx"
# session = rt.InferenceSession('./weights/RFB.onnx',sess_options)
# 
# input_name = session.get_inputs()[0].name
# print('Input Name:', input_name)
# 
# torch.set_grad_enabled(False)
# x_onnx=np.float32(x)
# for i in range(1000):
    # t1_onnx = time.time()
    # y_onnx = session.run([], {input_name: x_onnx})
    # t1_onnxend = time.time()- t1_onnx
    # onnx_time.append(t1_onnxend)
# average_time =  np.sum(np.array(onnx_time[1:]))/999
# print("ONNX inference time:{}\n".format(average_time))

# print(y_onnx)

#Prepare models for torch and tensorRT

net = RFB(cfg = None, phase = 'test')
trained_model = "./weights/RBF_Final.pth"
device = torch.device("cuda")






original_torch_time = []
tensorRT_time = []




model = load_model(net, trained_model, True).eval().to(device)
x = torch.ones((1, 3, 224, 224)).cuda()

size = [(480,600),(720,1280),(1920,1080)]
for sz in size:
    print("***** SIZE {} *****".format(sz))
    x = torch.rand((1,3,sz[0],sz[1])).to(device)
    x=x.to(device)
    model_trt = torch2trt(model, [x])
    
    
    for i in range(1000):
        # print(i)
        t1 = time.time()
        y_star = model(x)
        t1_end = time.time()- t1
        original_torch_time.append(t1_end)
    average_time =  np.sum(np.array(original_torch_time[1:]))/999
    print("Original torch inference time:{}\n".format(average_time))


    for i in range(1000):
        # print(i)   
        t3 = time.time()
        y = model_trt(x)
        t3_end =  time.time()-t3
        tensorRT_time.append(t3_end)
        # print("TensorRT inference time:", time.time() - t1)
    print(y[0].shape)
    print(y_star[0].shape)
    average_time_rt =  np.sum(np.array(tensorRT_time[1:]))/999
    print("Torch2TensorRT inference time:{}\n".format(np.sum(np.array(tensorRT_time[1:]))/999))
    print("Torch2TensorRT run {}x faster than original torch".format(round(average_time/average_time_rt) ) )






# print(y)
# print(y_star)
# print(torch.max(torch.abs(y[0] - y_star[0])))

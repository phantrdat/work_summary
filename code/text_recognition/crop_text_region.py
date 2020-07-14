import cv2
import numpy as np
import os
import utils
def crop():
    files = [x[:-4] for x in sorted(os.listdir('test_video_frames/result'))]
    # print(files)
    for fname in files:
        f = open(f'test_video_frames/box_res/{fname}.txt','r')
        coors = [line.strip('\n').split(',') for line in f.readlines() if line!='\n']
        # coors = np.array(coors[0]).reshape(-1,2)

        red = [0,0,255]
        im = cv2.imread(f'test_video_frames/result/{fname}.jpg')
        clone = im.copy()
        for i  in range(len(coors)):
            c = coors[i]
            p1 = int(c[0]) 
            p2 = int(c[1])
            p3 = int(c[4])
            p4 = int(c[5])
            # cv2.circle(im, (p1,p2), radius=0, color=(0, 0, 255), thickness=8)
            # cv2.circle(im, (p3,p4), radius=0, color=(0, 0, 255), thickness=8)
            cv2.rectangle(im, (p1,p2), (p3,p4), color=(0, 0, 255), thickness=0)
            cropped_im = clone[p2:p4, p1:p3]
            cbb = f'{c[0]}-{c[1]}_{c[2]}-{c[3]}_{c[4]}-{c[5]}_{c[6]}-{c[7]}'
            cv2.imwrite(f'test_video_frames/text_cropped/{fname},{cbb}.jpg',cropped_im)
def plot_one_box(img, coord, label=None, score=None, color=None, line_thickness=None):
    tl = int(round(0.001 * max(img.shape[0:2])))  # line thickness
    color = color
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        # s_size = cv2.getTextSize(str('{:.0%}'.format(score)),0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=2*float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0]+s_size[0]+15, c1[1] - t_size[1] -3
        cv2.rectangle(img, c1, c2 , color, -1)  # filled
        cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0],c1[1] - 2), 0, 2*float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)
def visualize():
    name_box_dict = {}
    recognize_results = open('/home/phantrdat/Desktop/Scene_Text_Detection/deep-text-recognition-benchmark/log_demo_result.txt','r').readlines()
    recognize_results = [line.strip('\n') for line in recognize_results]
    for res in recognize_results:
        _res = res.split()
        if _res[1].isdigit() and len(_res[1]) <=5:
            image_name = os.path.basename(_res[0])[:-4]
            chars = int(_res[1])
            # print(image_name, chars)
            coords = image_name[:-4].split(',')[1].split('_')
            ori_image_name = image_name[:-4].split(',')[0].strip('res_')
            coords = [coords[0], coords[2]]
            for i in range(len(coords)):
                coords[i] = coords[i].split('-')
                coords[i][0] = int(coords[i][0])
                coords[i][1] = int(coords[i][1])
                coords[i] = tuple(coords[i])
            if ori_image_name in name_box_dict:
                name_box_dict[ori_image_name].append((coords,chars))
            else:
                name_box_dict[ori_image_name] = [(coords,chars)]
    for k, v in name_box_dict.items():
        img = cv2.imread(f'/home/phantrdat/Desktop/athlete/{k}.jpg')
        for i in range(len(v)):
            coords, chars = v[i]
            utils.plot_one_box(img, coords, label=chars, color=(0, 0, 255), line_thickness=1)
        cv2.imwrite(f'/home/phantrdat/Desktop/Scene_Text_Detection/CRAFT-pytorch/test_video_frames/out/{k}.jpg', img)
# visualize()
def compare_tracking_and_visualize():
    
# for k, v in name_box_dict.items():
#     if (len(v)>1):
#         print(k)
# print(name_box_dict)
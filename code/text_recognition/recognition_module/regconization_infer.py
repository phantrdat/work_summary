import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter, TransLabelConverter
from dataset import StreamDataset, AlignCollate
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Params:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

opt = Params(FeatureExtraction='ResNet', PAD=False, Prediction='Attn', SequenceModeling='BiLSTM', Transformation='TPS', 
batch_max_length=25, batch_size=192, character='''0123456789abcdefghijklmnopqrstuvwxyz`!@#$^&*()+=-_[]/|\<>,.?:;''', 
hidden_size=256, image_folder='/home/phantrdat/Desktop/athlete/', imgH=32, imgW=100, input_channel=1, num_fiducial=20, num_gpu=1, output_channel=512, 
rgb=False, saved_model='./recognition_module/weights/TPS-ResNet-BiLSTM-Attn-Sensitive.pth', sensitive=True, workers=4)
if opt.sensitive:
    opt.character = string.printable[:-6]
""" model configuration """
if 'CTC' in opt.Prediction:
    converter = CTCLabelConverter(opt.character)
if 'Transformer' in opt.Prediction:
    converter = TransLabelConverter(opt.character)
else:
    converter = AttnLabelConverter(opt.character)
opt.num_class = len(converter.character)

if opt.rgb:
    opt.input_channel = 3

# prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
model = Model(opt)
print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
        opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
        opt.SequenceModeling, opt.Prediction)
model = torch.nn.DataParallel(model).to(device)

    # load model
print('loading pretrained model from %s' % opt.saved_model)
model.load_state_dict(torch.load(opt.saved_model, map_location=device))
model.eval()
def infer(bb_image_dict):

    demo_data = StreamDataset(opt, bb_image_dict)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    
    
    final_pred_str = []
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)


            log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            # log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            

            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' or 'Transformer' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]
                final_pred_str.append(pred)
                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{img_name:25s}\t{pred:25s}\n')

            # log.close()
    return final_pred_str
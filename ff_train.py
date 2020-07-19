"""
Evaluates a folder of video files or a single file with a xception binary
classification network.

Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>

Author: Andreas Rössler
"""
import time
import os
import sys
#sys.path.insert(0,'/home/liqian19/FaceForensics/dataset')
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image as pil_image
from tqdm import tqdm

from utils.args import argparse
from models.models import model_selection
from dataset.voc0712 import FaceDetection,detection_collate
from dataset.data_augment import preproc
from dataset.transform import xception_default_data_transforms
from termcolor import cprint
#import transform.xception_default_data_transforms as xception_default_data_transforms


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def load_weights(model_path):
    model_weight = torch.load(model_path)
    from collections import OrderedDict
    new_state_dict=OrderedDict()
    print("detection==========================")
    for key in model_weight.keys():
        #print(key)
        if 'pointwise' in key:
            #print(model_weight[key].size())
            new_state_dict['model.'+key]=model_weight[key].unsqueeze(-1).unsqueeze(-1)
        elif 'fc' in key:
            continue
        else:
            new_state_dict['model.'+key]=model_weight[key]
    return new_state_dict

def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = xception_default_data_transforms['train']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image


def train_with_model(image, model, post_function=nn.Softmax(dim=1)):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    #preprocessed_image = preprocess_image(image, cuda)

    # Model prediction
    output = model(image)
    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1) # argmax
    #prediction = prediction.astype(float)#float(prediction.cpu().numpy())
    prediction = torch.LongTensor([int(t.cpu().numpy()) for t in prediction])
    return prediction, output


def save_checkpoint(net, final=True, epoch=50):
    if final:
        torch.save(net.state_dict(),'/home/liqian19/weights/FaceXception/Final_xception_face.pth')
    else:
        torch.save(net.state_dict(),'/home/liqian19/weights/FaceXception/Xception_face_epoch{}.pth'.format(epoch))
        

def train_full_image_network(data_dir, model_path, output_path, beta1=0.9, beta2=0.99,
                             learning_rate=1e-5, batch_size=16, epochs=100, save_interval=50):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    """
    # Load model
    model, *_ = model_selection(modelname='xception', num_out_classes=2)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1,beta2))
    loss_func = torch.nn.CrossEntropyLoss().cuda()
    if model_path is not None:                
        model.load_state_dict(load_weights(model_path))
        model.last_linear = model.fc
        del model.fc
        model.last_linear = nn.Linear(1000, 2)
        print('model found in {}'.format(model_path))
        #print(model_weight)
    else:
        print('No model found, initializing random model.')

    # TODO: 
    # _preproc=preproc(299,[0.5]*3,0.1)
    # dataset=FaceDetection(data_root,"train",_preproc)
    # print("The number of dataset: ",len(dataset))
    for epoch in range(epochs):
        running_loss=0.0
        running_correct=0.0
        time_sum=0.0
        correct=0.0
        for i in range(0, epoch_size):
            batch_iterator=iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=1,
                                                collate_fn=detection_collate))
            t0=time.time()
            inputs, targets = next(batch_iterator)
            inputs = inputs.cuda()
            targets.cuda()
            prediction, output = train_with_model(inputs, model)
            
            optimizer.zero_grad()
            loss = loss_func(output, targets.cuda())  # compute loss for every net
            loss.backward()
            optimizer.step()
            correct=torch.sum(prediction.cuda()==targets.cuda().data)
            running_correct+=correct###
            t1=time.time()
            time_sum+=(t1-t0)###
            prnt=[time.ctime(), epoch, i, epoch_size, loss.item(), float(correct)/batch_size, t1-t0]
            cprint('Time:{}||Epoch:{}||EpochIter:{}/{}||Loss:{:.4f}||Accuracy:{:.4f}||Batch_Time:{:.4f}'.format(*prnt),'green')
        
        if (epoch+1) % save_interval == 0:
            save_checkpoint(model, final=False, epoch=epoch)

if __name__ == '__main__':
    args = parse_args()

    train_full_image_network(data_dir=args.data_dir, model_path=args.model_path,
                             output_path=args.output_path, batch_size=args.batch_size)
    
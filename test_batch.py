"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: test.py
about: main entrance for validating/testing the GridDehazeNet
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from val_data import ValData
from model import GateHazeNet
from utils import validation
import torchvision.transforms as transforms
import PIL.Image as Image
import scipy.misc
import skimage.io as sio
from torch.autograd import Variable

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-network_height', help='Set the network height (row)', default=3, type=int)
parser.add_argument('-network_width', help='Set the network width (column)', default=6, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-category', help='Set image category (indoor or outdoor?)', default='indoor', type=str)
args = parser.parse_args()

network_height = args.network_height
network_width = args.network_width
num_dense_layer = args.num_dense_layer
growth_rate = args.growth_rate
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
category = args.category

print('--- Hyper-parameters for testing ---')
print('val_batch_size: {}\nnetwork_height: {}\nnetwork_width: {}\nnum_dense_layer: {}\ngrowth_rate: {}\nlambda_loss: {}\ncategory: {}'
      .format(val_batch_size, network_height, network_width, num_dense_layer, growth_rate, lambda_loss, category))

# --- Set category-specific hyper-parameters  --- #
if category == 'indoor':
    val_data_dir = './data/test/SOTS/indoor/'
elif category == 'outdoor':
    val_data_dir = './data/test/SOTS/outdoor/'
else:
    raise Exception('Wrong image category. Set it to indoor or outdoor for RESIDE dateset.')


# --- Gpu device --- #
# device_ids = [Id for Id in range(torch.cuda.device_count())]
device_ids = [2]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Validation data loader --- #
val_data_loader = DataLoader(ValData(val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=24)


# --- Define the network --- #
net = GateHazeNet(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)


# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)


# --- Load the network weight --- #
net.load_state_dict(torch.load('{}_haze_best_{}_{}'.format(category, network_height, network_width)))

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ]
)

# --- Use the evaluation model in testing --- #
net.eval()

path = "/A-pool/wangke/dehaze-alg/measure/test_resize/"

dehaze_path = '/A-pool/wangke/dehaze-alg/measure/dehaze/gridDehazeNetOutdoor/'

import os

l = os.listdir(path)
tt = 0
count = 0
for i in l:
    count = count + 1
    print(count)
    name = i
    print(path + i)
    img = Image.open(path + i).convert('RGB')
    imgIn = transform(img).unsqueeze_(0)

    #===== Test procedures =====
    varIn = Variable(imgIn)
    varIn = varIn.cuda()
    prediction = net(varIn)

    prediction = prediction.data.cpu().numpy().squeeze().transpose((1,2,0))
    scipy.misc.toimage(prediction).save(dehaze_path+name)
#     print(dehaze_path+name)



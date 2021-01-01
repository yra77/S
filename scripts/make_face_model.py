# =============================================================================
# MIT License
# 
# Copyright (c) 2018 chuanqi305
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

import os
import sys 
import locale
locale.setlocale(locale.LC_ALL,"rus" )
#reload(sys)  # Reload does the trick!
#sys.setdefaultencoding('CP1251')
caffePath = 'C:/Users/User/source/repos/Python_Caffe/caffe-ssd/build/install/python'
#uu = caffePath.decode('CP1251')
sys.path.insert(0, caffePath)

import numpy as np
import caffe
import cv2
import time

import matplotlib.pyplot as plt
import math
os.environ['GLOG_minloglevel'] = '2' 
#from caffe.proto import caffe_pb2
#from google.protobuf import text_format

from caffe.proto.caffe_pb2 import NetParameter, LayerParameter
import google.protobuf.text_format as txtf

def transform_input(img, transpose=True, dtype=np.float32):
    """Return transformed input image 'img' for CNN input
    transpose: if True, channels in dim 0, else channels in dim 2
    dtype: type of returned array (sometimes important)    
    """
    inpt = cv2.resize(img, (300,300))
    inpt = inpt - 127.5
    inpt = inpt / 127.5
    inpt = inpt.astype(dtype)
    if transpose:
        inpt = inpt.transpose((2, 0, 1))
    return inpt

def get_masks(net, percentile=50):
    """Returns dict layer_name:channels_mask for convolutions.
    100%-percentile% channels are selected by maximum response (in blobs).
    net: caffe.Net network
    """
    bnames = [e for e in net.blobs.keys() if ('data' not in e) and ('split' not in e) 
              and ('mbox' not in e) and ('detection' not in e)]
    blobmask = {}
    prev = None
    for b in bnames:
        blob = net.blobs[b].data
        mean = blob.mean(axis=(0,2,3))
        perc = np.percentile(mean, percentile)
        mask = mean>perc
        blobmask[b] = mask
        if ('dw' in b) and (prev is not None):
            blobmask[prev] = mask
        prev = b
    return blobmask

def resize_network(netdef, name2num, verbose=True):
    """Change number of channels in convolutions
    netdef: network params
    name2num: maps from channel name to new number of channels
    verbose: if True, display changes
    """
    new_layers = []
    for l in netdef.layer:
        newl = LayerParameter()
        newl.CopyFrom(l)
        if (l.name in name2num):
            if (l.type == 'Convolution'):
                if verbose:
                    print(l.name+': \t'+
                          'Changing num_output from '+str(l.convolution_param.num_output)+' to '+str(name2num[l.name]))
                newl.convolution_param.num_output = name2num[l.name]
                if newl.convolution_param.group > 1:
                    newl.convolution_param.group = name2num[l.name]
            else:
                if verbose:
                    print('Layer '+l.name+' is not convolution, skipping')
        new_layers.append(newl)
    new_pnet = NetParameter()
    new_pnet.CopyFrom(netdef)
    del(new_pnet.layer[:])
    new_pnet.layer.extend(new_layers)
    return new_pnet

def set_params(model, newmodel, newnetdef, blob2mask):
    """Copy parameters from bigger network to smaller (with pruned channel).
    model: initial model (bigger)
    newmodel: pruned model (smaller)
    newnetdef: pruned model parameters
    blob2mask: maps blob name to channel mask
    """
    l2bot = {l.name:l.bottom for l in newnetdef.layer}
    l2top = {l.name:l.top for l in newnetdef.layer}
    l2group = {l.name:l.convolution_param.group for l in newnetdef.layer}
    
    for name in model.params.keys():
        #if 'mbox' in name:
        if ('perm' in name) or ('flat' in name) or ('priorbox' in name):
            continue
        top = l2top[name][0]
        bot = l2bot[name][0]
        topmask = blob2mask[top] if top in blob2mask else None
        botmask = blob2mask[bot] if bot in blob2mask else None
        conv = model.params[name][0].data
        bias = model.params[name][1].data
        if (topmask is not None) and (botmask is not None):
            print('Setting parameters for layer '+name)
        if topmask is not None:
            conv = conv[topmask,:,:,:]
            bias = bias[topmask]
        if (botmask is not None) and (l2group[name]==1):
            conv = conv[:,botmask,:,:]
        newmodel.params[name][0].data[...] = conv
        if name+'/scale' in newmodel.params:
            newmodel.params[name+'/scale'][1].data[...] = bias
        else:
            newmodel.params[name][1].data[...] = bias
            
if __name__ == "__main__":
    #get percents of pruned channels
    percentile = int(sys.argv[1])  
    
    #Task-specific: mask only classes 'background' and 'person'
    class_labels = ('background', '1', '2', '3', '4', 
    '5', '6', '7', '8', '9', '10', '11', '12', 
    '13', '14', '15', '16', '17', 
    '18', '19', '20','21', '22', '23','24', '25', '26','27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38')
    class_mask = [((e=='background') or (e=='38')) for e in class_labels]
    
    n_old_classes = 39
    n_coord = 4
    #create masks for coordinates and confidence layers
    #numbers are old/new numbers of boxes in those layers
    mboxes = {
    "conv11_mbox_loc"    : np.array([True,]*n_coord*2 + [False,]*n_coord*(3-2)),
    "conv13_mbox_loc"    : np.array([True,]*n_coord*2 + [False,]*n_coord*(6-2)),
    "conv14_2_mbox_loc"  : np.array([True,]*n_coord*2 + [False,]*n_coord*(6-2)),
    "conv15_2_mbox_loc"  : np.array([True,]*n_coord*2 + [False,]*n_coord*(6-2)),
    "conv16_2_mbox_loc"  : np.array([True,]*n_coord*2 + [False,]*n_coord*(6-2)),
    "conv17_2_mbox_loc"  : np.array([True,]*n_coord*2 + [False,]*n_coord*(6-2)),
    "conv11_mbox_conf"   : np.array(class_mask*2 + [False,]*n_old_classes*(3-2)),
    "conv13_mbox_conf"   : np.array(class_mask*2 + [False,]*n_old_classes*(6-2)),
    "conv14_2_mbox_conf" : np.array(class_mask*2 + [False,]*n_old_classes*(6-2)),
    "conv15_2_mbox_conf" : np.array(class_mask*2 + [False,]*n_old_classes*(6-2)),
    "conv16_2_mbox_conf" : np.array(class_mask*2 + [False,]*n_old_classes*(6-2)),
    "conv17_2_mbox_conf" : np.array(class_mask*2 + [False,]*n_old_classes*(6-2)) 
    }
    
    #reference network (bigger)
    ref_net = caffe.Net('C:/Users/User/source/repos/Python_Caffe/DataSet_Plate/Model/MobileNetSSD_deploy.prototxt', 
                    'C:/Users/User/source/repos/Python_Caffe/DataSet_Plate/Model/empty.caffemodel', 
                    caffe.TEST) 
    
    #reference network parameters
    with open('C:/Users/User/source/repos/Python_Caffe/DataSet_Plate/Model/MobileNetSSD_deploy.prototxt', 'r') as f:
        ref_par = NetParameter()
        txtf.Merge(f.read(), ref_par)
       
    #new network parameters: train,test,deploy
    with open('C:/Users/User/source/repos/Python_Caffe/DataSet_Plate/Model/MobileNetSSD_train.prototxt', 'r') as f:
        train_par = NetParameter()
        txtf.Merge(f.read(), train_par)   
    with open('C:/Users/User/source/repos/Python_Caffe/DataSet_Plate/Model/MobileNetSSD_test.prototxt', 'r') as f:
        test_par = NetParameter()
        txtf.Merge(f.read(), test_par)  
    with open('C:/Users/User/source/repos/Python_Caffe/DataSet_Plate/Model/MobileNetSSD_deploy.prototxt', 'r') as f:
        dep_par = NetParameter()
        txtf.Merge(f.read(), dep_par)
      
    #get faces collage and compute layer responses
    faces = cv2.imread('C:/Users/User/source/repos/Python_Caffe/DataSet_Plate/Images/5.jpg')
    inpt = transform_input(faces)
    ref_net.blobs['data'].data[...] = inpt
    output = ref_net.forward()
    
    #get masks for regular convolutions
    blobmask = get_masks(ref_net, percentile)
    
    #get masks for coordinate|confidence convolutions
    blobmask.update(mboxes)    
    
    #resize networks
    sizes = {k:sum(v) for k,v in blobmask.items()}
    train_par = resize_network(train_par, sizes)
    test_par = resize_network(test_par, sizes, verbose=False)
    dep_par = resize_network(dep_par, sizes, verbose=False)
    
    #write new parameters
    with open('C:/Users/User/source/repos/Python_Caffe/DataSet_Plate/train.prototxt', 'w') as f:
        f.write(txtf.MessageToString(train_par))
    with open('C:/Users/User/source/repos/Python_Caffe/DataSet_Plate/test.prototxt', 'w') as f:
        f.write(txtf.MessageToString(test_par))
    with open('C:/Users/User/source/repos/Python_Caffe/DataSet_Plate/deploy.prototxt', 'w') as f:
        f.write(txtf.MessageToString(dep_par))
    
    #load pruned net with empty parameters
    new_net = caffe.Net('C:/Users/User/source/repos/Python_Caffe/DataSet_Plate/Model/MobileNetSSD_deploy.prototxt', 
                    'C:/Users/User/source/repos/Python_Caffe/DataSet_Plate/Model/empty.caffemodel', caffe.TRAIN)
    
    #copy masked parameters to pruned net
    set_params(ref_net, new_net, train_par, blobmask)
    
    #save pruned net parameters
    new_net.save('C:/Users/User/source/repos/Python_Caffe/DataSet_Plate/full2.caffemodel')

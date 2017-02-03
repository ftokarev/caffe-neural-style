from __future__ import division, print_function

import os
import sys
sys.path.insert(0, os.environ['CAFFE_PATH'])

import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf as pb
import numpy as np
import skimage
import tempfile


GPU_ID = 0
CONTENT_IMAGE = './content.jpg'
STYLE_IMAGE = './style.jpg'
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
CONTENT_LAYERS = ['conv4_2']
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
GRAM_LAYERS = ['gram_'+layer for layer in STYLE_LAYERS]
CAFFE_MODEL = './vgg/VGG_ILSVRC_19_layers_deploy.prototxt'
CAFFE_WEIGHTS = './vgg/VGG_ILSVRC_19_layers.caffemodel'

# https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md
MEAN = np.array([103.939, 116.779, 123.68])

caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

def prune(net_param, used_layers):
    """Prune the network

    We assume that all layers after the last of used_layers are not useful
    """
    prune_from = None
    for i, layer in enumerate(net_param.layer):
        for top in layer.top:
            if prune_from and top in used_layers:
                prune_from = None
                break
            elif not prune_from and top not in used_layers:
                prune_from = i
    if prune_from:
        for i in reversed(range(prune_from, len(net_param.layer))):
            print('Pruning layer', net_param.layer[i].name)
            del net_param.layer[i]


def adjust(net_param, content_layers, style_layers):
    """Add additional layers to the network"""
    gram_layers = ['gram_' + layer for layer in style_layers]
    for style_layer, gram_layer in zip(style_layers, gram_layers):
        gram = net_param.layer.add()
        gram.type = 'Gram'
        gram.name = gram_layer
        gram.bottom.append(style_layer)
        gram.top.append(gram_layer)


net_param = caffe_pb2.NetParameter()
with open(CAFFE_MODEL, 'r') as f:
    pb.text_format.Merge(f.read(), net_param)


prune(net_param, CONTENT_LAYERS+STYLE_LAYERS)
adjust(net_param, CONTENT_LAYERS, STYLE_LAYERS)























with tempfile.NamedTemporaryFile() as tmp:
    tmp.write(pb.text_format.MessageToString(net_param))
    tmp.seek(0)
    net = caffe.Net(tmp.name, caffe.TEST, weights=CAFFE_WEIGHTS)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_channel_swap('data', (2,1,0))  # RGB -> BGR
transformer.set_transpose('data', (2,0,1))  # HxWxC -> CxHxW
transformer.set_raw_scale('data', 256.0)
transformer.set_mean('data', MEAN)

# 2.
# Take the style image and pass it through the network, storing the
# activations of the specified layers.

style_img = caffe.io.load_image(STYLE_IMAGE)
data_shape = (1, 3, style_img.shape[0], style_img.shape[1])

net.blobs['data'].reshape(*data_shape)
transformer.inputs['data'] = data_shape

net.blobs['data'].data[0] = transformer.preprocess('data', style_img)
net.forward()
style_activations = {}
style_layer_sizes = {}
gram_layer_sizes = {}
for layer in STYLE_LAYERS:
    style_layer_sizes[layer] = net.blobs[layer].data.size
    layer = 'gram_'+layer
    gram_layer_sizes[layer] = net.blobs[layer].data.size
    style_activations[layer] = net.blobs[layer].data.copy()

# 3.
# Take the content image and pass it through the network, storing the
# activations of the specified layers.

content_img = caffe.io.load_image(CONTENT_IMAGE)
data_shape = (1, 3, content_img.shape[0], content_img.shape[1])

net.blobs['data'].reshape(*data_shape)
transformer.inputs['data'] = data_shape

net.blobs['data'].data[0] = transformer.preprocess('data', content_img)
net.forward()
content_activations = {}
content_layer_sizes = {}
for layer in CONTENT_LAYERS:
    content_layer_sizes[layer] = net.blobs[layer].data.size
    content_activations[layer] = net.blobs[layer].data.copy()

# 4.
# Build a network for style transfer
# Add input layers for content and style activations, add loss layers

target_img = content_img
data_shape = (1, 3, target_img.shape[0], target_img.shape[1])

for name in CONTENT_LAYERS+GRAM_LAYERS:
    layer = net_param.layer.add()
    layer.type = 'Input'
    layer.name = 'input_'+name
    layer.top.append('input_'+name)
    shape = layer.input_param.shape.add()
    for dim in net.blobs[name].shape:
        shape.dim.append(dim)

for name in CONTENT_LAYERS:
    layer = net_param.layer.add()
    layer.type = 'EuclideanLoss'
    layer.name = 'loss_'+name
    ze_size = content_layer_sizes[name]
    layer.loss_weight.append(2*CONTENT_WEIGHT/ze_size)
    layer.bottom.append('input_'+name)
    layer.bottom.append(name)
    layer.top.append('loss_'+name)

for name in STYLE_LAYERS:
    gram_name = 'gram_'+name
    layer = net_param.layer.add()
    layer.type = 'EuclideanLoss'
    layer.name = 'loss_'+gram_name
    layer.loss_weight.append(2*STYLE_WEIGHT/gram_layer_sizes[gram_name]/np.square(style_layer_sizes[name])) # TODO document this while I still remember the details
    layer.bottom.append('input_'+gram_name)
    layer.bottom.append(gram_name)
    layer.top.append('loss_'+gram_name)

# replace InputLayer with ParameterLayer, so that we'll be able to backprop into the image
for layer in net_param.layer:
    if layer.name == 'input':
        layer.type = 'Parameter'
        for dim in data_shape:
            layer.parameter_param.shape.dim.append(dim)
        break


for layer in net_param.layer:
    if layer.type not in ['EuclideanLoss', 'Gram', 'Input', 'Parameter', 'Pooling', 'ReLU']:
        # do not backprop into weights and biases of conv layers
        param = layer.param.add()
        param.lr_mult = 0
        param = layer.param.add()
        param.lr_mult = 0


# 5.
# Create LBFGS solver

solver_param = caffe_pb2.SolverParameter()
solver_param.max_iter = 100
solver_param.lbfgs_corrections = 100
solver_param.display = 10

#net_param.debug_info = True
#solver_param.debug_info = True

del net

with tempfile.NamedTemporaryFile() as net_proto:
    net_proto.write(pb.text_format.MessageToString(net_param))
    net_proto.seek(0)
    with tempfile.NamedTemporaryFile() as solver_proto:
        solver_param.train_net = net_proto.name
        solver_proto.write(pb.text_format.MessageToString(solver_param))
        solver_proto.seek(0)
        solver = caffe.LBFGSSolver(solver_proto.name)
        solver.net.copy_from(CAFFE_WEIGHTS)

# 6.
# Set up input data
solver.net.params['input'][0].data[0] = transformer.preprocess('data', target_img)

for name in CONTENT_LAYERS:
    solver.net.blobs['input_'+name].data[0] = content_activations[name][0]

for name in GRAM_LAYERS:
    solver.net.blobs['input_'+name].data[0] = style_activations[name][0]

# 7.
# Optimize

solver.step(500)

# TODO deal with the warning: 'UserWarning: Possible precision loss when converting from float32 to uint8'
# TODO deal with the error: 'ValueError: Images of type float must be between -1 and 1.' (i.e. get rid of np.clip)
result = transformer.deprocess('data', solver.net.params['input'][0].data[0])
skimage.io.imsave('./output.jpg', np.clip(result, -1, 1))

print("Done")


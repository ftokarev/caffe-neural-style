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
CONTENT_IMAGE = 'content.jpg'
STYLE_IMAGE = 'style.jpg'
OUTPUT_IMAGE = 'output.jpg'
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_LOSS_WEIGHT = 1e-5
CONTENT_LAYERS = ['conv4_2']
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
GRAM_LAYERS = ['gram_'+layer for layer in STYLE_LAYERS]
CAFFE_MODEL = 'vgg/VGG_ILSVRC_19_layers_deploy.prototxt'
CAFFE_WEIGHTS = 'vgg/VGG_ILSVRC_19_layers.caffemodel'
DISPLAY_EVERY = 10
NUM_ITER = 500

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


def add_gram_layers(net_param, style_layers, gram_layers):
    for style_layer, gram_layer in zip(style_layers, gram_layers):
        gram = net_param.layer.add()
        gram.type = 'Gram'
        gram.name = gram_layer
        gram.bottom.append(style_layer)
        gram.top.append(gram_layer)

def get_transformer():
    # mean is taken from gist.github.com/ksimonyan/3785162f95cd2d5fee77
    transformer = caffe.io.Transformer({'data': (1,3,1,1)})
    transformer.set_channel_swap('data', (2,1,0))  # RGB -> BGR
    transformer.set_transpose('data', (2,0,1))  # HxWxC -> CxHxW
    transformer.set_raw_scale('data', 256.0)
    transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))
    return transformer

def prepare_data_blob(net, transformer, img):
    data_shape = (1, 3, img.shape[0], img.shape[1])
    transformer.inputs['data'] = data_shape
    net.blobs['data'].reshape(*data_shape)
    net.blobs['data'].data[0] = transformer.preprocess('data', img)

def prepare_input_param(net, transformer, img):
    data_shape = (1, 3, img.shape[0], img.shape[1])
    transformer.inputs['data'] = data_shape
    net.params['input'][0].reshape(*data_shape)
    net.params['input'][0].data[0] = transformer.preprocess('data', img)

#
# 1 get content and style activations
#

net_param = caffe_pb2.NetParameter()
with open(CAFFE_MODEL, 'r') as f:
    pb.text_format.Merge(f.read(), net_param)

prune(net_param, CONTENT_LAYERS+STYLE_LAYERS)
add_gram_layers(net_param, STYLE_LAYERS, GRAM_LAYERS)

with tempfile.NamedTemporaryFile() as tmp:
    tmp.write(pb.text_format.MessageToString(net_param))
    tmp.seek(0)
    net = caffe.Net(tmp.name, caffe.TEST, weights=CAFFE_WEIGHTS)

transformer = get_transformer()

style_img = caffe.io.load_image(STYLE_IMAGE)
prepare_data_blob(net, transformer, style_img)
net.forward()

gram_activations = {}
for layer in GRAM_LAYERS:
    gram_activations[layer] = net.blobs[layer].data.copy()

style_layer_sizes = {}
for layer in STYLE_LAYERS:
    style_layer_sizes[layer] = net.blobs[layer].data.size

content_img = caffe.io.load_image(CONTENT_IMAGE)
prepare_data_blob(net, transformer, content_img)
net.forward()

content_activations = {}
for layer in CONTENT_LAYERS:
    content_activations[layer] = net.blobs[layer].data.copy()

#
# 2 prepare the network by adding the necessary layers
#

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
    weight = 2 * CONTENT_WEIGHT / content_activations[name].size
    layer.type = 'EuclideanLoss'
    layer.name = 'loss_'+name
    layer.loss_weight.append(weight)
    layer.bottom.append('input_'+name)
    layer.bottom.append(name)
    layer.top.append('loss_'+name)

for name in STYLE_LAYERS:
    gram_name = 'gram_'+name
    layer = net_param.layer.add()
    weight = 2 * STYLE_WEIGHT / gram_activations[gram_name].size /np.square(style_layer_sizes[name])
    layer.type = 'EuclideanLoss'
    layer.name = 'loss_'+gram_name
    layer.loss_weight.append(weight)
    layer.bottom.append('input_'+gram_name)
    layer.bottom.append(gram_name)
    layer.top.append('loss_'+gram_name)

# add TV Loss
layer = net_param.layer.add()
weight = TV_LOSS_WEIGHT
layer.type = 'TVLoss'
layer.name = 'loss_tv'
layer.loss_weight.append(weight)
layer.bottom.append('data')
layer.top.append('loss_tv')

# replace InputLayer with ParameterLayer,
# so that we'll be able to backprop into the image
for layer in net_param.layer:
    if layer.name == 'input':
        layer.type = 'Parameter'
        for dim in net.blobs['data'].shape:
            layer.parameter_param.shape.dim.append(dim)
        break

# disable weights learning
for layer in net_param.layer:
    if layer.type not in ['EuclideanLoss', 'TVLoss', 'Gram', 'Input', 'Parameter', 'Pooling', 'ReLU']:
        param = layer.param.add()
        param.lr_mult = 0
        param = layer.param.add()
        param.lr_mult = 0

del net

#
# 3 create solver, assign inputs
#

solver_param = caffe_pb2.SolverParameter()
solver_param.display = DISPLAY_EVERY

with tempfile.NamedTemporaryFile() as net_proto:
    net_proto.write(pb.text_format.MessageToString(net_param))
    net_proto.seek(0)
    with tempfile.NamedTemporaryFile() as solver_proto:
        solver_param.train_net = net_proto.name
        solver_proto.write(pb.text_format.MessageToString(solver_param))
        solver_proto.seek(0)
        solver = caffe.LBFGSSolver(solver_proto.name)
        solver.net.copy_from(CAFFE_WEIGHTS)

target_img = content_img
prepare_input_param(solver.net, transformer, target_img)

for name in CONTENT_LAYERS:
    solver.net.blobs['input_'+name].data[0] = content_activations[name][0]

for name in GRAM_LAYERS:
    solver.net.blobs['input_'+name].data[0] = gram_activations[name][0]

#
# 4 optimize!
#

solver.step(NUM_ITER)

result = transformer.deprocess('data', solver.net.params['input'][0].data[0])
skimage.io.imsave(OUTPUT_IMAGE, np.clip(result, 0, 1))

print("Done")


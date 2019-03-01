"""
.. _tutorial-deploy-model-on-rasp:

Deploy the Pretrained Model on Raspberry Pi
===========================================
**Author**: `Ziheng Jiang <https://ziheng.org/>`_

This is an example of using NNVM to compile a ResNet model and deploy
it on Raspberry Pi.
"""

import tvm
import nnvm.compiler
import nnvm.testing
from tvm import rpc
from tvm.contrib import util, graph_runtime as runtime

# Tensorflow imports
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing


# Base location for model related files.
repo_base = 'https://github.com/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/'

#optimize config
batch_size = 1
num_classes = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape

#optimized target 
target = 'cuda'
target_host = 'llvm'
layout = "NCHW"
ctx = tvm.gpu(0)

# download model 
from mxnet.gluon.utils import download
import os.path

model_name = 'classify_image_graph_def-with_shapes.pb'
model_url = os.path.join(repo_base, model_name)

download(model_url, model_name)

# Creates tensorflow graph definition from protobuf file.

with tf.gfile.FastGFile(os.path.join("./", model_name), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # Add shapes to the graph.
    with tf.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, 'softmax')


######################################################################
# Import the graph to NNVM
# ------------------------
# Import tensorflow graph definition to nnvm.
#
# Results:
#   sym: nnvm graph for given tensorflow protobuf.
#   params: params converted from tensorflow params (tensor protobuf).
sym, params = nnvm.frontend.from_tensorflow(graph_def, layout=layout)

print ("Tensorflow protobuf imported as nnvm graph")

#prepare image data
img_name = 'elephant-299.jpg'
image_url = os.path.join(repo_base, img_name)
download(image_url, img_name)
from PIL import Image
image = Image.open(img_name).resize((299, 299))
x = np.array(image)

shape_dict = {'DecodeJpeg/contents': x.shape}
dtype_dict = {'DecodeJpeg/contents': 'uint8'}

with nnvm.compiler.build_config(opt_level=3):
    graph, lib, params = nnvm.compiler.build(sym, shape=shape_dict, target=target, target_host=target_host, dtype=dtype_dict, params=params)

# Save the library at local temporary directory.
tmp = util.tempdir()
lib_fname = tmp.relpath('net.tar')
lib.export_library(lib_fname)

######################################################################
# Deploy the Model Remotely by RPC
# --------------------------------
# With RPC, you can deploy the model remotely from your host machine
# to the remote device.

# The following is my environment, change this to the IP address of your target device
host = '0.0.0.0'
port = 9090
remote = rpc.connect(host, port)

# upload the library to remote device and load it
remote.upload(lib_fname)
rlib = remote.load_module('net.tar')

# create the remote runtime module
ctx = remote.gpu(0)
module = runtime.create(graph, rlib, ctx)


# set input data
dtype = 'uint8'
module.set_input('DecodeJpeg/contents', tvm.nd.array(x.astype(dtype)))
# set parameter (upload params to the remote device. This may take a while)
module.set_input(**params)

# run
module.run()
# get output
tvm_output = m.get_output(0, tvm.nd.empty(((1, 1008)), 'float32'))

######################################################################
# Process the output
# ------------------
# Process the model output to human readable text for InceptionV1.
predictions = tvm_output.asnumpy()
predictions = np.squeeze(predictions)

#download the table
download(map_proto_url, map_proto)
download(lable_map_url, lable_map)

# Creates node ID --> English string lookup.
node_lookup = tf_testing.NodeLookup(label_lookup_path=os.path.join("./", map_proto),
                                         uid_lookup_path=os.path.join("./", lable_map))
# Print top 5 predictions from TVM output.
top_k = predictions.argsort()[-5:][::-1]
for node_id in top_k:
    human_string = node_lookup.id_to_string(node_id)
    score = predictions[node_id]
    print('%s (score = %.5f)' % (human_string, score))

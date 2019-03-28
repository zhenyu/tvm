import tvm
import nnvm.compiler
import nnvm.testing
from tvm import rpc
# download model 
from mxnet.gluon.utils import download
import os.path

# Base location for model related files.
repo_base = 'https://github.com/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/'

#optimize config
batch_size = 1
num_classes = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape

#prepare image data
img_name = 'elephant-299.jpg'
image_url = os.path.join(repo_base, img_name)
download(image_url, img_name)
from PIL import Image
image = Image.open(img_name).resize((299, 299))

import numpy as np
x = np.array(image)

shape_dict = {'DecodeJpeg/contents': x.shape}
dtype_dict = {'DecodeJpeg/contents': 'uint8'}

# The following is my environment, change this to the IP address of your target device
host = '0.0.0.0'
port = 9090
remote = rpc.connect(host, port)

# upload the library to remote device and load it
lib_fname = '/home/ubuntu/tfnet.tar'
remote.upload(lib_fname)
rlib = remote.load_module('tfnet.tar')

#load json model to graph
loaded_json = open('/home/ubuntu/tf.json').read()

# create the remote runtime module
ctx = remote.gpu(0)
module = runtime.create(graph, rlib, ctx)

# set parameter (upload params to the remote device. This may take a while)
param_file = open(param_file_name, 'rb')
param_bytes = param_file.read()
params = nnvm.compiler.load_param_dict(param_bytes)
param_file.close()

module.set_input(**params)

# set input data
dtype = 'uint8'
module.set_input('DecodeJpeg/contents', tvm.nd.array(x.astype(dtype)))

# run
module.run()
# get output
tvm_output = module.get_output(0)

######################################################################
# Process the output
# ------------------
# Process the model output to human readable text for InceptionV1.
predictions = tvm_output.asnumpy()
predictions = np.squeeze(predictions)
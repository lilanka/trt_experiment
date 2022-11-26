#!/bin/python3
import os

import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms

import networks
from utils import download_model_if_doesnt_exist

import onnx
import onnxruntime as rt

import pycuda.driver as cuda
#from cuda import cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger()  

def convert_model_to_onnx(model, name, x):
  model_onnx_path = "models/onnx/" + name + ".onnx"

  input_names, output_names = ["actual_input"], ["output"]
  torch.onnx.export(model, x, model_onnx_path, verbose=False, input_names=input_names, output_names=output_names, export_params=True)

def load_model():
  model_name = "mono_640x192"

  download_model_if_doesnt_exist(model_name)
  encoder_path = os.path.join("models", model_name, "encoder.pth")
  depth_decoder_path = os.path.join("models", model_name, "depth.pth")

  # load pretrained model 
  encoder = networks.ResnetEncoder(18, False)
  depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

  loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
  filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
  encoder.load_state_dict(filtered_dict_enc)

  loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
  depth_decoder.load_state_dict(loaded_dict)

  encoder.eval()
  depth_decoder.eval()

  def test_data():
    image_path = "assets/test_image.jpg"

    input_image = pil.open(image_path).convert('RGB')
    original_width, original_height = input_image.size

    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)

    input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
    return input_image_pytorch

  encoder_in = test_data()
  with torch.no_grad():
    decoder_in = encoder(encoder_in)

  return encoder, encoder_in, depth_decoder, decoder_in

def build_engine(onnx_model_path, tensorrt_engine_path, fp16=False):
  logger = trt.Logger(trt.Logger.ERROR)
  builder = trt.Builder(logger)
  config = builder.create_builder_config()
  config.max_workspace_size = 1 << 30
  builder.max_batch_size = 1
  if fp16:
    config.set_flag(trt.BuilderFlag.FP16)

  network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  parser = trt.OnnxParser(network, logger) # onnx parser  

  with open(onnx_model_path, "rb") as model:
    if not parser.parse(model.read()):
      print("Failed parsing .onnx file!")
      for error in range(parser.num_errors):
        print(parser.get_error(error))
      exit()
    print("Succeeded parsing .onnx file!") 

  input_tensor = network.get_input(0)
  print("Input tensor_name: ", input_tensor.name)
  engine = builder.build_serialized_network(network, config)
  print("Engine building done!")
  with open(tensorrt_engine_path, "wb") as f:
    f.write(engine)

def inference(engine, data):
  height, width = data.shape[2], data.shape[3]
  with engine.create_execution_context() as context:
    # set input shape base on image dimensions for inference
    context.set_binding_shape(engine.get_binding_index("actual_input"), (1, 3, height, width))
    # allocate host and device buffers
    bindings = []
    for binding in engine:
      binding_idx = engine.get_binding_index(binding)
      size = trt.volume(context.get_binding_shape(binding_idx))
      dtype = trt.nptype(engine.get_binding_dtype(binding))
      if engine.binding_is_input(binding):
        input_buffer = np.ascontiguousarray(data)
        input_memory = cuda.mem_alloc(data.element_size() * data.nelement())
        bindings.append(int(input_memory))
      else:
        output_buffer = cuda.pagelocked_empty(size, dtype)
        output_memory = cuda.mem_alloc(output_buffer.nbytes)
        bindings.append(int(output_memory))

  stream = cuda.Stream()
  cuda.memcpy_htod_async(input_memory, input_buffer, stream)
  context.execute_async_v2(bindings=bindings, stream_handle=stream.handle) # running on cpu. getting memory out of range error
  # Transfer prediction output from the GPU.
  cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
  # Synchronize the stream
  stream.synchronize()

def main():
  encoder, encoder_in, decoder, decoder_in = load_model()
  print(encoder_in.shape)

  # convert model to onnx
  convert_model_to_onnx(encoder, "encoder", encoder_in)
  #convert_model_to_onnx(decoder, "decoder", decoder_in)

  # todo: sometimes onnx is still complex. simplify it if necessary

  # convert onnx model to TensorRT engine
  encoder_onnx_path = "models/onnx/encoder.onnx"

  tensorrt_engine_path = "models/fp32_implicit.engine" 
  build_engine(encoder_onnx_path, tensorrt_engine_path)

  # read the engine from the file and deserialize
  with open(tensorrt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

  # TensorRT inference
  tensorrt_output = inference(engine, encoder_in)

if __name__ == '__main__':
  main()
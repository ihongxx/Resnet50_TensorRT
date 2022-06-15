import argparse
from tkinter import N
from myCalibrator import Resnet50EntropyCalibrator

from pytools import T
import tensorrt as trt

def ONNX2TRT(args, calib=None):
    """
        convert onnx to tensorrt engine, use mode of ['fp32', 'fp16', 'int8']

    return: trt engine
    """

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING) 
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.max_workspace_size = 4 << 30

    if args.mode.lower() == 'int8':
        assert builder.platform_has_fast_int8 == True, 'not support int8'
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib
    elif args.mode.lower() == 'fp16':
        assert (builder.platform_has_fast_fp16 == True), 'not support fp16'
        config.set_flag(trt.BuilderFlag.FP16)

    with open(args.onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_error):
                print(parser.get_error(error))
            return None

    try:
        engine_bytes  = builder.build_serialized_network(network, config)
    except AttributeError:
        engine = builder.build_engine(network, config)
        engine_bytes = engine.serialize()
        del engine
    with open(args.engine_file_path, 'wb') as f:
        f.write(engine_bytes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pytorch to trt args')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--channel', type=int, default=3)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--cache_file', type=str, default='./resnet50_calibration.cache')
    parser.add_argument('--mode', type=str, default='int8')
    parser.add_argument("--onnx_file_path", type=str, default='./model/onnx/resnet50.onnx', help='onnx_file_path')
    parser.add_argument("--engine_file_path", type=str, default='./model/trt/resnet50_int8.engine', help='engine_file_path')
    args = parser.parse_args()

    if args.mode.lower() == 'int8':
        calib = Resnet50EntropyCalibrator(args)
    else:
        calib = None
    ONNX2TRT(args, calib)                          
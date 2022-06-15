import argparse
from time import time
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import torch
import tensorrt as trt
from collections import OrderedDict
import torchvision
import PIL.Image as Image
from tqdm import tqdm
from torchvision import transforms

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=False)
    
    def forward(self, x):
        feature = self.backbone(x)
        probability = torch.softmax(feature, dim=1)
        return probability

def load_checkpoint(net, pretrained):

    state_dict_total = torch.load(pretrained)

    net_state_dict_rename = OrderedDict()

    net_dict = net.state_dict()

    state_dict = state_dict_total
    for k, v in state_dict.items():

        # name = k[7:]
        name = k
        net_state_dict_rename[name] = v

    net_dict.update(net_state_dict_rename)
    net.load_state_dict(net_dict, strict=False)

    return net

def create_torch_model():
    model = Classifier()
    # model = load_checkpoint(model, pretrained='./model/pth/resnet50.pth')
    model.load_state_dict(torch.load('./model/pth/resnet50.pth'))

    return model

def loadEngine2TensorRT(filepath):
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # 反序列化引擎
    with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def load_normalized_test_case(test_image, pagelocked_buffer):
    # 将输入图片转换为CHW numpy数组
    def normalize_image(image):
        # 调整大小、平滑、转换图像为CHW
        c, h, w = 3, 224, 224
        image_arr = np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(trt.float32)).ravel()
        return (image_arr / 255.0 - 0.45) / 0.225
    
    # 规范化图像，并将图像复制到锁页内存中
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image

def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pytorch to trt args')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--channel', type=int, default=3)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    # parser.add_argument('--mode', type=str, default='int8')
    # parser.add_argument("--onnx_file_path", type=str, default='./model/onnx/resnet50.onnx', help='onnx_file_path')
    # parser.add_argument("--engine_file_path", type=str, default='./model/trt/resnet50_int8.engine', help='engine_file_path')
    args = parser.parse_args()

    inference_times = 100

    test_img = './data/binoculars.JPEG'
    labels_file = 'class_labels.txt'
    labels = open(labels_file, 'r').read().split('\n')


    model_torch = create_torch_model()
    model_torch.eval()
    model_torch.cuda()

    image = Image.open(test_img)
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )])
    image_t = preprocess(image)
    batch_t = torch.unsqueeze(image_t, 0).cuda()

    t_begine = time.time()
    with torch.no_grad():
        for i in tqdm(range(inference_times)):
            outputs_torch = model_torch(batch_t)
    t_end = time.time()
    torch_time = (t_end - t_begine)/inference_times  # 计算torch模型处理一张图片所需要时间
    _, index = torch.max(outputs_torch, 1)
    inferclass = index.cpu().numpy().tolist()[0]
    pred = labels[inferclass]
    print("Correctly recognized " + test_img + " as " + pred)


    engine_file = './model/trt/resnet50_fp32.engine'
    engine = loadEngine2TensorRT(engine_file)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()
    test_case = load_normalized_test_case(test_img, inputs[0].host)  # inputs[0].host表示锁业内存

    t_begine = time.time()
    for i in tqdm(range(inference_times)):
        trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    t_end = time.time()
    trt_fp32_time = (t_end - t_begine)/inference_times
    pred = labels[np.argmax(trt_outputs[0])]
    print("Correctly recognized " + test_case + " as " + pred)



    engine_file = './model/trt/resnet50_fp16.engine'
    engine = loadEngine2TensorRT(engine_file)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()
    test_case = load_normalized_test_case(test_img, inputs[0].host)  # inputs[0].host表示锁业内存

    t_begine = time.time()
    for i in tqdm(range(inference_times)):
        trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    t_end = time.time()
    trt_fp16_time = (t_end - t_begine)/inference_times
    pred = labels[np.argmax(trt_outputs[0])]
    print("Correctly recognized " + test_case + " as " + pred)


    engine_file = './model/trt/resnet50_int8.engine'
    engine = loadEngine2TensorRT(engine_file)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()
    test_case = load_normalized_test_case(test_img, inputs[0].host)  # inputs[0].host表示锁业内存

    t_begine = time.time()
    for i in tqdm(range(inference_times)):
        trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    t_end = time.time()
    trt_int8_time = (t_end - t_begine)/inference_times
    pred = labels[np.argmax(trt_outputs[0])]
    print("Correctly recognized " + test_case + " as " + pred)

    print('==> Torch time: {:.5f} ms'.format(torch_time))
    print('==> TRT fp32 time: {:.5f} ms'.format(trt_fp32_time))
    print('==> TRT fp16 time: {:.5f} ms'.format(trt_fp16_time))
    print('==> TRT int8 time: {:.5f} ms'.format(trt_int8_time))






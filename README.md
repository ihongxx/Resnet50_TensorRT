# Resnet50_TensorRT

本项目介绍了深度学习中TensorRT对Resnet50图像分类算法的推理加速,其中int8量化采用IMAGENET数据集进行校准，项目内容如下：
* 根据Resnet50预训练模型生成torch模型；
* 根据torch模型生成onnx模型；
* 根据onnx模型生成trt fp32 engine；
* 根据onnx模型生成trt fp16 engine；
* 根据onnx模型生成trt int8 engine；
* torch、trt fp32、trt fp16、trt int8模型进行推理加速对比；


## 环境配置：
  ```
  windows11
  CUDA11.6
  CUDNN8.4.0
  
  TensorRT8.4.0.6
  pytorch1.11
  numpy1.21.6
  Pillow9.0.1
  ```

## 一、Build torch model
```
# 利用Resnet50预训练模型resnet50-0676ba61.pth构建torch模型
python to_torch.py --pretrained_model ./model/pre/resnet50-0676ba61.pth --torch_file_path ./model/pth/resnet50.pth
```

## 二、Build onnx model
```
# 利用torch模型构建onnx模型
python to_onnx.py --torch_file_path ./model/pth/resnet50.pth --onnx_file_path ./model/onnx/resnet50.onnx
```

## 三、Build trt model
```
# 利用onnx模型构建trt fp32推理引擎
python to_trt.py --onnx_file_path ./model/onnx/resnet50.onnx --engine_file_path ./model/trt/resnet50_fp32.engine

# 利用onnx模型构建trt fp16推理引擎
python to_trt.py --mode fp16 --onnx_file_path ./model/onnx/resnet50.onnx --engine_file_path ./model/trt/resnet50_fp16.engine

# 利用onnx模型构建trt int8推理引擎
python to_trt.py --mode int8 --cache_file ./resnet50_calibration.cache --img_dir E:/Datasets/IMAGES/image_cls/ILSVRC2012_img_val/ \
                  --onnx_file_path ./model/onnx/resnet50.onnx --engine_file_path ./model/trt/resnet50_int8.engine

```

## 四、Inference
```
# 对torch模型以及fp32、fp16、int8引擎进行推理
python demo.py
```

## 五、Inference Results
```
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:09<00:00, 10.21it/s] 
Correctly recognized ./data/binoculars.JPEG as binoculars
[06/16/2022-00:13:58] [TRT] [W] TensorRT was linked against cuDNN 8.3.2 but loaded cuDNN 8.2.0
[06/16/2022-00:13:58] [TRT] [W] TensorRT was linked against cuDNN 8.3.2 but loaded cuDNN 8.2.0
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 160.36it/s]
Correctly recognized ./data/binoculars.JPEG as binoculars
[06/16/2022-00:13:59] [TRT] [W] TensorRT was linked against cuDNN 8.3.2 but loaded cuDNN 8.2.0
[06/16/2022-00:13:59] [TRT] [W] TensorRT was linked against cuDNN 8.3.2 but loaded cuDNN 8.2.0
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 441.88it/s]
Correctly recognized ./data/binoculars.JPEG as binoculars
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 680.44it/s]
Correctly recognized ./data/binoculars.JPEG as binoculars
==> Torch time: 0.09797 ms
==> TRT fp32 time: 0.00624 ms
==> TRT fp16 time: 0.00226 ms
==> TRT int8 time: 0.00148 ms
```

# Resnet50_TensorRT

本项目介绍了深度学习中TensorRT对Resnet50图像分类算法的推理加速。包括：\
1、根据预训练模型生成torch模型；\
2、根据torch模型生成onnx模型；\
3、根据onnx模型生成trt fp32 模型；\
4、根据onnx模型生成trt fp16模型；\
5、根据onnx模型生成trt int8模型；\
6、对torch、trt fp32、trt fp16、trt int8模型进行推理加速对比；

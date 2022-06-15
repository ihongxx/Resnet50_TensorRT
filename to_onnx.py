from statistics import mode
from torchvision import models
import torch
import torchvision
import argparse

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 使用torchvision自带的预训练模型
        self.backbone = torchvision.models.resnet50(pretrained=False)
        # self.backbone.load_state_dict(torch.load(model_path))
    
    def forward(self, x):
        feature = self.backbone(x)
        probability = torch.softmax(feature, dim=1)
        return probability

def get_onnx(args, example_tensor):

    net = Classifier()
    net.load_state_dict(torch.load(args.torch_file_path))
    net.eval()
    model = net.to(args.device)

    example_tensor = example_tensor.to(args.device)

    _ = torch.onnx.export(model, example_tensor, args.onnx_file_path, verbose=False,
                        #   training=False, 
                        #   do_constant_folding=True, 
                          input_names=['input'], output_names=['output'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='build resnet50 torch model args')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--torch_file_path", type=str, default='./model/pth/resnet50.pth', help='torch_file_path')
    parser.add_argument("--onnx_file_path", type=str, default='./model/onnx/resnet50.onnx', help='onnx_file_path')
    args = parser.parse_args()

    example_tensor = torch.randn(1, 3, 224, 224)

    get_onnx(args, example_tensor)
from operator import imod
import torch
import argparse
import torchvision

"""
    加载resnet50网络结构;
    使用预训练模型参数赋值;
    重新构建一个带参数的resnet50模型;
"""

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 使用torchvision自带的预训练模型
        self.backbone = torchvision.models.resnet50(pretrained=False)
    
    def forward(self, x):
        feature = self.backbone(x)
        probability = torch.softmax(feature, dim=1)
        return probability

def main():
    parser = argparse.ArgumentParser(description='build resnet50 torch model args')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--pretrained_model", type=str, default='./model/pre/resnet50-0676ba61.pth', help='engine_file_path')
    parser.add_argument("--torch_file_path", type=str, default='./model/pth/resnet50.pth', help='onnx_file_path')
    args = parser.parse_args()

    # 使用标准resnet50模型结构
    if args.pretrained:
        net = torchvision.models.resnet50(pretrained=True)
    # 官方resnet50模型结构添加softmax分类
    else:
        net = Classifier()
        net.load_state_dict(torch.load(args.torch_file_path))
    net = net.to(args.device)
    net.eval()
    torch.save(net.state_dict(), args.torch_file_path)

if __name__ == '__main__':
    main()
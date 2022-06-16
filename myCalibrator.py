from tkinter import N
from tkinter.tix import IMAGE
from matplotlib.pyplot import cla
import os
import numpy as np
import torchvision.transforms as transforms
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import tensorrt as trt


class Resnet50EntropyCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, args, files_path='./data/val_100.txt'):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = args.cache_file

        self.batch_size = args.batch_size
        self.Channel = args.channel
        self.Height = args.height
        self.Width = args.width
        self.transform = transforms.Compose([
            transforms.Resize([self.Height, self.Width]),
            transforms.ToTensor(),
        ])

        self._txt_file = open(files_path, 'r')
        self._lines = self._txt_file.readlines()
        self.imgs = []
        for line in self._lines:
            line = line.strip()
            # line = line.split(' ')[0]
            self.imgs.append(os.path.join(args.img_dir, line))
        self.current_idx = 0
        self.max_batch_idx = len(self.imgs)//self.batch_size  # 一共多少个块
        self.data_size = trt.volume([self.batch_size, self.Channel,self.Height, self.Width]) * trt.float32.itemsize
        self.device_input = cuda.mem_alloc(self.data_size)

    def next_batch(self):
        if self.current_idx < self.max_batch_idx:
            batch_imgs = np.zeros((self.batch_size, self.Channel, self.Height, self.Width))
            for i in range(self.batch_size):
                img = Image.open(self.imgs[self.current_idx + i])
                img = self.transform(img).numpy()
                assert (img.nbytes == self.data_size/self.batch_size), 'not valid img !' + i
                batch_imgs[i] = img
            self.current_idx += 1
            print('batch:[{}/{}]'.format(self.current_idx, self.max_batch_idx))
            return np.ascontiguousarray(batch_imgs)  # 将不连续的内存分配为连续的内存中
        else:
            return np.array([])

    def get_batch_size(self):
        return self.batch_size

    
    def get_batch(self, names):
        if self.current_idx + self.batch_size > len(self.imgs):
            return None
        
        current_batch = int(self.current_idx / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))
        
        # 获得一个块数据
        try:
            batch_imgs = self.next_batch()
            if batch_imgs.size == 0 or batch_imgs.size != self.batch_size*self.Channel*self.Height*self.Width:
                return None
            cuda.memcpy_htod(self.device_input, batch_imgs.astype(np.float32))  # 传输到GPU
            self.current_idx += self.batch_size
            return [int(self.device_input)]
        except:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
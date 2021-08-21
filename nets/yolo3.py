from collections import OrderedDict

import torch
import torch.nn as nn

import numpy as np
from PIL import Image
from nets.darknet import darknet53


'''
Author: TuZhou
Description: 一次DBL操作
param {*} filter_in
param {*} filter_out
param {*} kernel_size
return {*}
'''
def DBL(filter_in, filter_out, kernel_size):
    #1x1卷积操作不需要pad
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.ModuleList([
        DBL(in_filters, filters_list[0], 1),
        DBL(filters_list[0], filters_list[1], 3),
        DBL(filters_list[1], filters_list[0], 1),
        DBL(filters_list[0], filters_list[1], 3),
        DBL(filters_list[1], filters_list[0], 1),
        DBL(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                        stride=1, padding=0, bias=True)
    ])
    return m


class YoloBody(nn.Module):
    def __init__(self, anchor, num_classes = 20):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone = darknet53()

        # out_filters : [64, 128, 256, 512, 1024]
        out_filters = self.backbone.layers_out_filters

        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #------------------------------------------------------------------------#
        
        #输出13x13x75的特征图（对于voc数据集）
        final_out_filter0 = len(anchor[0]) * (5 + num_classes)
        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], final_out_filter0) 

        #输出26x26x75的特征图
        final_out_filter1 = len(anchor[1]) * (5 + num_classes)
        self.last_layer1_conv = DBL(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, final_out_filter1)

        #输出52x52x75的特征图
        final_out_filter2 = len(anchor[2]) * (5 + num_classes)
        self.last_layer2_conv = DBL(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, final_out_filter2)


    def forward(self, x):
        #在函数内定义函数必须在该函数内调用
        '''
        Author: TuZhou
        Description: 
        param {*} last_layer为输入的列表，包含DBL等操作
        param {*} layer_in为输入张量
        return {*}返回输出的最终特征图与待上采样的特征图
        '''
        def _branch(last_layer, layer_in):
            #enumerate为列表中每个元素都添加了一个索引，从零开始
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)  
                if i == 4:              #索引为4时说明完成了5次DBL操作，此时的输出应该准备用于上采样
                    out_branch = layer_in
            return layer_in, out_branch
        #---------------------------------------------------#   
        #   从darknet59中获得三个待拼接特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)

        #---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        #---------------------------------------------------#
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        out0, out0_branch = _branch(self.last_layer0, x0)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)

        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)
        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1, out1_branch = _branch(self.last_layer1, x1_in)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)
        #---------------------------------------------------#
        #   第一个特征层
        #   out3 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out2, _ = _branch(self.last_layer2, x2_in)
        # print("out is {}".format(out1))
        # print("out2 is {}".format(out2.size()))
        return out0, out1, out2





def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


if __name__ == "__main__":
    img = './img/street.jpg'
    image = Image.open(img)
    img = np.array(image)
    ten_image = torch.from_numpy(img)
    anchors_path = '../cfg/yolo_anchors.txt'
    anchors  = get_anchors(anchors_path)
    model = YoloBody(anchor=anchors)
    print(model(ten_image))
import os

import math
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from matplotlib import pyplot as plt



'''
Author: TuZhou
Description: 计算先验框与真实框的iou
param {*} _box_a为真实框
param {*} _box_b为先验框
return {*}返回iou值
'''
def jaccard(_box_a, _box_b):
    # 计算真实框的左上角和右下角
    #[:, 0]为numpy中的切片操作，取二维数组中所有数组中的索引为0的元素，等同于[..., 0]
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    # 计算先验框的左上角和右下角
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)

    inter = inter[:, :, 0] * inter[:, :, 1]
    # 计算先验框和真实框各自的面积
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    # 求IOU
    union = area_a + area_b - inter
    return inter / union  # [A,B]
    
def clip_by_tensor(t,t_min,t_max):
    t=t.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


#均方差损失函数
def MSELoss(pred,target):
    return (pred-target)**2


'''
Author: TuZhou
Description: 交叉熵损失函数
param {*} pred为预测的x或y坐标值
param {*} target为真实框与网格的x或y偏移值
return {*}
'''
def BCELoss(pred,target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


'''
Author: TuZhou
Description: 损失函数
param {*} self
param {*} anchors为所有锚的宽高，为二维数组，从前往后为大图到小图
param {*} num_classes为分类数量
param {*} img_size为输入的原图大小
param {*} cuda为是否使用cuda
param {*} normalize为是否使用标准化
return {*}
'''
class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, cuda, normalize):
        super(YOLOLoss, self).__init__()
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        #-----------------------------------------------------------#
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        #-------------------------------------#
        #   获得三个特征层的宽高
        #   13、26、52
        #-------------------------------------#
        self.feature_length = [img_size[0]//32,img_size[0]//16,img_size[0]//8]
        self.img_size = img_size

        self.ignore_threshold = 0.5         #iou阈值
        self.lambda_xy = 1.0
        self.lambda_wh = 1.0
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.cuda = cuda
        self.normalize = normalize

    #input为一个batch的一个尺寸特征图的所有张量，是一个四维数组(batchsize, 75, w, h)，targets为一组原图标签
    def forward(self, input, targets=None):
        # print("input:")
        # print(input)
        # print('\n')
        # print(targets)
        #----------------------------------------------------#
        #   input的shape为  bs, 3*(5+num_classes), 13, 13
        #                   bs, 3*(5+num_classes), 26, 26
        #                   bs, 3*(5+num_classes), 52, 52
        #----------------------------------------------------#
        
        #-----------------------#
        #   一共多少张图片
        #-----------------------#
        bs = input.size(0)
        #-----------------------#
        #   特征层的高
        #-----------------------#
        in_h = input.size(2)
        #-----------------------#
        #   特征层的宽
        #-----------------------#
        in_w = input.size(3)

        #-----------------------------------------------------------------------#
        #   计算步长
        #   每一个特征点对应原来的图片上多少个像素点
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        #   如果特征层为52x52的话，一个特征点就对应原来的图片上的8个像素点
        #   stride_h = stride_w = 32、16、8
        #-----------------------------------------------------------------------#
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w

        #-------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #   所以也要除上步长
        #-------------------------------------------------#
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        
        #-----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   batch_size, 3, 13, 13, 5 + num_classes
        #   batch_size, 3, 26, 26, 5 + num_classes
        #   batch_size, 3, 52, 52, 5 + num_classes
        #   将bs, 3x(5 + num_classes), 13x13的格式转化为上面的格式
        #   最终为8x3x13x13x长度为25的张量
        #-----------------------------------------------#
        prediction = input.view(bs, int(self.num_anchors/3),
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        # print(len(prediction[0]))
        # #print(prediction[0][0])
        # print(len(prediction[0][0]))
        # print('\n')
        # #print(prediction[0][0][0])
        # print(len(prediction[0][0][0]))
        # exit()
        
        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]
        h = prediction[..., 3]
        # 获得置信度，是否有物体
        conf = torch.sigmoid(prediction[..., 4])
        # 种类置信度，从第五个其后面全为种类预测置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])

        #---------------------------------------------------------------#
        #   找到哪些先验框内部包含物体
        #   利用真实框和先验框计算交并比
        #   mask        batch_size, 3, in_h, in_w   无目标的特征点
        #   noobj_mask  batch_size, 3, in_h, in_w   有目标的特征点
        #   tx          batch_size, 3, in_h, in_w   中心x偏移情况
        #   ty          batch_size, 3, in_h, in_w   中心y偏移情况
        #   tw          batch_size, 3, in_h, in_w   宽高调整参数的真实值
        #   th          batch_size, 3, in_h, in_w   宽高调整参数的真实值
        #   tconf       batch_size, 3, in_h, in_w   置信度真实值
        #   tcls        batch_size, 3, in_h, in_w, num_classes  种类真实值
        #----------------------------------------------------------------#
        mask, noobj_mask, tx, ty, tw, th, tconf, tcls, box_loss_scale_x, box_loss_scale_y =\
                                                                            self.get_target(targets, scaled_anchors,
                                                                                            in_w, in_h,
                                                                                            self.ignore_threshold)

        #---------------------------------------------------------------#
        #   将预测结果进行解码，判断预测结果和真实值的重合程度
        #   如果重合程度过大则忽略即标记为0，因为这些特征点属于预测比较准确的特征点
        #   作为负样本不合适
        #----------------------------------------------------------------#
        noobj_mask = self.get_ignore(prediction, targets, scaled_anchors, in_w, in_h, noobj_mask)

        if self.cuda:
            box_loss_scale_x = (box_loss_scale_x).cuda()
            box_loss_scale_y = (box_loss_scale_y).cuda()
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()
        box_loss_scale = 2 - box_loss_scale_x * box_loss_scale_y
        
        # 计算中心偏移情况的loss，使用BCELoss效果好一些，即交叉熵loss
        loss_x = torch.sum(BCELoss(x, tx) * box_loss_scale * mask)
        loss_y = torch.sum(BCELoss(y, ty) * box_loss_scale * mask)
        # 计算宽高调整值的loss，使用均方差Loss
        loss_w = torch.sum(MSELoss(w, tw) * 0.5 * box_loss_scale * mask)
        loss_h = torch.sum(MSELoss(h, th) * 0.5 * box_loss_scale * mask)
        # 计算置信度的loss
        loss_conf = torch.sum(BCELoss(conf, mask) * mask) + \
                    torch.sum(BCELoss(conf, mask) * noobj_mask)
                    
        loss_cls = torch.sum(BCELoss(pred_cls[mask == 1], tcls[mask == 1]))

        loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                loss_conf * self.lambda_conf + loss_cls * self.lambda_cls

        # print(loss, loss_x.item() + loss_y.item(), loss_w.item() + loss_h.item(), 
        #         loss_conf.item(), loss_cls.item(), \
        #         torch.sum(mask),torch.sum(noobj_mask))
        if self.normalize:
            num_pos = torch.sum(mask)
            num_pos = torch.max(num_pos, torch.ones_like(num_pos))
        else:
            num_pos = bs/3

        # print(torch.sum(mask))
        # print(num_pos)
        # exit()
        # print(loss)
        # print(num_pos)
        # exit()
        
        #loss为一个batch图片计算出的损失和
        return loss, num_pos


    '''
    Author: TuZhou
    Description: 
    param {*} self
    param {*} target为真实框
    param {*} scaled_anchors为每个anchor的高宽
    param {*} in_w特征图的宽
    param {*} in_h为特征图的高
    param {*} ignore_threshold
    return {*} 返回一堆矩阵，分别为是否有目标，是否无目标，x,y,w,h的调整参数，物体置信度，种类置信度，真实框宽高
    '''
    def get_target(self, target, scaled_anchors, in_w, in_h, ignore_threshold):
        #-----------------------------------------------------#
        #   计算一共有多少张图片
        #-----------------------------------------------------#
        bs = len(target)
        #-------------------------------------------------------#
        #   获得当前特征层先验框所属的编号，方便后面对先验框筛选
        #-------------------------------------------------------#
        anchor_index = [[0,1,2],[3,4,5],[6,7,8]][self.feature_length.index(in_w)]
        subtract_index = [0,3,6][self.feature_length.index(in_w)]
        #-------------------------------------------------------#
        #   创建全是0或者全是1的阵列
        #-------------------------------------------------------#
        mask = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)

        tx = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, self.num_classes, requires_grad=False)

        box_loss_scale_x = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        box_loss_scale_y = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        for b in range(bs):    
            #没有目标        
            if len(target[b]) == 0:
                continue
            #-------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            #   原ground truth的值在0,1之间，需要转化到特征图上的真实值
            #-------------------------------------------------------#
            gxs = target[b][:, 0:1] * in_w
            gys = target[b][:, 1:2] * in_h
            
            #-------------------------------------------------------#
            #   计算出正样本相对于特征层的宽高
            #-------------------------------------------------------#
            gws = target[b][:, 2:3] * in_w
            ghs = target[b][:, 3:4] * in_h
            # print(gws)
            # print(ghs)

            #-------------------------------------------------------#
            #   计算出特征图真实目标中心点所在的左上角网格坐标
            #   torch.floor返回小于等于其值的最大整数
            #-------------------------------------------------------#
            gis = torch.floor(gxs)
            gjs = torch.floor(gys)
            
            #-------------------------------------------------------#
            #   将真实框转换一个形式
            #   num_true_box, 4
            #   torch.zeros_like生成和括号维度相同的0矩阵
            #   最终张量长度为4，前两个值为0，代表坐标
            #-------------------------------------------------------#
            gt_box = torch.FloatTensor(torch.cat([torch.zeros_like(gws), torch.zeros_like(ghs), gws, ghs], 1))
            #print(gt_box)
            # print(gt_box.size())
            
            #-------------------------------------------------------#
            #   将先验框转换一个形式
            #   9个先验框均加入列表中，长度为4，前两个值为0，后面两个代表先验框的宽高，列表维度为9
            #-------------------------------------------------------#
            anchor_shapes = torch.FloatTensor(torch.cat((torch.zeros((self.num_anchors, 2)), torch.FloatTensor(scaled_anchors)), 1))
            # print(anchor_shapes)
            # print(anchor_shapes.size())
            # exit()
            #-------------------------------------------------------#
            #   计算交并比
            #   num_true_box, 9
            #-------------------------------------------------------#
            anch_ious = jaccard(gt_box, anchor_shapes)

            #-------------------------------------------------------#
            #   计算和每个真实框重合度最大的先验框是哪个
            #   num_true_box, 
            #-------------------------------------------------------#
            #print(anch_ious)
            best_ns = torch.argmax(anch_ious, dim=-1)
            # print(best_ns.size())
            # print(best_ns)
            # exit()
            for i, best_n in enumerate(best_ns):
                #若iou最大的先验框不属于该特征层
                #i为对比的真实框索引
                if best_n not in anchor_index:
                    continue
                #-------------------------------------------------------------#
                #   取出各类坐标：
                #   gi和gj代表的是真实框中心网格左上角x轴y轴坐标
                #   gx和gy代表真实框中心的x轴和y轴坐标
                #   gw和gh代表真实框的宽和高
                #-------------------------------------------------------------#
                gi = gis[i].long()
                gj = gjs[i].long()
                gx = gxs[i]
                gy = gys[i]
                gw = gws[i]
                gh = ghs[i]

                if (gj < in_h) and (gi < in_w):
                    #获取iou最大先验框对应该特征层的索引
                    best_n = best_n - subtract_index

                    #----------------------------------------#
                    #   负样本noobj_mask代表有无目标，原本是全1矩阵，b为batch中的索引
                    #   gi,gj为真实框的中心网格，此处一定有目标
                    #----------------------------------------#
                    noobj_mask[b, best_n, gj, gi] = 0
                    #----------------------------------------#
                    #   正样本mask代表有无目标，原本是全0矩阵
                    #----------------------------------------#
                    mask[b, best_n, gj, gi] = 1
                    #----------------------------------------#
                    #   tx、ty代表中心调整参数的真实值，真实框中心点减去其网格左上角坐标
                    #----------------------------------------#
                    tx[b, best_n, gj, gi] = gx - gi.float()
                    ty[b, best_n, gj, gi] = gy - gj.float()
                    #----------------------------------------#
                    #   tw、th代表宽高调整参数的真实值，为log(真实框的宽高除以iou最大的先验框的宽高)
                    #----------------------------------------#
                    tw[b, best_n, gj, gi] = math.log(gw / scaled_anchors[best_n+subtract_index][0])
                    th[b, best_n, gj, gi] = math.log(gh / scaled_anchors[best_n+subtract_index][1])
                    #----------------------------------------#
                    #   用于获得xywh的比例
                    #   大目标loss权重小，小目标loss权重大
                    #----------------------------------------#
                    # print(target[b][i, 2])
                    # print(target[b])
                    # exit()
                    #box_loss_scale_x代表对应真实框的宽
                    box_loss_scale_x[b, best_n, gj, gi] = target[b][i, 2]
                    box_loss_scale_y[b, best_n, gj, gi] = target[b][i, 3]
                    #----------------------------------------#
                    #   tconf代表物体置信度，gi,gi是相对于target的值，此处就应该有目标，所以设为1
                    #----------------------------------------#
                    tconf[b, best_n, gj, gi] = 1
                    #----------------------------------------#
                    #   tcls代表种类置信度，target[b][i, 4]表示第b张图片的第i个真实框的物体种类，4为种类
                    #   设置为1代表该种类处设置为1表明了此预测框为什么种类
                    #   target维度应该为bsx框框数x5
                    #----------------------------------------#
                    tcls[b, best_n, gj, gi, int(target[b][i, 4])] = 1
                    # print(target[b])
                    # print(target[b].size())
                    # print(target[b][i, 4])
                    # print((target[b][i, 4]).size())
                    # exit()
                else:
                    print('Step {0} out of bound'.format(b))
                    print('gj: {0}, height: {1} | gi: {2}, width: {3}'.format(gj, in_h, gi, in_w))
                    continue

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls, box_loss_scale_x, box_loss_scale_y


    '''
    Author: TuZhou
    Description: 
    param {*} self
    param {*} prediction为一组图片的一个特征图输出
    param {*} target为该组图片的标签
    param {*} scaled_anchors代表先验框的宽高
    param {*} in_w为原图宽
    param {*} in_h为原图高
    param {*} noobj_mask为负样本，有无物体，1为有，0为无
    return {*}
    '''
    def get_ignore(self,prediction,target,scaled_anchors,in_w, in_h,noobj_mask):
        #-----------------------------------------------------#
        #   计算一共有多少张图片
        #-----------------------------------------------------#
        bs = len(target)
        #-------------------------------------------------------#
        #   获得当前特征层先验框所属的编号，方便后面对先验框筛选
        #-------------------------------------------------------#
        anchor_index = [[0,1,2],[3,4,5],[6,7,8]][self.feature_length.index(in_w)]
        scaled_anchors = np.array(scaled_anchors)[anchor_index]

        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])  
        y = torch.sigmoid(prediction[..., 1])
        # print(x.shape)
        # print(y.shape)
        # exit()
        # 先验框的宽高调整参数
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格，先验框中心，网格左上角
        #torch.linspace(0,in_w - 1,in_w)划分矩阵，起始为0，终止为in_w - 1，划分成in_w个，即长度为in_w
        #repeat()对张量重复扩充，见https://blog.csdn.net/qq_34806812/article/details/89388210
        #x.shape = [8,3,13,13]
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs*self.num_anchors/3), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs*self.num_anchors/3), 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        # index_select按列索引第0列的值也就是宽
        #scaled_anchors维度为3x2
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        # print(anchor_w)
        # print(anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        
        #exit()
        
        #-------------------------------------------------------#
        #   计算调整后的先验框中心与宽高
        #   预测的是与左上角坐标的偏移值，加上左上角坐标就是中心坐标
        #   pred_boxes维度为8,3,13,13,4。8张图片，3个预测框，13x13个格子，4个框框预测值
        #-------------------------------------------------------#
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        #pre_w = e^w + anchor_w
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        #print(prediction[..., :4].size())

        for i in range(bs):
            pred_boxes_for_ignore = pred_boxes[i]
            #-------------------------------------------------------#
            #   将预测结果转换一个形式
            #   pred_boxes_for_ignore      num_anchors, 4
            #-------------------------------------------------------#
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)
            # print(pred_boxes_for_ignore)
            # exit()
            #-------------------------------------------------------#
            #   计算真实框，并把真实框转换成相对于特征层的大小
            #   gt_box      num_true_box, 4
            #-------------------------------------------------------#
            if len(target[i]) > 0:
                gx = target[i][:, 0:1] * in_w
                gy = target[i][:, 1:2] * in_h
                gw = target[i][:, 2:3] * in_w
                gh = target[i][:, 3:4] * in_h
                #torch.cat中dim = -1也是按列拼接，此时拼接的矩阵行数应该相同
                gt_box = torch.FloatTensor(torch.cat([gx, gy, gw, gh],-1)).type(FloatTensor)
                #print(gt_box)
                #a = torch.FloatTensor(torch.cat([gx, gy, gw, gh],1)).type(FloatTensor)
                # print(a)
                # exit()
                #-------------------------------------------------------#
                #   计算预测框与真实框的iou
                #   anch_ious       num_true_box, num_anchors
                #-------------------------------------------------------#
                anch_ious = jaccard(gt_box, pred_boxes_for_ignore)
                #-------------------------------------------------------#
                #   每个先验框对应真实框的最大重合度
                #   此处max取的是与真实框中预测重合度最高的，anch_ious_max维度应为1x507,其中507为3x13x13个预测框计算结果
                #   anch_ious_max   num_anchors
                #-------------------------------------------------------#
                #print(anch_ious.size())

                anch_ious_max, _ = torch.max(anch_ious,dim=0)
                # print(anch_ious_max.size())
                # print(anch_ious_max)
                #将anch_ious_max维度变换为3x13x13，pred_boxes的维度为8x3x13x13x4
                anch_ious_max = anch_ious_max.view(pred_boxes[i].size()[:3])
                #负样本中预测效果好的标记为0，noobj_mask维度为8x3x13x13
                noobj_mask[i][anch_ious_max>self.ignore_threshold] = 0
                # print(pred_boxes[i].size())
                # print(pred_boxes[i].size()[:3])
                # print(noobj_mask.size())
                # print(noobj_mask)
                #exit()
        return noobj_mask


#模型初始化参数
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

class LossHistory():
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = log_dir
        self.time_str   = time_str
        self.save_path  = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.save_path)

    def append_loss(self, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")

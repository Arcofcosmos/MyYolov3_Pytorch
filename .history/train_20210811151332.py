#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


from nets.yolo3 import YoloBody
from nets.yolo_training import YOLOLoss, LossHistory, weights_init
from utils.dataloader import YoloDataset, yolo_dataset_collate



'''
Author: TuZhou
Description: 读取anchors路径将anchors的尺度读取到列表中
param {*} anchors_path
return {*}返回值为3维列表，存放3个2维列表，每个二维列表存放3组一维列表，表示每个anchor的高宽
'''
def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
'''
Author: TuZhou
Description: 进行单周期的训练
param {*} net为网络
param {*} yolo_loss为损失函数
param {*} epoch为当前周期索引
param {*} train_iteration为训练迭代次数
param {*} val_iteration为验证迭代次数
param {*} train_dataLoder为训练集
param {*} val_dataLoader为验证集
param {*} Epoch为总周期数
param {*} cuda为是否使用cuda
return {*}
'''
def fit_one_epoch(net, yolo_loss, epoch, train_iteration, val_iteration, train_dataLoder, val_dataLoader, Epoch, cuda):
    total_loss = 0
    val_loss = 0

    #表示训练模式
    net.train()
    print('Start Train')
    #为每个周期训练添加进度条
    with tqdm(total=train_iteration,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_dataLoder):
            #一个周期结束
            if iteration >= train_iteration:            
                break

            images, targets = batch[0], batch[1]
            # print(targets)
            # exit()
            #无梯度计算
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            #outputs应为5维数组，三个尺寸特征图，8张图片，每个尺寸有75个，13x13,26x26,52x52的尺寸，维度顺序排列
            outputs     = net(images)
            # print(outputs[1][0].size())
            # print(outputs[1][0][0].size())
            # print(outputs[1].size())
            # print(outputs[0].size())
            # print(outputs[2].size())

            
            #exit()
            losses      = []
            num_pos_all = 0
            #----------------------#
            #   计算损失
            #   三个预测框均计算
            #----------------------#
            for i in range(3):
                loss_item, num_pos = yolo_loss(outputs[i], targets)
                losses.append(loss_item)
                #若无归一化，则num_pos_all的最终结果为batch大小
                num_pos_all += num_pos

            #loss应该为一张图片的三个特征层损失和
            loss = sum(losses) / num_pos_all
            # print(num_pos_all)
            # exit()
            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    net.eval()
    print('Start Validation')
    with tqdm(total=val_iteration, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_dataLoader):
            if iteration >= val_iteration:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val  = torch.from_numpy(images_val).type(torch.FloatTensor).cuda()
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                else:
                    images_val  = torch.from_numpy(images_val).type(torch.FloatTensor)
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                optimizer.zero_grad()

                outputs     = net(images_val)
                losses      = []
                num_pos_all = 0
                #----------------------#
                #   计算损失
                #----------------------#
                for i in range(3):
                    loss_item, num_pos = yolo_loss(outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos

                loss = sum(losses) / num_pos_all
                val_loss += loss.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    loss_history.append_loss(total_loss/(train_iteration+1), val_loss/(val_iteration+1))
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' %(total_loss / (train_iteration + 1), val_loss / (val_iteration + 1)))
    print('Saving state, iter:', str(epoch+1))
    #torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch + 1), total_loss / (train_iteration + 1), val_loss / (val_iteration + 1)))

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    Cuda = True
    #------------------------------------------------------#
    #   是否对损失进行归一化，用于改变loss的大小
    #   用于决定计算最终loss是除上batch_size还是除上正样本数量
    #------------------------------------------------------#
    normalize = False
    #------------------------------------------------------#
    #   输入的shape大小
    #------------------------------------------------------#
    input_shape = (416, 416)
    #------------------------------------------------------#
    #   视频中的Config.py已经移除
    #   需要修改num_classes直接修改此处的num_classes即可
    #   如果需要检测5个类, 这里就写5. 默认为20
    #------------------------------------------------------#
    num_classes = 20
    #----------------------------------------------------#
    #   先验框anchor的路径
    #----------------------------------------------------#
    anchors_path = './cfg/yolo_anchors.txt'
    anchors      = get_anchors(anchors_path)
    # print(anchors)
    # exit()
    #------------------------------------------------------#
    #   创建yolo模型
    #   训练前一定要修改Config里面的classes参数
    #------------------------------------------------------#
    model = YoloBody(anchors, num_classes)
    weights_init(model)

    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   将训练过的模型载入，再次微调使训练更快
    #------------------------------------------------------#
    model_path      = "./model/yolo_weights.pth"
    print('Loading weights into state dict...')
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict      = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished load_state_dict init_weights...!')

    net = model.train()

    if Cuda:
        #有多GPU则使用多GPU训练
        net = torch.nn.DataParallel(model) 
        #该值为True则程序自动搜寻适合卷积的最优算法，加快计算速度     
        cudnn.benchmark = True
        net = net.cuda()

    #np.reshape(anchors,[-1,2])将anchors三维数组装换为二维数组，从前往后为大图到小图
    yolo_loss    = YOLOLoss(np.reshape(anchors,[-1,2]), num_classes, (input_shape[1], input_shape[0]), Cuda, normalize)
    loss_history = LossHistory("logs/")

    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    annotation_path = './dataset/2007_train.txt'
    #----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    #----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Unfreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        lr              = 1e-3
        Batch_size      = 8
        Init_Epoch      = 0
        Freeze_Epoch    = 50
        
        optimizer       = optim.Adam(net.parameters(),lr)
        #lr_scheduler为学习率更新，一个周期更新一次，每次更新为lr*gamma
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset   = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), True)
        val_dataset     = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), False)
        train_dataLoder             = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        val_dataLoader         = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)
                        
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        for param in model.backbone.parameters():
            #模型骨干网络的参数不会更新
            param.requires_grad = False              

        train_iteration      = num_train // Batch_size
        val_iteration  = num_val // Batch_size
        
        if train_iteration == 0 or val_iteration == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(net, yolo_loss, epoch, train_iteration, val_iteration, train_dataLoder, val_dataLoader, Freeze_Epoch, Cuda)
            lr_scheduler.step()       
            
    if True:
        lr              = 1e-4
        Batch_size      = 4
        Freeze_Epoch    = 50
        Unfreeze_Epoch  = 100

        optimizer       = optim.Adam(net.parameters(),lr)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)
        
        train_dataset   = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), True)
        val_dataset     = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), False)
        train_dataLoder             = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        val_dataLoader         = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)
                        
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        train_iteration      = num_train//Batch_size
        val_iteration  = num_val//Batch_size
        
        if train_iteration == 0 or val_iteration == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            fit_one_epoch(net, yolo_loss, epoch, train_iteration, val_iteration, train_dataLoder, val_dataLoader, Unfreeze_Epoch, Cuda)
            lr_scheduler.step()

        torch.save(model.state_dict, './logs/yolov3.pth')

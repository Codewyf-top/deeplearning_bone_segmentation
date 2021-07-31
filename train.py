import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
import os, glob, cv2, time
# import segmentation_models_pytorch as smp
from albumentations import *
from lib.dataset import MyDataset
from lib.unet import unet_resnet
from lib.unet_base import U_Net
from lib.utils import diceCoeff
from torch.utils.tensorboard import SummaryWriter
import copy

# tensorboard路径，训练可视化
log_dir = 'result/Unet/log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_writer = SummaryWriter(log_dir=log_dir)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 模型存储路径
model_dir = 'result/Unet/checkpoint'


def make_dirs():  # 对应文件夹不存在的话就创建文件夹
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


# 损失函数
bce_fn = torch.nn.BCEWithLogitsLoss()


def bce_loss(outputs, targets):
    return bce_fn(outputs, targets)


# 数据增强
# 可以自己选择加不加，我感觉你的数据比较特殊，最多加一些噪声的增强，翻转/crop等操作会破坏原信息
def get_aug(p=0.7):
    return Compose([
        #VerticalFlip(p=0.4),  # flip-y
        #RandomRotate90(p=0.5),  # rorate
        #RandomResizedCrop(512, 512, p=0.6),
        ShiftScaleRotate(shift_limit=0, scale_limit=(0.8, 1.2), rotate_limit=(-60, 60), p=1),
        OneOf([
            RandomContrast(limit=0.2),
            RandomBrightness(limit=0.2),
            RandomGamma(),
        ]),
    ], p=p)  # 0.836


def train():
    n_save_iter = 50  # 每隔200 iter保存一下模型
    n_val_iter = 10  # 每隔10 iter计算测试及loss和accuracy
    load_iters = 0  # 加载之前训练的模型(指定iter数)
    # 实例化模型
    model = U_Net(3, 11).to(device)
    # model = unet_resnet('resnext50_32x4d', 3, 11, False).to(device)
    # model = unet_resnet('resnet18', 3, 11, False).to(device)
    # model = unet_resnet('resnet101', 3, 11, False).to(device)
    # model = smp.unet(encoder_name='resnet34', in_channels=3, classes=11)  # segmentation_models_pytorch库可以很方便的调用各种语义分割模型架构，可以github主页看具体支持哪些结构
    # model = smp.unet(encoder_name='efficientnet-b4', in_channels=3, classes=11)
    # model = smp.unet(encoder_name='se_resnet50', in_channels=3, classes=11)
    # 加载训练好的权重
    if (load_iters != 0):
        pre = torch.load(os.path.join('./result/checkpoint', str(load_iters) + '.pth'))
        model.load_state_dict(pre)
    model.train()  # 将模式设置为train（开启dropout、BN层等）
    opt = Adam(model.parameters(), lr=3e-4)  # 使用Adam优化器，并设置学习率

    # 加载训练数据
    dataset = MyDataset(tfms=get_aug(), n_class=11)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(MyDataset(tfms=None, n_class=11, train=False), batch_size=1, shuffle=True,
                                num_workers=0)

    # 训练
    iters = load_iters
    train_loss_log = open('result/Unet/train_loss.txt', 'a')
    train_accuracy_log = open('result/Unet/train_accuracy.txt', 'a')
    val_loss_log = open('result/Unet/val_loss.txt', 'a')
    val_accuracy_log = open('result/Unet/val_accuracy.txt', 'a')
    while iters < 3000:
        for imgs, masks in dataloader:
            iters += 1
            imgs = imgs.to(device=device, non_blocking=True)
            masks = masks.to(device=device, non_blocking=True)

            # 模型预测
            output = model(imgs)
            # 计算loss
            loss = bce_loss(output, masks)
            # 反向传播
            opt.zero_grad()
            loss.backward()
            opt.step()
            # 打印并记录损失值
            train_loss_log.writelines(str(iters) + ',' + str(round(loss.item(), 6)) + '\n')
            log_writer.add_scalar('Loss/train', loss.item(), iters)  # tensorboard
            print('iter{%d} ' % (iters) + ' loss= %.5f ' % (loss.item()))
            # 计算训练集accuracy/测试集loss和accuracy    这部分不参与反向传播，因此不用更新梯度
            with torch.no_grad():
                # 计算训练集accuracy
                accuracy = diceCoeff(output, masks)
                train_accuracy_log.writelines(str(iters) + ',' + str(round(accuracy.item(), 6)) + '\n')
                log_writer.add_scalar('Accuracy/train', accuracy.item(), iters)
                # 计算测试集loss和accuracy
                if (iters % n_val_iter == 0):
                    loss_val, accuracy_val, accuracy_class = val(copy.deepcopy(model), dataloader_val, bce_loss, diceCoeff)
                # 添加进tensorboard
                    val_loss_log.writelines(str(iters) + ',' + str(round(loss_val, 6)) + '\n')
                    val_accuracy_log.writelines(str(iters) + ',' + str(round(accuracy_val, 6)) + '\n')
                    log_writer.add_scalar('Loss/val', loss_val, iters)
                    log_writer.add_scalar('Accuracy/val', accuracy_val, iters)

                    pass

            # 保存模型
            if (iters % n_save_iter == 0):
                save_file_name = os.path.join(model_dir, '%d.pth' % iters)
                torch.save(model.state_dict(), save_file_name)
    for txt in [train_loss_log, train_accuracy_log, val_loss_log, val_accuracy_log]:
        txt.close()

def val(model, dataloader, loss_fn, metrix):
    with torch.no_grad():
        iters = 0
        loss = 0.0
        accuracy = 0.0
        accuracy_class = np.array([0.0] * 11)
        for imgs, masks in dataloader:
            iters += 1
            # cpu转gpu
            imgs = imgs.to(device=device, non_blocking=True)
            masks = masks.to(device=device, non_blocking=True)
            # 模型预测
            output = model(imgs)
            # 计算loss
            loss += loss_fn(output, masks).item()
            accuracy += metrix(output, masks).item()

            for i in range(11):
                accuracy_class[i] += metrix(output[:, i, :, :], masks[:, i, :, :]).item()
                # print('Val: accuracy_class{%d} ' % (i) + ' accuracy= %.5f ' % (accuracy_class[i].item()))
        loss /= iters
        accuracy /= iters

        accuracy_class /= iters
        print(f'Val: accuracy_class:{accuracy_class}\n average accuracy_class: {sum(accuracy_class)/11}')
    return loss, accuracy, accuracy_class


if __name__ == "__main__":
    make_dirs()  # 创建需要的文件夹并指定gpu
    train()
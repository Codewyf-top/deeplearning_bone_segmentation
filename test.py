import torch
import numpy as np
import os, cv2, time
import glob
from lib.unet_base import U_Net
from lib.unet import unet_resnet
import segmentation_models_pytorch as smp
# for mac os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_dir = 'result/checkpoint'


def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))



pth_epoch = 600 # 加载之前训练的模型(指定轮数)


def test():
    
    # 提取网络结构
    # model = U_Net(3, 11).to(device)
    model = unet_resnet('resnext50_32x4d', 3, 11, False).to(device)
    # model = unet_resnet('resnet18', 3, 11, False).to(device)
    # model = unet_resnet('resnet101', 3, 11, False).to(device)
    # model = smp.unet(encoder_name='resnet34', in_channels=3, classes=11)  # segmentation_models_pytorch库可以很方便的调用各种语义分割模型架构，可以github主页看具体支持哪些结构
    # model = smp.unet(encoder_name='efficientnet-b4', in_channels=3, classes=11)
    # model = smp.unet(encoder_name='se_resnet50', in_channels=3, classes=11)

    # 加载训练好的权重
    pre=torch.load(os.path.join('./result/checkpoint',str(pth_epoch)+'.pth'),map_location=torch.device('cpu'))
    model.load_state_dict(pre)
    model.eval()

    # 一张图片一张图片的预测
    fnames = glob.glob('data/image/*.png')
    for fname in fnames:
        print(fname)

        # 加载图片并进行预处理
        src = cv2.imread(fname)    # 加载图片
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)/255.0   # 归一化

        img = img2tensor((src - src.mean()) / src.std())
        img = img.unsqueeze(0)
        img = img.to(device=device, non_blocking=True)
        output = model(img)[0].detach().cpu().numpy()

        # 每个像素点有一个1x11的向量，代表六个类别概率，看哪个类别概率最大（即softmax的作用）
        output_argmax = np.argmax(output, axis=0)

        # 0-11分别代表11个类别，给他们赋予某种颜色，以实现可视化
        img = np.zeros((output.shape[1],output.shape[2],3),np.uint8)
        for i in range(11):
            img[np.where(output_argmax == i)] = [(i*20)%255, (i*30)%255, (i*40)%255]

        cv2.imwrite('result/test/' + os.path.basename(fname), img)


if __name__ == "__main__":
    test()
import torch,cv2,os
from torch.utils.data import Dataset, DataLoader
import numpy as np

TRAIN = 'data/image'
MASK = 'data/label'


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2: img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class MyDataset(Dataset):
    def __init__(self, tfms=None, n_class=11, train=True):
        # 所有图片
        fnames = [fname for fname in os.listdir(TRAIN)]
        # 划分训练集和测试集
        split_index = int(len(fnames)*0.8)
        if train:
            self.fnames = fnames[:split_index]
        else:
            self.fnames = fnames[split_index:]

        self.tfms = tfms
        self.n_class = n_class

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        # 加载图片和label
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(TRAIN, fname)), cv2.COLOR_BGR2RGB)
        mask_origin = cv2.cvtColor(cv2.imread(os.path.join(MASK, fname)), cv2.COLOR_BGR2GRAY)

        # 将label转化为shape为11x512x512的onehot矩阵
        mask = np.zeros((mask_origin.shape[0],mask_origin.shape[1],self.n_class), dtype=np.float)
        for i in range(self.n_class):
            mask[:,:,i] = mask_origin==i

        # 数据增强（目前没加）
        if self.tfms is not None:
            augmented = self.tfms(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        # 图像标准化
        img = img / 255.0
        return img2tensor((img - img.mean()) / img.std()), img2tensor(mask)

if __name__=='__main__':
    data = MyDataset()
    print(len(data))
    for i in range(10):
        img,label = data[i]
        print(img.shape,label.shape)

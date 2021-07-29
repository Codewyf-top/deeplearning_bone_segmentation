1. 新数据集的处理

1.1 运行 get_color.py（记得修改图片路径）, 鼠标点击某个像素会打印出该像素的RGB值。

1.2 在 preprocess.py 中，修改上一步记录下来的各类别的RGB值（背景对应[0, 0, 0]，第一类对应[46, 46, 46]，以此类推

然后运行，加载data/originlabel中的标签图像，生成新的label（以png形式存储在data/label中）

这一步是将原本的label图像中，每一个像素的RGB值映射为[0, 1, 2, 3, 4, ..., 10]中的某一个值，以方便dataset里加载label并转化为11x256x256的one-hot形式标签。

1.3 运行 train.py


2. 训练过程可视化： tensorboard --logdir=result/log
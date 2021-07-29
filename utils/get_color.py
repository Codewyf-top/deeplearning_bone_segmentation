import cv2,os


def capture_event(event,x,y,flags,params):

    if event==cv2.EVENT_LBUTTONDOWN:
        print('color:',image[y,x,:])


if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>>>>>>>  鼠标点击选取初始点  <<<<<<<<<<<<<<<<<<<<<<<<<<
    index = 0   # 从第几张图像上选取初始点
    image=cv2.imread('../data/originlabel/1.png') # 加载对应图片
    print(image.shape)
    window = 'choose seed'
    cv2.namedWindow(window)
    cv2.setMouseCallback(window,capture_event)
    while True:
        cv2.imshow(window,image)
        key = cv2.waitKey(1)
        if key==13 or key==27:
            break
    cv2.destroyAllWindows()
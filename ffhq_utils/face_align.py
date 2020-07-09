import torch
import numpy as np
import skimage.io as io
# from FaceSDK.face_sdk import FaceDetection
from face_sdk import FaceDetection
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from PIL import Image
import torch.nn.functional as F
import torchvision as tv
import torchvision.utils as vutils
import time
import cv2
import os
from skimage import img_as_ubyte
import json
import argparse

def _standard_face_pts():
    pts=np.array([
        196.0,226.0,
        316.0,226.0,
        256.0,286.0,
        220.0,360.4,
        292.0,360.4],np.float32) /256.0-1.0

    return np.reshape(pts,(5,2))

def _origin_face_pts():
    pts=np.array([
        196.0,226.0,
        316.0,226.0,
        256.0,286.0,
        220.0,360.4,
        292.0,360.4],np.float32)

    return np.reshape(pts,(5,2))


def compute_transformation_matrix(img,landmark,normalize,target_face_scale=1.0):

    std_pts=_standard_face_pts() #[-1,1]
    target_pts = (std_pts*target_face_scale +1)/2*512.


    # print(target_pts)

    h,w,c=img.shape
    if normalize == True:
        landmark[:,0]=landmark[:,0]/h*2-1.0
        landmark[:,1]=landmark[:,1]/w*2-1.0

    # print(landmark)

    affine=SimilarityTransform()

    affine.estimate(target_pts,landmark)

    return affine.params


def show_detection(image,box,landmark):
    plt.imshow(image)
    print(box[2]-box[0])
    plt.gca().add_patch(Rectangle((box[1], box[0]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none'))
    plt.scatter(landmark[0][0], landmark[0][1])
    plt.scatter(landmark[1][0], landmark[1][1])
    plt.scatter(landmark[2][0], landmark[2][1])
    plt.scatter(landmark[3][0], landmark[3][1])
    plt.scatter(landmark[4][0], landmark[4][1])
    plt.show()



def affine2theta(affine,input_w,input_h,target_w,target_h):
    # param = np.linalg.inv(affine)
    param=affine
    theta = np.zeros([2, 3])
    theta[0, 0] = param[0, 0]*input_h/target_h
    theta[0, 1] = param[0, 1]*input_w/target_h
    theta[0, 2] = (2*param[0,2]+param[0,0]*input_h+param[0,1]*input_w)/target_h-1
    theta[1, 0] = param[1, 0]*input_h/target_w
    theta[1, 1] = param[1, 1]*input_w/target_w
    theta[1, 2] = (2*param[1,2]+param[1,0]*input_h+param[1,1]*input_w)/target_w-1
    return theta

def main(sta, end):

    ### If the origin url is None, then we don't need to reid the origin image
    # origin_url="/home/jingliao/ziyuwan/reid_origin_bill_image_restored"
    # origin_url="/home/jingliao/ziyuwan/reid_origin_celebrities_image_restored"

    # if not os.path.exists(save_url):
    #     os.makedirs(save_url)

    # if not os.path.exists(origin_url):
    #     os.makedirs(origin_url)

    # img=cv2.imread("/home/d/ziyuwan/dataset/CelebAMask-HQ/test_imgs/9922.jpg")
    # cv2.imwrite('opencv.png',img)
    #
    # dst=cv2.resize(img,(256,256),interpolation=cv2.INTER_LANCZOS4)
    # cv2.imwrite('opencv_resize.png',dst)

    # detect=FaceDetection()

    detect = FaceDetection(algorithm='FaceDetectionSLN', model_name='Detection.011')
    # image=io.imread("/home/d/ziyuwan/dataset/CelebAMask-HQ/test_imgs/9922.jpg")

    count = 0

    map_id = {}
    base_path = '/data/home/v-had/github/ffhq-dataset/FFHQ'
    img_list = sorted(os.listdir('/data/home/v-had/github/ffhq-dataset/FFHQ'))
    thread_list = img_list[sta:end]
    for x in thread_list:
        if len(x) is not 5:
            continue
        img_url = os.path.join(base_path, x, '!input.jpg')
        pil_img = Image.open(img_url).convert('RGB')

        # pil_img = Image.open("/home/d/ziyuwan/dataset/CelebAMask-HQ/test_imgs/9922.jpg").convert("RGB")

        # pil_img=pil_img.resize((256,256),resample=Image.LANCZOS)
        # pil_img.save('pil_resize.png')

        image = np.array(pil_img)
        image = image[100:-100][:][:]
        # print('skimage')
        # print(image)

        start = time.time()
        det = detect.detect_and_align(image, 1)
        done = time.time()
        #        print(f'time cost: {done - start} seconds')

        # print(len(det[1]))

        # show_detection(image,det[0][0],det[1][0])

        if len(det[1]) == 0:
            print("Warning: There is no face in %s" % (x))
            continue

        affine = compute_transformation_matrix(image, det[1][0], False, target_face_scale=1.32)

        # F.affine_grid(affine)

        aligned_face = warp(image, affine, output_shape=(512, 512, 3))

        # print(aligned_face.dtype)
        # print(np.min(aligned_face))
        # break

        #
        # io.imshow(aligned_face)
        # io.show()

        # print(aligned_face[150,150])

        img_name = x

        io.imsave(os.path.join('/data/home/v-had/github/SfSNet_pytorch_modified/data/ffhq_1_32', img_name + '.png'),
                  img_as_ubyte(aligned_face))

        count += 1

        if count % 1000 == 0:
            print('%d have finished ...' % (count))

if __name__=='__main__':
    import threading
    import time

    exitFlag = 0


    class myThread(threading.Thread):
        def __init__(self, sta, end):
            threading.Thread.__init__(self)
            self.sta = sta
            self.end = end

        def run(self):
            print("start thread：" + self.name)
            main(self.sta, self.end)
            print("end thread：" + self.name)

    try:
        thr1 = myThread(50000, 60000)
        thr2 = myThread(60000, 70000)
#        thr4 = myThread(30000, 40000)
#        thr5 = myThread(40000, 50000)
#        thr6 = myThread(50000, 60000)
#        thr7 = myThread(60000, 70000)
        thr1.start()
        thr2.start()
#        thr4.start()
#        thr5.start()
#        thr6.start()
#        thr7.start()
        thr1.join()
        thr2.join()
#        thr4.join()
#        thr5.join()
#        thr6.join()
#        thr7.join()
        # splits = 70000//8
        # cur = 0
        # thr_list = []
        # while cur < 69999:
         #    end = cur + splits if cur+splits < 70000 else 69999
          #   thr = myThread(cur, end)
          #   thr.start()
          #   thr_list.append(thr)
          #   print(f"start thread from {cur} to {end}")
          #   cur = end
        # for thr in thr_list:
         #   thr.join()

    except:
        print("error")



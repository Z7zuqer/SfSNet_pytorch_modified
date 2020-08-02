import os
import cv2

def resize_imgs(path):
    imgs = os.listdir(path)
    for idx in imgs:
        if not idx.endswith('png'):
            continue
        img_path = os.path.join(path, idx)
        img = cv2.imread(img_path)
        print(img.shape)

def main():
    base_path = '/data/home/v-had/github/SfSNet_pytorch_modified/data/sfs-net/'

    train_path = os.path.join(base_path, 'train', '0003')
    test_path = os.path.join(base_path, 'test', '0603')

    resize_imgs(train_path)
    resize_imgs(test_path)

if __name__ == "__main__":
    main()
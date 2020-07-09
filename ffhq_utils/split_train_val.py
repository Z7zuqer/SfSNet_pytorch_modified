import os
import numpy as np

def main():
    base_path = '/data/home/v-had/github/SfSNet_pytorch_modified/data/ffhq_1_32'
    new_dir_path = '/data/home/v-had/github/SfSNet_pytorch_modified/data/ffhq_pipeline_test'
    train_ratio = 5
    test_ratio = 2

    imgs = os.listdir(base_path)
    total_num = len(imgs)

    train_num = int((train_ratio / (train_ratio+test_ratio)) * total_num)

    img_idx = np.arange(0, total_num)
    np.random.shuffle(img_idx)
    train_idx = img_idx[:train_num]
    test_idx = img_idx[train_num:]

    new_train_dir = os.path.join(new_dir_path, 'train')
    if not os.path.exists(new_train_dir):
        os.mkdir(new_train_dir)
    new_test_dir = os.path.join(new_dir_path, 'test')
    if not os.path.exists(new_test_dir):
        os.mkdir(new_test_dir)

    count = 0
    for idx in train_idx:
        # print(os.path.join(new_dir_path, 'train', "{:04d}.png".format(count)))
        os.system('cp {} {}'.format(os.path.join(base_path, imgs[idx]),
                                    os.path.join(new_dir_path, 'train', "{:06d}.png".format(count))))
        count += 1

    count = 0
    for idx in test_idx:
        # print(os.path.join(new_dir_path, 'train', "{:04d}.png".format(count)))
        os.system('cp {} {}'.format(os.path.join(base_path, imgs[idx]),
                                    os.path.join(new_dir_path, 'test', "{:06d}.png".format(count))))
        count += 1


if __name__ == '__main__':
    main()
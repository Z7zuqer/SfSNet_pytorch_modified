import os
import csv

def main():
    base_path = '/data/home/v-had/github/SfSNet_pytorch_modified/data/full_syn'

    train_csv_path = os.path.join(base_path, 'train.csv')
    test_csv_path = os.path.join(base_path, 'test.csv')

    classes = os.listdir(os.path.join(base_path, 'train'))
    imgs_list = []
    for item in classes:
        class_path = os.path.join(base_path, 'train', item)
        imgs = os.listdir(os.path.join(base_path, 'train', item))
#        imgs_list = []
        for img in imgs:
            if 'albedo' in img:
                imgs_list.append(os.path.join(item, img))

    imgs_list = sorted(imgs_list)
    lenimg = int(len(imgs_list) * (5.0 / 7.0))
    train_list = imgs_list[:lenimg]
    test_list = imgs_list[lenimg:]

    with open(train_csv_path, "w") as csv_wri_f:
        csv_wri = csv.writer(csv_wri_f)
        csv_wri.writerow(["", "albedo", "normal", "depth", "mask", "face", "light"])
        # img_list = sorted(os.listdir(os.path.join(base_path, 'train')))
        for idx, item in enumerate(train_list):
            csv_wri.writerow([idx, item, item.replace("albedo", "normal"), item.replace("albedo", "depth"), item.replace("albedo", "mask"), item.replace("albedo", "face"), item.replace("albedo", "light").replace("png", "txt")])
    
    with open(test_csv_path, "w") as csv_wri_f:
        csv_wri = csv.writer(csv_wri_f)
        csv_wri.writerow(["", "albedo", "normal", "depth", "mask", "face", "light"])
       #  csv_wri.writerow(["", "face"])
        # img_list = sorted(os.listdir(os.path.join(base_path, 'test')))
        for idx, item in enumerate(test_list):
            csv_wri.writerow([idx, item, item.replace("albedo", "normal"), item.replace("albedo", "depth"), item.replace("albedo", "mask"), item.replace("albedo", "face"), item.replace("albedo", "light").replace("png", "txt")])

            # csv_wri.writerow([idx, os.path.join("../data/{}/test/{}").format(base_path.split("/")[-1], item)])

if __name__ == "__main__":
    main()

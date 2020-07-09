import os
import csv

def main():
    base_path = '/data/home/v-had/github/SfSNet_pytorch_modified/data/ffhq_pipeline_test'

    train_csv_path = os.path.join(base_path, 'train.csv')
    test_csv_path = os.path.join(base_path, 'test.csv')

    with open(train_csv_path, "w") as csv_wri_f:
        csv_wri = csv.writer(csv_wri_f)
        csv_wri.writerow(["", "face"])
        img_list = sorted(os.listdir(os.path.join(base_path, 'train')))
        for idx, item in enumerate(img_list):
            csv_wri.writerow([idx, os.path.join("../data/{}/train/{}").format(base_path.split("/")[-1], item)])

    with open(test_csv_path, "w") as csv_wri_f:
        csv_wri = csv.writer(csv_wri_f)
        csv_wri.writerow(["", "face"])
        img_list = sorted(os.listdir(os.path.join(base_path, 'test')))
        for idx, item in enumerate(img_list):
            csv_wri.writerow([idx, os.path.join("../data/{}/test/{}").format(base_path.split("/")[-1], item)])

if __name__ == "__main__":
    main()
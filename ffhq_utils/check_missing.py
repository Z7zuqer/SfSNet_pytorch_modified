import os

def main():
    base_path = '/data/home/v-had/github/SfSNet_pytorch_modified/data/ffhq_1_32'
    count = 0
    for i in sorted(os.listdir(base_path)):
        img_idx = int(i.split(".")[0])
        print(img_idx, count)
        if img_idx != count:
            break
        count += 1

if __name__ == "__main__":
    main()
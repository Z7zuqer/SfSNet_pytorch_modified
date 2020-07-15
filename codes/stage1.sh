CUDA_VISIBLE_DEVICES=6,7 python stage1.py --epochs 10000 --lr 0.02 --batch_size 4 --read_first 10000 --log_dir ./results/skip_net/exp4/ --details 'Skipnet with normals'

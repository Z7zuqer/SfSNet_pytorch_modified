CUDA_VISIBLE_DEVICES=6,7 python stage1.py --epochs 100 --lr 0.0002 --batch_size 8 --read_first 10000 --log_dir ./results/skip_net/exp4/ --details 'Skipnet with normals'

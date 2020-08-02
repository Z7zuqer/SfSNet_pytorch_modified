CUDA_VISIBLE_DEVICES=2,3 python stage1.py --epochs 100 --lr 0.02 --batch_size 128 --read_first 20000  --log_dir ./results/skip_net/exp5/ --details 'Skipnet with normals'

#!/bin/bash
#default 1000
category_num=1000
#default 10
instance_num=10

#python3 fractal_renderer/ifs_search.py --rate=0.1 --category=${category_num} --numof_point=100000  --save_dir='./data'

#変換関数の数の違いで比較
mkdir ./data/function
python3 fractal_renderer/ifs_search.py --rate=0.1 --category=${category_num} --numof_point=100000 --func_num=1   --save_dir='./data/function/f1'
python3 fractal_renderer/ifs_search.py --rate=0.1 --category=${category_num} --numof_point=100000 --func_num=4   --save_dir='./data/function/f4'
python3 fractal_renderer/ifs_search.py --rate=0.1 --category=${category_num} --numof_point=100000 --func_num=8   --save_dir='./data/function/f8'
python3 fractal_renderer/ifs_search.py --rate=0.1 --category=${category_num} --numof_point=100000 --func_num=-1  --save_dir='./data/function/fall'
mkdir ./data/function/DB
python3 ./fractal_renderer/make_fractaldb.py --draw_type="gaussian_filter" --save_root="./data/function/DB/FractalDB_funcnum1" --load_root="./data/function/f1/csv"   --instance=${instance_num}
python3 ./fractal_renderer/make_fractaldb.py --draw_type="gaussian_filter" --save_root="./data/function/DB/FractalDB_funcnum4" --load_root="./data/function/f4/csv"   --instance=${instance_num}
python3 ./fractal_renderer/make_fractaldb.py --draw_type="gaussian_filter" --save_root="./data/function/DB/FractalDB_funcnum8" --load_root="./data/function/f8/csv"   --instance=${instance_num}
python3 ./fractal_renderer/make_fractaldb.py --draw_type="gaussian_filter" --save_root="./data/function/DB/FractalDB_allfunc"  --load_root="./data/function/fall/csv" --instance=${instance_num}

#filterの違いで比較
#mkdir ./data/filter
#python3 ./fractal_renderer/make_fractaldb.py --draw_type="gaussian_filter" --save_root="./data/filter/FractalDB+gaussian" --instance=${instance_num}
#python3 ./fractal_renderer/make_fractaldb.py --draw_type="gaussian_filter+noise" --save_root="./data/filter/FractalDB+gaussian_noise" --instance=${instance_num}
#python3 ./fractal_renderer/make_fractaldb.py --draw_type="uniform_filter" --save_root="./data/filter/FractalDB+uniform" --instance=${instance_num}
#python3 ./fractal_renderer/make_fractaldb.py --draw_type="random_type" --save_root="./data/filter/FractalDB+random_type" --instance=${instance_num}
#python3 ./fractal_renderer/make_fractaldb.py --draw_type="point_gray" --save_root="./data/filter/FractalDB+normal" --instance=${instance_num}

#データのサイズで比較
#mkdir ./data/size
#python3 ./fractal_renderer/make_fractaldb.py --draw_type="gaussian_filter" --save_root="./data/size/FractalDB+224" --instance=${instance_num} --image_size_x=224 --image_size_y=224
#python3 ./fractal_renderer/make_fractaldb.py --draw_type="gaussian_filter" --save_root="./data/size/FractalDB+512" --instance=${instance_num} --image_size_x=512 --image_size_y=512

mkdir ./data/function/weight
python3 ./train/pretrain.py --db_path="./data/function/DB" --weight_path="./data/function/weight"

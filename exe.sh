category_num=1
instance_num=1

python3 fractal_renderer/ifs_search.py --rate=0.1 --category=${category_num} --numof_point=100000  --save_dir='./data'

mkdir ./data/filter
python3 ./fractal_renderer/make_fractaldb.py --draw_type="point_gray" --save_root="./data/filter/FractalDB+normal" --instance=${instance_num}
python3 ./fractal_renderer/make_fractaldb.py --draw_type="gaussian_filter" --save_root="./data/filter/FractalDB+gaussian" --instance=${instance_num}
python3 ./fractal_renderer/make_fractaldb.py --draw_type="uniform_filter" --save_root="./data/filter/FractalDB+uniform" --instance=${instance_num}
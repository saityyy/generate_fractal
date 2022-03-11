category_num=1

python3 fractal_renderer/ifs_search.py --rate=0.1 --category=${category_num} --numof_point=100000  --save_dir='./data'

python3 ./fractal_renderer/make_fractaldb.py --draw_type="point_gray" --save_root="./data/FractalDB+normal" --instance=1
python3 ./fractal_renderer/make_fractaldb.py --draw_type="gaussian_filter" --save_root="./data/FractalDB+gaussian" --instance=1
python3 ./fractal_renderer/make_fractaldb.py --draw_type="uniform_filter" --save_root="./data/FractalDB+uniform" --instance=1
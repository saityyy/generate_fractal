# generate_fractal

- フラクタル図形の自動生成プログラム。  
- 様々なバリエーションでの非線形変換の実装  
- フラクタル図形の詳細や本データセットでの取り組みについてはinfo.ipynbに記載
- 画像サイズ、保存先パス、生成する画像枚数などのパラメータはmain.pyで変更可能

# 動作環境
- windows10  
- python 3.8.9


# 実行
- requirements.txtに記載されたライブラリのインストール
- パラメータとサンプル画像を生成する。
```
python fractal_renderer/ifs_search.py --rate=0.1 --category=1000 --numof_point=100000  --save_dir='./data'
```
- データセット作成
```
python .\fractal_renderer\make_fractaldb.py --draw_type="point_gray_filter" --save_root="./data/FractalDB+"
```





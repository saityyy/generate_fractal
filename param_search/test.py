import os
import numpy as np
import matplotlib.pyplot as plt
from ifs_search import generator
from functions import func_collection


for file in os.scandir("./sample/csv"):
    params = np.genfromtxt(file.path, dtype=np.str, delimiter=',')
    print(file.name.split(".")[0][-1])
    func = func_collection[int(file.name.split(".")[0][-1])]
    print(func)
    fractal_img = generator(params, func)
    plt.imshow(fractal_img)
    plt.show()

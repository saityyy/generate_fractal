# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%


def show_rate(csv_path):
    df = pd.read_csv(csv_path)
    a = np.array(df.iloc[:, 0])
    print(csv_path)
    plt.plot(np.sort(a))
    plt.show()


# %%
for file in os.scandir("./pixels_csv"):
    show_rate(file.path)

# %%

import os
import numpy as np
from tqdm.auto import tqdm

root = './data'
for cate in os.listdir(root):
    if cate[:5] != 'input':
        continue
    for pc_file in tqdm(os.listdir(os.path.join(root, cate))):
        if pc_file[-3:] != 'xyz':
            continue
        path = os.path.join(root, cate, pc_file)
        pc = np.loadtxt(path)
        np.savetxt(path, pc, fmt='%.6f')

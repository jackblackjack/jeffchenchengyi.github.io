import os
import re
from tqdm import tqdm

replacement = """"""

print('Cleaning Latex in .ipynb files in ./portfolio')
for dname, dirs, files in tqdm(os.walk('./portfolio')):
    for fname in files:
        fpath = os.path.join(dname, fname)
        if fpath[-5:] == 'ipynb':
            with open(fpath) as f:
                s = f.read()
            s = s.replace("{{", "{ {")
            s = s.replace("}}", "} }")
            with open(fpath, "w") as f:
                f.write(s)

print('Cleaning Latex in .ipynb files in ./machine-learning')
for dname, dirs, files in tqdm(os.walk('./machine-learning')):
    for fname in files:
        fpath = os.path.join(dname, fname)
        if fpath[-5:] == 'ipynb':
            with open(fpath) as f:
                s = f.read()
            s = s.replace("{{", "{ {")
            s = s.replace("}}", "} }")
            with open(fpath, "w") as f:
                f.write(s)
                
print('Latex Cleaned!')
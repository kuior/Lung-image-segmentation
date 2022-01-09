import os
import numpy as np
import PIL
from PIL import Image
from sklearn.metrics import accuracy_score
from time import time
from tqdm import tqdm

pre_path = './data/membrane/test1_pre_WBCE/'
gt_path = './data/membrane/test1_gt/'
a = []
target_size = (256,256)

def dice(gt, pre):
    ints = np.sum(((gt == 1) * 1) * ((pre == 1) * 1))
    sums = np.sum(((gt == 1) * 1) + ((pre == 1) * 1)) + 0.0001
    dice = 2.0 * ints / sums
    return dice

t1 = time()

for i in tqdm(range(56806)):
    img1 = PIL.Image.open(os.path.join(pre_path,"%d_predict.png"%i)).convert("L")
    pre = np.array(img1)
    pre = np.ndarray.flatten(pre)
    img2 = PIL.Image.open(os.path.join(gt_path,"%d.png"%i)).convert("L")
    img2 = img2.resize(target_size, resample=0, box=None)
    gt = np.array(img2)
    gt = np.ndarray.flatten(gt)
    for j in range(len(pre)):
        if pre[j]>0:
            pre[j]=1
        if gt[j]>0:
            gt[j]=1
#    if max(gt) != 0:
    a.append(dice(gt,pre))
#    a.append(accuracy_score(gt,pre))

mean_dice = np.mean(a)
print(mean_dice)

t2 = time()
time = t2 -t1
print(time/60)

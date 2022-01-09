import os
import random, shutil

fileDir1 = "./data/membrane/train/image1/"
fileDir2 = "./data/membrane/train/label1/"

image1 = "./data/membrane/train/image1_1/"
label1 = "./data/membrane/train/label1_1/"

val_img1 = "./data/membrane/train/val_img1_1/"
val_label1 = "./data/membrane/train/val_label1_1/"


image2 = "./data/membrane/train/image2_1/"
label2 = "./data/membrane/train/label2_1/"


image3 = "./data/membrane/train/image3_1/"
label3 = "./data/membrane/train/label3_1/"


image4 = "./data/membrane/train/image4_1/"
label4 = "./data/membrane/train/label4_1/"


def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    return all_files


def cross_validate(data_files, folds):

    fold_size = len(data_files) // folds
    for split_index in range(0, (len(data_files)-fold_size), fold_size):
        val = data_files[split_index:split_index + fold_size]
        training = data_files[:split_index] + data_files[split_index + fold_size:]
        yield training, val

datadir = fileDir1
data_files = get_file_list_from_dir(datadir)


i = 1
for train_set, val_set in cross_validate(data_files, 4):

    for name in train_set:
        if (i == 1):
            shutil.copyfile(fileDir1+name, image1+name)
            shutil.copyfile(fileDir2+name, label1+name)

    for name in val_set:
    	if (i == 1):
            shutil.copyfile(fileDir1+name, val_img1+name)
            shutil.copyfile(fileDir2+name, val_label1+name)
    i = i + 1

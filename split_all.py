import os
import random, shutil

fileDir1 = "./data/membrane/train/image/"
fileDir2 = "./data/membrane/train/label/"

image1 = "./data/membrane/train/image1/"
label1 = "./data/membrane/train/label1/"
test1 = "./data/membrane/test1/"
test1_label = "./data/membrane/test1_label/"

image2 = "./data/membrane/train/image2/"
label2 = "./data/membrane/train/label2/"
test2 = "./data/membrane/test2/"
test2_label = "./data/membrane/test2_label/"

image3 = "./data/membrane/train/image3/"
label3 = "./data/membrane/train/label3/"
test3 = "./data/membrane/test3/"
test3_label = "./data/membrane/test3_label/"

image4 = "./data/membrane/train/image4/"
label4 = "./data/membrane/train/label4/"
test4 = "./data/membrane/test4/"
test4_label = "./data/membrane/test4_label/"

def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    return all_files


def cross_validate(data_files, folds):

    fold_size = len(data_files) // folds
    for split_index in range(0, (len(data_files)-fold_size), fold_size):
        testing = data_files[split_index:split_index + fold_size]
        training = data_files[:split_index] + data_files[split_index + fold_size:]
        yield training, testing

datadir = fileDir1
#def ml_function(datadir, num_folds):
data_files = get_file_list_from_dir(datadir)

k1 = []
k2 = []
k3 = []
k4 = [] 
i = 1
for train_set, test_set in cross_validate(data_files, 4):
        #print(len(train_set))
        #print(len(test_set))

    for name in train_set:
        if (i == 1):
            shutil.copyfile(fileDir1+name, image1+name)
            shutil.copyfile(fileDir2+name, label1+name)
        if (i == 2):
            shutil.copyfile(fileDir1+name, image2+name)
            shutil.copyfile(fileDir2+name, label2+name)
        if (i == 3):
            shutil.copyfile(fileDir1+name, image3+name)
            shutil.copyfile(fileDir2+name, label3+name)   

        if (i == 4):
            shutil.copyfile(fileDir1+name, image4+name)
            shutil.copyfile(fileDir2+name, label4+name)
 
    for name in test_set:
    	if (i == 1):
            shutil.copyfile(fileDir1+name, test1+name)
            shutil.copyfile(fileDir2+name, test1_label+name)
            a = name.replace('.png', '')
            a = int(a)
            k1.append(a)
    	if (i == 2):
            shutil.copyfile(fileDir1+name, test2+name)
            shutil.copyfile(fileDir2+name, test2_label+name)
            a = name.replace('.png', '')
            a = int(a)
            k2.append(a)
    	if (i == 3):
            shutil.copyfile(fileDir1+name, test3+name)
            shutil.copyfile(fileDir2+name, test3_label+name)
            a = name.replace('.png', '')
            a = int(a)
            k3.append(a)
    	if (i == 4):
            shutil.copyfile(fileDir1+name, test4+name)
            shutil.copyfile(fileDir2+name, test4_label+name)
            a = name.replace('.png', '')
            a = int(a)
            k4.append(a)
    i = i + 1

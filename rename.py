import os
def rename(): 
    path = "./data/membrane/train/label1_copy/"
    path_new = "./data/membrane/train/label1_new/"
    filelist = os.listdir(path)
    count = 0

    for file in filelist: 
        if os.path.isdir(Olddir):
            continue
        filename = os.path.splitext(file)[0]  
        filetype = ".png"
        Newdir = os.path.join(path_new,str(count)+filetype)
        os.rename(Olddir,Newdir) 
        count += 1
rename()

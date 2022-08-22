import numpy as np
import os
import shutil

train_path = './train'
train_out = './train.txt'
val_path = './valid'
val_out = './val.txt'

data_train_out = './train_filelist'
data_val_out = './val_filelist'

def get_filelist(input_path,output_path):
    with open(output_path,'w') as f:
        for dir_path,dir_names,file_names in os.walk(input_path):
            if dir_path != input_path:
                label = int(dir_path.split('\\')[-1]) -1
            #print(label)
            for filename in file_names:
                f.write(filename +' '+str(label)+"\n")

def move_imgs(input_path,output_path):
    for dir_path, dir_names, file_names in os.walk(input_path):
        for filename in file_names:
            #print(os.path.join(dir_path,filename))
            source_path = os.path.join(dir_path,filename)

            shutil.copyfile(source_path, os.path.join(output_path,filename))

get_filelist(train_path,train_out)
get_filelist(val_path,val_out)
move_imgs(train_path,data_train_out)
move_imgs(val_path,data_val_out)
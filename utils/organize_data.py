import os
from shutil import copyfile
download_path = './datasets/market1501/Market-1501-v15.09.15'
save_path = download_path + '/pytorch'

if not os.path.isdir(save_path):
    os.mkdir(save_path)

train_path = download_path + '/bounding_box_train'
train_save_path = download_path + '/pytorch/train'
val_save_path = download_path + '/pytorch/val'

if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            dst_path = val_save_path + '/' + ID[0]
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
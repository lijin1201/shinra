import os,glob
import pandas as pd
from torch.utils.data import random_split
#from pathlib import Path

root_dir = '/data1/lidc-idri/slices' 
save_dir = '/data2/lijin/lidc-prep/kjs/splits'

pattern = os.path.join(root_dir, '**', '*.npy')
img_name_list = glob.glob(pattern, recursive=True)

train_dataset, test_dataset = random_split([os.path.relpath(path,root_dir) for path in img_name_list], [0.7, 0.3])

train_label_score = [ int(os.path.splitext(os.path.basename(tdata))[0].split("_")[-1]) for tdata in train_dataset]
test_label_score = [ int(os.path.splitext(os.path.basename(tdata))[0].split("_")[-1]) for tdata in test_dataset]

train_label_malB = [ 1 if score>3 else 0 for score in train_label_score]
test_label_malB = [ 1 if score>3 else 0 for score in test_label_score]

train_label_malA = [ 1 if score>3 else 0 if score<3 else "amb" for score in train_label_score]
test_label_malA = [ 1 if score>3 else 0 if score<3 else "amb" for score in test_label_score]

#store in csv files
train_malB = pd.DataFrame( list(zip(train_dataset,train_label_malB)), columns=["filename","label"] ) # label_score
train_malB.to_csv(save_dir + "/train_malB.csv",index=False)
test_malB = pd.DataFrame( list(zip(test_dataset,test_label_malB)), columns=["filename","label"] ) # label_score
test_malB.to_csv(save_dir + "/test_malB.csv",index=False)

train_malA = pd.DataFrame( list(zip(train_dataset,train_label_malA)), columns=["filename","label"] ) # label_score
train_malA.to_csv(save_dir + "/train_malA.csv",index=False)
test_malA = pd.DataFrame( list(zip(test_dataset,test_label_malA)), columns=["filename","label"] ) # label_score
test_malA.to_csv(save_dir + "/test_malA.csv",index=False)

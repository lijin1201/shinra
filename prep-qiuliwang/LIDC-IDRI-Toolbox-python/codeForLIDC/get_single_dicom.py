'''
Created by Wang Qiu Li
7/5/2018

get dicom info according to malignancy.csv and ld_scan.txt
get one dicom
'''

import csvTools
import os
import pandas as pd
import pydicom
import scipy.misc
import cv2
import numpy as np

basedir = 'D:/Data/LIDC-IDRI/DOI/'
resdir = 'D:/Data/LIDC-IDRI/NNPY/'
resdir2 = 'D:/Data/LIDC-IDRI/NNNPY/'

noduleinfo = csvTools.readCSV('files/malignancy.csv')
idscaninfo = csvTools.readCSV('files/id_scan.txt')

print(len(noduleinfo))
print(len(idscaninfo))

# LUNA2016 data prepare ,first step: truncate HU to -1000 to 400
def truncate_hu(image_array):
    image_array[image_array > 400] = 0
    image_array[image_array <-1000] = 0
    return image_array
    
# LUNA2016 data prepare ,second step: normalzation the HU
def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    avg = image_array.mean()
    image_array = image_array-avg
    return image_array   # a bug here, a array must be returned,directly appling function did't work



def cutTheImage(x, y, pix):
    temp = 16
    x1 = x - temp
    x2 = x + temp
    y1 = y - temp
    y2 = y + temp
    img_cut = pix[x1:x2, y1:y2]
    return img_cut

def caseid_to_scanid(caseid):
    returnstr = ''
    if caseid < 10:
        returnstr = '000' + str(caseid)
    elif caseid < 100:
        returnstr = '00' + str(caseid)
    elif caseid < 1000:
        returnstr = '0' + str(caseid)
    else:
        returnstr = str(caseid)
    return 'LIDC-IDRI-' + returnstr

for onenodule in noduleinfo:
    scanid = onenodule[1]
    scanid = caseid_to_scanid(int(scanid))
    print(scanid)
    scanpath = ''
    for idscan in idscaninfo:
        if scanid in idscan[0]:
            scanpath = idscan[0]
            break
        
    filelist1 = os.listdir(basedir + scanpath)
    filelist2 = []
    print(len(filelist1))
    for onefile in filelist1:
        if '.dcm' in onefile:
            filelist2.append(onefile)
        
    # print(len(filelist2))
    slices = [pydicom.dcmread(basedir + scanpath + '/' + s) for s in filelist2]
    slices.sort(key = lambda x : float(x.ImagePositionPatient[2]),reverse=True)
    x_loc = int(onenodule[6])
    y_loc = int(onenodule[7])
    z_loc = int(onenodule[8])
    # print(x_loc, y_loc, z_loc)
    pix = slices[z_loc - 1].pixel_array
    cutpix = cutTheImage(y_loc, x_loc, pix)
    scipy.misc.imsave(resdir + scanid + '.jpeg', cutpix)

from uwimg import *
import os
import shutil
arguments_strIn = './pytorch-hed/crop_image/'
arguments_out='./pytorch-hed/edge_image/'
#dr='./data/doge_test/'
#print(os.listdir(arguments_strIn))

for i in range(len(os.listdir(arguments_strIn))):
    edge = load_image(arguments_out+'testdata_hed_'+str(i+1)+".jpg")
    ori=load_image(arguments_strIn+'testdata_hed_' +str(i+1)+".jpg")
    mag=cc_images(ori,edge)
    save_image(mag, str(i))
    source=str(i)+'.jpg'
    des='pytorch-CycleGAN-and-pix2pix/datasets/Demo_hed/test/'+source
    shutil.move(source, des)

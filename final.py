from uwimg import *
import os
import shutil
#arguments_strIn = './pytorch-hed/crop_image/'
dr='./data/doge_test/'
#print(os.listdir(arguments_strIn))

for i in range(len(os.listdir(dr))):
    im = load_image("data/doge_test/img ("+str(i+1)+").jpg")
    res = colorize_sobel(im)
    mag = res
    save_image(mag, str(i))
    source=str(i)+'.jpg'
    des='pytorch-CycleGAN-and-pix2pix/datasets/mydog/testpre/'+source
    shutil.move(source, des)
#for i in range(1,62):
#    im = load_image("data/anime/pn ("+str(i)+").png")
#    res = colorize_sobel(im)
#    #res = sobel_image(im)
#    mag = res
#    #feature_normalize(mag)
#    save_image(mag, str(i+27))
#im = load_image("data/anime/test.png")
#res = colorize_sobel(im)
#save_image(res, "test")
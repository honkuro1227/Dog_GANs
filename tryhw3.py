from uwimg import *

def draw_corners():
    im = load_image("data/resize1.jpg")
    detect_and_draw_corners(im, 2, 50, 3)
    save_image(im, "corners")

def draw_matches():
    a = load_image("data/field_panorama_0123.jpg")
    b = load_image("data/prtn04.jpg")
    m = find_and_draw_matches(b, a, 2, 50, 3)
    
    save_image(m, "matches")

def easy_panorama():
    im1 = load_image("data/field_panorama_0123.jpg")
    im2 = load_image("data/prtn04.jpg")
    
#    im1=cylindrical_project(im1, 20000)
    im2=cylindrical_project(im2, 20000)
    
    #pan = panorama_image(im1, im2, thresh=500)
    pan = panorama_image(im2, im1, thresh=2, iters=50000, inlier_thresh=3)
    save_image(pan, "field_panorama_01234")

def rainier_panorama():
    im1 = load_image("data/Rainier1.png")
    im2 = load_image("data/Rainier2.png")
    im3 = load_image("data/Rainier3.png")
    im4 = load_image("data/Rainier4.png")
    im5 = load_image("data/Rainier5.png")
    im6 = load_image("data/Rainier6.png")
    pan = panorama_image(im1, im2, thresh=5)
    save_image(pan, "rainier_panorama_1")
    pan2 = panorama_image(pan, im5, thresh=5)
    save_image(pan2, "rainier_panorama_2")
    pan3 = panorama_image(pan2, im6, thresh=5)
    save_image(pan3, "rainier_panorama_3")
    pan4 = panorama_image(pan3, im3, thresh=5)
    save_image(pan4, "rainier_panorama_4")
    pan5 = panorama_image(pan4, im4, thresh=5)
    save_image(pan5, "rainier_panorama_5")
def resize_apple():
    im1 = load_image("data/resize0.jpg")
    im2 = load_image("data/resize1.jpg")
    im3 = load_image("data/resize2.jpg")
    im4 = load_image("data/resize3.jpg")
    im5 = load_image("data/resize4.jpg")
    im6 = load_image("data/resize5.jpg")
    
    pan = panorama_image(im1, im2, thresh=5)
    save_image(pan, "re1")
    pan2 = panorama_image(pan, im3, thresh=5)
    save_image(pan2, "re2")
    pan3 = panorama_image(pan2, im4, thresh=5)
    save_image(pan3, "re3")
    pan4 = panorama_image(pan3, im5, thresh=5)
    save_image(pan4, "re4")
    pan5 = panorama_image(pan4, im6, thresh=5)
    save_image(pan5, "re5")
def Cylinder_resize_apple():
    images=[]
    print("load image")
    for i in range(18):
        if i<10:
            tmp='data/prtn0'+str(i) +'.jpg'
        else:
            tmp='data/prtn'+str(i) +'.jpg'
        images.append(load_image(tmp))
    print("end load",len(images))
    
    for im in range(18):
        images[im]=cylindrical_project(images[im], 12000)
    save_image(images[0], "cylindrical_projection")
    j=0
    pans=[]
    while(j<len(images)):
        print("time"+str(j))
        tmp = panorama_image(images[j],images[j+1], thresh=2, iters=50000, inlier_thresh=4)
#        name='field_panorama_'+str(j)+str(j+1)
#        save_image(tmp, name)
        pans.append(tmp)
        free_image(images[j])
        free_image(images[j+1])
        j+=2
    j2=0
    while(j2<len(pans)):
        print("time2:"+str(j2))
        tmp = panorama_image(pans[j2],pans[j2+1], thresh=2, iters=50000, inlier_thresh=4)
        name='field_panorama_22222_'+str(j2)+str(j2+1)
        save_image(tmp, name)
        free_image(pans[j2])
        free_image(pans[j2+1])
        j2+=2
        
#def Cylinder_resize_apple2():
#    images=[]
#    print("load image")
#    for i in range(18):
#        if i<10:
#            tmp='data/prtn0'+str(i) +'.jpg'
#        else:
#            tmp='data/prtn'+str(i) +'.jpg'
#        images.append(load_image(tmp))
#    print("end load",len(images))
#    
#    for im in range(18):
#        images[im]=cylindrical_project(images[im], 12000)
#    save_image(images[0], "cylindrical_projection")
#    j=0
#    while(j<len(images)):
#        print("time"+str(j))
#        tmp = panorama_image(images[j],images[j+1], thresh=2, iters=50000, inlier_thresh=4)
#        name='field_panorama_'+str(j)+str(j+1)
#        save_image(tmp, name)
#        free_image(images[j])
#        free_image(images[j+1])
#        j+=2
        
    
    
def field_panorama():
    im1 = load_image("data/field1.jpg")
    im2 = load_image("data/field2.jpg")
    im3 = load_image("data/field3.jpg")
    im4 = load_image("data/field4.jpg")
    im5 = load_image("data/field5.jpg")
    im6 = load_image("data/field6.jpg")
    im7 = load_image("data/field7.jpg")
    im8 = load_image("data/field8.jpg")

    im1 = cylindrical_project(im1, 1200)
    im2 = cylindrical_project(im2, 1200)
    im3 = cylindrical_project(im3, 1200)
    im4 = cylindrical_project(im4, 1200)
    im5 = cylindrical_project(im5, 1200)
    im6 = cylindrical_project(im6, 1200)
    im7 = cylindrical_project(im7, 1200)
    im8 = cylindrical_project(im8, 1200)
    save_image(im1, "cylindrical_projection")
    pan = panorama_image(im5, im6, thresh=2, iters=50000, inlier_thresh=3)
    save_image(pan, "field_panorama_1")
    pan2 = panorama_image(pan, im7, thresh=2, iters=50000, inlier_thresh=3)
    save_image(pan2, "field_panorama_2")
    pan3 = panorama_image(pan2, im8, thresh=2, iters=50000, inlier_thresh=3)
    save_image(pan3, "field_panorama_3")
    pan4 = panorama_image(pan3, im4, thresh=2, iters=50000, inlier_thresh=3)
    save_image(pan4, "field_panorama_4")
    pan5 = panorama_image(pan4, im3, thresh=2, iters=50000, inlier_thresh=3)
    save_image(pan5, "field_panorama_5")

#draw_corners()
#print("draw don")
draw_matches()
#print("2")
#resize_apple()
print("start")
easy_panorama()
#print("finish")
#Cylinder_resize_apple()

#rainier_panorama()
#Cylinder_resize_apple2()

print("finish cy")
import cv2
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter 
import math
import copy

diff = []
n_p = 8
def Theta(x,f):
    return math.atan2(x,f)

def High(x,y,f):
    return y/math.sqrt(x*x+f*f)

def Model(p1,p2):
    #translation
    length = len(p1)
    A = np.zeros((2*length,8))
    for i in range(length):
        A[i*2,0] = p2[i][1]
        A[i*2,1] = p2[i][0]
        A[i*2,2] = 1
        A[i*2,6] = -p2[i][1]*p1[i][1]
        A[i*2,7] = -p2[i][0]*p1[i][1]
        A[i*2+1,3] = p2[i][1]
        A[i*2+1,4] = p2[i][0]
        A[i*2+1,5] = 1
        A[i*2+1,6] = -p2[i][1]*p1[i][0]
        A[i*2+1,7] = -p2[i][0]*p1[i][0]
       
    b = np.zeros((length*2))
    for i in range(length):
        b[i*2] = p1[i][1]
        b[i*2+1] = p1[i][0]
    #print(np.linalg.det(A))
    if np.linalg.det(A) == 0:
        x = np.linalg.lstsq(A,b.T,rcond=None)[0]
    else:
        x = np.linalg.solve(A,b.T)
    return x

def Bilinear(M,b):
    H = np.linalg.inv(M)
    print(M,H)
    hei = b.shape[0]
    wid = b.shape[1]
    c = np.zeros_like(b)
    for i in range(hei):
        for j in range(wid):
            p_m = np.dot(H,np.array([j,i,1]).T)
            p_m/=p_m[2]
            p = [p_m[1],p_m[0]]
            p[1] = wid-p[1]
            if p[0] >= 0 and p[0] < hei and p[1] >= 0 and p[1] < wid:
                sml = np.floor(p)
                lar = np.ceil(p)
                if sml[0] < 0:
                    sml[0] = 0
                if sml[1] < 0:
                    sml[1] = 0
                if lar[0] > hei-1:
                    lar[0] = hei-1
                if lar[1] > wid-1:
                    lar[1] = wid-1
                a1 = lar[1]-sml[1]
                a2 = p[1]-sml[1]
                b1 = lar[0]-sml[0]
                b2 = p[0]-sml[0]
                color = [0,0,0]
                if a1*b1 == 0:
                    if a1 != 0:
                        if sum(b[int(sml[0])][int(sml[1])])!= 0 and sum( b[int(sml[0])][int(lar[1])] ) != 0:
                            color = (a1-a2/a1)*b[int(sml[0])][int(sml[1])]+(a2/a1)*b[int(sml[0])][int(lar[1])]
                        elif sum(b[int(sml[0])][int(sml[1])])!= 0:
                            color = b[int(sml[0])][int(sml[1])]
                        else:
                            color = b[int(sml[0])][int(lar[1])]
                    if b1 != 0:
                        if sum( b[int(sml[0])][int(sml[1])] ) != 0 and sum( b[int(lar[0])][int(sml[1])] ) != 0:
                            color = (b1-b2/b1)*b[int(sml[0])][int(sml[1])]+(b2/b1)*b[int(lar[0])][int(sml[1])]
                        elif sum( b[int(sml[0])][int(sml[1])] ) != 0:
                            color = b[int(sml[0])][int(sml[1])]
                        else:
                            color = b[int(lar[0])][int(sml[1])]
                elif sum(b[int(sml[0])][int(sml[1])])> 0 and sum(b[int(lar[0])][int(sml[1])]) > 0 and sum(b[int(sml[0])][int(lar[1])]) > 0 and sum(b[int(lar[0])][int(lar[1])])> 0:
                    color = ((a1-a2)*(b1-b2)*b[int(sml[0])][int(sml[1])] + a2*(b1-b2)*b[int(lar[0])][int(sml[1])] + b2*(a1-a2)*b[int(sml[0])][int(lar[1])] +a2*b2*b[int(lar[0])][int(lar[1])])/(a1*b1)
                else:
                    if sum(b[int(sml[0])][int(sml[1])])> 10:
                        color = b[int(sml[0])][int(sml[1])]
                    elif sum(b[int(lar[0])][int(sml[1])]) > 10:
                        color = b[int(lar[0])][int(sml[1])]
                    elif sum(b[int(sml[0])][int(lar[1])]) > 10:
                        color = b[int(sml[0])][int(lar[1])]
                    elif sum(b[int(lar[0])][int(lar[1])]) > 10:
                        color = b[int(lar[0])][int(lar[1])]
                c[i][j] = color
    return c
def Drift_M(x_max):
    length = len(diff)
    a = (diff[length-1][0]-diff[0][0])/x_max
    x = np.array([ [ 1,0,0 ],[ a,1,0 ],[ 0,0,1 ] ])
    return x
    
def Vote(x,voters,Err):
    score = 0
    for v in voters:
        A = np.zeros((2,8))
        A[0,0] = v[1][1]
        A[0,1] = v[1][0]
        A[0,2] = 1
        A[0,6] = -v[1][1]*v[0][1]
        A[0,7] = -v[1][0]*v[0][1]
        A[1,3] = v[1][1]
        A[1,4] = v[1][0]
        A[1,5] = 1
        A[1,6] = -v[1][1]*v[0][0]
        A[1,7] = -v[1][0]*v[0][0]
        b = np.zeros(2)
        b_c = np.dot(A,x.T)
        b[0] = v[0][1]
        b[1] = v[0][0]
        error = math.sqrt( (b[0]-b_c[0])**2+(b[1]-b_c[1])**2 )
        if error <= Err:
            score+=1
    return score
def Faith(score,total):
    P = 0.99
    return int(math.log10(1-P)/math.log10(1-(score/total)**4))
def Blend(MatchP = []):
    #Ransac
    K = 0
    c = 5
    n = 3
    d = 100
    Voting = -1
    BestModel = None
    while(K < 300):
        L = []
        for p in range(n):
            L.append(np.random.randint(0,len(MatchP)))
        p1 = []
        p2 = []
        voters = []
        sample = []
        for point in L:
            p1.append([MatchP[point][0][0],MatchP[point][0][1]])
            p2.append([MatchP[point][1][0],MatchP[point][1][1]])
            sample.append(MatchP[point])
        for i in range(len(MatchP)):
            for j in L:
                if i == j:
                    break
            voters.append(MatchP[i])
        x = Model(p1,p2)
        score = Vote(x,voters,c)
        if score > Voting:
            BestModel = x
            Voting = score
            if score > d:
                return x
            
        K+=1
   
    print("MaxVoters = ",Voting)
    result = BestModel
    return result   
def Blend_sift(Match,Fea,idx):
    c = 3
    n = 4
    d = len(Match)/2
    Voting = -1
    BestModel = None
    k_i = 0
    iters = 1000
    while(k_i < iters):
        L = []
        for p in range(n):
            L.append(np.random.randint(0,len(Match)))
        p1 = []
        p2 = []
        voters = []
        for p in L:
            #print(Match[p].queryIdx,Match[p].trainIdx)
            p1.append([Fea[idx-1][Match[p].queryIdx].pt[0],Fea[idx-1][Match[p].queryIdx].pt[1]])
            p2.append([Fea[idx][Match[p].trainIdx].pt[0],Fea[idx][Match[p].trainIdx].pt[1]])
        for i in range(len(Match)):
            for j in L:
                if i == j:
                    break
            voters.append([Fea[idx-1][Match[i].queryIdx].pt,Fea[idx][Match[i].trainIdx].pt])
        x = Model(p1,p2)
        score = Vote(x,voters,c)
        if score > Voting:
            BestModel = x
            Voting = score
            #iters = Faith(score,len(Match))
            #print(iters)
            '''
            if score > d:
                print(k_i," ,score = ",score)
                return x
            '''
            
        k_i+=1
    print(k_i," , MaxVoters = ",Voting)
    result = BestModel
    return result

def Cylin(img,f):
    height = img.shape[0]
    width = img.shape[1]
    twist = np.zeros_like(img)
    xc = width/2
    yc = height/2
    for i in range(height):
        for j in range(width):
           x = j - xc
           y = i - yc
           x1 = s*Theta(x,f)+xc
           y1 = s*High(x,y,f)+yc
           twist[int(y1)][int(x1)] = img[i][j]
    '''
    for i in range(height):
        for j in range(width):
            if twist[i][j][0] == 0 and twist[i][j][1] == 0 and twist[i][j][2] == 0:
                total = 0
                node = []
                for k in range(i-1,i+2):
                    for m in range(j-1,j+2):
                        if k >= 0 and k < height and m >= 0 and m < width:
                            if twist[k][m][0] != 0 or twist[k][m][1]!=0 or twist[k][m][2]!=0:
                                total+=1
                                node.append([k,m])
                t = 0
                for k in range(total):
                    t += twist[node[k][0]][node[k][1]]
                if total != 0:
                    twist[i][j] = t/total
    '''           
    return twist

        
def Normalize(fea = []):
    #print(fea[0])
    for F in fea:
        Max_n = max(F[2])
    F[2]/=Max_n
    return fea
def Dis(a,b):
    total  = 0
    for i in range(len(a)):
        total += (a[i]-b[i])**2
    return math.sqrt(total)         
def Match(p,others):
    min_d = Dis(p[2],others[0][2])
    final = others[0]
    last2 = min_d
    flag = True
    for px in others:
        if Dis(p[2],px[2]) < min_d:
            last2 = min_d
            min_d = Dis(p[2],px[2])
            final = px
    if min_d >= last2*0.8:
        flag = False
    return flag,final
def ChanG(features,f):
    xc = width/2
    yc = height/2
    for fea in features:
        i = fea[0]
        j = fea[1]
        x = j - xc
        y = i - yc
        x1 = s*Theta(x,f)+xc
        y1 = s*High(x,y,f)+yc
        fea[0] = int(y1)
        fea[1] = int(x1)
    return features
def combine(a,b,M,idx):
    wid = b.shape[1]
    hei = b.shape[0]
    c1_m = np.dot(M,np.array([0,0,1]).T)
    c2_m = np.dot(M,np.array([hei,0,1]).T)
    c3_m = np.dot(M,np.array([0,wid,1]).T)
    c4_m = np.dot(M,np.array([hei,wid,1]).T)
    c1_m/=c1_m[2]
    c2_m/=c2_m[2]
    c3_m/=c3_m[2]
    c4_m/=c4_m[2]
    c1 = [c1_m[0],c1_m[1]]
    c2 = [c2_m[0],c2_m[1]]
    c3 = [c3_m[0],c3_m[1]]
    c4 = [c4_m[0],c4_m[1]]
    H = np.linalg.inv(M)
    Do_Ri,Do_Rj = max(c4[0],max(c3[0],max(c1[0],c2[0]))),max(c4[1],max(c3[1],max(c1[1],c2[1])))
    Up_Li,Up_Lj = min(c4[0],min(c3[0],min(c1[0],c2[0]))), min(c4[1],min(c3[1],min(c1[1],c2[1])))
    Do_Ri = math.ceil(Do_Ri)
    Do_Rj = math.ceil(Do_Rj)
    Up_Li = math.floor(Up_Li)
    Up_Lj = math.floor(Up_Lj)
    di = min(0,Up_Li)
    di = math.floor(di)
    dj = min(0,Up_Lj)
    dj = math.floor(dj)
    h = max(Do_Ri,a.shape[0])-di
    w = max(Do_Rj,a.shape[1])-dj
    h = math.ceil(h)
    w = math.ceil(w)
    c = np.zeros((h,w,3))
    L_j = 0-dj
    temp = []
    diff.append([0-di,0-dj])
    for i in range(h):
        p = np.dot(M,np.array([i,w-1,1]).T)
        p/=p[2]
        temp.append(p[1])
    R_j = math.ceil(max(temp))
    temp.clear
    for i in range(w):
        p = np.dot(M,np.array([h-1,i,1]).T)
        p/=p[2]
        temp.append(p[0])
    D_i = math.ceil(max(temp))
    if D_i < a.shape[0]-di:
        D_i = a.shape[0]-di
    temp.clear
    for i in range(2):
        p = np.dot(M,np.array([0,i,1]).T)
        p/=p[2]
        temp.append(p[0])
    U_i = math.floor(min(temp))
    if U_i > 0-di:
        U_i = 0-di
    #Dis = math.sqrt((D_i-U_i)**2+(R_j-L_j)**2)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
           
            c[i-di][j-dj] = a[i][j]
                    #c[i-di][j-dj] += (0.5)*a[i][j]
            
    for i in range(Up_Li,Do_Ri):
        for j in range(Up_Lj,Do_Rj):
            p_m = np.dot(H,np.array([i,j,1]).T)
            p_m/=p_m[2]
            p = [p_m[0],p_m[1]]
            if p[0] >= 0 and p[0] < hei and p[1] >= 0 and p[1] < wid:
                sml = np.floor(p)
                lar = np.ceil(p)
                if sml[0] < 0:
                    sml[0] = 0
                if sml[1] < 0:
                    sml[1] = 0
                if lar[0] > hei-1:
                    lar[0] = hei-1
                if lar[1] > wid-1:
                    lar[1] = wid-1
                a1 = lar[1]-sml[1]
                a2 = p[1]-sml[1]
                b1 = lar[0]-sml[0]
                b2 = p[0]-sml[0]
                color = [0,0,0]
                if sum(b[int(sml[0])][int(sml[1])])> 0 and sum(b[int(lar[0])][int(sml[1])]) > 0 and sum(b[int(sml[0])][int(lar[1])]) > 0 and sum(b[int(lar[0])][int(lar[1])])> 0:
                    color = ((a1-a2)*(b1-b2)*b[int(sml[0])][int(sml[1])] + a2*(b1-b2)*b[int(lar[0])][int(sml[1])] + b2*(a1-a2)*b[int(sml[0])][int(lar[1])] +a2*b2*b[int(lar[0])][int(lar[1])])/(a1*b1)
                else:
                    if sum(b[int(sml[0])][int(sml[1])])> 50:
                        color = b[int(sml[0])][int(sml[1])]
                    elif sum(b[int(lar[0])][int(sml[1])]) > 50:
                        color = b[int(lar[0])][int(sml[1])]
                    elif sum(b[int(sml[0])][int(lar[1])]) > 50:
                        color = b[int(sml[0])][int(lar[1])]
                    elif sum(b[int(lar[0])][int(lar[1])]) > 50:
                        color = b[int(lar[0])][int(lar[1])]
                if sum(color) != 0 :
                    if i-di < D_i and i-di > U_i and j-dj < R_j and j-dj > L_j and sum(c[i-di][j-dj]) > 40:
                        dis_j = (j-dj)-L_j
                        dis_i = D_i-(i-di)
                        c[i-di][j-dj] = (dis_j/(R_j-L_j))*color
                        dis_j = R_j - (j-dj)
                        dis_i = (i-di) - U_i
                        c[i-di][j-dj] += (dis_j/(R_j-L_j))*color
                        #c[i-di][j-dj] += (0.5)*color
                    else:
                        c[i-di][j-dj] = color
        
    if idx == n_p:
        cal = np.dot(M,np.array([wid-1,hei-1,1]).T)
        cal/=cal[2]
        diff.append([math.floor(cal[1]),math.floor(cal[0])])
                            
    return c    
def GetFeatures(img, radius=16):

    height = img.shape[0]
    width =  img.shape[1]
    
    img_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
    img_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)

    S_xx = cv2.GaussianBlur(img_x ** 2, (3,3), cv2.BORDER_DEFAULT)
    S_yy = cv2.GaussianBlur(img_x * img_y, (3,3), cv2.BORDER_DEFAULT)
    S_xy = cv2.GaussianBlur(img_y ** 2, (3,3), cv2.BORDER_DEFAULT)

    response = (S_xx * S_yy - S_xy ** 2 ) - k * (S_xx + S_yy) ** 2

    #choose strongest 250 points with r, (r=16 in default)
    dtype = [('value',float), ('row',int), ('col',int)]
    R_list = np.zeros(height*width, dtype=dtype)

    for h in range(height):
        for w in range(width):
            R_list[h*width+w] = (response[h,w], h, w)

    R_list = np.sort(R_list, order='value')[::-1]

    # features entry = (h, w, (3x3)descriptor)
    features = []
    for res in R_list:
        TooClose = False
        if len(features) >= 250:
            break
        for f in features:
            if (res[1]-f[0])**2+(res[2]-f[1])**2 < radius**2:
                TooClose = True 
                break 
        if not TooClose:
            h,w = res[1], res[2]
            des = []
            for i in range(3):
                for j in range(3):
                    if h-1+i < 0 or h-1+i >= height or w-1+j < 0 or w-1+j >= width:
                        des.append(-1)
                    else:
                        des.append(img[h-1+i,w-1+j])
            features.append([h,w,des])
            
    return features
k = 0.04
r = 16


# cv2.imshow('after',response)

name = "prtn"
photos = []
fea = []
FeaNor = []
Feature = []
Des = []
Match = []
te = []
Gray = []
cylin = []
s = 703
foc_len= [704.896,706.389,705.946,706.732,706.657,705.703,705.399,704.736,703.821,704.319,704.63,703.786,704.212,704.616,704.766,704.438,704.99,705.456]
#foc_len = [407.332,408.411,410.123,409.618,410.095,410.298,406.256,406.018]
#s = 406
for i in range(n_p):
    if(i <= 9):
        temp = name+'0'+str(i)+".jpg"
    else:
        temp = name+str(i)+".jpg"
    img = cv2.imread(temp,cv2.IMREAD_GRAYSCALE)
    # print(im.shape)
    
    height = img.shape[0]
    width =  img.shape[1]
    '''
    fea = GetFeatures(img)
    fea = Normalize(fea)
    fea = ChanG(fea,foc_len[i])
    FeaNor.append(fea)
    '''

for i in range(0, n_p):
    print(i)
    if(i <= 9):
        temp = name+'0'+str(i)+".jpg"
    else:
        temp = name+str(i)+".jpg"
    img = cv2.imread(temp)
    height = img.shape[0]
    width = img.shape[1]
    photos.append(img)
    twist = Cylin(photos[i],703)
    cylin.append(twist)
cylin.append(cylin[0])
photos.append(photos[0])
for i in range(n_p+1):
    
    twist = cylin[i]
    gray= cv2.cvtColor(twist,cv2.COLOR_BGR2GRAY)
    siftDetector=cv2.xfeatures2d.SIFT_create()
    kp = siftDetector.detect(gray,None)
    psd_kp1, psd_des1 = siftDetector.compute(gray,kp)
    if i == 0:
        print(np.array(psd_des1).shape)
    Feature.append(psd_kp1)
    Des.append(psd_des1)
    im=cv2.drawKeypoints(gray,kp, outImage = None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('sift_keypoints'+str(i)+'.jpg',im)
    cylin.append(twist)
    if i > 0:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(Des[i-1], Des[i], k=2)
        goodMatch = []
        for m, n in matches:
	# goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
            if m.distance < 0.50*n.distance:
                goodMatch.append(m)
        
        #print(goodMatch[0].distance,goodMatch[0].queryIdx,goodMatch[0].trainIdx)
        #print(psd_kp1[goodMatch[0].queryIdx].pt,psd_kp2[goodMatch[0].trainIdx].pt)
        img_out = cv2.drawMatches(cylin[i-1], Feature[i-1], cylin[i], Feature[i], goodMatch , None, flags=2)

        #cv2.imshow('image', img_out)
        cv2.imwrite('sift'+str(i)+'.jpg',img_out)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        Match.append(goodMatch)
    
    
#cylindral coordinate


    
    '''        
    cv2.imshow('after',twist)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
#compare

#blendings
for i in range(n_p+1):
    if i > 0:
        M_p = Blend_sift(Match[i-1],Feature,i)
        M = np.zeros((3,3))
        for j in range(8):
            M[math.floor(j/3)][j%3] = M_p[j]
        M[2][2] = 1
        print(i)
        if i == 1:
            final = cylin[0].copy()
        final = combine(final,cylin[i],M,i)
        if i == n_p:
            Mod = Drift_M(final.shape[1])
            Drift = Bilinear(Mod,final)
                
print(diff)
cv2.imwrite('final.jpg',final)
cv2.imwrite('Drift.jpg',Drift)
        
print("finish")
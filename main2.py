# %%
import cv2 as cv

from matplotlib import pyplot as plt
import networkx as nx
import itertools
import numpy as np
import math

# %%
nonoise_vid = cv.VideoCapture('./1tagvideo.mp4')
if (nonoise_vid.isOpened() == False):
	print("Error opening the video file")
else:
  # Get frame rate information

  fps = int(nonoise_vid.get(5))
  print("Frame Rate : ",fps,"frames per second")	

  # Get frame count
  frame_count = nonoise_vid.get(7)
  print("Frame count : ", frame_count)

# %% [markdown]
# # SCRATHPAD

# %%
def detect_rectangles(corners:list()):
    list_of_point_pairs = []
    for point_pairs in itertools.combinations(list_of_corners,2):
        p1,p2= point_pairs[0],point_pairs[1]
        dist = np.linalg.norm(p1-p2)
        if(dist>50):
            center =  np.array([(p1[0]+p2[0])/2,(p1[1]+p2[1])/2])
            list_of_point_pairs.append([p1,p2,center,dist])
               
    array_of_corners = np.array(list_of_point_pairs,dtype=object)
    array_of_corners = array_of_corners[np.argsort(array_of_corners[:,-1])][::-1]
    
    list_of_rectangles = []
    list_of_center_dist = []
    for line_pairs in itertools.combinations(array_of_corners,2):
        n,m = line_pairs[0],line_pairs[1]
        if(np.abs(n[-1]-m[-1])>10):
            continue
        if (np.linalg.norm(m[3]-n[3])>10):
            continue
        list_of_rectangles.append(np.array([tuple(n[0].tolist()),tuple(m[0].tolist()),tuple(n[1].tolist()),tuple(m[1].tolist())]))
        list_of_center_dist.append(np.array([n[3],np.linalg.norm(n[0]-m[0])],dtype = object))
    return list_of_rectangles,list_of_center_dist

# %%
def plot_polygon(img, lov):
    for i in lov:
        pts = np.array(i, np.int32)
        pts = pts.reshape((-1,1,2))
        img_rect = cv.polylines(img,[pts],True,(0,255,255))
    return img_rect

# %%
def calculate_homography(world_corners,image_corner):
    Xw = world_corners
    xc = image_corner
    matrix_A = np.matrix(np.zeros((8,9)))
    for a,b in zip(enumerate(Xw),enumerate(xc)):
        i,j,n,m = a[0],b[0],a[1],b[1]
        matrix_A[i+j,:] = -n[0],-n[1],-1,0,0,0,m[0]*n[0],m[0]*n[1],m[0]
        matrix_A[(i+j)+1,:] = 0,0,0,-n[0],-n[1],-1,m[1]*n[0],m[1]*n[1],m[1]
    U,S,V = np.linalg.svd(matrix_A)
    H = np.reshape(V[-1],[3,3])
    return H/H[-1,-1]

# %%
def decode_tag(tag):
    tag = cv.cvtColor(tag,cv.COLOR_BGR2GRAY)
    _,tag = cv.threshold(tag,127,255,cv.THRESH_OTSU+cv.THRESH_OTSU)
    tag_grids = np.array_split(tag,8)
    msg = np.zeros((8,8))
    for i,tag_grid in enumerate(tag_grids):
        for j,grid in  enumerate(np.array_split(tag_grid,8,axis=1)):
            if np.count_nonzero(grid) < 0.5*grid.size:
                msg[i][j] = 0
            else:
                msg[i][j] =1
    if msg[2][2]:
        ori = 180
    elif msg[2][5]:
        ori = 90
    elif msg[5][2]:
        ori = 270
    elif msg[5][5]:
        ori = 0
    else :
        ori = None
    idx = (int(msg[3,3])*1+int(msg[3,4])*2+int(msg[4,4])*4+int(msg[4,3])*8)
    idx = ((idx<<(ori//90)&0b1111)|(idx>>(4-(ori//90))))
    if((msg[0:2]!=0).any() and (msg[:][0:2]!=0).any() and (msg[6:-1][:]!=0).any() and (msg[:][6:-1]!=0).any()):
        return np.nan,None,None
    return msg,ori,idx

# %%
def Warping(im, H, size, tes = None):
    Yt, Xt = np.indices((size[0], size[1]))
    
    cam_pts = np.stack((Xt.ravel(), Yt.ravel(), np.ones(Xt.size)))

    H_inv = np.linalg.inv(H)
    # print(H_inv)
    cam_pts = H_inv.dot(cam_pts)
    cam_pts /= cam_pts[2,:]

    Xi, Yi = cam_pts[:2,:].astype(int)
    # padding
    Xi[Xi >=  im.shape[1]] = im.shape[1]
    Xi[Xi < 0] = 0
    Yi[Yi >=  im.shape[0]] = im.shape[0]
    Yi[Yi < 0] = 0
    
    # warped_image = np.zeros((size[0], size[1], 3))
    # im[Yt.ravel(), Xt.ravel(), :] = tes[Yi, Xi, :]
    if (type(tes)==np.ndarray):
        im[Yi, Xi, :] = tes[Yt.ravel(), Xt.ravel(), :]
        return im
    else:
        warped_image = np.zeros((size[0],size[1], 3))
        warped_image[Yt.ravel(), Xt.ravel(), :]= im[Yi, Xi, :]
        return warped_image

# %%
def sort_points(points):
    
    left_most = points[np.argmin(points[:,0])]
    right_most = points[np.argmax(points[:,0])]
    topmost = points[np.argmin(points[:,1])]
    bottommost = points[np.argmax(points[:,1])]
    return [left_most,bottommost,right_most,topmost]
    pass

# %%

frames = []
out = cv.VideoWriter('outpy.avi',cv.VideoWriter_fourcc('M','J','P','G'), 26, (1920,1080))
testudo  =cv.imread('testudo.jpg')
while(nonoise_vid.isOpened()):
	# nonoise_vid.read() methods returns a tuple, first element is a bool 
	# and the second is frame
    ret, frame = nonoise_vid.read()
    if ret == True:
        #Split the channels and invert color of the frame as we are only interested in the red channel
        frames.append(frame)
    
        sharpen = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])## Sharpen the image to hightlight edges
        img = frame #assign the frame of interest
        img_modify = cv.medianBlur(img, 5) # Blur the image slightly
        img_modify = cv.filter2D(img_modify,-1,sharpen) ##Sharpen the image to get the edges more promanent 
        gray = cv.cvtColor(img_modify,cv.COLOR_BGR2GRAY) #convert to grayscale

        #Perform corner detection
        corners = cv.goodFeaturesToTrack(gray,30,0.01,35) 
        corners = np.int0(corners)
        list_of_corners = []
        for i in corners:
            x,y = i.ravel()
            list_of_corners.append(np.array([x,y]))

        ## detect all the rectangle in the image
        rectangles,d=detect_rectangles(corners)

        tags = []
        for i, rectangle in enumerate(rectangles):
            #### MAKE IT TO THE CLOSEST MULTIPLE OF 8 the side length of tag
            max_frame_size = int(d[i][1])+(int(d[i][1])%8)
            PoF = [[0,0],[max_frame_size,0],[max_frame_size,max_frame_size],[0,max_frame_size]]
            rectangle = sort_points(rectangle)
            homo = calculate_homography(rectangle[::-1],PoF)
            rotation_matrix =calculate_rotation(homo)
            try:
                # warped_img = cv.warpPerspective(frames[0], homo,(max_frame_size,max_frame_size))
                warped_img = np.uint8(Warping(img,homo,(max_frame_size,max_frame_size)))
                msg,ori,idx = decode_tag(warped_img)
            except:
                continue
            tags.append([rectangle,msg,ori,idx,max_frame_size,homo])

        for i,tag in enumerate(tags):
            ## For every quad that returned a valid tag info superimpose a image
            if(np.isnan(tag[1]).any() or type(tag[2])==type(None) or type(tag[3])==type(None)):
                continue
            if(tag[3]!=7):
                continue
            #Resize the image to fit the size of the AR Tag
            testudo_rot = np.rot90(testudo,tag[2]//90,(0,1))
            # Rotate the image so that its aligned with the tags ori
            testudo_rot = cv.resize(testudo_rot,(tag[4],tag[4]))
            # Construct a clean white image the same size as the tag to clear the area
            clear = np.uint8(np.ones((tag[4],tag[4],3)))*255
            ## Warp the clearing image to the tags. 
            clear = Warping(img, tag[5],(tag[4],tag[4]),clear)
            # Finally warp the testudo image onto the frame
            final = Warping(clear,tag[5], (tag[4],tag[4]),testudo_rot)
            ## Smoothen out the image to eliminate holes
            final = cv.medianBlur(final,3)
            break

    else:
        break
    print(len(frames))
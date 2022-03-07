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


# %%


# %% [markdown]
# # SCRATHPAD

# %%
## Fit rectangles given the list of corners
def detect_rectangles(corners:list()):
    list_of_point_pairs = []
    ## Create a list of point pairs
    for point_pairs in itertools.combinations(corners,2):
        p1,p2= point_pairs[0],point_pairs[1]
        dist = np.linalg.norm(p1-p2)
        if(dist>50):
            slope = 0
            center =  np.array([(p1[0]+p2[0])/2,(p1[1]+p2[1])/2])
            list_of_point_pairs.append([p1,p2,slope,center,dist])
               
    array_of_corners = np.array(list_of_point_pairs,dtype=object)
    array_of_corners = array_of_corners[np.argsort(array_of_corners[:,-1])][::-1]
    
    list_of_rectangles = []
    list_of_center_dist = []
    ## Create a list of points such that the pair of lines have similar lengths and similar mid point coordinates
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
    print(lov)
    for i in lov:
        print(i)
        pts = np.array(i, np.int32)
        pts = pts.reshape((-1,1,2))
        img_rect = cv.polylines(img,[pts],True,(0,255,255))
    return img_rect
def plot_circles(img, lop):
    img_rect = np.copy(img)
    lop = np.asarray(lop)
    center_coordinates = (120, 50)
    # Radius of circle
    radius = 20
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    # print(lop)
    for i in lop[0]:
        i = tuple(i[0].astype(int).tolist())
        print(i)
        img_rect = cv.circle(img_rect,i,2,(0,255,0),2)
    return img_rect
# %%
## Calculate Homography
def calculate_homography(world_corners,image_corner):
    ## 4 points in the world frame
    Xw = world_corners
    ## 4 points in the camera frame
    xc = image_corner
    ## intermediary matrix to calculate the homography
    matrix_A = np.matrix(np.zeros((8,9)))
    ## Populate the martrix
    for a,b in zip(enumerate(Xw),enumerate(xc)):
        i,j,n,m = a[0],b[0],a[1],b[1]
        matrix_A[i+j,:] = -n[0],-n[1],-1,0,0,0,m[0]*n[0],m[0]*n[1],m[0]
        matrix_A[(i+j)+1,:] = 0,0,0,-n[0],-n[1],-1,m[1]*n[0],m[1]*n[1],m[1]
    ## Calculate the psudo inverse of the matrix
    U,S,V = np.linalg.svd(matrix_A)
    ## The vector with the smallest eigen value is the solution to the equation
    H = np.reshape(V[-1],[3,3])
    return H/H[-1,-1]

# %%
## Decode the tag given a cropped image 
def decode_tag(tag):
    ## Threshold the image to a binary image
    tag = cv.cvtColor(tag,cv.COLOR_BGR2GRAY)
    _,tag = cv.threshold(tag,127,255,cv.THRESH_OTSU+cv.THRESH_OTSU)
    ## Split the image into 8 equal segments
    tag_grids = np.array_split(tag,8)
    msg = np.zeros((8,8))
    ## Iterate over all segments and average each segment into the binary value
    for i,tag_grid in enumerate(tag_grids):
        for j,grid in  enumerate(np.array_split(tag_grid,8,axis=1)):
            if np.count_nonzero(grid) < 0.5*grid.size:
                msg[i][j] = 0
            else:
                msg[i][j] =1
    ## assign the orientation of the tag based on the 3rd row/column
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
    ## Assign the ID of the tag
    idx = (int(msg[3,3])*1+int(msg[3,4])*2+int(msg[4,4])*4+int(msg[4,3])*8)
    ## Rotate the ID given the orientation
    idx = ((idx<<(ori//90)&0b1111)|(idx>>(4-(ori//90))))
    ## If the border condition is not met then return None
    if((msg[0:2]!=0).any() and (msg[:][0:2]!=0).any() and (msg[6:-1][:]!=0).any() and (msg[:][6:-1]!=0).any()):
        return np.nan,None,None
    
    return msg,ori,idx

# %%
## Warp the image with the given size and template image
def Warping(im, H, size, tes = None):
    Yt, Xt = np.indices((size[0], size[1]))
    ## Create an array with values equal to coordinates of the point
    cam_pts = np.stack((Xt.ravel(), Yt.ravel(), np.ones(Xt.size)))
    H_inv = np.linalg.inv(H)
    
    ## Find the transformation that maps the camera point to the world point
    cam_pts = H_inv.dot(cam_pts)
    ## Normalize so that the Z is 1
    cam_pts /= cam_pts[2,:]

    ## Floor the float values to a interger value
    Xi, Yi = cam_pts[:2,:].astype(int)
    # padding
    Xi[Xi >=  im.shape[1]] = im.shape[1]
    Xi[Xi < 0] = 0
    Yi[Yi >=  im.shape[0]] = im.shape[0]
    Yi[Yi < 0] = 0
    ## If a template image is provided then map that to world frame
    if (type(tes)==np.ndarray):
        im[Yi, Xi, :] = tes[Yt.ravel(), Xt.ravel(), :]
        return im
    ## Map the world frame to the given the camera frame
    else:
        warped_image = np.zeros((size[0],size[1], 3))
        warped_image[Yt.ravel(), Xt.ravel(), :]= im[Yi, Xi, :]
        return warped_image
# %%
 ## Sort the points so that the are in anti clockwise 
def sort_points(points):
    left_most = points[np.argmin(points[:,0])]
    right_most = points[np.argmax(points[:,0])]
    topmost = points[np.argmin(points[:,1])]
    bottommost = points[np.argmax(points[:,1])]
    return [left_most,bottommost,right_most,topmost]
    pass
    
    
def paste_img(tag_info,img,src):
    tag = tag_info 
    img_rot = np.rot90(img,tag[2]//90,(0,1))
    # Rotate the image so that its aligned with the tags ori
    img_rot = cv.resize(img_rot,(tag[4],tag[4]))
    # Construct a clean white image the same size as the tag to clear the area
    clear = np.uint8(np.ones((tag[4],tag[4],3)))*255
    ## Warp the clearing image to the tags. 
    clear = Warping(src, tag[5],(tag[4],tag[4]),clear)
    # Finally warp the testudo image onto the frame
    final = Warping(clear,tag[5], (tag[4],tag[4]),img_rot)
    ## Smoothen out the image to eliminate holes
    final = cv.medianBlur(final,3)
    return final

def calculate_rotation(K,homography):
    B_tilde = np.linalg.inv(K)@homography
    h1,h2,h3 = np.array_split(homography,3,axis=1)
    lb = (np.linalg.norm(np.linalg.inv(K)@h1)+np.linalg.norm(np.linalg.inv(K)@h2))/2
    lb = 1/lb
    if(np.linalg.det(B_tilde)<0):
        B = -lb*B_tilde
    else:
        B = lb*B_tilde
    b1,b2,t = np.array_split(B,3,axis=1)
    # r1,r2 = b1/np.linalg.norm(b1),b2/np.linalg.norm(b2)
    b3 = np.cross(b1.T,b2.T)
    # r3 = b3/np.linalg.norm(b3)
    r1, r2, r3 = b1, b2, b3

    rotation_matrix = np.hstack((r1,r2,r3.T,t))

    perspective_transform = (K@rotation_matrix)
    
    return t,rotation_matrix,perspective_transform
    

# %%
def process_video():
    frames = []
    out = cv.VideoWriter('outpy.avi',cv.VideoWriter_fourcc('M','J','P','G'), 26, (1920,1080))
    testudo  =cv.imread('testudo.jpg')
    previous_tag=None
    K = np.array([[1346.100595,0,932.1633975],
    [0,1355.933136,654.8986796],
    [0,0,1]])
    while(nonoise_vid.isOpened()):
        # nonoise_vid.read() methods returns a tuple, first element is a bool 
        # and the second is frame
        ret, frame = nonoise_vid.read()
        if ret == True:
        #Split the channels and invert color of the frame as we are only interested in the red channel
            frames.append(frame)
            sharpen = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])## Sharpen the image to hightlight edges
            img = frame #assign the frame of interest
            img_modify = cv.medianBlur(img, 5) # Blur the image slightly to remove noise
            img_modify = cv.filter2D(img_modify,-1,sharpen) ##Sharpen the image to get the edges more promanent 
            gray = cv.cvtColor(img_modify,cv.COLOR_BGR2GRAY) #convert to grayscale

            #Perform corner detection
            corners = cv.goodFeaturesToTrack(gray,30,0.1,35) ## only 30 points are selected
            corners = np.int0(corners)
            list_of_corners = []
            for i in corners:
                x,y = i.ravel()
                list_of_corners.append(np.array([x,y]))
            ## Given the list of corners, fit quads to the images
            rectangles,d=detect_rectangles(list_of_corners)
            tags = []
            ## For every quad calculate the Homography matrix that converts the world point to the camera plane

            for i, rectangle in enumerate(rectangles):
                #### MAKE IT TO THE CLOSEST MULTIPLE OF 8 the side length of tag
                max_frame_size = int(d[i][1])+(int(d[i][1])%8)
                ## The points on the image plane that will be mapped to the world plane
                PoF = [[0,0],[max_frame_size,0],[max_frame_size,max_frame_size],[0,max_frame_size]]
                ## Sort the list of points such that the points are in the clockwise sense
                rectangle = sort_points(rectangle)
                ## Calculate the homography using SVD
                homo = calculate_homography(rectangle[::-1],PoF)
                tvec,rvec,_ = calculate_rotation(K,np.linalg.inv(homo))
                try:
                    warped_img = np.uint8(Warping(img,homo,(max_frame_size,max_frame_size)))
                    ## Decode the same image assuming its a TAG
                    msg,ori,idx = decode_tag(warped_img)
                except:
                    continue
                ## Add it to the list of tags
                tags.append([msg,ori,idx,max_frame_size,homo])
                
                ## For every quad that returned a valid tag info superimpose a image
                tag = tags[-1]
                if(np.isnan(tag[0]).any() or type(tag[1])==type(None) or type(tag[2])==type(None)):
                    continue
                if(tag[2]!=7):
                    continue
                cv.imshow("cube",warped_img)
                cv.waitKey()
                cv.destroyAllWindows()
                points_to_project = np.float32([[0, 0, 0],[tag[3], 0, 0],[tag[3], tag[3], 0], [0, tag[3], 0],[0, 0, 0], 
                                                    [0, 0, -tag[3]], [tag[3], 0, -tag[3]],[tag[3],0,0],[tag[3],0,-tag[3]], [tag[3], tag[3], -tag[3]],
                                                    [tag[3],tag[3],0],[tag[3],tag[3],-tag[3]],[tag[3],0,-tag[3]],[tag[3],0,0],[tag[3],0,-tag[3]],[0, 0, -tag[3]]])
                    # points = np.array([[[0,0,0],[tag[3],0,0],[tag[3],tag[3],0],[0,tag[3],0],\
                # [0,0,-1],[tag[3],0,-1],[tag[3],tag[3],-1],[0,tag[3],-1]]],dtype = np.float64)
                projected_corners,_ = cv.projectPoints(points_to_project, rvec, tvec, K, np.zeros((1, 4)))
                img_rect = np.copy(img)
                img_rect = plot_polygon(img_rect,projected_corners)
                # out.write(final)
                cv.imshow("cube",img_rect)
                cv.waitKey()
                cv.destroyAllWindows()
                break
            else:
                if(previous_tag):
                    tag = previous_tag
                    points_to_project = np.float32([[0, 0, 0],[tag[3], 0, 0],[tag[3], tag[3], 0], [0, tag[3], 0],[0, 0, 0], 
                                                    [0, 0, -tag[3]], [tag[3], 0, -tag[3]],[tag[3],0,0],[tag[3],0,-tag[3]], [tag[3], tag[3], -tag[3]],
                                                    [tag[3],tag[3],0],[tag[3],tag[3],-tag[3]],[tag[3],0,-tag[3]],[tag[3],0,0],[tag[3],0,-tag[3]],[0, 0, -tag[3]]])
                    # points = np.array([[[0,0,0],[tag[3],0,0],[tag[3],tag[3],0],[0,tag[3],0],\
                    # [0,0,-1],[tag[3],0,-1],[tag[3],tag[3],-1],[0,tag[3],-1]]],dtype = np.float64)
                    projected_corners,_ = cv.projectPoints(points_to_project, rvec, tvec, K, np.zeros((1, 4)))
                    img_rect = np.copy(img)
                    img_rect = plot_polygon(img_rect,projected_corners)
                    # out.write(final)
                    cv.imshow("cube",img_rect)
                    cv.waitKey()
                    cv.destroyAllWindows()
        else:
            break
        print(len(frames)) 
        
if __name__=="__main__":
    process_video()
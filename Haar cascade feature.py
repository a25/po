#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# ## Trying slefie with eye

# In[5]:


import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img_path='my.jpeg'
frame =  cv2.imread(img_path)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
x,y=0,0
w,h=gray.shape[0],gray.shape[1]
faces=[(x,y,w,h)]
for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        i=0
        for (x_eye,y_eye,w_eye,h_eye) in eyes:
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
            radius = int(0.3 * (w_eye + h_eye))
            color = (0, 255, 0)
            thickness = 2
            square=frame[y+y_eye:y+y_eye+h_eye, x+x_eye:x+x_eye+w_eye]
            a=cv2.cvtColor(square, cv2.COLOR_BGR2HLS)
            flat=np.reshape(a[:,:,1],(1,-1))
            b=flat[0][flat[0]>0]
            percent=np.percentile(flat[0,flat[0,:]>0],40)
            mask1=cv2.inRange(a[:,:,1], percent, 255)
            contours, hierarchy = cv2.findContours(mask1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                ellipse = cv2.fitEllipse(c)
                (xc,yc),(MA,ma),angle = ellipse
                square=cv2.ellipse(square,ellipse,(0,255,0),2)
            
        cv2.imshow('Eyes',square)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


cap = cv2.VideoCapture(0)
ds_factor = 0.5
x1s,y1s,x2s,y2s=[],[],[],[]
sampling=30
s=0
quit=0
while True:
    ret, frame = cap.read()
#     frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    w1,h1=frame.shape[0],frame.shape[1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    square=None
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if quit:
        break
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        i=0
        for (x_eye,y_eye,w_eye,h_eye) in eyes:
            if i==2:
                break
            
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
            y1s.append(y+y_eye)
            x1s.append(x+x_eye)
            radius = int(0.3 * (w_eye + h_eye))
            color = (0, 255, 0)
            thickness = 2
            square=frame[y+y_eye:y+y_eye+h_eye, x+x_eye:x+x_eye+w_eye]
            if square is not None:
                if square.any():
                    cv2.imshow('Eyes',square)
                    cv2.imwrite('eyej.jpg',square)
#                     quit=1
#                     break

    cv2.imshow('Eye Detector', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()


# In[6]:


im2=cv2.imread('eyej.jpg')


# In[8]:


cv2.imshow('Eye Detector', im2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ## DLIB code

# In[3]:


import dlib
import cv2
import numpy as np


# In[4]:


filename='shape_predictor_68_face_landmarks.dat'


# In[6]:


img = cv2.imread('boy1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale detector = dlib.get_frontal_face_detector()
detector = dlib.get_frontal_face_detector()
rects = detector(gray, 1) # rects contains all the faces detected


# In[7]:


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
left_eye=[]
right_eye=[]
predictor = dlib.shape_predictor(filename)
for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)
    
    print(left_eye,right_eye)
    for (x, y) in shape:
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    
cv2.imshow('hjs',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[9]:


left_eye_ids = np.arange(36,42)
right_eye_ids=np.arange(42,48)


# In[10]:


mask = np.zeros(img.shape[:2], dtype=np.uint8)


# In[11]:


def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


# In[12]:


left_eye_ids


# In[13]:


mask = np.zeros(img.shape[:2], dtype=np.uint8)
mask = eye_on_mask(mask, left_eye_ids)
mask = eye_on_mask(mask, right_eye_ids)


# In[14]:


# cv2.imshow('a',mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[15]:


kernel = np.ones((9, 9), np.uint8)
mask = cv2.dilate(mask, kernel, 5)
eyes = cv2.bitwise_and(img, img, mask=mask)
mask = (eyes == [0, 0, 0]).all(axis=2)
eyes[mask] = [255, 255, 255]
eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)


# In[16]:


cv2.imshow('a',eyes)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[20]:


def nothing(x):
    pass
cv2.namedWindow('image')
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
while 1:
    threshold = cv2.getTrackbarPos('threshold', 'image')
    _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.medianBlur(thresh, 3)
    cv2.imshow('image',thresh)
    c = cv2.waitKey(1)
    if c == 27:
        break

cv2.destroyAllWindows()


# In[25]:


# mask = (eyes == [0, 0, 0]).all(axis=2)


# In[ ]:


def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key = cv2.contourArea) # finding contour with #maximum area
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    if right:
        cx += mid # Adding value of mid to x coordinate of centre of #right eye to adjust for dividing into two parts
    cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)# drawing over #eyeball with red
mid = (shape[39][0] + shape[42][0]) // 2
contouring(thresh[:, 0:mid], mid, img)
contouring(thresh[:, mid:], mid, img, True)
cv2.imshow('image',img)


# In[55]:



import cv2
import dlib
import numpy as np
filename='shape_predictor_68_face_landmarks.dat'
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]
kernel = np.ones((9, 9), np.uint8)
path='boy4.jpg'
width,height=480,480
img = cv2.imread(path)
img=cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA) 

    
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(filename)

   
def nothing(x):
    global img
    img = cv2.imread(path)
    pass

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)

for rect in rects:
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask = eye_on_mask(mask, left)
    mask = eye_on_mask(mask, right)
    eyes = cv2.bitwise_and(img, img, mask=mask)
    eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

def do_nothing(x):
    pass

while(1):
    a=cv2.cvtColor(eyes, cv2.COLOR_BGR2HLS)
    flat=np.reshape(a[:,:,1],(1,-1))
    b=flat[0][flat[0]>0]
    percent=np.percentile(flat[0,flat[0,:]>0],40)
    mask1=cv2.inRange(a[:,:,1], percent, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,kernel)
    cv2.imshow('contours1',mask1)
    cv2.imshow('contours2',eyes)
    cv2.imshow('patient1',img)
    
    if(cv2.waitKey(1)==ord('q')):
        break
cv2.destroyAllWindows()


# In[4]:


import matplotlib.pyplot as plt


# In[24]:


x=sorted(flat[0,:],reverse=True)


# In[38]:


b=flat[0][flat[0]>0]


# In[48]:


b


# In[45]:


l=b.shape[0]
l=np.floor(l*0.9)


# In[64]:


np.percentile(b,90)


# In[32]:


flat[0][::-1].sort()


# In[5]:


plt.hist(flat[0,:])


# ## Backup

# In[60]:



import cv2
import dlib
import numpy as np
filename='shape_predictor_68_face_landmarks.dat'
max_value = 255
max_value_H = 255
low_H = 0
low_S = 0
low_V = 0

high_H = max_value_H
high_S = max_value
high_V = max_value

low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
window_detection_name = 'Object Detection1'

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)
    
    
    
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(filename)

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]
################## window creation
cv2.namedWindow(window_detection_name,cv2.WINDOW_AUTOSIZE)

cv2.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv2.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
kernel = np.ones((9, 9), np.uint8)


# cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
path='boy2.jpg'
width,height=480,480
img = cv2.imread(path)
img=cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)    
def nothing(x):
    global img
    img = cv2.imread(path)
    pass
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)



for rect in rects:

    shape = predictor(gray, rect)
    shape = shape_to_np(shape)
    mask = np.ones(img.shape[:2], dtype=np.uint8)
    mask = eye_on_mask(mask, left)
    mask = eye_on_mask(mask, right)
    eyes = cv2.bitwise_and(img, img, mask=mask)
    eyes= cv2.bitwise_not(eyes)
    eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

def do_nothing(x):
    pass


# cv2.namedWindow('contours')
# cv2.createTrackbar('threshold','contours',low_thresh,high_thresh,do_nothing)
# cv2.createTrackbar('threshold','contours',low_thresh,high_thresh,do_nothing)

while(1):

#     frame_HSV = eyes#cv2.cvtColor(eyes, cv2.COLOR_BGR2HSV)

    a=cv2.cvtColor(eyes, cv2.COLOR_BGR2HLS)
    flat=np.reshape(a[:,:,1],(1,-1))
    b=flat[0][flat[0]>0]
    percent=np.percentile(flat[0,flat[0,:]>0],40)
    mask1=cv2.inRange(a[:,:,1], percent, 255)

#         contours, hierarchy = cv2.findContours(mask1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask1 = cv2.inRange(a, (low_H, low_S, low_V), (high_H, high_S, high_V))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,kernel)
    cv2.imshow(window_detection_name,mask1)
    cv2.imshow('contours2',eyes)

    cv2.imshow('patient1',img)
    
    if(cv2.waitKey(1)==ord('q')):
        break
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# ## Backup-2

# In[ ]:



import cv2
import dlib
import numpy as np
filename='shape_predictor_68_face_landmarks.dat'
max_value = 255
max_value_H = 255
low_H = 0
low_S = 0
low_V = 0

high_H = max_value_H
high_S = max_value
high_V = max_value

low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
window_detection_name = 'Object Detection1'

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)
    
    
    
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(filename)

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]
################## window creation
cv2.namedWindow(window_detection_name,cv2.WINDOW_AUTOSIZE)

# cv2.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
# cv2.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
# cv2.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
# cv2.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
# cv2.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
# cv2.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
# cap = cv2.VideoCapture(0)
# ret, img = cap.read()
# thresh = img.copy()

# cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)


# cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
path='my1.jpg'
width,height=480,480
img = cv2.imread(path)
img=cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)    
def nothing(x):
    global img
    img = cv2.imread(path)
    pass
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)



for rect in rects:

    shape = predictor(gray, rect)
    shape = shape_to_np(shape)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask = eye_on_mask(mask, left)
    mask = eye_on_mask(mask, right)
#     mask = cv2.dilate(mask, kernel, 5)
    eyes = cv2.bitwise_and(img, img, mask=mask)
    eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

def do_nothing(x):
    pass


# cv2.namedWindow('contours')
# cv2.createTrackbar('threshold','contours',low_thresh,high_thresh,do_nothing)
# cv2.createTrackbar('threshold','contours',low_thresh,high_thresh,do_nothing)

while(1):

#     frame_HSV = eyes#cv2.cvtColor(eyes, cv2.COLOR_BGR2HSV)

    a=cv2.cvtColor(eyes, cv2.COLOR_BGR2HLS)
    flat=np.reshape(a[:,:,1],(1,-1))
    b=flat[0][flat[0]>0]
    percent=np.percentile(flat[0,flat[0,:]>0],40)
    mask1=cv2.inRange(a[:,:,1], percent, 255)

#         contours, hierarchy = cv2.findContours(mask1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#     mask1 = cv2.inRange(a, (low_H, low_S, low_V), (high_H, high_S, high_V))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,kernel)
    cv2.imshow(window_detection_name,mask1)
    cv2.imshow('contours2',eyes)

    cv2.imshow('patient1',img)
    
    if(cv2.waitKey(1)==ord('q')):
        break
cv2.destroyAllWindows()


# In[ ]:





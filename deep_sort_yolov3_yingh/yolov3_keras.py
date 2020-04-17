
import os
import sys
from imutils.object_detection import non_max_suppression
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import time
# import logging.config
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from PIL import Image

sys.path.insert(1, r'H:/Admin/unsw/COMP9517/project/Group_Component/Object-Detection-and-Tracking/OneStage/yolo/deep_sort_yolov3')
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque

from keras import backend

backend.clear_session()

# path
working_dir = r'H:/Admin/unsw/COMP9517/project/Group_Component'
os.chdir(working_dir) 

output_path = r'output/deep_sort_yolov3'


pts = [deque(maxlen=30) for _ in range(9999)]
# warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(123)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

start = time.time()
#Definition of the parameters
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.3 

counter = []

#deep_sort
model_filename = r'Object-Detection-and-Tracking/OneStage/yolo/deep_sort_yolov3/model_data/market1501.pb'
encoder = gdet.create_box_encoder(model_filename,batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

def read_filenames(img_dir):

    files = []
    
    for (dirpath, dirnames, filenames) in os.walk(img_dir):
        for file in filenames:
            files.append(file)

    return files

frame_files = read_filenames(r'sequence/')

# frame = cv2.imread(r'sequence/000040.jpg')

# test = frames[:10]

for seq in frame_files[:20]:

    if not seq.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    print(seq)
    
    # start = time.time()
    
    frame = cv2.imread(os.path.join('sequence/',seq))

    image = Image.fromarray(frame)
    boxs,class_names = YOLO().detect_image(image)

    features = encoder(frame,boxs)
    # score to 1.0 here).
    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    # Call the tracker
    tracker.predict()
    tracker.update(detections)

    i = int(0)
    indexIDs = []
    c = []
    boxes = []
    for det in detections:
        bbox = det.to_tlbr()
        cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        #boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track.track_id))
        counter.append(int(track.track_id))
        bbox = track.to_tlbr()
        color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

        x1 = int(bbox[0])
        y1 = int(bbox[1])

        x2 = int(bbox[2])
        y2 = int(bbox[3])

        cv2.rectangle(frame, (x1, y1), (x2, y2),(color), 2)
        
        # cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),1)
        if len(class_names) > 0:
            class_name = class_names[0][0]
            (text_width, text_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 2)
            cv2.putText(frame, class_name + str(track.track_id), (x1, y1-baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (color), 2)

        i += 1
        #bbox_center_point(x,y)
        center = (int((x1+x2)/2),int((y1+y2)/2))

        #track_id[center]
        pts[track.track_id].append(center)
        thickness = 2
        #center point
        cv2.circle(frame,  (center), 1, color, thickness)

    #draw motion path
        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                continue
            thickness = 3
            cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
            #cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

    count = len(set(counter))
    cv2.putText(frame, "Total Object Count: "+str(count),(int(20), int(120)),0, 0.75, (0,255,0),2)
    cv2.putText(frame, "Current Object Count: "+str(i),(int(20), int(80)),0, 0.75, (0,255,0),2)
    # cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)
    # cv2.namedWindow("YOLO3_Deep_SORT", 0);
    # cv2.resizeWindow('YOLO3_Deep_SORT', 1024, 768);
    # cv2.imshow('YOLO3_Deep_SORT', frame)
    cv2.imwrite(os.path.join(output_path, seq), frame)


# plt.imshow(frames[7])
# plt.show()

height , width , layers =  frames[0].shape 
fourcc =  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# fourcc = 0
fps = 10

out = cv2.VideoWriter('video_yolo_tracking.avi', fourcc, fps, (width, height))
 
for i in range(len(frames)):
    out.write(frames[i])

cv2.destroyAllWindows()
out.release()

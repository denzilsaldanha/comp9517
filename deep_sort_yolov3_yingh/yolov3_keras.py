
import os
import sys
from imutils.object_detection import non_max_suppression
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import time
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import argparse
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

sys.path.insert(1, r'H:/Admin/unsw/COMP9517/project/Group_Component')
from people_in_box import People_In_Box

# parser = argparse.ArgumentParser()
# parser.add_argument("-i", "--working_dir", dest="working_dir", help="Path to project directory.", default="H:/Admin/unsw/COMP9517/project/Group_Component")
# parser.add_argument("-o", "--output_file", dest="output_file", help="Path to output video file.", default="video_yolov3_tracking.mp4")
# parser.add_argument("-d", "--input_dir", dest="input_dir", help="Path to input working directory.", default="H:/Admin/unsw/COMP9517/project/Group_Component/sequence")
# parser.add_argument("-u", "--output_dir", dest="output_dir", help="Path to output working directory.", default="H:/Admin/unsw/COMP9517/project/Group_Component/output")
# parser.add_argument("-r", "--frame_rate", dest="frame_rate", help="img rate of the output video.", default=20)

# args = vars(parser.parse_args())
# if not options.input_dir:   # if filename is not given
# 	parser.error('Error: path to image input_file must be specified. Pass --input-file to command line')

# output_video = options.output_file
# img_path = os.path.join(args['input_dir'], '')
# output_path = os.path.join(options.output_dir, 'deep_sort_yolov3')
# fps = float(options.frame_rate)

# def cleanup():
# 	print("cleaning up...")
# 	os.popen('rm -f ' + img_path + '*')
# 	os.popen('rm -f ' + output_path + '*')




# path
working_dir = r'H:/Admin/unsw/COMP9517/project/Group_Component'
os.chdir(working_dir) 

out_path = r'C:/Users/85698/output/yolov3'
out_video = 'video_yolov3.avi'

# save_video = False

def get_file_names(img_path):

    files = []
    
    for (dirpath, dirnames, filenames) in os.walk(img_path):
        for file in filenames:
            files.append(file)

    return files

def save_to_video():
    
    out_files = get_file_names(out_path)
    out_files.sort()
    img = cv2.imread(os.path.join(out_path, out_files[0]))
    height , width , layers =  img.shape
    
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc =  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    
    out = cv2.VideoWriter(out_video, fourcc, fps, (width,height))
    
    print("output video...")
    for f in out_files:
        img = cv2.imread(os.path.join(out_path, f))
        out.write(img)
        
    out.release()
    cv2.destroyAllWindows()
    


pts = [deque(maxlen=30) for _ in range(9999)]
# warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(1234)
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

frame_files = get_file_names(r'sequence/')

# img = cv2.imread(r'sequence/000040.jpg')

# test = frames[:10]


    
# Establishing coordinates and dimension of box
bxx = 150
bxy = 100
bxw = 500
bxh = 400

for f in frame_files:

    if not f.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    print(f)
    
    # start = time.time()
    
    img = cv2.imread(os.path.join('sequence/',f))

    image = Image.fromarray(img)
    boxs,class_names = YOLO().detect_image(image)

    features = encoder(img,boxs)
    # score to 1.0 here).
    detections = [Detection(bbox, 0.8, feature) for bbox, feature in zip(boxs, features)]
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
    rects = []
    rects_centers = []
    centers = []
    n = 0
    people_array = []

    for det in detections:
        bbox = det.to_tlbr()
        if (bbox[3] - bbox[1]) > 30:
            cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            
            rects.append(bbox)
            rects_centers.append([int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)])

            n += 1

    # for track in tracker.tracks:
    #     bbox = track.to_tlbr()
    #     if not track.is_confirmed() or track.time_since_update or (bbox[3] - bbox[1]) <= 40:
    #         continue
    #     #boxes.append([track[0], track[1], track[2], track[3]])
    #     indexIDs.append(int(track.track_id))
    #     counter.append(int(track.track_id))
    #     color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

    #     x1 = int(bbox[0])
    #     y1 = int(bbox[1])

    #     x2 = int(bbox[2])
    #     y2 = int(bbox[3])

    #     cv2.rectangle(img, (x1, y1), (x2, y2),(color), 2)
        
    #     # cv2.putText(img,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),1)
    #     if len(class_names) > 0:
    #         class_name = class_names[0][0]
    #         (text_width, text_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    #         cv2.putText(img, class_name + str(track.track_id), (x1, y1-baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (color), 1)

    #     i += 1
    #     #bbox_center_point(x,y)
    #     center = (int((x1+x2)/2),int((y1+y2)/2))
    #     centers.append(center)

    #     #track_id[center]
    #     pts[track.track_id].append(center)
    #     thickness = 2
    #     #center point
    #     cv2.circle(img,  (center), 2, color, thickness)

    # #draw motion path
    #     for j in range(1, len(pts[track.track_id])):
    #         if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
    #             continue
    #         thickness = 2
    #         cv2.line(img,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)

    # count = len(set(counter))

    # # draw a rectangle
    # cv2.rectangle(
    #     img, (bxx, bxy), (bxx+bxw, bxy+bxh), (140,230,240), 1)

    # # Start Task 2 - return the count of the people inside the drawn box

    # # Initialise Constructor for the People in Box Tracker
    # people_inside = People_In_Box(img, rects_centers, rects)

    # peopleInBox, _ = people_inside.count_people_in_box(bxx,bxy,bxx+bxw,bxy+bxh)
                                             
    # # draw user defined bounding rectangle
    # cv2.rectangle(img, (bxx,bxy), (bxx+bxw, bxy+bxh), 
    #               (255,255,204), 1)
    
    # print('finish counting in the box..')
    
    # # End Task 2


    # # Start Task 3

    # # select centroid distance of 37 as group detection threshold
    # groups = people_inside.detect_group(40)
    # peopleInGroup, peopleAlone, group_boxs = people_inside.count_people_in_group(groups)
    
    # # draw group bounding boxes
    # if len(group_boxs) > 0:
    #   for box in group_boxs:
    #     cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(225, 225, 51),2)

    # Show the count on img
    text_color = [150, 255, 0]
    (text_width, text_height), baseline = cv2.getTextSize(
            'people detected: ', cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(img, 'people detected: ' + str(n), 
                (10, 15+(text_height+5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (text_color), 2)
    # cv2.putText(img, 'people in box: ' + str(peopleInBox), 
    #             (10, 15+(text_height+5)*2),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (text_color), 2)
    # cv2.putText(img, 'groups detected: ' + str(len(group_boxs)), 
    #             (10, 15+(text_height+5)*3),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (text_color), 2)
    # cv2.putText(img, 'people in groups: ' + str(peopleInGroup), 
    #             (10, 15+(text_height+5)*4),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (text_color), 2)
    # cv2.putText(img, 'people alone: ' + str(peopleAlone), 
    #             (10, 15+(text_height+5)*5),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (text_color), 2)

    people_array.append(n)

    cv2.imwrite(os.path.join(out_path, f), img)


# plt.imshow(frames[7])
# plt.show()


print("saving to video...")
save_to_video()


if __name__ == '__main__':
	main()

import os, sys
# from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import math
import cv2
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import time
import logging.config
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

sys.path.insert(1, r'H:/Admin/unsw/COMP9517/project/Group_Component/pedestrian-detection-master/yoloV2')
from models import yolo
from log_config import LOGGING



fig = plt.figure(figsize=(2, 2))

# path
working_dir = r'H:/Admin/unsw/COMP9517/project/Group_Component'
os.chdir(working_dir) 

# sys.path.insert(1, r'H:/Admin/unsw/COMP9517/project/Group_Component/keras-frcnn-master')


# path
working_dir = r'H:/Admin/unsw/COMP9517/project/Group_Component'
os.chdir(working_dir) 

out_path = r'output/yolov2'
out_video = 'video_yolo.avi'

# save_video = False

def get_file_names(img_path):

    files = []
    
    for (dirpath, dirnames, filenames) in os.walk(img_path):
        for file in filenames:
            files.append(file)

    return files

fps = 10

def save_to_video():
    
    out_files = get_file_names(out_path)
    out_files.sort()
    img = cv2.imread(os.path.join(out_path, out_files[0]))
    height , width , layers =  img.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(out_video, fourcc, fps, (width,height))
    
    print("output video...")
    for f in out_files:
        img = cv2.imread(os.path.join(out_path, f))
        out.write(img)
        
    out.release()
    cv2.destroyAllWindows()
    

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

frame_files = get_file_names(r'sequence/')
img = cv2.imread(os.path.join('sequence/',frame_files[0]))

source_h, source_w, channels = img.shape

logging.config.dictConfig(LOGGING)
logger = logging.getLogger('detector')

FLAGS = tf.flags.FLAGS
# tf.compat.v1.flags.DEFINE_string('video', "0", 'Path to the video file.')
tf.flags.DEFINE_string('model_name', 'Yolo2Model', 'Model name to use.')


model_cls = find_class_by_name(FLAGS.model_name, [yolo])
model = model_cls(input_shape=(source_h, source_w, channels))
model.init()



begin = time.time()

# loop over the frames

# img_array = frames[:100].copy()
people_array = []

for f in frame_files:

    if not f.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    print(f)
    
    # start = time.time()
    
    img = cv2.imread(os.path.join('sequence/',f))

    # detect people in the image
    preds = model.evaluate(img)
    # print(preds)
    cogs = []
    n = 0
    

    for o in preds:
        
        class_name = o['class_name']

        if class_name != 'person' or int(o['box']['bottom'] - o['box']['top']) < 30:
                continue

        x1 = o['box']['left']
        x2 = o['box']['right']

        y1 = o['box']['top']
        y2 = o['box']['bottom']

        color = o['color']
        
        # cogs.append([math.floor((o['box']['left']+o['box']['right'])/2),
        #                 math.floor(o['box']['bottom']),  # red point : fits quite good
        #                 1])
        
        # print(cogs)
        # cv2.circle(img, (cogs[-1][0], cogs[-1][1]), 5, (0, 0, 255), -1)
        
        n += 1

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        (test_width, text_height), baseline = cv2.getTextSize(
            class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)
        cv2.rectangle(img, (x1, y1),
                    (x1+test_width, y1-text_height-baseline),
                    color, thickness=cv2.FILLED)
        cv2.putText(img, class_name, (x1, y1-baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    # Show the count on frame
    cv2.putText(img, 'people detected: ' + str(n), (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1, cv2.LINE_AA)
    
    people_array.append(n)

    cv2.imwrite(os.path.join(out_path,f), img)

    # esc to quit
    # if cv2.waitKey(1) == 27:
    #     break

# End time
end = time.time()

# Time elapsed
seconds = end - begin

print(seconds)

cv2.imshow('test',img_array[0])

height , width , layers =  frames[0].shape 
fourcc =  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# fourcc = 0
fps = 20

out = cv2.VideoWriter('video_yolov2.avi', fourcc, fps, (width, height))
 
for i in range(len(img_array)):
    out.write(img_array[i])

cv2.destroyAllWindows()
out.release()



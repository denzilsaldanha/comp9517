
# Import python libraries
import cv2
import copy
import os
import argparse
from human_detectors import Human_Detectors
from tracker import Tracker
from people_in_box import People_In_Box


def create_roi(videopath, roi):
    x = input("Please type 'Y' or 'N' ")

    if x.lower() == 'y':
        f = open("Input/"+videoname+"_pre-testedROI.txt", "w+")
        f.write(str(roi))
        f.close()
    elif x.lower() == 'n':
        pass
    else:
        create_roi(videoname, roi)


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-vid', '--video', required=True,
                        default="Input/project.avi", help="Video File Path")
    parser.add_argument('-roi', '--roi creation mode', required=False, default="manually", help="Create region of interest-do it 'manually'," +
                        "or use the 'pre-tested' one which gives good results")
    args = vars(parser.parse_args())

    video = args['video']
    roi_mode = args['roi creation mode']
    # to run the code, execute the below command
    # python object_tracking.py -vid Input/project.avi -roi pre-tested
    # press escape to exit the video
    videopath, __ = video.split(".", -1)
    __, videoname = videopath.split('/', -1)

    camera = cv2.VideoCapture(video)
    ret, frame = camera.read()

    # video = cv2.VideoCapture('project.avi')

    # Initialise Human Detector
    human_detect = Human_Detectors()

    # Initialise Tracker
    tracker = Tracker(160, 30, 5, 100)

    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]

    if roi_mode == 'manually':
        roi = cv2.selectROI(frame, showCrosshair=False)
    elif roi_mode == 'pre-tested':
        try:
            roi_file = open(videopath+'_pre-testedROI.txt', 'r')
            rois = roi_file.read()
            rois = rois[1:-1]
            roi = rois.split(", ")
            for i in range(len(roi)):
                roi[i] = int(roi[i])
        except:
            print(
                "The pre-tested Region of Interest file does not exist yet. Please create it manually.")
            roi = cv2.selectROI(frame, showCrosshair=False)

    cv2.destroyWindow('ROI selector')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vid_out = cv2.VideoWriter('Output_Video/'+videoname+'.avi', fourcc,
                              20.0, (int(camera.get(3)), int(camera.get(4))), isColor=True)

    # rootdir = "./sequence/"
    # # # loop over the image paths
    # for subdir, dirs, files in os.walk(rootdir):
    #     for file in files:
    #         path = os.path.join(subdir, file)
    #         frame = cv2.imread(path)

    # Run through video frames
    while(camera.isOpened()):
        ret, frame = camera.read()

        # Detect and return centeroids of the objects in the frame
        centers = human_detect.Detect(frame)
        total_centers = len(centers)

        # Track centroids, if found
        if (total_centers > 0):
            # Track object using Kalman Filter
            tracker.Update(centers)
            count = len(tracker.tracks)

            # Start Task 2
            # Establishing coordinates and dimension of box
            x_coord = roi[0]
            y_coord = roi[1]
            width = roi[2]
            height = roi[3]

            # Declaring Constructor for the People in Box Tracker
            people_inside = People_In_Box(
                frame, centers)

            # Return the count of the people inside the drawn box
            peopleInBox = people_inside.count_people(
                x_coord, y_coord, x_coord + width, y_coord + height)

            # # draw a rectangle
            # cv2.rectangle(
            #     frame, (x_coord, y_coord), (x_coord + width, y_coord + height), (255, 0, 0), 2)
            # End Task 2

            # Start Task 3 - Fix the function in people_in_box.py
            peopleInGroup, peopleAlone, frame = people_inside.count_people_in_group(
                75)

            # drawing tracks with different colors
            for i in range(len(tracker.tracks)):
                if (len(tracker.tracks[i].trace) > 1):
                    for j in range(len(tracker.tracks[i].trace)-1):
                        # Coordinated of predicted line
                        x_1 = int(tracker.tracks[i].trace[j][0][0])
                        y_1 = int(tracker.tracks[i].trace[j][1][0])
                        x_2 = int(tracker.tracks[i].trace[j+1][0][0])
                        y_2 = int(tracker.tracks[i].trace[j+1][1][0])
                        clr = tracker.tracks[i].track_id % 9
                        cv2.line(frame, (x_1, y_1), (x_2, y_2),
                                 track_colors[clr], 2)

            # Show the tracked video frame
            vid_out.write(frame)
            cv2.rectangle(
                frame, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 1)
            cv2.putText(frame, 'people detected: ' + str(count), (10, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'people in box: ' + str(peopleInBox), (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'people in groups: ' + str(peopleInGroup), (10, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'people alone: ' + str(peopleAlone), (10, 550),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('Tracking', frame)

            """
            For everyone running it, press "Spacebar" to go to the next frame
            """
            # cv2.waitKey(
            #     0)  # Comment out if you want to run like a video instead of frame by frame
            # print("total number people in the frame: ", count)

        key = cv2.waitKey(50) & 0xff
        # Escape key to exit
        if key == 27:
            break

    camera.release()
    vid_out.release()
    cv2.destroyAllWindows()
    if roi_mode == 'manually':
        print("Do you wish to save the created roi to be used as an optional pre-tested file for this video next run onwards (if it gave good results)")
        wr = create_roi(videoname, roi)


if __name__ == "__main__":
    main()

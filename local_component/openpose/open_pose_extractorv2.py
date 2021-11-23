# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse


try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #print(dir_path)
    photo_path = '../doc/mini_dataset2/WIN_20210205_10_26_35_Pro_273.jpg'
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '\\openpose\\build\\python\\openpose\\Release');
            print(dir_path + '\\openpose\\build\\python\\openpose\\Release')
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '\\openpose\\build\\x64\\Release;' +  dir_path + '\\openpose\\build\\bin;'
            print(os.environ['PATH'])
            import pyopenpose as op
        else:
            print(sys.path)
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('openpose/openpose/build/python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')

    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e
except Exception as e:
    print(e)
    sys.exit(-1)


class OpenPoseWrapper:
    def __init__(self):

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--image_path", default="./openpose/examples/media/COCO_val2014_000000000241.jpg",
                            help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")

        args = self.parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        self.params = dict()
        self.params["model_folder"] = dir_path +"/openpose/models/"
        self.params["face"] = False  # True
        self.params["hand"] = False
        self.params["number_people_max"] = 1
        self.params["disable_blending"] = True

        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1]) - 1:
                next_item = args[1][i + 1]
            else:
                next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-', '')
                if key not in self.params:  self.params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-', '')
                if key not in self.params: self.params[key] = next_item

        # Starting OpenPose
        #print("1")
        self.opWrapper = op.WrapperPython()
        #print("2")
        self.opWrapper.configure(self.params)
        self.opWrapper.start()
    def tag_person(self, image):
        try:
            datum = op.Datum()
            imageToProcess = image
            #print(image.shape)
            datum.cvInputData = imageToProcess
            self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            # Display Image
            '''print("Body keypoints: \n" + str(datum.poseKeypoints))
            print("Face keypoints: \n" + str(datum.faceKeypoints))
            print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
            print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))'''
            #cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
            image = datum.cvOutputData
            #my_image = imageToProcess
            pose_keypoints = datum.poseKeypoints
            '''for i in datum.poseKeypoints:
                #print("hello")
                #print(i[0][0])
                if (i[0][2] > 0.5):
                    cv2.circle(my_image, (i[0][0], i[0][1]), 12, (0, 0, 255))
            '''
            #cv2.imshow("my_image", datum.cvOutputData)
            #cv2.waitKey(0)
            return pose_keypoints, image
        except Exception as e:
            print(e)
            return None


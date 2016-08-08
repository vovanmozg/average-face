#!/usr/bin/python

# open image
# detect shapes
# crop faces
# detect landmarks
# 
#
#
#


# download http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2



import sys
import os
import dlib
import glob
from skimage import io



predictor_path = '/Users/vovanmozg/Downloads/bigdata/shape_predictor_68_face_landmarks.dat'
face_path = '/Users/vovanmozg/Downloads/bigdata/socialfaces/3_files/images(19).jpg'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
#win = dlib.image_window()

f = face_path

print("Processing file: {}".format(f))
img = io.imread(f)

#win.clear_overlay()
#win.set_image(img)

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
dets = detector(img, 1)
    


print("Number of faces detected: {}".format(len(dets)))
for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, d)
    print(shape.parts)
    print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                              shape.part(1)))



    # Draw the face landmarks on the screen.
    #win.add_overlay(shape)

#win.add_overlay(dets)
#dlib.hit_enter_to_continue()

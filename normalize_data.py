# -*- coding: utf-8 -*-
"""
######################################################################################################################################
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Any publications arising from the use of this software, including but
not limited to academic journal and conference publications, technical
reports and manuals, must cite at least one of the following works:

Revisiting Data Normalization for Appearance-Based Gaze Estimation
Xucong Zhang, Yusuke Sugano, Andreas Bulling
in Proc. International Symposium on Eye Tracking Research and Applications (ETRA), 2018
######################################################################################################################################
"""

import os
import cv2
import numpy as np
import csv
import argparse
import glob

    
def normalize(img, landmarks, focal_norm, distance_norm, roi_size, center, hr, ht, cam, gc=None):
    ## universal function for data normalization
    hR = cv2.Rodrigues(hr)[0] # rotation matrix

    ## ---------- normalize image ----------
    distance = np.linalg.norm(center) # actual distance between eye and original camera

    z_scale = distance_norm/distance
    cam_norm = np.array([
        [focal_norm, 0, roi_size[0]/2],
        [0, focal_norm, roi_size[1]/2],
        [0, 0, 1.0],
    ])
    S = np.array([ # scaling matrix
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])
    
    hRx = hR[:,0]
    forward = (center/distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T # rotation matrix R

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam))) # transformation matrix

    img_warped = cv2.warpPerspective(img, W, roi_size) # image normalization

    ## ---------- normalize rotation ----------
    hR_norm = np.dot(R, hR) # rotation matrix in normalized space
    # hr_norm = cv2.Rodrigues(hR_norm)[0] # convert rotation matrix to rotation vectors

    ## ---------- normalize gaze vector ----------
    gc_normalized = None

    num_point = landmarks.shape[0]
    landmarks_warped = cv2.perspectiveTransform(landmarks.reshape(-1,1,2).astype('float32'), W)
    landmarks_warped = landmarks_warped.reshape(num_point, 2)
    if gc is not None:
        gc_normalized = gc.reshape((3,1)) - center # gaze vector
        # For modified data normalization, scaling is not applied to gaze direction (only R applied).
        # For original data normalization, here should be:
        # "M = np.dot(S,R)
        # gc_normalized = np.dot(R, gc_normalized)"
        gc_normalized = np.dot(R, gc_normalized)
        gc_normalized = gc_normalized/np.linalg.norm(gc_normalized)

    return [img_warped, R, hR_norm, gc_normalized, landmarks_warped]



def read_image(img_path, camera_matrix, camera_distortion):
    # load input image and undistort
    img_original = cv2.imread(img_path)
    img = cv2.undistort(img_original, camera_matrix, camera_distortion)

    return img

def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec
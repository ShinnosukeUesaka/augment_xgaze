import os
import argparse
import math
import numpy as np
import imageio
import scipy.io
import glob
import cv2
import matplotlib.pyplot as plt
import time
from read_xgaze import read_xml,read_csv_as_dict, read_lm_gc
from normalize_data import read_image, normalize, estimateHeadPose
from utils import vector_to_pitchyaw, add, to_h5 #, angular_error, pitchyaw_to_vector
from pathlib import Path

def lm68_to_50(lm_68):
	'''
	lm_68: (68,2)
	'''
	lm_50 = np.zeros((50,2))
	lm_50[0] = lm_68[8]
	lm_50[1:44] = lm_68[17:60]
	lm_50[44:47] = lm_68[61:64]
	lm_50[47:50] = lm_68[65:68]
	return lm_50

def generate_normalized_xgaze(images_folder_path, annotations_folder_path, calibrations_folder_path, subject_number, frame_number, camera_number, roi_size, distance_norm, focal_norm=960):
	subject_name = f"subject{subject_number}"
	frame_name = f"frame{frame_number}"
	camera_name = f"cam{camera_number}"
	
	images_folder_path = Path(images_folder_path)
	annotations_folder_path = Path(annotations_folder_path)
	calibrations_folder_path = Path(calibrations_folder_path)

	csv_path = annotations_folder_path / f"{subject_name}.csv"
	sub_dict = read_csv_as_dict(str(csv_path))

	camera_path = calibrations_folder_path / f"{camera_name}.xml"
	camera_matrix, camera_distortion, camera_translation, camera_rotation = read_xml(str(camera_path))

	image_path = images_folder_path / subject_name / frame_name / f"{camera_name}.JPG"
	image = read_image(str(image_path), camera_matrix, camera_distortion)
 
	#newly added.
	if camera_name in ['cam03', 'cam06','cam13']:
		img = cv2.rotate(image, cv2.ROTATE_180)
  
	lm_gt, gc, _, _ = read_lm_gc(sub_dict, image_path)

	# #  Data Normalization
	# # --------------------------------------------  estimate head pose --------------------------------------------
	face_model_load = np.loadtxt('./face_model.txt')
	use_68 = True 
	'''
	Use 50 landmarks to estimate head pose, or only use 6 landmarks
	'''
	if use_68:
		landmarks_sub = lm68_to_50(lm_gt) 
		landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
		landmarks_sub = landmarks_sub.reshape(50, 1, 2)  # input to solvePnP requires such shape
		facePts = face_model_load.reshape(50, 1, 3)
		hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion, iterate=True)
	else:
		face_model = face_model_load[[20, 23, 26, 29, 15, 19], :]  # the eye and nose landmarks
		facePts = face_model.reshape(6, 1, 3)
		landmarks_sub = lm_gt[[36, 39, 42, 45, 31, 35], :]
		landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
		landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape
		hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion, iterate=True)

	# compute estimated 3D positions of the landmarks
	ht = ht.reshape((3,1))
	hR = cv2.Rodrigues(hr)[0] # rotation matrix
	face_model = face_model_load[[20, 23, 26, 29, 15, 19], :]  # the eye and nose landmarks
	Fc = np.dot(hR, face_model.T) + ht # 3D positions of facial landmarks
	# get the face center
	two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
	nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
	face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

	# -------------------------------------------- normalize image --------------------------------------------
	img_face, R, hR_norm, gaze_norm, landmarks_norm = normalize(img,lm_gt, focal_norm, distance_norm, roi_size, face_center, hr, ht, camera_matrix, gc)
	'''
	img_face: is the normalized face

	R: normalization matrix
	hR_norm: normliazed head rotation matrix
	gaze_norm: normalized gaze direction (vector form)

	landmarks_norm: the landmarks in the normalized face image

	'''	
	face_gaze = vector_to_pitchyaw(-gaze_norm.reshape((1,3))).flatten()
	face_head_pose = np.array([np.arcsin(hR_norm[1, 2]),np.arctan2(hR_norm[0, 2], hR_norm[2, 2])]) #hr_norm
	face_mat_norm = R

	return img_face, landmarks_norm, face_gaze, face_head_pose, face_mat_norm
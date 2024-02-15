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


raw_path = '/home/s-uesaka/datasets/xgazeraw/data/train'
calibration_path = '/home/s-uesaka/datasets/xgazeraw/calibration/cam_calibration'
annatation_path = '/home/s-uesaka/datasets/xgazeraw/data/annotation_train'
output_path = '/home/s-uesaka/datasets/normalized_test'

# Normalization parameters
focal_norm = 960 # focal length of normalized camera
distance_norm = 280 # normalized distance between eye and camera
roi_size = (704, 704) # size of cropped eye image


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


def main(args, sub_dict):

	image_list = sorted(glob.glob(args.frame_path + '/' + '*.JPG'))
	for input_path in image_list:
		img_name = os.path.basename(input_path) 
		frame = input_path.split('/')[-2]
		subject = input_path.split('/')[-3]

		camera_path = os.path.join(calibration_path , img_name.replace('.JPG','.xml'))
		camera_matrix, camera_distortion, camera_translation, camera_rotation = read_xml(camera_path)

		img_path = os.path.join(raw_path, subject, frame, img_name)
		img = read_image(img_path, camera_matrix, camera_distortion)

		if img_name in ['cam03.JPG', 'cam06.JPG','cam13.JPG']:
			img = cv2.rotate(img, cv2.ROTATE_180)
   
		lm_gt, gc, _, _ = read_lm_gc(sub_dict, os.path.join(frame,img_name))

		# #  Data Normalization
		# # --------------------------------------------  estimnate head pose --------------------------------------------
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
		hr_norm = np.array([np.arcsin(hR_norm[1, 2]),np.arctan2(hR_norm[0, 2], hR_norm[2, 2])])
		to_write = {}
		add(to_write, 'face_patch', img_face)
		add(to_write, 'frame_index', int(frame[-4:]) )
		add(to_write, 'cam_index', int(img_name[-6:-4])+1 )
		add(to_write, 'face_gaze', vector_to_pitchyaw(-gaze_norm.reshape((1,3))).flatten())
		add(to_write, 'face_head_pose', hr_norm)
		add(to_write, 'face_mat_norm', R)
		add(to_write, 'landmarks_norm', landmarks_norm)

		to_h5(to_write, args.save_path)





if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()

	current_dir = os.path.dirname(os.path.realpath(__file__))
	subject_list = sorted(glob.glob(raw_path + '/' + 'subject*'))
	print('total subjects: ', len(subject_list))

	for subject_path in subject_list[:]:
		frame_list = sorted(glob.glob(subject_path + '/' + 'frame*'))
		subject = subject_path.split('/')[-1]
		csv_path = os.path.join(annatation_path, subject+'.csv')

		# read csv and save it as a dictionary to save time (read csv every time is slow)
		sub_dict = read_csv_as_dict(csv_path)
		for frame_path in frame_list:
			print(frame_path + 'of total {} frame '.format(len(frame_list)))
			frame = frame_path.split('/')[-1]
			config, _ = parser.parse_known_args()
			config.frame_path =  os.path.join(raw_path, subject, frame)
			config.save_path = os.path.join(output_path, subject + '.h5')
			main(config, sub_dict)

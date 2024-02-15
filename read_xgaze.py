import csv
import os
import numpy as np
import scipy.io
import cv2

def read_csv_as_dict(csv_path):
	with open(csv_path, newline='') as csvfile:
		data = csvfile.readlines()
	reader = csv.reader(data)
	sub_dict = {}
	for row in reader:
		frame = row[0]
		cam_index = row[1]
		sub_dict[frame+'/'+cam_index] = row[2:]
	return sub_dict
    
def read_lm_gc(sub_dict, index):
    gaze_point_screen = np.array([int(float(i)) for i in sub_dict[index][0:2]])
    gaze_point_cam = np.array([float(i) for i in sub_dict[index][2:5]])
    head_rotation_cam = np.array([float(i) for i in sub_dict[index][5:8]])
    head_translation_cam = np.array([float(i) for i in sub_dict[index][8:11]])
    lm_2d = np.array([int(float(i)) for i in sub_dict[index][11:]]).reshape(68,2)
    return lm_2d, gaze_point_cam, head_rotation_cam, head_translation_cam


def read_xml(xml_path):
    if not os.path.isfile(xml_path):
        print('no camera calibration file is found.')
        exit(0)
    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
    camera_distortion = fs.getNode('Distortion_Coefficients').mat()
    camera_translation = fs.getNode('cam_translation').mat()
    camera_rotation = fs.getNode('cam_rotation').mat()
    return camera_matrix, camera_distortion, camera_translation, camera_rotation

def draw_lm(image, landmarks, color= (0, 0, 255),  print_idx=False):
	i = 0
	for x,y in landmarks:
		# Radius of circle
		radius = 20
		# Line thickness of 2 px
		thickness = -1
		image = cv2.circle(image, (int(x), int(y)), radius, color, thickness)
		if print_idx:
			image = cv2.putText(image,
				text=str(i),
				org=(int(x), int(y)),
				fontFace=cv2.FONT_HERSHEY_SIMPLEX,
				fontScale=2.0,
				color=color,
				thickness=2,
				lineType=cv2.LINE_4)
		
		i += 1
	return image
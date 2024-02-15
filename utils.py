import os
import numpy as np
import imageio
import cv2
import h5py
from PIL import Image
import dlib
from imutils import face_utils
import cv2


def pitchyaw_to_vector(pitchyaws):
	r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

	Args:
		pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

	Returns:
		:obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
	"""
	n = pitchyaws.shape[0]
	sin = np.sin(pitchyaws)
	cos = np.cos(pitchyaws)
	out = np.empty((n, 3))
	out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
	out[:, 1] = sin[:, 0]
	out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
	return out
	
def vector_to_pitchyaw(vectors):
	"""Convert given gaze vectors to pitch (theta) and yaw (phi) angles."""
	n = vectors.shape[0]
	out = np.empty((n, 2))
	vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
	out[:, 0] = np.arcsin(vectors[:, 1])  # theta
	out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
	return out

def angular_error(a, b):
	"""Calculate angular error (via cosine similarity)."""
	a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
	b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

	ab = np.sum(np.multiply(a, b), axis=1)
	a_norm = np.linalg.norm(a, axis=1)
	b_norm = np.linalg.norm(b, axis=1)

	# Avoid zero-values (to avoid NaNs)
	a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
	b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

	similarity = np.divide(ab, np.multiply(a_norm, b_norm))

	return np.arccos(similarity) * 180.0 / np.pi


def add(to_write, key, value):  # noqa
	if key not in to_write:
		to_write[key] = [value]
	else:
		to_write[key].append(value)

def to_h5(to_write, output_path):
	for key, values in to_write.items():
		to_write[key] = np.asarray(values)
		# print('%s: ' % key, to_write[key].shape)
	
	if not os.path.isfile(output_path):
		with h5py.File(output_path, 'w', libver='latest') as f:
			for key, values in to_write.items():
				print("values.shape: ", values.shape)
				f.create_dataset(
					key, data=values,
					chunks=(
						tuple([1] + list(values.shape[1:]))
						if isinstance(values, np.ndarray)
						else None
					),
					compression='lzf',
					maxshape=tuple([None] + list(values.shape[1:])),
				)
				print("chunks: ", f[key].chunks)
	else:
		with h5py.File(output_path, 'a', libver='latest') as f:
			for key, values in to_write.items():
				if key not in list(f.keys()):
					print('write it to f {}'.format(output_path))
					f.create_dataset(
						key, data=values,
						chunks=(
							tuple([1] + list(values.shape[1:]))
							if isinstance(values, np.ndarray)
							else None
						),
						compression='lzf',
						maxshape=tuple([None] + list(values.shape[1:])),
					)
				else:
					data = f[key]
					data.resize(data.shape[0] + values.shape[0], axis=0)
					data[-values.shape[0]:] = values
     

def crop_center(pil_img, crop_width, crop_height):
        img_width, img_height = pil_img.size
        return pil_img.crop(((img_width - crop_width) // 2,
                            (img_height - crop_height) // 2,
                            (img_width + crop_width) // 2,
                            (img_height + crop_height) // 2))
        
def convert_img(img, source_roi: float, source_focal: float, source_distance: float, target_roi: float, target_focal: float, target_distance: float):
    """Only supports square image."""
    img = Image.fromarray(img)
    source_face_resolution = source_focal/source_distance
    target_face_resolution = target_focal/target_distance
    target_face_ratio = target_face_resolution/target_roi
    
    crop_size = source_face_resolution/target_face_ratio
    cropped_image = crop_center(img, crop_size,crop_size)
    return np.array(cropped_image.resize((target_roi, target_roi)))


def get_lm(image: Image):
	p = "shape_predictor_68_face_landmarks.dat"
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(p)

	gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)

	if len(rects) == 0:
		rect = dlib.rectangle(0,0,image.size[0], image.size[1])
	else:
		rect = rects[0]

	lm = predictor(gray, rect)
	lm = face_utils.shape_to_np(lm)
	return lm

def eye_squares_are_valid(cam_index, eye_squares, image_size, face_resolution):
	"""note that cam_index starts from 1 to 18. not 0 to 17
	returns False when there is an anomaly"""
	threshold_array = [[-0.07259259, -0.05703704],
       [-0.07648148, -0.05314815],
       [-0.07907407, -0.06351852],
       [-0.07259259, -0.04666667],
       [-0.05703704, -0.02203704],
       [-0.05962963, -0.02203704],
       [-0.06611111, -0.04018519],
       [-0.07907407, -0.05314815],
       [-0.07907407, -0.06351852],
       [-0.07648148, -0.05962963],
       [-0.07907407, -0.05962963],
       [-0.07907407, -0.06611111],
       [-0.04018519, -0.01296296],
       [-0.04018519, -0.00777778],
       [-0.02722222, -0.0012963 ],
       [-0.07907407, -0.05314815],
       [-0.07907407, -0.06611111],
       [-0.07907407, -0.05314815]]
	eye_line = (np.average(eye_squares[0], axis=0)[1] + np.average(eye_squares[1], axis=0)[1])/2
	face_size = face_resolution*225
	image_center = image_size[1]/2
	upper_limit = image_center + face_size*threshold_array[cam_index-1][0]
	lower_limit = image_center + face_size*threshold_array[cam_index-1][1]
	print(f"Lower: {lower_limit}, Upper: {upper_limit}")
	return upper_limit < eye_line and eye_line < lower_limit, lower_limit, upper_limit

def get_squares_cv2(image: Image):
    src = np.array(image, dtype=np.uint8)[:,:,::-1]

    eye_cascade_path = './haarcascade_eye.xml'

    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    src = np.array(src)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(src_gray)

    squares = []
    for x, y, w, h in eyes:
        squares.append(((x,y),(x+w,y+h)))
    return squares

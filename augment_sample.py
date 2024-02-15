from normalize_xgaze import generate_normalized_xgaze
from stable_diffusion_utils import DepthToImage, Inpainting
from PIL import Image, ImageDraw
import numpy as np


def getsquare(eye_left_edge,eye_right_edge,scale_horizontal=1.2,scale_vertical=0.7):
    eye_center = ((eye_left_edge[0]+eye_right_edge[0])/2, (eye_left_edge[1]+eye_right_edge[1])/2)
    eye_size = eye_right_edge[1] - eye_left_edge[0]
    eye_size = abs(eye_size)
    start_point = (int(eye_center[0] - eye_size*scale_horizontal/2), int(eye_center[1] - eye_size*scale_vertical/2))
    end_point = (int(eye_center[0] + eye_size*scale_horizontal/2), int(eye_center[1] + eye_size*scale_vertical/2))
    return start_point, end_point


def gets_eye_squares(landmarks, scale_horizontal=1.3, scale_vertical=0.9):
    left_eye_landmarks = landmarks[36:42]
    right_eye_landmarks = landmarks[42:48]

    return get_eye_square_from_eye_landmarks(left_eye_landmarks, scale_horizontal, scale_vertical), get_eye_square_from_eye_landmarks(right_eye_landmarks,scale_horizontal,scale_vertical)


def get_eye_square_from_eye_landmarks(eye_landmarks, scale_horizontal=1.5, scale_vertical=1.5):
    eye_landmarks = np.array(eye_landmarks)
    eye_center =  np.average(eye_landmarks, axis=0)
    eye_size = np.average(np.absolute(eye_landmarks - eye_center), axis=0)
    eye_size = [eye_size[0]*scale_horizontal, eye_size[1]*scale_vertical]
    
    start_point = (eye_center[0]- eye_size[0], eye_center[1] - eye_size[1])
    end_point = (eye_center[0] + eye_size[0], eye_center[1] + eye_size[1])
    
    return (start_point, end_point)
    


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                        (img_height - crop_height) // 2,
                        (img_width + crop_width) // 2,
                        (img_height + crop_height) // 2))



class SampleGenerator():
    def __init__(self):
        self.depth2image_model = DepthToImage()
        self.inpainting_model = Inpainting()

    def generate_augmentated_image(self, image_path="/home/s-uesaka/datasets/xgazeraw/data/train/subject0000/frame0001/cam05.JPG"):
        ROI_SIZE = 704
        FINAL_ROI_SIZE = 480
        TRAINING_IMG_SIZE = 224
        DISTANCE_NORM=280
        

        
        image,lm_gt_normalized, _, _, _, = generate_normalized_xgaze(image_path,
                                            roi_size=(ROI_SIZE,ROI_SIZE),
                                            distance_norm=DISTANCE_NORM,
                                            calibration_path = '/home/s-uesaka/datasets/xgazeraw/calibration/cam_calibration',
                                                annotation_path = '/home/s-uesaka/datasets/xgazeraw/data/annotation_train')
        image = Image.fromarray(image[:,:,::-1])
        
        generated_image = self.depth2image_model.generate_image(image, prompt = "photo of a person's face", seed=0, scale=9, steps=50, strength=0.6)    
        
        left_eye_square = self.getsquare(lm_gt_normalized[36], lm_gt_normalized[39])
        img_left_eye = image.crop([left_eye_square[0][0], left_eye_square[0][1],left_eye_square[1][0], left_eye_square[1][1]])
        
        right_eye_square = self.getsquare(lm_gt_normalized[42], lm_gt_normalized[45])
        img_right_eye = image.crop([right_eye_square[0][0], right_eye_square[0][1],right_eye_square[1][0], right_eye_square[1][1]])
        
        generated_image.paste(img_left_eye, left_eye_square[0])
        generated_image.paste(img_right_eye, right_eye_square[0])
        
        mask = Image.new("RGB", (ROI_SIZE,ROI_SIZE), (0, 0, 0))
        rect_d = ImageDraw.Draw(mask)
        rect_d.rectangle(
            right_eye_square, outline=(255, 255, 255), width=10
        )
        rect_d.rectangle(
            left_eye_square, outline=(255, 255, 255), width=10
        )
    
        inpainted_image = self.inpainting_model.generate_image(generated_image, mask)
        
        
        augmented_image = self.crop_center(inpainted_image, FINAL_ROI_SIZE, FINAL_ROI_SIZE)
        augmented_image = augmented_image.resize((TRAINING_IMG_SIZE,TRAINING_IMG_SIZE))
        
        intermediate_results = {
            'normalized': image,
            'depth': generated_image,
            'inpainted': inpainted_image,
            'cropped': self.crop_center(inpainted_image, FINAL_ROI_SIZE, FINAL_ROI_SIZE)
        }
        
        return augmented_image, intermediate_results
    
    @classmethod
    def generate_not_augmented(cls, image_path="/home/s-uesaka/datasets/xgazeraw/data/train/subject0000/frame0001/cam05.JPG"):
        ROI_SIZE = 224
        DISTANCE_NORM=600
        image,lm_gt_normalized, _, _, _, = generate_normalized_xgaze(image_path,
                                            roi_size=(ROI_SIZE,ROI_SIZE),
                                            distance_norm=DISTANCE_NORM,
                                            calibration_path = '/home/s-uesaka/datasets/xgazeraw/calibration/cam_calibration',
                                                annotation_path = '/home/s-uesaka/datasets/xgazeraw/data/annotation_train')
        return Image.fromarray(image[:,:,::-1])
        
    
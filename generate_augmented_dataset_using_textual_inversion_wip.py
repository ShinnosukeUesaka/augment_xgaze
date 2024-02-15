from normalize_xgaze import generate_normalized_xgaze
from stable_diffusion_utils import DepthToImage, Inpainting
from diffusers import StableDiffusionDepth2ImgPipeline, StableDiffusionInpaintPipeline
from transformers import CLIPTextModel
from PIL import Image, ImageDraw
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import shutil
from utils import convert_img, get_lm, eye_squares_are_valid, get_squares_cv2
import random
from utils import add, to_h5
import time
import h5py
import face_alignment
import torch
import json
import copy


@dataclass
class NormalizationParameters():
    roi_size: int
    focal_length: int
    distance: int
    
    @property
    def face_resolution(self):
        return self.focal_length/self.distance
    
    @property 
    def face_ratio(self):
        return self.face_resolution/self.roi_size


def get_eye_square_from_eye_landmarks(eye_landmarks, scale_horizontal=3.1, scale_vertical=3.7, min_square_size=(40,35), max_square_size=(120,120), horizontal_offset=0.1):
    eye_landmarks = np.array(eye_landmarks)
    eye_center =  np.average(eye_landmarks, axis=0).astype(int)
    eye_size = np.average(np.absolute(eye_landmarks - eye_center), axis=0) #eye_size = squre_size/2 (eye size is the size measured from the eye center.)
    eye_size = np.array([eye_size[0]*scale_horizontal, eye_size[1]*scale_vertical]).astype(int)
    eye_size = np.maximum(eye_size, np.array(min_square_size)/2).astype(int)
    eye_size = np.minimum(eye_size, np.array(max_square_size)/2).astype(int)
    
    eye_center = np.array([eye_center[0], eye_center[1]+eye_size[1]*horizontal_offset]).astype(int)
    
    start_point = (eye_center[0]- eye_size[0], eye_center[1] - eye_size[1])
    end_point = (eye_center[0] + eye_size[0], eye_center[1] + eye_size[1])
    
    return (start_point, end_point)

def get_eye_squares(landmarks, scale_horizontal=3.3, scale_vertical=3, min_square_size=[40,50], max_square_size=[150,100], horizontal_offset=0.1):
    left_eye_landmarks = landmarks[36:42]
    right_eye_landmarks = landmarks[42:48]

    return get_eye_square_from_eye_landmarks(left_eye_landmarks, scale_horizontal, scale_vertical, min_square_size, max_square_size, horizontal_offset), get_eye_square_from_eye_landmarks(right_eye_landmarks,scale_horizontal,scale_vertical, min_square_size, max_square_size, horizontal_offset)



    
    
    
    
class ImageAugmentor():
    def __init__(self, depth2image_model_path="stabilityai/stable-diffusion-2-depth", text_encoder_model_path=None):
        self.lm_detection_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')
       
        if text_encoder_model_path is None:
            self.depth2image_model = StableDiffusionDepth2ImgPipeline.from_pretrained(
                                        depth2image_model_path,
                                        torch_dtype=torch.float16,
                                    ).to("cuda")
        else:
            text_encoder = CLIPTextModel.from_pretrained(
                text_encoder_model_path, subfolder="text_encoder"
            )
            self.depth2image_model = StableDiffusionDepth2ImgPipeline.from_pretrained(
                depth2image_model_path,
                text_encoder=text_encoder,
                #torch_dtype=torch.float16,
            ).to("cuda")
        
        self.inpainting_model = StableDiffusionInpaintPipeline.from_pretrained(
                                    "stabilityai/stable-diffusion-2-inpainting",
                                    torch_dtype=torch.float16
                                ).to("cuda")
        
        self.generator = torch.Generator(device='cuda')

    def generate_augmentated_image(self, image: Image, landmarks, stablediffusion_config={'prompt': "photo of a person's face", "negative_prompt": "deformed, bad anotomy", "seed": 0, "scale": 9, "steps": 50, "strength": 0.7}):
        
        self.generator.manual_seed(stablediffusion_config['seed'])
        
        generated_image = self.depth2image_model(image = image,
                                                prompt = stablediffusion_config['prompt'],
                                                negative_prompt = stablediffusion_config['negative_prompt'],
                                                guidance_scale = stablediffusion_config['scale'],
                                                num_inference_steps = stablediffusion_config['steps'],
                                                strength = stablediffusion_config['strength'],
                                                generator=self.generator).images[0]
        
        after_depth = copy.deepcopy(generated_image)
                
        #Will use official landmarks from the datasets since the quality is bad. Recalculate it using neural based model.
        
        recalculated_landmarks = self.lm_detection_model.get_landmarks(np.array(image))
        
        if recalculated_landmarks is None:
            recalculated_landmarks = landmarks
        else:
            recalculated_landmarks = recalculated_landmarks[0]
            
        left_eye_square, right_eye_square = get_eye_squares(recalculated_landmarks, scale_horizontal=3.3, scale_vertical=4, min_square_size=[40,45], max_square_size=[160,90], horizontal_offset=0.1)
        
        img_left_eye = image.crop([left_eye_square[0][0], left_eye_square[0][1],left_eye_square[1][0], left_eye_square[1][1]])
        
        img_right_eye = image.crop([right_eye_square[0][0], right_eye_square[0][1],right_eye_square[1][0], right_eye_square[1][1]])
        
        generated_image.paste(img_left_eye, left_eye_square[0])
        generated_image.paste(img_right_eye, right_eye_square[0])
        
        
        mask = Image.new("RGB", image.size, (0, 0, 0))
        rect_d = ImageDraw.Draw(mask)
        rect_d.rectangle(
            right_eye_square, outline=(255, 255, 255), width=9
        )
        rect_d.rectangle(
            left_eye_square, outline=(255, 255, 255), width=9
        )
        
        inpainted_image = self.inpainting_model(prompt=stablediffusion_config['prompt'],
                                                negative_prompt = stablediffusion_config['negative_prompt'],
                                                image=generated_image,
                                                mask_image=mask,
                                                num_inference_steps=25,
                                                generator=self.generator).images[0]
        
        return after_depth, generated_image, inpainted_image, mask


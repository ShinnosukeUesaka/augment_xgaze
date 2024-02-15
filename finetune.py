import os, sys
import os.path as osp
sys.path.append( osp.dirname(osp.realpath(__file__)) + '/stablediffusion')

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

import cv2


class GeneratePipeline(object):

     def __init__(self, 
            depth2image_model_path="stabilityai/stable-diffusion-2-depth", 
            text_encoder_model_path=None,

        ):

        if text_encoder_model_path is None:
            self.depth2image_model = StableDiffusionDepth2ImgPipeline.from_pretrained(
                                        depth2image_model_path,
                                        torch_dtype=torch.float16,
                                    ).to("cuda")
        else:
            text_encoder = CLIPTextModel.from_pretrained(text_encoder_model_path, subfolder="text_encoder"
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

    
class ImageAugmentor():
    def __init__(self, depth2image_model_path="stabilityai/stable-diffusion-2-depth", text_encoder_model_path=None):

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
        
        return inpainted_image


if __name__ == "__main__":
    start_time = time.time()

    dataset_dir = '/work/jqin/Datasets/xgaze_448_v2'
    output_dir = '/work/jqin/CVPR24/augment-xgaze/outputs'
    generate_augmented_dataset(dataset_dir=Path(dataset_dir),
                               output_dir=Path(output_dir),
                               source_normalization_parameters=NormalizationParameters(roi_size=512, distance=300, focal_length=750),
                               target_normalization_parameters = NormalizationParameters(roi_size=224, distance=600, focal_length=960),
                               sample_ratio=1,
                               cam_index_to_use=[1,2],
                               stablediffusion_config={'prompt': "photo of a person's face", "negative_prompt": "deformed, bad anotomy", "seed": 0, "scale": 9, "steps": 50, "strength": 1},
                               prompts_file_path=None)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
    
    
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





def generate_augmented_dataset(dataset_dir: Path,
                               output_dir: Path,
                               source_normalization_parameters: NormalizationParameters,
                               target_normalization_parameters: NormalizationParameters = NormalizationParameters(roi_size=224, focal_length=960, distance=600),
                               sample_ratio = 1,
                               cam_index_to_use=[1,2],
                               stablediffusion_config={'prompt': "photo of a person's face", "negative_prompt": "deformed, bad anotomy", "seed": 0, "scale": 9, "steps": 50, "strength": 0.7},
                               prompts_file_path: Path = None):
    
    
    image_augmentor = ImageAugmentor()

    MIN_SEED = 1000000000
    MAX_SEED = 10000000000
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for h5_path in dataset_dir.iterdir():
        
        if h5_path.suffix != '.h5':
            continue
        
        output_path = output_dir / h5_path.name
        
        if output_path.is_file(): #if the file already exists, then skip. (This is a feature that allows you to continue the process after interuption.)
            continue
        

        with h5py.File(h5_path, 'r', swmr=True) as source_f:
            
            print(" all keys: ", list(source_f.keys()) )
            number_of_samples = len(source_f['cam_index'])

            
            print(f"processing subject:{h5_path.name}, Number of images: {number_of_samples}")
            
            for index, cam_index, face_gaze, face_head_pose, face_mat_norm, face_patch, frame_index, landmarks_norm in zip(
                                                        range(number_of_samples),
                                                        source_f['cam_index'],
                                                        source_f['face_gaze'],
                                                        source_f['face_head_pose'],
                                                        source_f['face_mat_norm'],
                                                        source_f['face_patch'],
                                                        source_f['frame_index'],
                                                        source_f['landmarks_norm'],):
                if index % 18 not in cam_index_to_use:
                    continue

            # cam_index_data = source_f['cam_index'][:]

            # cam_index_selected = np.isin(cam_index_data, cam_index_to_use)
            
            # number_of_samples = int(np.count_nonzero(cam_index_selected)*sample_ratio)
            # sample_index = random.sample(sorted(np.where(cam_index_selected)[0]), number_of_samples)
            
            
            # print(f"processing subject:{h5_path.name}, Number of images: {number_of_samples}")
            
            # for index, cam_index, face_gaze, face_head_pose, face_mat_norm, face_patch, frame_index, landmarks_norm in zip(
            #                                             range(len(cam_index_data)),
            #                                             source_f['cam_index'],
            #                                             source_f['face_gaze'],
            #                                             source_f['face_head_pose'],
            #                                             source_f['face_mat_norm'],
            #                                             source_f['face_patch'],
            #                                             source_f['frame_index'],
            #                                             source_f['landmarks_norm'],):
            #     if index not in sample_index:
            #         continue
                
                image = Image.fromarray(face_patch[:,:,::-1])

                
                print(f"processing subject:{h5_path.name} Frame: {frame_index} Cam {cam_index}")
                
                
                if prompts_file_path is not None:
                    with open(prompts_file_path, 'r') as f:
                        prompts = json.load(f)
                        
                    stablediffusion_config['prompt'] = random.choice(prompts)['prompt']
                
                
                stablediffusion_config['seed'] = random.randint(MIN_SEED, MAX_SEED)

                
                augmented_image, before_painted_image =  image_augmentor.generate_augmentated_image(image,
                                                                              landmarks_norm,
                                                                              stablediffusion_config, return_generated_image=True)
                
                augmented_image = np.array(augmented_image)[:,:,::-1]
                
            
                # face_patch = convert_img(face_patch,
                #                         source_roi=source_normalization_parameters.roi_size,
                #                         source_focal=source_normalization_parameters.focal_length,
                #                         source_distance=source_normalization_parameters.distance,
                                        
                #                         target_roi=target_normalization_parameters.roi_size,
                #                         target_distance=target_normalization_parameters.distance,
                #                         target_focal=target_normalization_parameters.focal_length)
                
                # augmented_face_patch =  convert_img(augmented_image,
                #                         source_roi=source_normalization_parameters.roi_size,
                #                         source_focal=source_normalization_parameters.focal_length,
                #                         source_distance=source_normalization_parameters.distance,
                                        
                #                         target_roi=target_normalization_parameters.roi_size,
                #                         target_distance=target_normalization_parameters.distance,
                #                         target_focal=target_normalization_parameters.focal_length)


                face_patch = np.array(face_patch)
                before_painted_face_patch = np.array(before_painted_image)
                augmented_face_patch =  np.array(augmented_image) 

                to_write = {}
                add(to_write, 'face_patch', face_patch)
                add(to_write, 'augmented_face_patch', [augmented_face_patch])
                add(to_write, 'frame_index', frame_index )
                add(to_write, 'cam_index', cam_index )
                add(to_write, 'face_gaze', face_gaze)
                add(to_write, 'face_head_pose', face_head_pose)
                add(to_write, 'face_mat_norm', face_mat_norm)
                add(to_write, 'landmarks_norm', landmarks_norm)
                

                face_patch = face_patch.astype(np.uint8)
                augmented_face_patch = augmented_face_patch.astype(np.uint8)
                before_painted_face_patch = before_painted_face_patch.astype(np.uint8)
                augmented_face_patch = cv2.resize(augmented_face_patch, (face_patch.shape[1], face_patch.shape[0])  )
                before_painted_face_patch = cv2.resize(before_painted_face_patch, (face_patch.shape[1], face_patch.shape[0])  )
                im_show = cv2.hconcat([face_patch, augmented_face_patch, before_painted_face_patch])
                cv2.imwrite( osp.join(output_dir, f"{h5_path.name.split('.')[0]}_{index}.jpg"), im_show )
                # to_h5(to_write, output_path)
    
    
    
    
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

    def generate_augmentated_image(self, 
        image: Image, 
        landmarks, 
        stablediffusion_config={'prompt': "photo of a person's face", "negative_prompt": "deformed, bad anotomy", "seed": 0, "scale": 9, "steps": 50, "strength": 0.7},
        return_generated_image=False
        ):
        
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
        
        if return_generated_image:
            return inpainted_image, generated_image
        else:
            return inpainted_image


if __name__ == "__main__":
    start_time = time.time()

    dataset_dir = '/work/jqin/Datasets/xgaze_512'
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
    
    
    
    

# def get_best_square_using_references(candidate_squares, reference_squares):
    
#     candidate_squares_np = np.array(candidate_squares)
#     reference_squares = np.array(reference_squares)
#     candidate_centers = np.average(candidate_squares_np, axis=1)
#     reference_centers = np.average(reference_squares, axis=1)
#     candidate_centers_from_references = []
#     for candidate_center in candidate_centers:
#         candidate_centers_from_references.append(np.array([candidate_center - reference_center for reference_center in reference_centers]))
    
#     candidate_centers_from_references = np.array(candidate_centers_from_references)
#     candidate_length_from_references = np.linalg.norm(candidate_centers_from_references, axis=2)
#     candidate_min_length_from_references = np.min(candidate_length_from_references, axis=1)
#     best_candidate_index = np.argmin(candidate_min_length_from_references)
#     return candidate_squares[best_candidate_index]

# def get_best_eye_squares(image, lm, scale_horizontal=3.3, scale_vertical=4, min_square_size=[40,45], max_square_size=[160,90], horizontal_offset=0.1):
#     squares = get_eye_squares(lm, scale_horizontal, scale_vertical, min_square_size, max_square_size, horizontal_offset)
#     recalculated_lm = get_lm(image)
#     recalculated_squares = get_eye_squares(recalculated_lm, scale_horizontal, scale_vertical, min_square_size, max_square_size, horizontal_offset)
#     reference_squares =  get_squares_cv2(image)
#     if len(reference_squares) == 0:
#         return squares
#     return get_best_square_using_references([squares[0], recalculated_squares[0]],reference_squares), get_best_square_using_references([squares[1], recalculated_squares[1]],reference_squares)


#print file sizes in a directory. (do not print sum, print each file size)
directory = Path('directory')
# print(sum(f.stat().st_size for f in directory.glob('**/*') if f.is_file())) #sum of file size in a directory
# print(sum(f.stat().st_size for f in directory.glob('**/*') if f.is_file())/1024/1024/1024) #sum of file size in a directory in GB

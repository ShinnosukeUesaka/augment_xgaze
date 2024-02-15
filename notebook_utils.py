from pathlib import Path
from PIL import Image, ImageDraw
import h5py
import random
import cv2
import numpy as np

def get_normalized_image_and_lm(dataset_dir: Path = Path("/work/jqin/diffusion_iccv/xgaze_512/train"), subject_number: int = 0, image_index = 0, random_sample=False, max_image_index=100):
    max_try = 20
    tries = 0
    
    if random_sample:
        while True: 
            image_index = random.randint(0,max_image_index)
            subject_path = random.choice(list(dataset_dir.iterdir()))
            
            if subject_path.suffix == ".h5":
                break
            if tries > max_try:
                raise Exception
            tries += 1
    else:
        subject_path = dataset_dir / f"subject{subject_number:04}.h5"
    assert subject_path.is_file()
    print(subject_path)
    with h5py.File(subject_path, 'r', libver='latest', swmr=True) as f:
        image = f['face_patch'][image_index]
        landmark = (f['landmarks_norm'][image_index])
        print(f"Image Index: {image_index}, Path: {subject_path} CameraIndex: {f['cam_index'][image_index]}")

        
    return Image.fromarray(image[:,:,::-1]), landmark


def get_normalized_image_mpii(dataset_dir: Path = Path("/work/jqin/diffusion_iccv/xgaze_512/train"), subject_number: int = 0, image_index = 0, random_sample=False, max_image_index=100):
    max_try = 20
    tries = 0
    
    if random_sample:
        while True: 
            image_index = random.randint(0,max_image_index)
            subject_path = random.choice(list(dataset_dir.iterdir()))
            
            if subject_path.suffix == ".h5":
                break
            if tries > max_try:
                raise Exception
            tries += 1
    else:
        subject_path = dataset_dir / f"subject{subject_number:04}.h5"
    assert subject_path.is_file()
    print(subject_path)
    with h5py.File(subject_path, 'r', libver='latest', swmr=True) as f:
        image = f['face_patch'][image_index]
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        image = np.uint8(image)
        print(f"Image Index: {image_index}, Path: {subject_path}")

        
    return Image.fromarray(image)

def get_augmented_image_and_lm(dataset_dir: Path = Path("/work/s-uesaka/xgaze_224_augmented/train"), subject_number: int = 0, image_index = 0, random_sample=False, max_image_index=50):
    max_try = 20
    tries = 0
    
    if random_sample:
        while True: 
            image_index = random.randint(0,max_image_index)
            subject_path = random.choice(list(dataset_dir.iterdir()))
            
            if subject_path.suffix == ".h5":
                break
            if tries > max_try:
                raise Exception
            tries += 1
        print(subject_path)
    else:
        subject_path = dataset_dir / f"subject{subject_number:04}.h5"
         
    with h5py.File(subject_path, 'r', libver='latest', swmr=True) as f:
        image = f['face_patch'][image_index]
        augmented_image = f['augmented_face_patch'][(image_index,0)]
        landmark = (f['landmarks_norm'][(image_index)])
        
    return Image.fromarray(image[:,:,::-1]), Image.fromarray(augmented_image[:,:,::-1]), landmark

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def draw_lm(image: Image, landmarks, color= (0, 0, 255),  print_idx=False):
    image = np.array(image)
    i = 0
    for x,y in landmarks:
        # Radius of circle
        radius = 10
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
                thickness=1,
                lineType=cv2.LINE_4)
        
        i += 1
    return Image.fromarray(image)

def draw_horizontal_line(image: Image, line_cordinate):
    img = image.copy()
    draw = ImageDraw.Draw(img) 
    draw.line((0 ,line_cordinate, image.size[0], line_cordinate), fill=128)
    return img
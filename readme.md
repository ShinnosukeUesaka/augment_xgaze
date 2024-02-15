# This is a normalization code and augmentation code specific for ETH-XGaze

modify the code in `generate_augmented_dataset.py` line 244 ~ 251:

change these values according to your use cases. If prompts_file_path is provided, it will override stablediffusion_config['prompt'].
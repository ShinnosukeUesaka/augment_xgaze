import sys
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat, rearrange
from pytorch_lightning import seed_everything
# from imwatermark import WatermarkEncoder

# from stablediffusion.scripts.txt2img import put_watermark
from stablediffusion.ldm.util import instantiate_from_config
from stablediffusion.ldm.models.diffusion.ddim import DDIMSampler
from stablediffusion.ldm.data.util import AddMiDaS
import cv2

torch.set_grad_enabled(False)

class DepthToImage():
    def __init__(self):
        self.sampler = self.initialize_model('stablediffusion/configs/stable-diffusion/v2-midas-inference.yaml','stablediffusion/models/512-depth-ema.ckpt')
            
    def initialize_model(self, config, ckpt):
        config = OmegaConf.load(config)
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        sampler = DDIMSampler(model)
        return sampler

    def generate_image(self, image: Image, prompt = "photo of a person's face", seed=0, scale=9, steps=50, strength=0.6):
        #image = Image.fromarray(image)
        w, h = image.size
        print(f"loaded input image of size ({w}, {h})")
        width, height = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        image = image.resize((width, height))
        print(f"resized input image to size ({width}, {height} (w, h))")

        

        #num_samples = st.number_input("Number of Samples", min_value=1, max_value=64, value=1)
        #scale = st.slider("Scale", min_value=0.1, max_value=30.0, value=9.0, step=0.1)
        #steps = st.slider("DDIM Steps", min_value=0, max_value=50, value=50, step=1)
        #strength = st.slider("Strength", min_value=0., max_value=1., value=0.9)


        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        do_full_sample = strength == 1.
        t_enc = min(int(strength * steps), steps-1)
        
        self.sampler.make_schedule(steps, ddim_eta=0., verbose=True)
        result = self.paint(
            sampler=self.sampler,
            image=image,
            prompt=prompt,
            t_enc=t_enc,
            seed=seed,
            scale=scale,
            num_samples=1,
            do_full_sample=do_full_sample,
        )
        return result[0]

    def make_batch_sd(
            self,
            image,
            txt,
            device,
            num_samples=1,
            model_type="dpt_hybrid"
    ):
        image = np.array(image.convert("RGB"))
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
        # sample['jpg'] is tensor hwc in [-1, 1] at this point
        midas_trafo = AddMiDaS(model_type=model_type)
        batch = {
            "jpg": image,
            "txt": num_samples * [txt],
        }
        batch = midas_trafo(batch)
        batch["jpg"] = rearrange(batch["jpg"], 'h w c -> 1 c h w')
        batch["jpg"] = repeat(batch["jpg"].to(device=device), "1 ... -> n ...", n=num_samples)
        batch["midas_in"] = repeat(torch.from_numpy(batch["midas_in"][None, ...]).to(device=device), "1 ... -> n ...", n=num_samples)
        return batch


    def paint(self, sampler, image, prompt, t_enc, seed, scale, num_samples=1, callback=None,
            do_full_sample=False):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = sampler.model
        seed_everything(seed)

        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        wm = "SDV2"
        wm_encoder = None #WatermarkEncoder()
        # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

        with torch.no_grad(),\
                torch.autocast("cuda"):
            batch = self.make_batch_sd(image, txt=prompt, device=device, num_samples=num_samples)
            z = model.get_first_stage_encoding(model.encode_first_stage(batch[model.first_stage_key]))  # move to latent space
            c = model.cond_stage_model.encode(batch["txt"])
            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck]
                cc = model.depth_model(cc)
                depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                            keepdim=True)
                display_depth = (cc - depth_min) / (depth_max - depth_min)
                #st.image(Image.fromarray((display_depth[0, 0, ...].cpu().numpy() * 255.).astype(np.uint8)))
                cc = torch.nn.functional.interpolate(
                    cc,
                    size=z.shape[2:],
                    mode="bicubic",
                    align_corners=False,
                )
                depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                            keepdim=True)
                cc = 2. * (cc - depth_min) / (depth_max - depth_min) - 1.
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)
            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
            if not do_full_sample:
                # encode (scaled latent)
                z_enc = sampler.stochastic_encode(z, torch.tensor([t_enc] * num_samples).to(model.device))
            else:
                z_enc = torch.randn_like(z)
            # decode it
            samples = sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale,
                                    unconditional_conditioning=uc_full, callback=callback)
            x_samples_ddim = model.decode_first_stage(samples)
            result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
        return [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]
    
def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img
  
class Inpainting():
    def __init__(self):
        self.sampler = self.initialize_model('stablediffusion/configs/stable-diffusion/v2-inpainting-inference.yaml','stablediffusion/models/512-inpainting-ema.ckpt')
    

    def generate_image(self, image, mask, prompt='a photo of a face of a woman', seed=0, scale=10, ddim_steps=50, eta=0):
        #image = Image.fromarray(image)
        w, h = image.size
        print(f"loaded input image of size ({w}, {h})")
        width, height = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32
        image = image.resize((width, height))


        #num_samples = st.number_input("Number of Samples", min_value=1, max_value=64, value=1)
        num_samples = 1
        #scale = st.slider("Scale", min_value=0.1, max_value=30.0, value=10., step=0.1)
        #ddim_steps = st.slider("DDIM Steps", min_value=0, max_value=50, value=50, step=1)
        #eta = st.sidebar.number_input("eta (DDIM)", value=0., min_value=0., max_value=1.)

        mask = np.array(mask)
        mask = mask[:, :, -1] > 0
        if mask.sum() > 0:
            mask = Image.fromarray(mask)

            result = self.inpaint(
                sampler=self.sampler,
                image=image,
                mask=mask,
                prompt=prompt,
                seed=seed,
                scale=scale,
                ddim_steps=ddim_steps,
                num_samples=num_samples,
                h=height, w=width, eta=eta
            )
            return result[0]

    def initialize_model(self, config, ckpt):
        config = OmegaConf.load(config)
        model = instantiate_from_config(config.model)

        model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        sampler = DDIMSampler(model)

        return sampler


    def make_batch_sd(
            self,
            image,
            mask,
            txt,
            device,
            num_samples=1):
        image = np.array(image.convert("RGB"))
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32) / 255.0
        mask = mask[None, None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

        masked_image = image * (mask < 0.5)

        batch = {
            "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
            "txt": num_samples * [txt],
            "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
            "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
        }
        return batch


    def inpaint(self, sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512, eta=1.):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = sampler.model

        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        wm = "SDV2"
        wm_encoder = None # WatermarkEncoder()
        # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

        prng = np.random.RandomState(seed)
        start_code = prng.randn(num_samples, 4, h // 8, w // 8)
        start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

        with torch.no_grad(), \
                torch.autocast("cuda"):
                batch = self.make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)

                c = model.cond_stage_model.encode(batch["txt"])

                c_cat = list()
                for ck in model.concat_keys:
                    cc = batch[ck].float()
                    if ck != model.masked_image_key:
                        bchw = [num_samples, 4, h // 8, w // 8]
                        cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                    else:
                        cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                    c_cat.append(cc)
                c_cat = torch.cat(c_cat, dim=1)

                # cond
                cond = {"c_concat": [c_cat], "c_crossattn": [c]}

                # uncond cond
                uc_cross = model.get_unconditional_conditioning(num_samples, "")
                uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

                shape = [model.channels, h // 8, w // 8]
                samples_cfg, intermediates = sampler.sample(
                    ddim_steps,
                    num_samples,
                    shape,
                    cond,
                    verbose=False,
                    eta=eta,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc_full,
                    x_T=start_code,
                )
                x_samples_ddim = model.decode_first_stage(samples_cfg)

                result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                    min=0.0, max=1.0)

                result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
        return [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]
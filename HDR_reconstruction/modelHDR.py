import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
import torchTools as tt
from copy import deepcopy 
import math
import torchvision
from PIL import Image
import numpy as np
import os
from torch.hub import download_url_to_file
import safetensors.torch as sf

lr = 0.00001
weight_decay = 0.0

def load_model (expt_log_dir, do_not_load_from_ckpt=False, explicit_ckpt_name=None) : 
    ckpt_dir = osp.join(expt_log_dir, 'checkpoints')
    ckpt_paths = ot.listdir(ckpt_dir)
    if explicit_ckpt_name is not None :
        valid_paths = [_ for _ in ckpt_paths if (ot.getBaseName(_) == explicit_ckpt_name) or (osp.split(_)[1] == explicit_ckpt_name)]
        assert len(valid_paths) > 0, f'No ckpt path with name - {explicit_ckpt_name}'
        ckpt_path = valid_paths[0]
    elif any('last.ckpt' in _ for _ in ckpt_paths) : 
        # check for last.ckpt
        ckpt_path = [_ for _ in ckpt_paths if 'last.ckpt' in _][0]
    else  :
        # find the checkpoint with maximum steps 
        valid_paths = [_ for _ in ckpt_paths if parse_ckpt_path(_) is not None]
        ckpt_path = sorted(valid_paths, key=lambda x: parse_ckpt_path(x)[1])[-1]

    print('Loading from checkpoint ...', ckpt_path)

    # load model args
    args_dict = osp.join(expt_log_dir, 'args.pkl') 
    with open(args_dict, 'rb') as fp :
        data = pickle.load(fp) 
        try :
            data['global_step'] = parse_ckpt_path(ckpt_path)[1]
        except Exception:
            pass
        args = argparse.Namespace(**data)
        #args = DictWrapper(pickle.load(fp))

    # make model
    if do_not_load_from_ckpt :
        print("NOT LOADING FROM CKPT")
        model = DiffusionHDR(args)
    else: 
        model = DiffusionHDR.load_from_checkpoint(ckpt_path, args=args)
    return model


def get_device (args) :
    device = torch.device("cuda") if (torch.cuda.is_available() and args.gpus > 0) else "cpu"
    return device


@torch.inference_mode()
def encode_prompt_inner(device, text_encoder, tokenizer, txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    if len(tokens) > 0: 
        chunks = [[id_start] + tokens[:chunk_length] + [id_end]] 
        #chunks = [[id_start] + tokens[i:i+chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    else :
        chunks = [[id_start] + [id_end]]

    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds

@torch.inference_mode()
def encode_prompt_pair(device, text_encoder, tokenizer, positive_prompt, negative_prompt):
    c  = encode_prompt_inner(device, text_encoder, tokenizer, positive_prompt)
    uc = encode_prompt_inner(device, text_encoder, tokenizer, negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc

class DiffusionHDR(pl.LightningModule):

    def __init__(self, args, EVs=[-2,0,2], decode=False):
        super().__init__()

        self.EVs = EVs
        self.batch_size = args.batch_size
        self.args = args

        num_EVs = len(EVs)
        sd_name = 'stablediffusionapi/realistic-vision-v51'
        #sd_name = './models/realistic-vision-v51'
        #sd_name = 'runwayml/stable-diffusion-v1-5'
        self.tokenizer = CLIPTokenizer.from_pretrained(sd_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_name, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(sd_name, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(sd_name, subfolder="unet")

        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(self.unet.conv_in.in_channels * 2, self.unet.conv_in.out_channels, self.unet.conv_in.kernel_size, self.unet.conv_in.stride, self.unet.conv_in.padding)
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, 0:4, :, :].copy_(self.unet.conv_in.weight)
            # for i in range(num_EVs - 1):
            #     new_conv_in.weight[:, i * 4:(i + 1) * 4, :, :].copy_(self.unet.conv_in.weight)
            new_conv_in.bias = self.unet.conv_in.bias
            self.unet.conv_in = new_conv_in

            # new_conv_out = torch.nn.Conv2d(self.unet.conv_out.in_channels, self.unet.conv_out.out_channels * (num_EVs-1), self.unet.conv_out.kernel_size, self.unet.conv_out.stride, self.unet.conv_out.padding)
            # new_conv_out.weight.zero_()
            # # new_conv_out.weight[4:8, :, :, :].copy_(self.unet.conv_out.weight)
            # # new_conv_out.bias[4:8].copy_(self.unet.conv_out.bias)
            # for i in range(num_EVs-1):
            #     new_conv_out.weight[i * 4:(i + 1) * 4, :, :, :].copy_(self.unet.conv_out.weight)
            #     new_conv_out.bias[i * 4:(i + 1) * 4].copy_(self.unet.conv_out.bias)
            # self.unet.conv_out = new_conv_out

        unet_original_forward = self.unet.forward
        self.unet.config.in_channels = self.unet.config.in_channels

        def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
            c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
            c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
            new_sample = torch.cat([sample, c_concat], dim=1)
            kwargs['cross_attention_kwargs'] = {}
            return unet_original_forward(new_sample, timestep, encoder_hidden_states.float(), **kwargs)

        self.unet.forward = hooked_unet_forward

        # Load
        if hasattr(args, 'init_from_sd') and args.init_from_sd == 'false':
            model_path = './models/iclight_sd15_fc.safetensors'
            os.makedirs('./models', exist_ok=True)

            if not os.path.exists(model_path):
                download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', dst=model_path)

            sd_offset = sf.load_file(model_path)
            sd_origin = self.unet.state_dict()
            keys = sd_origin.keys()
            sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
            self.unet.load_state_dict(sd_merged, strict=True)
            del sd_offset, sd_origin, sd_merged, keys

        # Device
        device = self.device
        self.text_encoder = self.text_encoder.to(device=device, dtype=torch.float32)
        #self.vae = self.vae.to(device=device, dtype=torch.bfloat16)
        self.vae = self.vae.to(device=device, dtype=torch.float32)
        self.unet = self.unet.to(device=device, dtype=torch.float32)

        # self.unet.set_attn_processor(AttnProcessor2_0())
        # self.vae.set_attn_processor(AttnProcessor2_0())

        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
            
        self.dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            steps_offset=1
        )

        self.t2i_pipe = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=deepcopy(self.dpmpp_2m_sde_karras_scheduler),
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None
        )

        self.i2i_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=deepcopy(self.dpmpp_2m_sde_karras_scheduler),
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None
        )

        self.noise_scheduler = self.ddim_scheduler
    
    def forward(self, batch, stage='train'):
        prefix = '' if stage == 'train' else 'val_'

        latent_imgs = []
        for EV in self.EVs:
            img = batch[EV].to(dtype=torch.float32)
            latent = self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor

            if EV != 0:
                latent_imgs.append(latent)

        latent_imgs = torch.cat(latent_imgs, dim=0)
        
        noise = torch.randn_like(latent_imgs, )
        bsz = latent_imgs.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latent_imgs.device)
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(latent_imgs, noise, timesteps)

        EVs_no_0 = [EV for EV in self.EVs if EV != 0]
        words = ['dark', 'bright']
        if len(self.EVs) == 5:
            words = ['darkest', 'dark', 'bright', 'brightest']
        EV_txt = [words[i] for i in range(len(EVs_no_0))]

        with torch.no_grad() :
            prompt_cond_pairs = []
            for txt in EV_txt:
                for i in range(batch[0].shape[0]):
                    prompt_cond_pairs.append(encode_prompt_pair(
                        device=self.device,
                        text_encoder=self.text_encoder, 
                        tokenizer=self.tokenizer,
                        positive_prompt=txt, 
                        # the following is not used while training. keeping here so that copy-pasta works well
                        negative_prompt='lowres, bad anatomy, bad hands, cropped, worst quality'
                    ))

            conds = torch.cat([a for a, b in prompt_cond_pairs], dim=0)
        
        cond_latent = []
        for EV in EVs_no_0:
            img = batch[0].to(dtype=torch.float32)
            latent = self.vae.encode(img).latent_dist.mode() * self.vae.config.scaling_factor
            cond_latent.append(latent)
        cond_latent = torch.cat(cond_latent, dim=0)
            


        model_pred = self.unet(
            noisy_latents.float(),
            timesteps,
            encoder_hidden_states=conds.float(),
            cross_attention_kwargs={ 'concat_conds': cond_latent.float() },
            return_dict=False,
            added_cond_kwargs={},
        )[0]


        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latent_imgs, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return {f'{prefix}loss': loss}


    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        for k in outputs.keys() : 
            if 'loss' in k: 
                self.log(k, outputs[k], batch_size=self.batch_size)
        return outputs


    def trainable_parameters (self) : 
        return list(self.unet.parameters()) \

    def count_trainable_parameters(self) : 
        return sum(p.numel() for p in self.trainable_parameters() if p.requires_grad)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.trainable_parameters(), 
            lr=lr,
            weight_decay=weight_decay
        )

    @torch.no_grad()
    def inference (self, LDR, cfg=7, seed=1)  :
        """ 
        input_fg: np.array - 512 512 3 np.uint8
        input_bg: np.array - 512 512 3 np.uint8
        prompt: str
        """
        num_samples = 1
        seed = seed
        image_width=512
        image_height=512
        steps=20
        #steps=50
        highres_scale=1.5
        highres_denoise=0.5
        ips = [LDR, image_width, image_height, num_samples, seed, steps, cfg, highres_scale, highres_denoise]
        result = self.process(*ips)
        return result

    def process(self, LDR, image_width, 
            image_height, num_samples, seed, steps, 
            cfg, highres_scale, 
            highres_denoise):

        rng = torch.Generator(device=self.device).manual_seed(seed)


        EVs_no_0 = [EV for EV in self.EVs if EV != 0]
        words = ['dark', 'bright']
        if len(self.EVs) == 5:
            words = ['darkest', 'dark', 'bright', 'brightest']
        EV_txt = [words[i] for i in range(len(EVs_no_0))]

        with torch.no_grad() :
            prompt_cond_pairs = [
                encode_prompt_pair(
                    device=self.device,
                    text_encoder=self.text_encoder, 
                    tokenizer=self.tokenizer,
                    positive_prompt=txt, 
                    # the following is not used while training. keeping here so that copy-pasta works well
                    negative_prompt='lowres, bad anatomy, bad hands, cropped, worst quality'
                )
                for txt in EV_txt
            ]

            conds = torch.cat([a for a, b in prompt_cond_pairs], dim=0)
            unconds = torch.cat([b for a, b in prompt_cond_pairs], dim=0)

        cond_imgs = []
        cond_latent = []
        for EV in EVs_no_0:
            img = LDR.to(dtype=torch.float32)
            cond_imgs.append(img)
            latent = self.vae.encode(img).latent_dist.mode() * self.vae.config.scaling_factor
            cond_latent.append(latent)
        cond_latent = torch.cat(cond_latent, dim=0)


        # shape = (num_samples * len(self.), 4*(len(self.EVs)-1), cond_latent.shape[2], cond_latent.shape[3])
        # latents_init = torch.randn(shape, generator=rng, device=self.device).to(self.device)

        latents = self.t2i_pipe(
            #latents=latents_init.float(),
            prompt_embeds=conds.float(),
            negative_prompt_embeds=unconds.float(),
            width=image_width,
            height=image_height,
            num_inference_steps=steps,
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': cond_latent.float()},
        ).images.to(self.vae.dtype) / self.vae.config.scaling_factor

        imgs = []
        for i in range(len(self.EVs)-1):
            img = self.vae.decode(latents[i:i+1, :, :, :]).sample
            imgs.append(img)


        return {'imgs': imgs, 'guides': cond_imgs}

if __name__ == "__main__" :
    # standalone model test
    from args import get_parser
    from tqdm import tqdm
    from datamodule import DiffusionHDRDataModule

    parser = get_parser()
    args = parser.parse_args()

    args.batch_size = 1
    EVs = [-2, 0 , 2]
    
    model = DiffusionHDR(args, EVs=EVs).to(torch.device('cuda'))

    data_module = DiffusionHDRDataModule(args, EVs=EVs)
    data_module.setup() 

    # model = restart_training(model, 'iclight-ft-bg-alpha-8-1-1-llava-rcrop-smode-eks01')
    opt = model.configure_optimizers()

    for batch in data_module.train_dataloader() : 
        break

    # def uint8_to_normed_tensor (x) : 
    #     x = x.astype(float)
    #     x = (x / 127.5) - 1.
    #     return torch.from_numpy(x).float()

    # fg_img = np.array(Image.open('./gt_0.png').convert('RGB').resize((512, 512)))
    # tgt_img = np.array(Image.open('./gt_-2.png').convert('RGB').resize((512, 512)))

    # bg = uint8_to_normed_tensor(fg_img).unsqueeze(0).movedim(-1, 1)
    # tgt = uint8_to_normed_tensor(tgt_img).unsqueeze(0).movedim(-1, 1)
    # batch = {-2: tgt, 0: bg}
    
    device = torch.device("cuda") if (torch.cuda.is_available() and args.gpus > 0) else "cpu"
    tt.tensorApply(batch, lambda x : x.to(device))
    losses = []
    for i in tqdm(range(2000)) : 
        if i % 10 == 0 : 
            model.eval()
            dicts = model.inference(batch[0], only_stage_1=True)
            imgs = dicts['imgs']
            img = (batch[-2].to(dtype=torch.bfloat16) + 1) / 2
            img = torch.clamp(img, 0, 1)
            torchvision.utils.save_image(img, f"gt_-2.png")
            img = (batch[0].to(dtype=torch.bfloat16) + 1) / 2
            img = torch.clamp(img, 0, 1)
            torchvision.utils.save_image(img, f"gt_0.png")
            # img = (batch[2].to(dtype=torch.bfloat16) + 1) / 2
            # torchvision.utils.save_image(img, f"gt_2.png")
            for j, img in enumerate(imgs):
                img = (img + 1) / 2
                img = torch.clamp(img, 0, 1)
                torchvision.utils.save_image(img, f"img_txt_{i:02d}_{j:02d}.png")
            guides = dicts['guides']
            for j, guide in enumerate(guides):
                img = (guide + 1) / 2
                img = torch.clamp(img, 0, 1)
                torchvision.utils.save_image(img, f"guide_{i:02d}_{j:02d}.png")
            model.train()

        opt.zero_grad()
        loss = model.training_step(batch, 0)['loss']
        losses.append(loss.item())
        loss.backward()
        opt.step()
        print(f'Loss = {loss.item():.3f}')

    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.savefig('loss.png')

from PIL import Image
from pytorch_lightning import Callback
import torchTools as tt
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch
import pickle
import osTools as ot
from datamodule import *
from dictOps import *
from listOps import *
from torchTools import *
import random
from modelHDR import get_device
from piq import ssim, psnr, LPIPS
import torchvision
import torchvision.transforms.functional as TF
from uuid import uuid4
from pytorch_lightning.loggers import TensorBoardLogger
import os
import os.path as osp
from imageOps import make_image_grid
from tqdm import tqdm



def lpips (x, y, *args, **kwargs) : 
    return LPIPS(reduction='mean')(x, y) 

class AccNum:
    def __init__(self, initial_value=0.0):
        self.total = 0.0
        self.count = 0
        self.value = initial_value

    def __iadd__(self, other):
        self.total += other
        self.count += 1
        self.value = self.total / self.count
        return self

    def __repr__(self):
        return str(self.value)

def get_object_by_rel_path (obj, rel_path) :
    paths = rel_path.split('.')
    for p in paths :
        obj = getattr(obj, p)
    return obj

def txt_as_pil(wh, x, size=10):
    txt = Image.new("RGB", wh, color="white")
    draw = ImageDraw.Draw(txt)
    font = ImageFont.truetype('font/DejaVuSans.ttf', size=size)
    nc = int(40 * (wh[0] / 256))
    lines = "\n".join(x[start:start + nc] for start in range(0, len(x), nc))
    try:
        draw.text((0, 0), lines, fill="black", font=font)
    except UnicodeEncodeError:
        print("Cant encode string for logging. Skipping.")
    return txt

def to_PIL(x):
    x = x.detach().cpu().float().squeeze(0).permute(1,2,0).numpy()
    x = np.clip(x, 0, 1)
    x = (x * 255).astype(np.uint8)
    return Image.fromarray(x)
    #Image.fromarray(img.detach().float().squeeze(0).permute(1, 2, 0).cpu().numpy())

@torch.no_grad()
def log_batch (pl_module, save_path_base, batch, global_step, **kwargs) : 
    ot.mkdir(save_path_base)
    B = batch[0].shape[0] 
    imgs = []
    #for i in range(B) : 
    save_to = osp.join(save_path_base, f'img_{global_step:06d}.png')       
    dicts = pl_module.inference(batch[0][:1,:,:,:], only_stage_1=True, cfg=1)
    imgs_pred = dicts['imgs']
    img = (batch[0][:1,:,:,:].to(dtype=torch.bfloat16) + 1) / 2
    imgs.append(to_PIL(img))
    for j, img in enumerate(imgs_pred):
        img = (img + 1) / 2
        imgs.append(to_PIL(img))
    make_image_grid(imgs).save(save_to)

    #GT
    save_to = osp.join(save_path_base, f'img_{global_step:06d}_gt.png') 
    imgs = []
    img = (batch[0][:1,:,:,:].to(dtype=torch.bfloat16) + 1) / 2
    imgs.append(to_PIL(img))
    for k in batch.keys():
        if k != 0:
            img = (batch[k][:1,:,:,:].to(dtype=torch.bfloat16) + 1) / 2
            imgs.append(to_PIL(img))
    make_image_grid(imgs).save(save_to)

@torch.no_grad()
def infer_batch (pl_module, batch, **kwargs) : 
    B = batch[0].shape[0] 
    outs, tgts = [], []
    for i in range(B) : 
        # fg = ((0.5 + (batch['fg'] / 2)) * 255.).detach().cpu().numpy().astype(np.uint8)[i]
        # bg = ((0.5 + (batch['bg'] / 2)) * 255.).detach().cpu().numpy().astype(np.uint8)[i]
        # tgt = ((0.5 + (batch['tgt'] / 2)) * 255.).detach().cpu().numpy().astype(np.uint8)[i]
        # prompt = batch['txt'][i]
        dicts = pl_module.inference(batch[0][i:i+1,:,:,:], cfg=1, **kwargs)
        imgs_pred = dicts['imgs']
        out = imgs_pred[-2]
        outs.append(to_PIL(out).convert('RGB').resize((512, 512)))
        target = (batch[-2][i:i+1,:,:,:] + 1) / 2
        tgts.append(to_PIL(target).convert('RGB').resize((512, 512)))
    return dict(outputs=outs, originals=tgts) 

class DebugLogBatch (Callback) : 

    def __init__ (self) : 
        super().__init__() 
        self.steps = [1, 4, 16, 64, 256]
        job_name = os.environ.get('JOB_NAME', 'job')
        self.debug_dir = f'./debug/{job_name}'
        os.makedirs(self.debug_dir, exist_ok=True)

    def on_train_batch_end (self, trainer, pl_module, outputs, batch, batch_idx) : 
        global_step = trainer.global_step
        rank = pl_module.global_rank
        if global_step in self.steps :
            save_path_base = osp.join(self.debug_dir, f'rank_{rank:03d}')
            log_batch(pl_module, save_path_base, batch, global_step) 
        

class VisualizePredictions(Callback):

    def __init__ (self, args) : 
        super().__init__()
        self.args = args
        self.fixed_batch_train = None
        self.fixed_batch_val   = None

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.args.fast == 'true' : 
            return 
        global_step = trainer.global_step
        if global_step % self.args.frequency == 0 :
            pl_module.eval()
         
            if self.fixed_batch_train is None : 
                self.fixed_batch_train = deepcopy(batch) 

            log_dir = '' if pl_module.logger is None else pl_module.logger.log_dir

            # # do for regular batch
            # save_path_base = osp.join(log_dir, 'images', 'train')
            # log_batch(pl_module, save_path_base, batch, global_step) 

            # # do for fixed batch
            # save_path_base = osp.join(log_dir, 'images', 'train_fixed_batch')
            # log_batch(pl_module, save_path_base, self.fixed_batch_train, global_step)

            # do for regular batch, first stage only
            save_path_base = osp.join(log_dir, 'images', 'train_first_stage')
            log_batch(pl_module, save_path_base, batch, global_step, only_stage_1=True)

            # do for fixed batch
            save_path_base = osp.join(log_dir, 'images', 'train_fixed_batch_first_stage')
            log_batch(pl_module, save_path_base, self.fixed_batch_train, global_step, only_stage_1=True)

            pl_module.train()
                
    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        pass
        # if self.args.fast == 'true' : 
        #     return 
        # global_step = trainer.global_step
        # if batch_idx in [0] and trainer.current_epoch % 20 == 0 :
        #     pl_module.eval()

        #     if self.fixed_batch_val is None : 
        #         self.fixed_batch_val = deepcopy(batch) 

        #     log_dir = '' if pl_module.logger is None else pl_module.logger.log_dir

        #     # do for regular batch
        #     save_path_base = osp.join(log_dir, 'images', 'val')
        #     log_batch(pl_module, save_path_base, batch, global_step) 

        #     # do for fixed batch
        #     save_path_base = osp.join(log_dir, 'images', 'val_fixed_batch')
        #     log_batch(pl_module, save_path_base, self.fixed_batch_val, global_step)

        #     # do for regular batch, first stage only
        #     save_path_base = osp.join(log_dir, 'images', 'val_first_stage')
        #     log_batch(pl_module, save_path_base, batch, global_step, only_stage_1=True)

        #     # do for fixed batch
        #     save_path_base = osp.join(log_dir, 'images', 'val_fixed_batch_first_stage')
        #     log_batch(pl_module, save_path_base, self.fixed_batch_val, global_step, only_stage_1=True)

        #     pl_module.train()

class LogMetrics (Callback) : 
    """
    Log different helpful metrics.

    Currently we are logging: 
     
      * lpips, ssim, psnr
    """

    def __init__ (self, args, frequency=500) : 
        super().__init__() 
        self.frequency = frequency
        self.args = args

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.args.fast == 'true' : 
            return 
        global_step = trainer.global_step
        if global_step % self.frequency == 0 :
            # # do for final result
            # out_dict = infer_batch(pl_module, batch)

            # originals_ten = [TF.pil_to_tensor(_) for _ in tqdm(out_dict['originals'])]
            # targets_ten = [TF.pil_to_tensor(_) for _ in tqdm(out_dict['outputs'])]

            # originals_ten = (torch.stack(originals_ten).float() / 255.).cuda()
            # targets_ten = (torch.stack(targets_ten).float() / 255.).cuda()
            
            # metrics = [psnr, ssim, lpips] 
            # for metric in metrics : 
            #     val = metric(originals_ten, targets_ten, data_range=1.)
            #     trainer.logger.experiment.add_scalar(metric.__name__, val, global_step)


            # now do only for stage 1
            out_dict = infer_batch(pl_module, batch, only_stage_1=True)

            originals_ten = [TF.pil_to_tensor(_) for _ in tqdm(out_dict['originals'])]
            targets_ten = [TF.pil_to_tensor(_) for _ in tqdm(out_dict['outputs'])]

            originals_ten = (torch.stack(originals_ten).float() / 255.).cuda()
            targets_ten = (torch.stack(targets_ten).float() / 255.).cuda()
            
            metrics = [psnr, ssim, lpips] 
            for metric in metrics : 
                val = metric(originals_ten, targets_ten, data_range=1.)
                trainer.logger.experiment.add_scalar(f'stage-1-{metric.__name__}', val, global_step)

    @rank_zero_only
    @torch.no_grad()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        pass
        # if self.args.fast == 'true' : 
        #     return 
        # global_step = trainer.global_step
        # if batch_idx % self.frequency == 0 :
        #     out_dict = infer_batch(pl_module, batch)

        #     originals_ten = [TF.pil_to_tensor(_) for _ in tqdm(out_dict['originals'])]
        #     targets_ten = [TF.pil_to_tensor(_) for _ in tqdm(out_dict['outputs'])]

        #     originals_ten = (torch.stack(originals_ten).float() / 255.).cuda()
        #     targets_ten = (torch.stack(targets_ten).float() / 255.).cuda()
            
        #     metrics = [psnr, ssim, lpips] 
        #     for metric in metrics : 
        #         val = metric(originals_ten, targets_ten, data_range=1.)
        #         trainer.logger.experiment.add_scalar(f'val-{metric.__name__}', val, global_step)

        #     for model_name in FACE_EMBEDDING_MODELS : 
        #         val = face_embedding_distance(out_dict['originals'], out_dict['outputs'], model_name)
        #         trainer.logger.experiment.add_scalar(f'val-{model_name}-distance', val, global_step)

        #     # now do only for stage 1 
        #     out_dict = infer_batch(pl_module, batch, only_stage_1=True)

        #     originals_ten = [TF.pil_to_tensor(_) for _ in tqdm(out_dict['originals'])]
        #     targets_ten = [TF.pil_to_tensor(_) for _ in tqdm(out_dict['outputs'])]

        #     originals_ten = (torch.stack(originals_ten).float() / 255.).cuda()
        #     targets_ten = (torch.stack(targets_ten).float() / 255.).cuda()
            
        #     metrics = [psnr, ssim, lpips] 
        #     for metric in metrics : 
        #         val = metric(originals_ten, targets_ten, data_range=1.)
        #         trainer.logger.experiment.add_scalar(f'val-stage-1-{metric.__name__}', val, global_step)

        #     for model_name in FACE_EMBEDDING_MODELS : 
        #         val = face_embedding_distance(out_dict['originals'], out_dict['outputs'], model_name)
        #         trainer.logger.experiment.add_scalar(f'val-stage-1-{model_name}-distance', val, global_step)

class ParameterTracker (Callback) :
    """
    Sees how parameters are changing by logging the norms

    Helpful for tracking bugs in gradient setting

    """

    def __init__ (self, rel_paths, frequency=100) :
        super().__init__()
        self.rel_paths = rel_paths
        self.frequency = frequency

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step
        if global_step % self.frequency == 0 :
            model_parts = [get_object_by_rel_path(pl_module, _) for _ in self.rel_paths]
            param_norms = []
            for part in model_parts : 
                if isinstance(part, torch.nn.Parameter) : 
                    param_norms.append(part.norm())
                else :
                    param_norms.append(sum(p.norm() for p in part.parameters()))
            for pnorm, part_name in zip(param_norms, self.rel_paths) :
                trainer.logger.experiment.add_scalar(f'{part_name}_norm', pnorm, global_step)

class SaveArgs (Callback) :

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        ot.mkdir(pl_module.logger.log_dir)
        save_path = osp.join(pl_module.logger.log_dir, 'args.pkl')
        with open(save_path, 'wb') as fp :
            pickle.dump(vars(pl_module.args), fp)

if __name__ == "__main__" : 
    ###################################
    ## UNIT TESTS!!!!
    ###################################
    from args import get_parser, DictWrapper
    from datamodule import RelightDataModule
    from model import RelightModel, get_device, load_model

    parser = get_parser() 
    args = parser.parse_args() 

    print(args)

    vis_callback = VisualizePredictions(args)
    metric_callback = LogMetrics()

    data_module = RelightDataModule(args)
    data_module.setup() 

    model = load_model('old_ckpts/version_0').to(torch.device('cuda'))

    logger = TensorBoardLogger(save_dir='/tmp/log_dir', name="my_experiment")

    trainer = pl.Trainer(
        gpus=1,
        precision=32,
        accelerator='gpu',
        callbacks=[
            VisualizePredictions(args), 
            LogMetrics(),
        ],
        log_every_n_steps=1,
        max_steps=10,
        logger=logger
    )

    trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)

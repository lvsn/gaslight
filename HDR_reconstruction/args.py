import argparse
from pathlib import Path


def dict_as_table_str(d):
    # Determine the maximum width of the keys and values for alignment
    max_key_width = max(len(str(key)) for key in d.keys())
    max_value_width = max(len(str(value)) for value in d.values())

    # Print the table header
    res = '' 
    res += f"{'Key':<{max_key_width}} | {'Value':<{max_value_width}}\n"
    res += ('-' * (max_key_width + max_value_width + 3)) + "\n"
    for key, value in d.items():
        res += f"{str(key):<{max_key_width}} | {str(value):<{max_value_width}}\n"
    return res

class DictWrapper:
    def __init__(self, d):
        self._dict = d

    def __getattr__(self, name):
        if name == '_dict' : 
            return self._dict
        elif name in self._dict:
            return self._dict[name]
        raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        if name == '_dict':
            super().__setattr__(name, value)
        else:
            self._dict[name] = value

    def __delattr__(self, name):
        if name in self._dict:
            del self._dict[name]
        else:
            raise AttributeError(f"No such attribute: {name}")

    def __contains__ (self, name) : 
        return name in self._dict

    def __repr__ (self) : 
        return dict_as_table_str(self._dict)

def get_parser () : 
    parser = argparse.ArgumentParser(description='Relight')
    #Inference
    parser.add_argument('--image', type=Path, default=None, help='Image to process')
    parser.add_argument('--images_dir', type=Path, default=None, help='Directory of images to process')
    parser.add_argument('--out_dir', type=Path, default=None, help='Directory to save HDRs')
    parser.add_argument('--ext', type=str, default='png', help='Image extension')
    parser.add_argument('--ckpt_path', type=Path, default=None, help='Path to the checkpoint')
    parser.add_argument("--save_stacks", action='store_true', default=False, help='Whether to save the stacks of images')
    parser.add_argument("--predict_ev_plus", action='store_true', default=False, help='Whether to predict EVs 2 and 4 or not')
    
    
    #Training 
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0000, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--prefetch_factor', type=int, default=16, help='number of batches to be prefetched')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--n_more_steps', type=int, default=None, help='number of more steps to train for')
    parser.add_argument('--gpus', type=int, default=8, help='number of GPUs')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='number of batches to accumulate gradient over')
    parser.add_argument('--ckpt_path_resume', type=str, default=None, help='where to resume training from')
    parser.add_argument('--frequency', type=int, default=1000, help='Plotting frequency')
    parser.add_argument('--restart', type=str, default='false', help='Whether to restart or start fresh')
    parser.add_argument('--fast', type=str, default='false', help='Whether to start fast')
    parser.add_argument('--llava_prompts', type=str, default='csv_files/llava_prompts.csv', help='synthetic prompts for training')
    # parser.add_argument('--llava_prompts', type=str, default=None, help='synthetic prompts for training')
    parser.add_argument('--seperate_mode', type=str, default='true', help='whether to use specific prompts for training and reconstruction')
    parser.add_argument('--every_n_train_steps', type=int, default=5000, help='save checkpoint every n train steps')
    parser.add_argument('--no_prompt', type=str, default='false', help='whether to use extra prompts during training') 
    parser.add_argument('--init_from_sd', type=str, default='false', help='whether to init from stable diffusion instead of ic light') 
    parser.add_argument('--data_config', type=str, default='configs/with_preset_conf.yaml', help='data config file') 
    parser.add_argument('--ldr_dir', type=str, default='/sensei-fs/users/suchatur/ldrs', help='path to ldrs') 
    parser.add_argument('--hdr_dir', type=str, default='/sensei-fs/users/suchatur/hdrs', help='path to hdrs') 
    parser.add_argument('--sd15_name', type=str, default='stablediffusionapi/realistic-vision-v51', help='base sd ckpt') 
    parser.add_argument('--name', type=str, default='GaSLight', help='Name of the job') 

    return parser

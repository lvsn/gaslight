import logging
import os
import os.path as osp
from subprocess import call

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything

import listOps as lo
import osTools as ot
import strOps as so
from args import get_parser
from callbacks import *
from datamodule import DiffusionHDRDataModule
from modelHDR import DiffusionHDR


def print_dict_as_table(d):
    # Determine the maximum width of the keys and values for alignment
    max_key_width = max(len(str(key)) for key in d.keys())
    max_value_width = max(len(str(value)) for value in d.values())

    # Print the table header
    print(f"{'Key':<{max_key_width}} | {'Value':<{max_value_width}}")
    print("-" * (max_key_width + max_value_width + 3))

    # Print the key-value pairs
    for key, value in d.items():
        print(f"{str(key):<{max_key_width}} | {str(value):<{max_value_width}}")


def resolve_checkpoint_from_default_pl_log_dir(log_dir):
    paths = list(ot.allFilesWithSuffix(log_dir, "ckpt"))
    last_ckpt_maybe = next(filter(lambda x: "last" in x, paths), None)
    if last_ckpt_maybe is not None:
        return last_ckpt_maybe
    steps = [
        so.reverse_f_string(osp.split(_)[1], "epoch={epoch}-step={step}.ckpt", int)[
            "step"
        ]
        for _ in paths
    ]
    path = paths[argmax(steps)]
    print("Using checkpoint path", path)
    return path


def restart_training(model, log_dir):
    """
    finds the relevant path to resume training from, and sets the checkpoint
    """
    if not osp.exists(log_dir):
        return model, None

    ver_cands = []
    for dir_ in ot.listdir(log_dir):
        if "version" in dir_:
            ver_cands.append((int(dir_.split("_")[-1]), dir_))
    ver_cands = sorted(ver_cands)

    if len(ver_cands) == 0:
        return model, None

    to_be_rem_idx = []
    for i in range(len(ver_cands)):
        ckpt_dir = osp.join(ver_cands[i][1], "checkpoints")
        if not osp.exists(ckpt_dir) or len(ot.listdir(ckpt_dir)) == 0:
            to_be_rem_idx.append(i)

    lo.removeIndices(ver_cands, to_be_rem_idx)

    if len(ver_cands) == 0:
        return model, None

    resume_path = resolve_checkpoint_from_default_pl_log_dir(ver_cands[-1][1])

    model = DiffusionHDR.load_from_checkpoint(resume_path, args=model.args)
    return model, resume_path


def main():
    
    torch.set_float32_matmul_precision("medium")

    parser = get_parser()
    args = parser.parse_args()
    
    job_name = args.name

    EVs = [-2, 0, 2]
    

    print_dict_as_table(vars(args))

    # Determine how long to train
    kwargs = {}

    if (
        args.ckpt_path is not None and args.n_more_steps is not None
    ) or args.epochs is None:
        global_step = 0
        if args.ckpt_path is not None:
            global_step = parse_ckpt_path(args.ckpt_path)[1]
        kwargs["max_steps"] = global_step + args.n_more_steps
    else:
        kwargs["max_epochs"] = args.epochs
    # en

    # seed_everything(args.seed) # don't seed since job may get preempted and i don't have proper restart at the moment

    data_module = DiffusionHDRDataModule(args, EVs=EVs)
    model = DiffusionHDR(args, EVs=EVs)

    resume_path = None
    if args.restart == "true":
        model, resume_path = restart_training(model, job_name)

    params_to_track = ["unet"]

    print_dict_as_table(kwargs)
    print(f"Trainable Params: {model.count_trainable_parameters() / 10**6:.2f}M")

    # FOR RUNAI MULTI-GPU
    gpus = args.gpus
    num_nodes = 1

    if gpus > 1:
        kwargs["strategy"] = "ddp"

    logger = pl.loggers.TensorBoardLogger(".", name=job_name)

    every_n_train_steps = args.every_n_train_steps
    model_checkpointer = pl.callbacks.ModelCheckpoint(
        monitor="loss",
        every_n_train_steps=every_n_train_steps,
        save_last=True,
        save_top_k=-1,
        dirpath="/home/chritoto/scratch/checkpoints",
    )

    trainer = pl.Trainer(
        gpus=gpus,
        num_nodes=num_nodes,
        precision=32,
        accelerator="gpu",
        callbacks=[
            VisualizePredictions(args),
            ParameterTracker(params_to_track),
            SaveArgs(),
            LogMetrics(args),
            DebugLogBatch(),
            model_checkpointer,
        ],
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=1,
        logger=logger,
        **kwargs,
    )

    trainer.fit(model, datamodule=data_module, ckpt_path=resume_path)


if __name__ == "__main__":
    main()

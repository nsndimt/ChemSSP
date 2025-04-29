from argparse import ArgumentParser
from functools import partial
import math
import pytorch_lightning as pl
import logging
from fs_chatgpt_utils import add_uniner_specific_args, load_UniNER_episode
from fs_ner_utils import add_fewshot_specific_args, add_multidataset_specific_args, add_train_specific_args, init_argparser, load_episode, load_sent, set_logging
import os
import datetime
from fs_ner_model import MultiSpan, ProtoSpan
import torch
import sys

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = init_argparser(parser)
    parser = add_fewshot_specific_args(parser)
    parser = add_train_specific_args(parser)
    parser = add_multidataset_specific_args(parser)
    parser = add_uniner_specific_args(parser)
    parser.add_argument("--task", type=str, required=True, help="model mode")
    parser.add_argument("--model", type=str, required=True, help="model name")
    args, _ = parser.parse_known_args()

    if args.model == 'protospan':
        model_class = ProtoSpan
    elif args.model == "multispan":
        model_class = MultiSpan
    else:
        raise Exception('Unknown model type')

    parser = model_class.add_model_specific_args(parser)
    args = parser.parse_args()
    root, formatter = set_logging(args.logging_level)

    if args.task in ["all_avg"]:
        pl.seed_everything(args.seed)
        train_dsnames = args.source.split(',')
        test_dsnames = args.target.split(',')
        # breakpoint()
        assert len(train_dsnames) > 0, train_dsnames
        assert len(test_dsnames) > 0, test_dsnames
        assert len(set(train_dsnames).intersection(set(test_dsnames))) == 0, set(train_dsnames).intersection(set(test_dsnames))
        if train_dsnames != ['UniNER']:
            train_dataset, train_data_loader = load_episode(train_dsnames, args, training=True)
        else:
            train_dataset, train_data_loader = load_UniNER_episode(args, training=True, debug_firstk=2000)
        dev_dataset, dev_data_loader = load_episode(test_dsnames, args, training=False, debug_firstk=100)
        test_dataset, test_data_loader = load_episode(test_dsnames, args, training=False)
        pl.seed_everything(args.seed)
        model = model_class(**vars(args))
    elif args.task in ["multi_supervised"]:
        pl.seed_everything(args.seed)
        train_dsnames = args.target.split(',')
        # breakpoint()
        assert len(train_dsnames) > 0, train_dsnames
        train_dataset, train_data_loader, train_types = load_sent(train_dsnames, "train", args, training=True)
        dev_dataset, dev_data_loader, dev_types = load_sent(train_dsnames, "test", args, training=False)
        test_dataset, test_data_loader = dev_dataset, dev_data_loader
        pl.seed_everything(args.seed)
        model = model_class(**vars(args), types=train_types)
    else:
        raise RuntimeError("unknown mode")
    
    precision = 32
    if args.tf32:
        torch.set_float32_matmul_precision('high')
    if args.bf16:
        precision = 'bf16-mixed'
    elif args.fp16:
        precision = '16-mixed'

    val_interval = math.floor(args.train_step / args.val_times)
    round_train_step = val_interval * args.val_times

    trainer_kwarg = {
        "accelerator": 'gpu',
        "devices": args.gpu,
        "precision": precision,
        "max_steps": round_train_step,
        "num_sanity_val_steps": 20,
        "check_val_every_n_epoch": None,
        "val_check_interval": val_interval * args.grad_acc,
        "accumulate_grad_batches": args.grad_acc,
        "gradient_clip_val": args.clip,
    }
    if type(args.gpu) == list and len(args.gpu) > 1:
        trainer_kwarg["strategy"] = "ddp"

    if args.nan_detect:
        trainer_kwarg["detect_anomaly"] = True

    if args.profile:
        trainer_kwarg["profiler"] = "simple"
        trainer_kwarg["max_steps"] = 100
        trainer_kwarg["val_check_interval"] = 50
        trainer_kwarg["accumulate_grad_batches"] = 1
        trainer_kwarg["num_sanity_val_steps"] = 0

    logger = None
    if not args.debug:
        if args.output_dir == '':
            filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            if 'SLURM_JOB_ID' in os.environ:
                checkpoint_dir = os.path.join('checkpoint', f"{filename}-{os.environ['SLURM_JOB_ID']}")
            else:
                checkpoint_dir = os.path.join('checkpoint', filename)
            assert not os.path.isdir(checkpoint_dir), checkpoint_dir
            os.mkdir(checkpoint_dir)
        else:
            checkpoint_dir = args.output_dir
            assert not os.path.isdir(checkpoint_dir)

        
        handler = logging.FileHandler(os.path.join(checkpoint_dir, f'shell_output.log'))
        handler.setFormatter(formatter)
        root.addHandler(handler)

        if args.wandb_proj == "None" or args.wandb_entity == "None":
            # CSVLogger
            trainer_kwarg['logger'] = True # CSVLogger
        else:
            logger = pl.loggers.wandb.WandbLogger(project=args.wandb_proj, entity=args.wandb_entity)
            logger.experiment.config.update(args)
            logger.experiment.config['checkpoint_dir'] = checkpoint_dir
            logger.experiment.config['val_interval'] = val_interval
            logger.experiment.config['source'] = args.source
            logger.experiment.config['target'] = args.target
            trainer_kwarg['logger'] = logger
    else:
        # no logger
        trainer_kwarg['logger'] = False


    if args.checkpoint:
        if args.best_checkpoint:
            checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="{epoch}-{step}",
                monitor='valid/f1',
                mode='max',
            )
        else:
            checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="{epoch}-{step}",
            )
        trainer_kwarg['callbacks'] = [checkpoint_callback]
    else:
        trainer_kwarg['enable_checkpointing'] = False

    logging.info(sys.argv)
    line = ''
    for k, v in vars(args).items():
        line += f"{k}:{v} "
        if len(line) > 80:
            logging.info(line)
            line = ''
    if line:
        logging.info(line)

    logging.info(f'train len: {len(train_dataset)} loader:{len(train_data_loader)}')
    logging.info(f'dev len: {len(dev_dataset)} loader:{len(dev_data_loader)}')

    logging.info(f"train step {args.train_step} round to:{round_train_step}")

    trainer = pl.Trainer(**trainer_kwarg)
    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=dev_data_loader)

    if not args.debug and args.best_checkpoint:
        test_res = trainer.test(model, ckpt_path='best', dataloaders=dev_data_loader)
    else:
        test_res = trainer.test(model, dataloaders=test_data_loader)
        train_dsnames = args.source.split(',')
        test_dsnames = args.target.split(',')
        if args.task in ["all_avg"] and train_dsnames == ['UniNER'] and len(test_dsnames) > 0:
            for dsname in test_dsnames:
                test_dataset, test_data_loader = load_episode(dsname, args, training=False)
                pred_output = trainer.predict(model, test_data_loader)
                ret = model.eval_epoch_end(pred_output, dsname, False)
                if logger is not None:
                    logger.log_metrics(ret)
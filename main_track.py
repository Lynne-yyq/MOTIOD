import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine_track import evaluate, train_one_epoch
from models import build_tracktrain_model, build_tracktest_model, build_model
from models import Tracker



def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=195, type=int)  # 300
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=True, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=500, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--id_loss_coef', default=1, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='mot')
    parser.add_argument('--coco_path',
                        default='/path/data/',
                        type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')


    parser.add_argument('--checkpoint_enc_ffn', default=False, action='store_true')
    parser.add_argument('--checkpoint_dec_ffn', default=False, action='store_true')

    # appended for track.
    parser.add_argument('--track_train_split', default='train', type=str)
    parser.add_argument('--track_eval_split', default='test', type=str)
    parser.add_argument('--track_thresh', default=0.7, type=float)
    parser.add_argument('--reid_shared', default=False, type=bool)
    parser.add_argument('--reid_dim', default=128, type=int)
    parser.add_argument('--num_ids', default=360, type=int)

    # detector for track.
    parser.add_argument('--det_val', default=False, action='store_true')

    return parser

def main(args):

    setattr(args, 'distributed', False)
    validate_args(args)
    device = torch.device(args.device)
    setup_seed(args.seed, utils.get_rank())

    model, criterion, postprocessors = setup_model(args)  # Ensure this returns postprocessors
    model.to(device)

    dataset_train, dataset_val = setup_datasets(args)
    data_loader_train, data_loader_val = setup_dataloaders(args, dataset_train, dataset_val)

    optimizer, lr_scheduler = configure_optimization(args, model)

    if args.frozen_weights:
        load_frozen_weights(model, args.frozen_weights)

    output_dir = prepare_output_dir(args.output_dir)
    if args.resume:
        resume_training(model, optimizer, lr_scheduler, args.resume, args)

    if args.eval:
        perform_evaluation(model, criterion, postprocessors, data_loader_val, dataset_val, device, output_dir)
        return

    train(model, criterion, data_loader_train, data_loader_val, optimizer, lr_scheduler, device, output_dir, args, postprocessors)

def validate_args(args):
    if args.frozen_weights:
        assert args.masks, "Frozen training is meant for segmentation only"

def setup_seed(seed, rank):
    seed = seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup_model(args):
    if args.det_val:
        assert args.eval, 'Detector validation mode is only supported in eval'
        return build_model(args)
    elif args.eval:
        return build_tracktest_model(args)
    else:
        return build_tracktrain_model(args)

def setup_datasets(args):
    return (build_dataset(args.track_train_split, args), build_dataset(args.track_eval_split, args))

def setup_dataloaders(args, dataset_train, dataset_val):
    samplers = get_samplers(args, dataset_train, dataset_val)
    return (create_dataloader(dataset_train, samplers['train'], args), create_dataloader(dataset_val, samplers['val'], args))

def get_samplers(args, dataset_train, dataset_val):
    if args.distributed:
        sampler_train = samplers.DistributedSampler(dataset_train) if not args.cache_mode else samplers.NodeDistributedSampler(dataset_train)
        sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False) if not args.cache_mode else samplers.NodeDistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    return {'train': sampler_train, 'val': sampler_val}

def create_dataloader(dataset, sampler, args):
    batch_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=not args.eval)
    return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)

def configure_optimization(args, model):
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in args.lr_backbone_names + args.lr_linear_proj_names) and p.requires_grad], "lr": args.lr},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in args.lr_backbone_names) and p.requires_grad], "lr": args.lr_backbone},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in args.lr_linear_proj_names) and p.requires_grad], "lr": args.lr * args.lr_linear_proj_mult}
    ]
    optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay) if args.sgd else torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    return optimizer, lr_scheduler

def load_frozen_weights(model, path):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

def prepare_output_dir(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return Path(output_dir)

def resume_training(model, optimizer, lr_scheduler, resume_path, args):
    checkpoint = torch.load(resume_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and not args.eval:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

def perform_evaluation(model, criterion, postprocessors, data_loader_val, dataset_val, device, output_dir):
    base_ds = get_coco_api_from_dataset(dataset_val)
    tracker = Tracker(score_thresh=args.track_thresh)
    evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, output_dir, tracker=tracker, det_val=args.det_val)

def train(model, criterion, data_loader_train, data_loader_val, optimizer, lr_scheduler, device, output_dir, args, postprocessors):
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir and (epoch % 10 == 0 or epoch >= args.epochs - 5):
            evaluate(model, criterion, postprocessors, data_loader_val, device, output_dir)
    print('Training completed in {}'.format(datetime.timedelta(seconds=int(time.time() - start_time))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR Detector and Trainer', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)


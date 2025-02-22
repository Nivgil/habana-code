# coding=utf-8
# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

"""BERT Pretraining script."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import dataclasses
import re
import dllogger
import h5py
import numpy as np
import os
import time

import pandas as pd
from tqdm import tqdm
from typing import Union, Optional
import json
from concurrent.futures import ProcessPoolExecutor
import random
import signal
import warnings

import torch
import torch.distributed
import torch.distributed.optim
from torch.utils.data import Dataset

import modeling
import schedulers
import lamb

import utils
from compute_timer import DeviceTimer
from compute_timer import ComputeTimeout

try:
    import apex
    from apex import amp
    from apex.optimizers import FusedLAMB
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel.distributed import flat_dist_call
    import amp_C
    import apex_C
    from apex.amp import _amp_state
except ImportError:
    if torch.cuda.is_available():
        raise ImportError("Please install apex from "
                          "https://www.github.com/nvidia/apex")
    else:
        from torch.nn.parallel import DistributedDataParallel as DDP


try:
    import habana_frameworks.torch as ht
except ImportError:
    assert False, "Could not import habana_frameworks.torch"


torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

skipped_steps = 0
avg_seq_per_pack = 1.0

# Track whether a SIGTERM (cluster time up) has been handled
timeout_sent = False
global_drop_timer = DeviceTimer(use_hpu = True)


def signal_handler(sig, frame):
    global timeout_sent
    timeout_sent = True


signal.signal(signal.SIGTERM, signal_handler)


# Workaround because python functions are not picklable
class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, idx):
        np.random.seed(self.seed + idx)
        random.seed(self.seed + idx)


def create_pretraining_dataset(input_file, max_pred_length, shared_list, args,
                               worker_init):
    num_workers = 0 if args.use_habana else 4
    train_data = PretrainingDataset(
        input_file=input_file,
        max_pred_length=max_pred_length,
        enable_packed_data_mode=args.enable_packed_data_mode
    )
    train_sampler = torch.utils.data.RandomSampler(train_data)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=args.train_batch_size * args.n_pu,
        num_workers=num_workers,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=True
    )
    return train_dataloader, input_file


class PretrainingDataset(torch.utils.data.Dataset):
    def __init__(self, input_file, max_pred_length,
                 enable_packed_data_mode: bool = False):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        if enable_packed_data_mode:
            keys = ['input_ids', 'input_mask', 'segment_ids', 'positions',
                    'masked_lm_positions', 'masked_lm_ids',
                    'next_sentence_positions', 'next_sentence_labels',
                    'next_sentence_weights']
        else:
            keys = ['input_ids', 'input_mask', 'segment_ids',
                    'masked_lm_positions', 'masked_lm_ids',
                    'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()
        self.enable_packed_data_mode = enable_packed_data_mode

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.inputs[0])

    def __getitem__(self, index):
        if self.enable_packed_data_mode:
            [
                input_ids,
                input_mask,
                segment_ids,
                positions,
                masked_lm_positions,
                masked_lm_ids,
                next_sentence_positions,
                next_sentence_labels,
                next_sentence_weights
            ] = [torch.from_numpy(
                sample[index].astype(np.int64)) for sample in self.inputs]
        else:
            [
                input_ids,
                input_mask,
                segment_ids,
                masked_lm_positions,
                masked_lm_ids,
                next_sentence_labels
            ] = [torch.from_numpy(sample[index].astype(np.int64)) if indice < 5 else torch.from_numpy(np.asarray(sample[index].astype(np.int64))) for indice, sample in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        if self.enable_packed_data_mode:
            next_sentence_labels = (next_sentence_weights == 1) * next_sentence_labels + (next_sentence_weights == 0) * -1
            return [input_ids,
                    segment_ids,
                    input_mask,
                    positions,
                    masked_lm_labels,
                    next_sentence_positions,
                    next_sentence_labels]
        else:
            return [input_ids,
                    segment_ids,
                    input_mask,
                    masked_lm_labels,
                    next_sentence_labels]


class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, seq_relationship_score,
                masked_lm_labels, next_sentence_labels):
        masked_lm_loss = self.loss_fn(
            prediction_scores.view(-1, self.vocab_size),
            masked_lm_labels.view(-1)
        )
        next_sentence_loss = self.loss_fn(
            seq_relationship_score.view(-1, 2),
            next_sentence_labels.view(-1)
        )
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss


@dataclasses.dataclass
class ComputeState:
    computed_batch_size: torch.Tensor
    start_compute: float = 0
    threshold: float = 0
    enable_drop: bool = False
    mini_batch_size: int = 0

    def __init__(self, device: Union[int, torch.device]):
        self.computed_batch_size = torch.zeros(1, device=device)

    def reset_state(self, compute_threshold: float, enable_drop: bool,
                    start_compute: float, mini_batch_size: int):
        self.threshold = compute_threshold
        self.enable_drop = enable_drop
        self.start_compute = start_compute
        self.mini_batch_size = mini_batch_size
        self.computed_batch_size.zero_()


def parse_arguments():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files for the task.")

    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")

    parser.add_argument("--bert_model", default="bert-large-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--checkpoint_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="Directory where the model checkpoint will be loaded.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="The initial checkpoint to start training from.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_seq",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=1000,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('OMPI_COMM_WORLD_RANK', -1),
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Mixed precision training")
    parser.add_argument('--amp',
                        default=False,
                        action='store_true',
                        help="Mixed precision training")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=1.0,
                        help='frequency of logging loss.')
    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint.")
    parser.add_argument('--resume_step',
                        type=int,
                        default=-1,
                        help="Step to resume training from.")
    parser.add_argument('--num_steps_per_checkpoint',
                        type=int,
                        default=100,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--phase2',
                        default=False,
                        action='store_true',
                        help="Whether to train with seq len 512")
    parser.add_argument('--allreduce_post_accumulation',
                        default=False,
                        action='store_true',
                        help="Whether to do allreduces during gradient accumulation steps.")
    parser.add_argument('--allreduce_post_accumulation_fp16',
                        default=False,
                        action='store_true',
                        help="Whether to do fp16 allreduce post accumulation.")
    parser.add_argument('--phase1_end_step',
                        type=int,
                        default=7038,
                        help="Number of training steps in Phase1 - seq len 128")
    parser.add_argument('--init_loss_scale',
                        type=int,
                        default=2**20,
                        help="Initial loss scaler value")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument('--log-dir',
                        type=str,
                        default='results',
                        help='If provided, the json summary will be written to'
                             'the specified file.')
    parser.add_argument("--use_env",
                        action='store_true',
                        help="Whether to read local rank from ENVVAR")
    parser.add_argument('--disable_progress_bar',
                        default=False,
                        action='store_true',
                        help='Disable tqdm progress bar')
    parser.add_argument('--steps_this_run', type=int, default=-1,
                        help='If provided, only run this many steps before exiting')
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether to use CPU when available")
    parser.add_argument("--use_habana",
                        action="store_true",
                        help="Whether not to use Habana device when available")
    parser.add_argument('--hmp',
                        dest='hmp',
                        action='store_true',
                        help='enable hmp mode')
    parser.add_argument('--hmp_bf16',
                        default="",
                        help='path to bf16 ops list in hmp O1 mode')
    parser.add_argument('--hmp_fp32',
                        default="",
                        help='path to fp32 ops list in hmp O1 mode')
    parser.add_argument('--hmp_opt_level',
                        default='O1',
                        help='choose optimization level for hmp')
    parser.add_argument('--hmp_verbose',
                        action='store_true',
                        help='enable verbose mode for hmp')
    parser.add_argument("--use_fused_lamb",
                        action='store_true',
                        help='use FusedLamb optimizer')
    parser.add_argument("--use_lazy_mode",
                        default='True', type=lambda x: x.lower() == 'true',
                        help='Whether to run model in lazy or eager execution mode, default=True for lazy mode')
    parser.add_argument('--enable_packed_data_mode', default='True', type=lambda x: x.lower() == 'true',
                        help='enable/disable training with packed data. Default is True, --input_dir should be set accordingly')
    parser.add_argument('--use_zero_optimizer',
                        default='False', type=lambda x: x.lower() == 'true',
                        help='use zero optimizer')
    parser.add_argument('--compute_threshold',
                        default=-1,
                        type=float,
                        help='Stop FWD/BWD when the threshold is reached.'
                             ' units in seconds')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Debug mode. print more data related to drop'
                             ' compute. Might cause runtime overhead.')

    args = parser.parse_args()
    args.fp16 = args.fp16 or args.amp


    if args.steps_this_run < 0:
        args.steps_this_run = args.max_steps

    return args


def unflatten_tensor(flat, tensor_list):
    outputs = []
    offset = 0
    for tensor in tensor_list:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return outputs


def update_tensors(grad_tensors, outputs):
    idx=0
    for grad in grad_tensors:
        grad.copy_(outputs[idx])
        idx+=1
    return outputs


def setup_training(args):
    if args.use_habana:
        device = torch.device('hpu')

        if args.hmp:
            print(args.hmp_bf16)
            from habana_frameworks.torch.hpex import hmp
            hmp.convert(opt_level=args.hmp_opt_level,
                        bf16_file_path=args.hmp_bf16,
                        fp32_file_path=args.hmp_fp32,
                        isVerbose=args.hmp_verbose)

        args.n_pu = 1
        from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu
        args.world_size, args.rank, args.local_rank = initialize_distributed_hpu()
        if args.local_rank != -1:
            torch.distributed.init_process_group('hccl',
                    rank=args.rank, world_size=args.world_size)
        if args.local_rank != -1:
            args.allreduce_post_accumulation = True
            args.allreduce_post_accumulation_fp16 = True
        else:
            args.allreduce_post_accumulation = False
            args.allreduce_post_accumulation_fp16 = False

    elif args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        if device == torch.device("cuda"):
            args.n_pu = torch.cuda.device_count()
        else:
            args.n_pu = 1

        args.allreduce_post_accumulation = False
        args.allreduce_post_accumulation_fp16 = False
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
        os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
        os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.n_pu = 1

    if args.gradient_accumulation_steps == 1:
        args.allreduce_post_accumulation = False
        args.allreduce_post_accumulation_fp16 = False

    if utils.is_main_process():
        dllogger.init(backends=[
            dllogger.JSONStreamBackend(
                verbosity=dllogger.Verbosity.VERBOSE,
                filename=os.path.join(args.log_dir, 'dllogger.json')),
            dllogger.StdOutBackend(
                verbosity=dllogger.Verbosity.VERBOSE,
                step_format=utils.format_step)
        ])
    else:
        dllogger.init(backends=[])

    print(f'Rank: {utils.get_rank()} online.\t world size: '
          f'{utils.get_world_size()},\t n_pu: {args.n_pu},\t device: {device}, '
          f'distributed training: {bool(args.local_rank != -1)}, '
          f'16-bits training: {args.fp16 or args.hmp}')

    if args.gradient_accumulation_steps < 1:
        raise ValueError('Invalid gradient_accumulation_steps parameter: '
                         f'{args.gradient_accumulation_steps}, should be >= 1')
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError('Invalid gradient_accumulation_steps parameter: '
                         f'{args.gradient_accumulation_steps}, batch size '
                         f'{args.train_batch_size} should be divisible')

    args.train_batch_size = (
            args.train_batch_size // args.gradient_accumulation_steps)

    if args.enable_packed_data_mode:
        args.gradient_accumulation_steps = round(
            args.gradient_accumulation_steps / avg_seq_per_pack)

    if not args.do_train:
        raise ValueError('`do_train` must be True.')

    if not args.resume_from_checkpoint and os.path.exists(args.output_dir) and (
            any([i.startswith('ckpt') for i in os.listdir(args.output_dir)])):
        raise ValueError(f'Output directory ({args.output_dir}) already exists '
                         'and is not empty.')

    if (not args.resume_from_checkpoint or (
            not os.path.exists(args.output_dir))) and utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args


def set_hooks(module, models_to_hook):
    try:
        import habana_frameworks.torch.core as htcore
    except ImportError:
        assert False, "Could Not import habana_frameworks.torch.core"

    def get_hook_func(module_name: str):
        def log_time(*args):
            htcore.mark_step()
            global_drop_timer.check_drop_compute_throw(module_name)
        return log_time

    expression = re.compile("|".join(models_to_hook))
    for name, module in module.named_modules():
        if expression.fullmatch(name) is not None:
            if utils.is_main_process():
                print(f"Hooking module: {name}")
            module.register_forward_hook(get_hook_func('_'.join([name, 'fwd'])))
            #module.register_backward_hook(get_hook_func('_'.join([name, 'bwd'])))


def prepare_model_and_optimizer(args, device):

    # Prepare model
    config = modeling.BertConfig.from_json_file(args.config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForPreTraining(config)

    checkpoint = None
    if not args.resume_from_checkpoint:
        global_step = 0
    else:
        if args.resume_step == -1 and not args.init_checkpoint:
            model_names = [
                f for f in os.listdir(args.checkpoint_dir) if f.endswith('.pt')]
            args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])

        global_step = args.resume_step if not args.init_checkpoint else 0

        if not args.init_checkpoint:
            checkpoint = torch.load(
                os.path.join(args.checkpoint_dir, f'ckpt_{global_step}.pt'),
                map_location='cpu'
            )
        else:
            checkpoint = torch.load(args.init_checkpoint, map_location='cpu')

        model.load_state_dict(checkpoint['model'], strict=False)

        if args.phase2 and not args.init_checkpoint:
            global_step -= args.phase1_end_step
        if utils.is_main_process():
            print(f'resume step from {args.resume_step}')

    model.to(device)
    # BERT modeling  uses weight sharing between word embedding and prediction
    # decoder. So make sure the storage is pointing properly even after model is
    # moved to device.
    if args.use_habana:
        model.cls.predictions.decoder.weight = model.bert.embeddings.word_embeddings.weight

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    if args.use_habana:
        if args.use_fused_lamb:
            try:
                from habana_frameworks.torch.hpex.optimizers import FusedLamb
            except ImportError:
                raise ImportError('Please install hbopt.')
            optimizer_cls = FusedLamb
        else:
            optimizer_cls = lamb.NVLAMB
    else:
        if torch.cuda.is_available():
            optimizer_cls = apex.optimizers.FusedLAMB
        else:
            optimizer_cls = lamb.NVLAMB
    if args.local_rank != -1 and args.use_zero_optimizer:
        optimizer = torch.distributed.optim.ZeroRedundancyOptimizer(
            optimizer_grouped_parameters[0]['params'],
            optimizer_class=optimizer_cls,
            lr=args.learning_rate,
            weight_decay=optimizer_grouped_parameters[0]['weight_decay']
        )
        for pg in optimizer_grouped_parameters[1:]:
            optimizer.add_param_group(pg)
    else:
        optimizer = optimizer_cls(optimizer_grouped_parameters,
                                  lr=args.learning_rate)

    lr_scheduler = schedulers.PolyWarmUpScheduler(optimizer,
                                                  warmup=args.warmup_proportion,
                                                  total_steps=args.max_steps)
    if args.fp16:
        if args.loss_scale == 0:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2',
                                              loss_scale='dynamic',
                                              cast_model_outputs=torch.float16)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2',
                                              loss_scale=args.loss_scale,
                                              cast_model_outputs=torch.float16)
        amp._amp_state.loss_scalers[0]._loss_scale = args.init_loss_scale

    model.checkpoint_activations(args.checkpoint_activations)

    if args.resume_from_checkpoint:
        if args.phase2 or args.init_checkpoint:
            keys = list(checkpoint['optimizer']['state'].keys())
            # override hyperparameters from previous checkpoint
            for key in keys:
                checkpoint['optimizer']['state'][key]['step'] = global_step
            # TODO(ngiladi): change enumerate to range, only idx is used.
            for idx, _ in enumerate(checkpoint['optimizer']['param_groups']):
                checkpoint['optimizer']['param_groups'][idx]['step'] = global_step
                checkpoint['optimizer']['param_groups'][idx]['t_total'] = args.max_steps
                checkpoint['optimizer']['param_groups'][idx]['warmup'] = args.warmup_proportion
                checkpoint['optimizer']['param_groups'][idx]['lr'] = args.learning_rate
        optimizer.load_state_dict(checkpoint['optimizer'])  # , strict=False)

        # Restore AMP master parameters
        if args.fp16:
            optimizer._lazy_init_maybe_master_weights()
            optimizer._amp_stash.lazy_init_called = True
            optimizer.load_state_dict(checkpoint['optimizer'])
            for param, saved_param in zip(amp.master_params(optimizer),
                                          checkpoint['master params']):
                param.data.copy_(saved_param.data)

    if args.local_rank != -1:
        if not args.allreduce_post_accumulation:
            if args.use_habana:
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    bucket_cap_mb=230
                )
            else:
                model = apex.parallel.DistributedDataParallelDDP(
                    model,
                    message_size=250_000_000,
                    gradient_predivide_factor=utils.get_world_size()
                )
        else:
            if args.use_habana:
                for param in model.parameters():
                    torch.distributed.broadcast(param.data, 0)
            else:
                flat_dist_call([param.data for param in model.parameters()],
                               torch.distributed.broadcast, (0,))
    elif args.n_pu > 1:
        model = torch.nn.DataParallel(model)

    criterion = BertPretrainingCriterion(config.vocab_size)

    hooked_modules = [
        "bert.embeddings",
        "bert.encoder.layer.\d+",
        "bert.pooler",
        "loss_fn"
    ]

    set_hooks(model, hooked_modules)

    return model, optimizer, lr_scheduler, checkpoint, global_step, criterion


def take_optimizer_step(args, optimizer, model, overflow_buf, global_step):

    global skipped_steps
    if args.allreduce_post_accumulation and not args.use_habana:
        # manually allreduce gradients after all accumulation steps
        # check for Inf/NaN
        # 1. allocate an uninitialized buffer for flattened gradient
        loss_scale = _amp_state.loss_scalers[0].loss_scale() if args.fp16 else 1
        master_grads = [p.grad for p in amp.master_params(optimizer) if p.grad is not None]
        flat_grad_size = sum(p.numel() for p in master_grads)
        allreduce_dtype = torch.float16 if args.allreduce_post_accumulation_fp16 else torch.float32
        flat_raw = torch.empty(flat_grad_size, device='cuda', dtype=allreduce_dtype)
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(
            65536,
            overflow_buf,
            [master_grads, allreduced_views],
            loss_scale / (
                    utils.get_world_size() * args.gradient_accumulation_steps)
        )
        # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
        torch.distributed.all_reduce(flat_raw)
        # 4. combine unscaling and unflattening of allreduced gradient
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(65536,
                                 overflow_buf,
                                 [allreduced_views, master_grads],
                                 1./loss_scale)
        # 5. update loss scale
        if args.fp16:
            scaler = _amp_state.loss_scalers[0]
            old_overflow_buf = scaler._overflow_buf
            scaler._overflow_buf = overflow_buf
            had_overflow = scaler.update_scale()
            scaler._overfloat_buf = old_overflow_buf
        else:
            had_overflow = 0
        # 6. call optimizer step function
        if had_overflow == 0:
            if args.use_habana and args.hmp:
                from habana_frameworks.torch.hpex import hmp
                with hmp.disable_casts():
                    optimizer.step()
            else:
                optimizer.step()
            global_step += 1
        else:
            # Overflow detected, print message and clear gradients
            skipped_steps += 1
            if utils.is_main_process():
                scaler = _amp_state.loss_scalers[0]
                dllogger.log(step="PARAMETER", data={"loss_scale": scaler.loss_scale()})
            if _amp_state.opt_properties.master_weights:
                for param in optimizer._amp_stash.all_fp32_from_fp16_params:
                    param.grad = None
        for param in model.parameters():
            param.grad = None
    else:
        # In case of parameter tying allreduce was called twice for the
        # parameters. Manually adding allreduce for the parameters.
        if args.use_habana and args.allreduce_post_accumulation:
            grad_tensors = [param.grad for param in model.parameters() if param.grad is not None]
            flat_tensor = torch.cat([t.contiguous().view(-1) for t in grad_tensors], dim=0)
            flat_tensor.div_(float(torch.distributed.get_world_size() * args.gradient_accumulation_steps))
            torch.distributed.all_reduce(flat_tensor)
            outputs = unflatten_tensor(flat_tensor, grad_tensors)
            updated_outputs = update_tensors(grad_tensors, outputs)

        if args.use_habana and args.hmp:
            from habana_frameworks.torch.hpex import hmp
            with hmp.disable_casts():
                optimizer.step()
        else:
            optimizer.step()
        #optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None
        global_step += 1

    return global_step


def get_metadata_file_path(input_dir: str) -> str:
    norm_path = os.path.normpath(input_dir)
    head_tail = os.path.split(norm_path)
    metadata_file_name = head_tail[1]
    metadata_file_name = metadata_file_name + '_metadata.json'
    metadata_file_path = os.path.join(head_tail[0],metadata_file_name)
    return metadata_file_path


def read_avg_seq_per_sample(input_dir: str, max_sequence_length) -> float:
    metadata = None
    metadata_file_path = get_metadata_file_path(input_dir)
    print(f"Reading dataset metadata from: {metadata_file_path}")
    if os.path.exists(metadata_file_path):
        file_handle = open(metadata_file_path, mode='r')
        json_content = file_handle.read()
        metadata = json.loads(json_content)
    else:
        print("Packed dataset metadata file not accessible, falling back to default values of avg_seq_per_sample")
        if max_sequence_length == 128:
            return 1.2
        elif max_sequence_length == 512:
            return 2.0
        else:
            assert f"invalid max_sequence_length"
    avg_seq_per_sample_key = "avg_seq_per_sample"
    if metadata is not None and avg_seq_per_sample_key in metadata.keys():
        avg_seq_per_sample = metadata[avg_seq_per_sample_key]
    else:
        assert False, f"Key {avg_seq_per_sample_key} not present in packed dataset metadata file: {metadata_file_path}"
    print(f"AVG_SEQ_PER_SAMPLE: {avg_seq_per_sample}")
    return avg_seq_per_sample


def main():
    global timeout_sent
    global avg_seq_per_pack

    args = parse_arguments()

    random.seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)
    worker_init = WorkerInitObj(args.seed + args.local_rank)
    if args.enable_packed_data_mode:
        avg_seq_per_pack = read_avg_seq_per_sample(args.input_dir,
                                                   args.max_seq_length)
    elif args.local_rank <= 0:
        warnings.warn('--enable_packed_data_mode flag will be deprecated and '
                      'usage of packed and unpacked dataset will be decided '
                      'based on metadata file availability at input_dir')
        avg_seq_per_pack = 1.0
    device, args = setup_training(args)

    if args.use_habana:
        try:
            import habana_frameworks.torch as ht
        except ImportError:
            assert False, "Could not import habana_frameworks.torch"
        if args.use_lazy_mode:
            try:
                import habana_frameworks.torch.core as htcore
            except ImportError:
                assert False, "Could Not import habana_frameworks.torch.core"
        else:
            os.environ["PT_HPU_LAZY_MODE"] = "2"

    dllogger.log(step="PARAMETER", data={"Config": [str(args)]})

    # Prepare optimizer
    (
        model,
        optimizer,
        lr_scheduler,
        checkpoint,
        global_step,
        criterion
    ) = prepare_model_and_optimizer(args, device)

    if utils.is_main_process():
        dllogger.log(step="PARAMETER", data={"SEED": args.seed})

    raw_train_start = None
    if not args.do_train:
        if args.use_lazy_mode and args.use_habana:
            os.environ.pop("PT_HPU_LAZY_MODE")
        return

    if utils.is_main_process():
        dllogger.log(step='PARAMETER',
                     data={'train_start': True})
        dllogger.log(step='PARAMETER',
                     data={'batch_size_per_pu': args.train_batch_size})
        dllogger.log(step='PARAMETER',
                     data={'learning_rate': args.learning_rate})

    model.train()
    most_recent_ckpts_paths = []
    average_loss = 0.0  # averaged loss every args.log_freq steps
    epoch = 0
    training_steps = 0
    average_training_time_per_step = 0
    average_perf_per_step = 0
    loss_list = []

    if device.type == 'cuda':
        pool = ProcessPoolExecutor(1)

    compute_state = ComputeState(device)
    starting_time = time.time()
    # loop infinitely over epochs, termination is handled via iteration count
    time_logs = []
    while True:
        restored_data_loader = None
        if not args.resume_from_checkpoint or epoch > 0 or (args.phase2 and global_step < 1) or args.init_checkpoint:
            files = []
            for file in os.listdir(args.input_dir):
                if os.path.isfile(os.path.join(args.input_dir, file)):
                    # Packed files have no 'training' pre/postfix.
                    if args.enable_packed_data_mode or 'training' in file:
                        files.append(os.path.join(args.input_dir, file))
            files.sort()
            num_files = len(files)
            random.Random(args.seed + epoch).shuffle(files)
            f_start_id = 0
        else:
            f_start_id = checkpoint['files'][0]
            files = checkpoint['files'][1:]
            args.resume_from_checkpoint = False
            num_files = len(files)
            # may not exist in all checkpoints
            epoch = checkpoint.get('epoch', 0)
            restored_data_loader = checkpoint.get('data_loader', None)

        shared_file_list = {}

        if torch.distributed.is_initialized() and (
                utils.get_world_size() > num_files):
            remainder = utils.get_world_size() % num_files
            data_file = files[(f_start_id * utils.get_world_size() + utils.get_rank() + remainder * f_start_id) % num_files]
        else:
            data_file = files[(f_start_id * utils.get_world_size() + utils.get_rank()) % num_files]

        previous_file = data_file

        if restored_data_loader is None:
            num_workers = 0 if args.use_habana else 4
            train_data = PretrainingDataset(data_file,
                                            args.max_predictions_per_seq,
                                            args.enable_packed_data_mode)
            train_sampler = torch.utils.data.RandomSampler(train_data)
            train_dataloader = torch.utils.data.DataLoader(
                train_data, sampler=train_sampler,
                batch_size=args.train_batch_size * args.n_pu,
                num_workers=num_workers,
                worker_init_fn=worker_init,
                drop_last=True,
                pin_memory=True
            )
            # shared_file_list["0"] = (train_dataloader, data_file)
        else:
            train_dataloader = restored_data_loader
            restored_data_loader = None

        overflow_buf = None
        if args.allreduce_post_accumulation and not args.use_habana:
            overflow_buf = torch.cuda.IntTensor([0])

        for f_id in range(f_start_id + 1, len(files)):

            if utils.get_world_size() > num_files:
                data_file = files[(f_id * utils.get_world_size() + utils.get_rank() + remainder * f_id) % num_files]
            else:
                data_file = files[(f_id * utils.get_world_size() + utils.get_rank()) % num_files]

            previous_file = data_file

            if device.type == 'cuda':
                dataset_future = pool.submit(create_pretraining_dataset,
                                             data_file,
                                             args.max_predictions_per_seq,
                                             shared_file_list,
                                             args,
                                             worker_init)

            if utils.is_main_process():
                train_iter = tqdm(
                    train_dataloader,
                    desc="Iteration",
                    disable=args.disable_progress_bar)
            else:
                train_iter = train_dataloader

            if raw_train_start is None:
                raw_train_start = time.time()
            for batch in train_iter:  # delayed update loop

                training_steps += 1
                local_step = training_steps % args.gradient_accumulation_steps
                is_optimizer_step = (local_step == 0)

                batch = [t.to(device) for t in batch]
                if args.enable_packed_data_mode:
                    input_ids, segment_ids, input_mask, positions, masked_lm_labels, next_sentence_positions, next_sentence_labels = batch
                else:
                    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
                    next_sentence_positions = None
                    positions = None

                batch = input_mask.shape[0]
                sentence_length = input_mask.shape[1]
                try:
                    if local_step == 1:
                        fwd_start = compute_state.start_compute
                    else:
                        fwd_start = time.time()
                    compute_dropped = False
                    # Forward pass
                    if args.local_rank != -1 and not is_optimizer_step and (
                            not args.allreduce_post_accumulation):
                        with model.no_sync():
                            prediction_scores, seq_relationship_score = model(
                                input_ids=input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask,
                                enable_packed_data_mode=(
                                    args.enable_packed_data_mode),
                                positions=positions,
                                next_sentence_positions=next_sentence_positions
                            )
                    else:
                        prediction_scores, seq_relationship_score = model(
                            input_ids=input_ids,
                            token_type_ids=segment_ids,
                            attention_mask=input_mask,
                            enable_packed_data_mode=(
                                args.enable_packed_data_mode),
                            positions=positions,
                            next_sentence_positions=next_sentence_positions
                        )

                    loss = criterion(prediction_scores, seq_relationship_score,
                                     masked_lm_labels, next_sentence_labels)
                    if args.n_pu > 1:
                        loss = loss.mean()  # mean() to average on multi-pu.

                    # Backward pass
                    divisor = args.gradient_accumulation_steps
                    if divisor > 1:
                        if not args.allreduce_post_accumulation:
                            # this division was merged into predivision
                            loss = loss / divisor
                    if args.fp16:
                        with amp.scale_loss(
                                loss,
                                optimizer,
                                delay_overflow_check=(
                                        args.allreduce_post_accumulation)
                        ) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    compute_state.computed_batch_size += (
                        compute_state.mini_batch_size)
                except ComputeTimeout:
                    fwd_start = time.time()
                    is_optimizer_step = True  # just straight to all-reduce
                    if args.debug:
                        print(f'Rank {utils.get_rank()} dropped '
                              f'{local_step}/'
                              f'{args.gradient_accumulation_steps}')
                    pass
                # End Compute

                if args.use_lazy_mode and args.use_habana:
                    htcore.mark_step()  # not a blocking step
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                loss_list.append(loss)
                if is_optimizer_step:
                    step_end = time.time()
                    lr_scheduler.step()  # learning rate warmup
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(
                            compute_state.computed_batch_size)
                    global_step = take_optimizer_step(
                        args, optimizer, model, overflow_buf, global_step)
                    if utils.is_main_process() and args.debug:
                        print(f'Rank {utils.get_rank()} STEP'
                              f' {global_step} compute logs '
                              f'{compute_state.computed_batch_size}')
                    if args.use_lazy_mode and args.use_habana:
                        htcore.mark_step()
                    global_drop_timer.reset()
                    global_drop_timer.start()
                    global_drop_timer.debug = args.debug
                    global_drop_timer.drop_threshold = args.compute_threshold
                    global_drop_timer.enable_drop_compute = (global_step > 5 and (compute_state.threshold > 0))
                    compute_state.reset_state(
                        compute_threshold=args.compute_threshold,
                        enable_drop=(global_step > 5 and (
                                compute_state.threshold > 0)),
                        start_compute=time.time(),
                        mini_batch_size=len(input_ids)
                    )
                    if global_step == 6:
                        start_train_timestamp = time.time()
                else:
                    computed_batch = compute_state.computed_batch_size.item()
                    step_end = time.time()
                if global_step > 5:
                    time_logs.append((global_step,
                                      local_step,
                                      utils.get_world_size(),
                                      batch,
                                      sentence_length,
                                      compute_state.computed_batch_size.item(),
                                      compute_dropped,
                                      fwd_start,
                                      step_end))
                if global_step >= args.steps_this_run or timeout_sent or training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                    for loss_t in loss_list:
                        average_loss += loss_t.item()
                    loss_list.clear()
                    train_time = time.time() - starting_time
                    starting_time = time.time()
                    average_training_time_per_step = train_time/(args.gradient_accumulation_steps * args.log_freq)
                    average_perf_per_step = args.train_batch_size*avg_seq_per_pack/average_training_time_per_step

                if global_step >= args.steps_this_run or timeout_sent:  # end of training
                    train_time_raw = time.time() - raw_train_start
                    last_num_steps = int(training_steps / args.gradient_accumulation_steps) % args.log_freq
                    last_num_steps = args.log_freq if last_num_steps == 0 else last_num_steps
                    average_loss = average_loss / (last_num_steps * divisor)
                    average_loss = torch.tensor(average_loss, dtype=torch.float32).to(device)
                    if torch.distributed.is_initialized():
                        average_loss /= utils.get_world_size()
                        torch.distributed.barrier()  # TODO(ngiladi): not necessary
                        torch.distributed.all_reduce(average_loss)  # TODO(ngiladi): why necessary?
                    final_loss = average_loss.item()
                    net_train_time = time.time() - start_train_timestamp
                    if utils.is_main_process():
                        dllogger.log(step=(epoch, global_step, ), data={
                            'final_loss':
                                f'{final_loss:3.4}',
                            'average_training_time_step':
                                f'{average_training_time_per_step:3.4}',
                            'average_perf_per_step':
                                f'{average_perf_per_step:3.4}',
                            'train_time_net':
                                f'{net_train_time:3.4f}'
                        })
                elif training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                    if utils.is_main_process():
                        dllogger.log(step=(epoch, global_step, ), data={
                            'average_loss':
                                f'{average_loss / (args.log_freq * divisor):3.4}',
                            'step_loss':
                                f'{loss.item() * args.gradient_accumulation_steps / divisor:3.4}',
                            'learning_rate':
                                f'{optimizer.param_groups[0]["lr"]:3.4}',
                            'average_training_time_step':
                                f'{average_training_time_per_step:3.4}',
                            'average_perf_per_step':
                                f'{average_perf_per_step:3.4}'
                        })
                    average_loss = 0

                if global_step >= args.steps_this_run or training_steps % (
                        args.num_steps_per_checkpoint * args.gradient_accumulation_steps) == 0 or timeout_sent:
                    if isinstance(
                            optimizer,
                            torch.distributed.optim.ZeroRedundancyOptimizer
                    ):
                        optimizer.consolidate_state_dict()
                    if utils.is_main_process() and not args.skip_checkpoint:
                        # Save a trained model
                        dllogger.log(step="PARAMETER", data={"checkpoint_step": global_step})
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Only save the model it-self
                        if args.resume_step < 0 or not args.phase2:
                            output_save_file = os.path.join(
                                args.output_dir,
                                "ckpt_{}.pt".format(global_step)
                            )
                        else:
                            output_save_file = os.path.join(
                                args.output_dir,
                                "ckpt_{}.pt".format(
                                    global_step + args.phase1_end_step)
                            )
                        checkpoint_dict = {}
                        if args.do_train:
                            if args.use_habana or args.no_cuda:
                                checkpoint_dict = {
                                    'model': model_to_save.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'files': [f_id] + files,
                                    'epoch': epoch,
                                    'data_loader': None if global_step >= args.max_steps else train_dataloader
                                }
                            else:
                                checkpoint_dict = {
                                    'model': model_to_save.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'master params':
                                        list(amp.master_params(optimizer)),
                                    'files': [f_id] + files,
                                    'epoch': epoch,
                                    'data_loader': None if global_step >= args.max_steps else train_dataloader}

                            torch.save(checkpoint_dict, output_save_file)
                            most_recent_ckpts_paths.append(output_save_file)
                            if len(most_recent_ckpts_paths) > 3:
                                ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                                os.remove(ckpt_to_be_removed)

                    # Exiting the training due to hitting max steps, or being sent a
                    # timeout from the cluster scheduler
                    if global_step >= args.steps_this_run or timeout_sent:
                        with open(os.path.join(
                                args.log_dir,
                                f'compute_logs_{utils.get_rank()}.csv'
                        ), 'w') as file:
                             file.write(pd.DataFrame(time_logs, columns=(
                                 'global_step',
                                 'local_step',
                                 'world_size',
                                 'batch',
                                 'sentence_length',
                                 'computed_batch',
                                 'compute_dropped',
                                 'fwd_start',
                                 'step_end'
                             )).to_csv())
                        del train_dataloader
                        return args, final_loss, train_time_raw, global_step
            del train_dataloader
            # Make sure pool has finished and switch train_dataloader
            # NOTE: Will block until complete
            if device.type == 'cuda':
                train_dataloader, data_file = dataset_future.result(timeout=None)
            else:
                train_dataloader, data_file = create_pretraining_dataset(
                    data_file,
                    args.max_predictions_per_seq,
                    shared_file_list,
                    args,
                    worker_init
                )
        epoch += 1


if __name__ == "__main__":

    now = time.time()
    args, final_loss, train_time_raw, global_step = main()
    pu_count = args.n_pu
    global_step += args.phase1_end_step if (args.phase2 and args.resume_step > 0) else 0
    if args.resume_step == -1:
        args.resume_step = 0
    if torch.distributed.is_initialized():
        pu_count = utils.get_world_size()
    if utils.is_main_process():
        e2e_time = time.time() - now
        training_perf = args.train_batch_size * args.gradient_accumulation_steps * pu_count * avg_seq_per_pack\
                        * (global_step - args.resume_step + skipped_steps) / train_time_raw
        dllogger.log(step=tuple(),
                     data={'e2e_train_time': e2e_time,
                           'training_sequences_per_second': training_perf,
                           'final_loss': final_loss,
                           'raw_train_time': train_time_raw})
    dllogger.flush()

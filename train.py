# Adapted from https://github.com/NVIDIA/waveglow under the BSD 3-Clause License.

# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import os
import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor

from dataset import load_cleanunet_dataset
from stft_loss import MultiResolutionSTFTLoss
from util import rescale, find_max_epoch, print_size
from util import LinearWarmupCosineDecay, loss_cleanunet
from util import prepare_directories_and_logger, save_checkpoint, load_checkpoint
from network import CleanUNet
from logger import Logger

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

#torch.autograd.set_detect_anomaly(True)


def validate(model, val_loader, loss_fn, iteration, trainset_config, logger, rank, device):
    model.eval()
    val_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for i, (noisy_audio, clean_audio) in enumerate(val_loader):
            noisy_audio, clean_audio = noisy_audio.to(device), clean_audio.to(device)
            denoised_audio= model(noisy_audio)  
            loss, loss_dict = loss_fn(clean_audio, denoised_audio)
            val_loss += loss
            num_batches += 1
        val_loss /= num_batches

    model.train()

    if rank == 0 and logger is not None:
        print(f"Validation loss at iteration {iteration}: {val_loss:.6f}")
        # save to tensorboard
        logger.add_scalar("Validation/Loss", val_loss, iteration)
        num_samples = min(4, clean_audio.size(0))

        for i in range(num_samples):
            clean_audio_i = clean_audio[i].squeeze()
            denoised_audio_i = denoised_audio[i].squeeze()
            noisy_audio_i = noisy_audio[i].squeeze()

            clean_audio_np = clean_audio_i.cpu().numpy()
            denoised_audio_np = denoised_audio_i.cpu().numpy()
            noisy_audio_np = noisy_audio_i.cpu().numpy()                      
            
           # Plot waveforms
            fig_waveform, axs_waveform = plt.subplots(1, 3, figsize=(15, 5))
            axs_waveform[0].plot(clean_audio_np)
            axs_waveform[0].set_title('Clean Waveform')
            axs_waveform[1].plot(denoised_audio_np)
            axs_waveform[1].set_title('Denoised Waveform')
            axs_waveform[2].plot(noisy_audio_np)
            axs_waveform[2].set_title('Noisy Waveform')
            plt.tight_layout()
            logger.add_figure('Waveforms/Sample_{}'.format(i), fig_waveform, iteration)
            plt.close(fig_waveform)

            # Log audio samples to TensorBoard
            sample_rate = trainset_config['sample_rate']
            logger.add_audio('Audio/Clean_{}'.format(i), clean_audio_np, iteration, sample_rate=sample_rate)
            logger.add_audio('Audio/Denoised_{}'.format(i), denoised_audio_np, iteration, sample_rate=sample_rate)
            logger.add_audio('Audio/Noisy_{}'.format(i), noisy_audio_np, iteration, sample_rate=sample_rate)


def train(num_gpus, rank, group_name, exp_path, checkpoint_path, log, optimization, loss_config, device=None):
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup local experiment path
    if rank == 0:
        print('exp_path:', exp_path)
    
    # Create tensorboard logger.
    output_dir = os.path.join(log["directory"], exp_path)
    log_directory = os.path.join(output_dir, 'logs')
    ckpt_directory = os.path.join(output_dir, 'checkpoint')

    if rank == 0:
        logger = prepare_directories_and_logger(
            log_directory, log_directory, ckpt_directory, rank=0)
    
    # distributed running initialization
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)

    # Get shared ckpt_directory ready
    ckpt_directory = os.path.join(log_directory, 'checkpoint')
    if rank == 0:
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
            os.chmod(ckpt_directory, 0o775)
        print("ckpt_directory: ", ckpt_directory, flush=True)

    # load training data
    trainloader, testloader = load_cleanunet_dataset(**trainset_config, 
                                batch_size=optimization["batch_size_per_gpu"], 
                                num_gpus=num_gpus)
    print('Data loaded')
    # initialize the model
    model = CleanUNet(**network_config).to(device)
    model.train()
    print_size(model)

    # apply gradient all reduce
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=optimization["learning_rate"], weight_decay=optimization["weight_decay"])
    
    # load checkpoint
    global_step = 0
    if checkpoint_path is not None:
        try:
            model, optimizer, _learning_rate, iteration = load_checkpoint(checkpoint_path, model)
            print('Model at %s has been loaded' % (checkpoint_path))
            print('Checkpoint model loaded successfully')            
            global_step = iteration + 1
        except:
            print(f'No valid checkpoint model found at {checkpoint_path}, start training from initialization.')           

    # define learning rate scheduler
    scheduler = LinearWarmupCosineDecay(
                    optimizer,
                    lr_max=optimization["learning_rate"],
                    n_iter=optimization["n_iters"],
                    iteration=global_step,
                    divider=25,
                    warmup_proportion=0.05,
                    phase=('linear', 'cosine'),
                )

    # define multi resolution stft loss
    if loss_config["stft_lambda"] > 0:
        mrstftloss = MultiResolutionSTFTLoss(**loss_config["stft_config"]).to(device)
    else:
        mrstftloss = None

    loss_fn = loss_cleanunet(**loss_config, mrstftloss=mrstftloss)

    # training
    epoch = 1
    print("Starting training...")    
    while global_step < optimization["n_iters"] + 1:
        # for each epoch
        for step, (noisy_audio, clean_audio) in enumerate(trainloader): 
            
            noisy_audio = noisy_audio.to(device)
            clean_audio = clean_audio.to(device)

            optimizer.zero_grad()
            # forward propagation
            denoised_audio = model(noisy_audio)  
            # calculate loss
            loss, _ = loss_fn(clean_audio, denoised_audio)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()

            # back-propagation
            loss.backward()
            # gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), optimization['max_norm'])
            # update learning rate
            scheduler.step()
            # update model parameters
            optimizer.step()

            print(f"Epoch: {epoch:<5} step: {step:<6} global step {global_step:<7} loss: {loss.item():.7f}", flush=True)

            if global_step > 0 and global_step % 10 == 0 and rank == 0: 
                # save to tensorboard
                logger.add_scalar("Train/Train-Loss", reduced_loss, global_step)
                #logger.add_scalar("Train/Train-Reduced-Loss", reduced_loss, global_step)
                logger.add_scalar("Train/Gradient-Norm", grad_norm, global_step)                
                logger.add_scalar("Train/learning-rate", optimizer.param_groups[0]["lr"], global_step)                    

            if global_step > 0 and global_step % log["iters_per_valid"] == 0 and rank == 0:
                validate(model, testloader, loss_fn, global_step, trainset_config, logger, rank, device)

            # save checkpoint
            if global_step > 0 and global_step % log["iters_per_ckpt"] == 0 and rank == 0:
                checkpoint_name = '{}.pkl'.format(global_step)
                checkpoint_path = os.path.join(ckpt_directory, checkpoint_name)
                save_checkpoint(model, optimizer, optimization['learning_rate'], global_step, checkpoint_path)
                print('model at iteration %s is saved' % global_step)

            global_step += 1

        epoch += 1            

    # After training, close TensorBoard.
    if rank == 0:
        logger.close()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config.json', 
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)

    train_config            = config["train_config"]        # training parameters
    global dist_config
    dist_config             = config["dist_config"]         # to initialize distributed training
    global network_config
    network_config          = config["network_config"]      # to define network
    global trainset_config
    trainset_config         = config["trainset_config"]     # to load trainset

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU. Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    train(num_gpus, args.rank, args.group_name, **train_config)

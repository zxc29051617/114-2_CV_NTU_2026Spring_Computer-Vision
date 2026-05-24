# ============================================================================
# File: config.py
# Date: 2026-03-27
# Author: TA
# Description: Experiment configurations.
# ============================================================================

################################################################
# NOTE:                                                        #
# You can modify these values to train with different settings #
# p.s. this file is only for training                          #
################################################################

# Experiment Settings
exp_name = 'default'  # name of experiment
# exp_name = 'resnet18_best'  # my best resnet18 model
# exp_name = 'mynet_best'  # my best mynet model

# Model Options
model_type = 'resnet18'  # 'mynet' or 'resnet18'
# model_type = 'mynet'

# Learning Options
epochs = 50                # train how many epochs
batch_size = 32            # batch size for dataloader 
use_adam = False           # Adam or SGD optimizer
lr = 1e-2                  # learning rate
milestones = [16, 32, 45]  # reduce learning rate at 'milestones' epochs
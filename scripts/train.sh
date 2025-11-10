#!/bin/bash
# Training script for Adaptive Teaching System (ATS)
# cd /home/xxxxxxxx/Adaptive-Teaching-System-main
BRAIN_BACKBONE="Shared_Temporal_Attention_Encoder"
VISION_BACKBONE="RN50"
ADAPTOR_BACKBONE="ShrinkAdapter"
SUB="01"
SEED=0
TRAIN_BATCH_SIZE=1024
EPOCH=150
LR=1e-4
LAMBDA_IMG2IMG=0.0
LAMBDA_IMG2EEG=0.0
LAMBDA_MMD=0.0
LAMBDA_MAP=0.0
EXP_SETTING="intra-subject"
DEVICE="cuda:0"
CONFIG="configs/ats.yaml"

python main.py \
    --config $CONFIG \
    --subjects sub-$SUB \
    --seed $SEED \
    --exp_setting $EXP_SETTING \
    --brain_backbone $BRAIN_BACKBONE \
    --vision_backbone $VISION_BACKBONE \
    --adaptor_backbone $ADAPTOR_BACKBONE \
    --epoch $EPOCH \
    --lr $LR \
    --device $DEVICE \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --lambda_img2img $LAMBDA_IMG2IMG \
    --lambda_img2eeg $LAMBDA_IMG2EEG \
    --lambda_mmd $LAMBDA_MMD \
    --lambda_map $LAMBDA_MAP

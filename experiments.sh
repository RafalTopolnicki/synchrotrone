#!/usr/bin/env bash
# Proposed experiments for the synchrotron object detection pipeline.
# Run with: bash experiments.sh
# Each run saves its own timestamped folder under runs/.
# Monitor all runs live with: tensorboard --logdir runs

set -e
source ~/activate.sh madrid

# ── shared settings ───────────────────────────────────────────────────────────
EPOCHS=100        # cap; early stopping (patience=10) will usually fire first
PATIENCE=10
BS=12             # safe for RTX 4090 at 640×640; bump to 8 if VRAM allows

# ── experiment 1: ResNet50 baseline ──────────────────────────────────────────
# Standard choice — good balance of accuracy and training speed.
#echo "=== EXP 1: ResNet50 baseline ==="
#python train.py \
#    --backbone resnet50 \
#    --epochs   $EPOCHS \
#    --patience $PATIENCE \
#    --batch_size $BS \
#    --lr 0.005

# ── experiment 2: ResNet50v2 ──────────────────────────────────────────────────
# Improved version of ResNet50: better normalization layers and training recipe.
# Typically 1-2 AP points better than v1 at the same speed.
#echo "=== EXP 2: ResNet50v2 ==="
#python train.py \
#    --backbone resnet50v2 \
#    --epochs   $EPOCHS \
#    --patience $PATIENCE \
#    --batch_size $BS \
#    --lr 0.005

# ── experiment 3: MobileNet — fast iteration ──────────────────────────────────
# 2× fewer parameters → faster training and inference.
# Good to run first to quickly check data/pipeline is working.
#echo "=== EXP 3: MobileNet (fast) ==="
#python train.py \
#    --backbone mobilenet \
#    --epochs   $EPOCHS \
#    --patience $PATIENCE \
#    --batch_size $BS \
#    --lr 0.005

# ── experiment 4: ResNet50 with lower LR ─────────────────────────────────────
# Smaller LR = more stable training, may converge to a better minimum.
# Useful if exp 1 shows noisy / oscillating val loss.
#echo "=== EXP 4: ResNet50, lr=0.001 ==="
#python train.py \
#    --backbone resnet50 \
#    --epochs   $EPOCHS \
#    --patience $PATIENCE \
#    --batch_size $BS \
#    --lr 0.001

# ── experiment 5: ResNet50v2 with larger batch ────────────────────────────────
# Larger batch = more stable gradient estimates.
# RTX 4090 has 24 GB — batch 8 should fit comfortably.
#echo "=== EXP 5: ResNet50v2, bs=8 ==="
#python train.py \
#    --backbone   resnet50v2 \
#    --epochs     $EPOCHS \
#    --patience   $PATIENCE \
#    --batch_size 8 \
#    --lr 0.005

# ── experiment 6: merged dunes — ResNet50v2 champion config ──────────────────
# CoR_dune_up + CoR_dune_down → single CoR_dune class.
# Simpler task: 2 classes instead of 5, more positive examples per class,
# smaller classifier head. Should reduce overfitting and improve AP on dunes.
#echo "=== EXP 6: ResNet50v2, merge_dunes ==="
#python train.py \
#    --backbone   resnet50v2 \
#    --epochs     $EPOCHS \
#    --patience   $PATIENCE \
#    --batch_size $BS \
#    --lr 0.005 \
#    --merge_dunes

# ── experiment 7: merged dunes — lower LR ────────────────────────────────────
# Given early overfitting in all runs, try a gentler learning rate.
#echo "=== EXP 7: ResNet50v2, merge_dunes, lr=0.001 ==="
#python train.py \
#    --backbone   resnet50v2 \
#    --epochs     $EPOCHS \
#    --patience   $PATIENCE \
#    --batch_size $BS \
#    --lr 0.001 \
#    --merge_dunes

# ── experiment 8: adamw_rot180_mergedunes ────────────────────────────────────
# Baseline for new optimizer + augmentation stack.
# SGD → AdamW: smoother fine-tuning of pretrained weights.
# +rot180: free augmentation (tiles have no preferred orientation).
# merge_dunes: simpler 2-class problem to isolate optimizer/aug effect.
#echo "=== EXP 8 (rerun with checkpoint): adamw_rot180_mergedunes ==="
#python train.py \
#    --backbone         resnet50v2 \
#    --epochs           $EPOCHS \
#    --patience         $PATIENCE \
#    --batch_size       $BS \
#    --lr               0.001 \
#    --merge_dunes \
#    --rot180 \
#    --save_checkpoints \
#    --name             adamw_rot180_mergedunes

# ── experiment 9: adamw_rot180_allclasses ────────────────────────────────────
# Same AdamW + rot180 stack on the full 4-class problem.
# Tests whether the optimizer/aug gains transfer to the harder task.
#echo "=== EXP 9: adamw_rot180_allclasses ==="
#python train.py \
#    --backbone   resnet50v2 \
#    --epochs     $EPOCHS \
#    --patience   $PATIENCE \
#    --batch_size $BS \
#    --lr         0.001 \
#    --rot180 \
#    --name       adamw_rot180_allclasses

# ── experiment 10: wd_high_adamw_rot180_mergedunes ───────────────────────────
# Target: large train/val gap (train mAP 0.45 vs val 0.21 in best SGD run).
# AdamW weight decay acts as true L2 regularization (decoupled from LR).
# Increasing wd from 5e-4 → 5e-3 should penalise complex weights more.
#echo "=== EXP 10: wd_high_adamw_rot180_mergedunes ==="
#python train.py \
#    --backbone      resnet50v2 \
#    --epochs        $EPOCHS \
#    --patience      $PATIENCE \
#    --batch_size    $BS \
#    --lr            0.001 \
#    --weight_decay  0.005 \
#    --merge_dunes \
#    --rot180 \
#    --name          wd_high_adamw_rot180_mergedunes

# ── experiment 11: wd_high_adamw_rot180_allclasses ───────────────────────────
# Same high weight-decay regularization on the full 4-class problem.
#echo "=== EXP 11: wd_high_adamw_rot180_allclasses ==="
#python train.py \
#    --backbone      resnet50v2 \
#    --epochs        $EPOCHS \
#    --patience      $PATIENCE \
#    --batch_size    $BS \
#    --lr            0.001 \
#    --weight_decay  0.005 \
#    --rot180 \
#    --name          wd_high_adamw_rot180_allclasses

# ── experiment 12: nofreeze_adamw_rot180_mergedunes ──────────────────────────
# The backbone freeze was designed for SGD where a large initial LR could
# destroy pretrained weights. With AdamW at lr=0.001 this risk is lower.
# Training all layers from epoch 1 may let backbone and head co-adapt faster.
#echo "=== EXP 12: nofreeze_adamw_rot180_mergedunes ==="
#python train.py \
#    --backbone      resnet50v2 \
#    --epochs        $EPOCHS \
#    --patience      $PATIENCE \
#    --batch_size    $BS \
#    --lr            0.001 \
#    --freeze_epochs 0 \
#    --merge_dunes \
#    --rot180 \
#    --name          nofreeze_adamw_rot180_mergedunes

# ── experiment 13: longrun_adamw_rot180_mergedunes ───────────────────────────
# patience=10 may cut training before AdamW has fully converged.
# Double patience and epoch cap to rule out premature stopping as a bottleneck.
#echo "=== EXP 13: longrun_adamw_rot180_mergedunes ==="
#python train.py \
#    --backbone   resnet50v2 \
#    --epochs     200 \
#    --patience   20 \
#    --batch_size $BS \
#    --lr         0.001 \
#    --merge_dunes \
#    --rot180 \
#    --name       longrun_adamw_rot180_mergedunes

# ── experiment 14: wd_mid_adamw_rot180_mergedunes ────────────────────────────
# Exp 8 (WD=5e-4): test=0.583, val=0.283, gap=0.215
# Exp 10 (WD=5e-3): test=0.565, val=0.338, gap=0.161
# WD=1e-3 sits between them — expect to recover most of the test mAP while
# keeping the narrower train/val gap that WD=5e-3 gave us.
#echo "=== EXP 14: wd_mid_adamw_rot180_mergedunes ==="
#python train.py \
#    --backbone      resnet50v2 \
#    --epochs        $EPOCHS \
#    --patience      $PATIENCE \
#    --batch_size    $BS \
#    --lr            0.001 \
#    --weight_decay  0.001 \
#    --merge_dunes \
#    --rot180 \
#    --name          wd_mid_adamw_rot180_mergedunes

# ── experiment 15: score threshold sweep ─────────────────────────────────────
# The model fires ~3-4× more boxes than GT (n_pred >> n_gt across all runs).
# Root cause: box_score_thresh defaults to 0.05 in Faster R-CNN.
# Sweep thresholds on the best checkpoint (Exp 8) to find the precision/recall
# sweet spot without any retraining. Each run is eval-only (resumes best.pt).
# EXP 15: score threshold sweep — needs Exp 8 rerun (with --save_checkpoints) to complete first
#echo "=== EXP 15: score threshold sweep ==="
#for THRESH in 0.1 0.2 0.3 0.4 0.5; do
#    python train.py \
#        --backbone        resnet50v2 \
#        --merge_dunes \
#        --rot180 \
#        --resume          <path-to-exp8-rerun>/checkpoints/best.pt \
#        --score_threshold $THRESH \
#        --eval_only \
#        --name            thresh_${THRESH}_mergedunes
#done

# ── experiment 16: nofreeze_wd_mid_adamw_rot180_mergedunes ───────────────────
# Exp 12 (no-freeze, default WD) had the tightest train/val gap (0.146) at
# test=0.577. Combining no-freeze with WD=1e-3 may push both val and test up.
#echo "=== EXP 16: nofreeze_wd_mid_adamw_rot180_mergedunes ==="
#python train.py \
#    --backbone      resnet50v2 \
#    --epochs        $EPOCHS \
#    --patience      $PATIENCE \
#    --batch_size    $BS \
#    --lr            0.001 \
#    --weight_decay  0.001 \
#    --freeze_epochs 0 \
#    --merge_dunes \
#    --rot180 \
#    --name          nofreeze_wd_mid_adamw_rot180_mergedunes

# ── experiment 18: score threshold sweep on Exp 14 checkpoint ────────────────
# Exp 14 (best, test mAP=0.622) predicts ~2.6× more boxes than GT on test and
# ~5.8× more on val. Precision is only 0.31 — most predictions are false positives.
# Sweep score_threshold to find the precision/recall sweet spot without retraining.
# Checkpoint: runs/20260527_000336_resnet50v2_adamw_rot180_mergedunes/checkpoints/best.pt
EXP14_CKPT="runs/20260527_000336_resnet50v2_adamw_rot180_mergedunes/checkpoints/best.pt"
for THRESH in 0.1 0.2 0.3 0.4 0.5; do
    echo "=== EXP 18: score threshold ${THRESH} ==="
    python train.py \
        --backbone        resnet50v2 \
        --score_threshold $THRESH \
        --resume          $EXP14_CKPT \
        --eval_only \
        --name            exp18_thresh_${THRESH}
done

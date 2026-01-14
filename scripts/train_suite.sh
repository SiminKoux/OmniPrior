#!/bin/bash
SCENE_DIR="./data/OmniFisheye_plus"
SCENE_LIST="suite"
CAP_MAX=1000000

for SCENE in $SCENE_LIST; do
    echo "Running $SCENE, MCMCStrategy"
    RESULT_DIR="./results/$SCENE/init_colmap_metric"
    CUDA_VISIBLE_DEVICES=0 python runner/trainer.py mcmc \
        --data_factor 1 \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR \
        --camera_model fisheye \
        --test_every 10 \
        --batch_size 1 \
        --init_type metric \
        --filter \
        --opacity_reg 0.001 \
        --scale_reg 0.03 \
        --ranking_reg 0.03 \
        --metric_depth_reg 0.01 \
        --depth_smooth_reg 0.01 \
        --strategy.cap-max $CAP_MAX \
        --strategy.refine-start-iter 500 \
        --strategy.refine-stop-iter 20000 \
        --strategy.noise_lr 5.0e4 \
        --strategy.min_opacity 0.01 \
        --max_steps 30000 \
        --deform_opt \
        --deform.tv_loss \
        --deform.enable_ddyn \
        --deform.guidance_reg 5.0e-2 \
        --init_steps 18000 \
        --gaussian_phase_length 22000 \
        --deform_phase_length 8000 \
        --save_steps 18000 22000 30000 \
        --eval_steps 18000 22000 30000 \
        --save_ply \
        --ply_steps 18000 22000 30000 \
        --with_eval3d \
        --with_ut \
        --disable_viewer
done

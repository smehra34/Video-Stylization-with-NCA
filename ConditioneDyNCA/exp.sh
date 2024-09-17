WANDB_API_KEY=55cda04bc8e196e576724e5be8baeebf6b8bd40c


# ---------------------------------------------

# Test Experiments

# python train.py --style_name comic\
#     --exp_name server_test_run \
#     --no_wandb \
#     --no_index \
#     --target_images full \
#     --batch_size 2 \
#     --vector_field_motion_loss_weight 6.0 \
#     --appearance_loss_weight 6.0 \
#     --auxillary_loss_weight .1 \
#     --overflow_loss_weight 1000.0

python generate_videos.py --exp_description server_test_run --style_name comic


# ---------------------------------------------

# Baseline Experiments

# # Full Dataset

# python train.py --style_name comic\
#     --exp_name comic-baseline_full \
#     --no_index \
#     --target_images full \
#     --batch_size 4 \
#     --vector_field_motion_loss_weight 6.0 \
#     --appearance_loss_weight 6.0 \
#     --auxillary_loss_weight .1 \
#     --overflow_loss_weight 1000.0

# python generate_videos.py --exp_description comic-baseline_full --style_name comic


# python train.py --style_name starry-night\
#     --exp_name starry-night-baseline_full \
#     --no_index \
#     --target_images full \
#     --batch_size 4 \
#     --vector_field_motion_loss_weight 6.0 \
#     --appearance_loss_weight 6.0 \
#     --auxillary_loss_weight .1 \
#     --overflow_loss_weight 1000.0

# python generate_videos.py --exp_description starry-night-baseline_full --style_name starry-night

# python train.py --style_name pencil-sketch\
#     --exp_name pencil-sketch-baseline_full \
#     --no_index \
#     --target_images full \
#     --batch_size 4 \
#     --vector_field_motion_loss_weight 6.0 \
#     --appearance_loss_weight 6.0 \
#     --auxillary_loss_weight .1 \
#     --overflow_loss_weight 1000.0

# python generate_videos.py --exp_description pencil-sketch-baseline_full --style_name pencil-sketch

# # # 5 Images Dataset

# python train.py --style_name comic\
#     --exp_name comic-baseline_5_images \
#     --no_index \
#     --target_images 5_images \
#     --batch_size 4 \
#     --vector_field_motion_loss_weight 6.0 \
#     --appearance_loss_weight 6.0 \
#     --auxillary_loss_weight .1 \
#     --overflow_loss_weight 1000.0

# python generate_videos.py --exp_description comic-baseline_5_images --style_name comic

# python train.py --style_name starry-night\
#     --exp_name starry-night-baseline_5_images \
#     --no_index \
#     --target_images 5_images \
#     --batch_size 4 \
#     --vector_field_motion_loss_weight 6.0 \
#     --appearance_loss_weight 6.0 \
#     --auxillary_loss_weight .1 \
#     --overflow_loss_weight 1000.0

# python generate_videos.py --exp_description starry-night-baseline_5_images --style_name starry-night

# python train.py --style_name pencil-sketch\
#     --exp_name pencil-sketch-baseline_5_images \
#     --no_index \
#     --target_images 5_images \
#     --batch_size 4 \
#     --vector_field_motion_loss_weight 6.0 \
#     --appearance_loss_weight 6.0 \
#     --auxillary_loss_weight .1 \
#     --overflow_loss_weight 1000.0

# python generate_videos.py --exp_description pencil-sketch-baseline_5_images --style_name pencil-sketch

# python train.py --style_name japanese-wave-art\
#     --exp_name japanese-wave-art-baseline_5_images \
#     --no_index \
#     --target_images 5_images \
#     --batch_size 4 \
#     --vector_field_motion_loss_weight 6.0 \
#     --appearance_loss_weight 6.0 \
#     --auxillary_loss_weight .1 \
#     --overflow_loss_weight 1000.0

# python generate_videos.py --exp_description japanese-wave-art-baseline_5_images --style_name japanese-wave-art


# # # 2 Images Dataset

# python train.py --style_name comic\
#     --exp_name comic-baseline_2_images \
#     --no_index \
#     --target_images 2_images \
#     --batch_size 4 \
#     --vector_field_motion_loss_weight 6.0 \
#     --appearance_loss_weight 6.0 \
#     --auxillary_loss_weight .1 \
#     --overflow_loss_weight 1000.0

# python generate_videos.py --exp_description comic-baseline_2_images --style_name comic

# python train.py --style_name starry-night\
#     --exp_name starry-night-baseline_2_images \
#     --no_index \
#     --target_images 2_images \
#     --batch_size 4 \
#     --vector_field_motion_loss_weight 6.0 \
#     --appearance_loss_weight 6.0 \
#     --auxillary_loss_weight .1 \
#     --overflow_loss_weight 1000.0

# python generate_videos.py --exp_description starry-night-baseline_2_images --style_name starry-night

# python train.py --style_name pencil-sketch\
#     --exp_name pencil-sketch-baseline_2_images \
#     --no_index \
#     --target_images 2_images \
#     --batch_size 4 \
#     --vector_field_motion_loss_weight 6.0 \
#     --appearance_loss_weight 6.0 \
#     --auxillary_loss_weight .1 \
#     --overflow_loss_weight 1000.0

# python generate_videos.py --exp_description pencil-sketch-baseline_2_images --style_name pencil-sketch

# python train.py --style_name japanese-wave-art\
#     --exp_name japanese-wave-art-baseline_2_images \
#     --no_index \
#     --target_images 2_images \
#     --batch_size 4 \
#     --vector_field_motion_loss_weight 6.0 \
#     --appearance_loss_weight 6.0 \
#     --auxillary_loss_weight .1 \
#     --overflow_loss_weight 1000.0

# python generate_videos.py --exp_description japanese-wave-art-baseline_2_images --style_name japanese-wave-art
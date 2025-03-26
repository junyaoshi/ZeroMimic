import os
import hydra
import torch
import cv2
from utils import unnormalize, generate_visualization_image, load_data, video_set
import numpy as np

task = 'open_drawer'  # 'open_drawer', 'open_vertical_hinge', 'close_drawer', 'close_vertical_hinge'
example_id = 1 # Check the example_data folder for more examples
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@hydra.main(config_path='configs', config_name=task, version_base=None)
def main(cfg):
    # Load data
    norm_stats, policy, current_qpos, current_qpos_normalized, current_img, goal_img, current_img_path, goal_img_path, camera_matrix, dist_coeffs = load_data(cfg, task, example_id, device)

    # Set video output parameters
    video_writer, video_output_path, frame_size = video_set(task)

    with torch.inference_mode():
        policy.eval()
        actions_normalized = policy(current_qpos_normalized, current_img, goal_img)
    
    # Post-process actions
    if cfg.mode == 'reltraj_relori':
        actions_rel = unnormalize(actions_normalized, norm_stats, pose='rel_poses', policy_class=cfg.policy.policy_class)
        actions_abs = actions_rel + current_qpos
    else:
        raise NotImplementedError(f'Mode {cfg.mode} not implemented') 

    print(f'current_qpos: {current_qpos}, predicted actions: {actions_abs}')

    # Generate visualization frames for each action step
    for i in range(len(actions_abs[0])):
        vis_img_bgr = generate_visualization_image(
            current_hand_pose=current_qpos[0].cpu().numpy(),
            pred_hand_pose=actions_abs[0,:i+1,:].cpu().numpy(),
            current_img_path=current_img_path,
            goal_img_path=goal_img_path,
            task_name=task,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        
        # Resize the image to match the video frame size
        vis_img_bgr_resized = cv2.resize(vis_img_bgr, frame_size)
        
        # Write the frame to the video
        video_writer.write(vis_img_bgr_resized)

    # Release the VideoWriter object
    video_writer.release()
    print(f'Visualization video saved to {video_output_path}')

if __name__ == '__main__':
    main()

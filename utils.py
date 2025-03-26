import os
import pickle
import logging
from copy import deepcopy
import json
from omegaconf import open_dict
import torch
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def np2torch_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = torch.from_numpy(v).float().cuda()
    return new_d


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def get_policy(cfg, ckpt_dir, ckpt_name):
    seed = cfg.seed
    policy_class = cfg.policy.policy_class

    # dim=3 for 3d model, dim=6 for 6d model
    if 'ori' in cfg.mode:
        cfg.policy.state_dim = 6  
        if cfg.policy.policy_class == 'Diffusion':
            cfg.policy.action_dim = 6

    # set policy_cfg
    with open_dict(cfg):
        cfg.policy.chunk_size = cfg.chunk_size
        cfg.policy.lr = cfg.lr
        cfg.policy.goal_conditioned = cfg.goal_conditioned
        cfg.policy.kl_weight = cfg.kl_weight
        cfg.policy.loss = cfg.loss
        cfg.policy.cos_sim_weight = cfg.cos_sim_weight
        cfg.policy.mag_diff_weight = cfg.mag_diff_weight
        cfg.policy.backbone = cfg.backbone
        cfg.policy.parallel = cfg.parallel
        if policy_class == 'Diffusion':
            cfg.policy.prediction_horizon = cfg.chunk_size
    policy_cfg = cfg.policy
    set_seed(seed)

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_cfg)
    policy_dict = torch.load(ckpt_path)
    if 'policy' in policy_dict:
        # Fix the state dict keys if needed
        state_dict = policy_dict['policy']
        if cfg.parallel == 'DataParallel':
            # Add 'module.' prefix if model is parallel but state dict isn't
            if not any(k.startswith('model.module.') for k in state_dict.keys()):
                state_dict = {'model.module.' + k.replace('model.', ''): v for k, v in state_dict.items()}
            
        loading_status = policy.deserialize(state_dict)
    else:
        loading_status = policy.deserialize(policy_dict)
    
    print(f'Loading status: {loading_status}')
    policy.cuda()
    policy.eval()

    stats_path = os.path.join(ckpt_dir, f'dataset_norm_stats.pkl')
    with open(stats_path, 'rb') as f:
        norm_stats = pickle.load(f)
    print(f'Successfully loaded norm stats')

    if 'rel_poses_mean' not in norm_stats:
        norm_stats['rel_poses_mean'] = np.zeros((cfg.chunk_size, cfg.policy.state_dim))
        norm_stats['rel_poses_std'] = np.ones((cfg.chunk_size, cfg.policy.state_dim))
    if cfg.use_eval_norm_stats:
        print('Use eval norm stats...')
        print(f'Eval norm stats: {eval_norm_stats}')
    else:
        eval_norm_stats = norm_stats
    
    norm_stats = np2torch_dict(norm_stats)
    eval_norm_stats = np2torch_dict(eval_norm_stats)
    
    return norm_stats, policy


def normalize(data, norm_stats, pose='current_poses', policy_class='ACT'):
    mean = norm_stats[f'{pose}_mean']
    std = norm_stats[f'{pose}_std']
    min = norm_stats[f'{pose}_min']
    max = norm_stats[f'{pose}_max']
    if pose == 'current_poses' or policy_class == 'ACT':
        return (data - mean) / std
    elif policy_class == 'Diffusion':
        return ((data - min) / (max - min)) * 2 - 1
    else:
        logging.error("INVALID POLICY CLASS")
        return None
    
def unnormalize(data, norm_stats, pose='current_poses', policy_class='ACT', idx=None):
    mean = norm_stats[f'{pose}_mean']
    std = norm_stats[f'{pose}_std']
    min = norm_stats[f'{pose}_min']
    max = norm_stats[f'{pose}_max']
    if idx is not None and pose == 'rel_poses':
        mean = mean[idx]
        std = std[idx]

    if pose == 'current_poses' or policy_class == 'ACT':
        return (data * std) + mean
    elif policy_class == 'Diffusion':
        return ((data + 1) / 2) * (max - min) + min
    else:
        logging.error("INVALID POLICY CLASS")
        return None


def load_img(img_path):
    """Load and preprocess an image for the policy.
    
    Args:
        img_path (str): Path to the image file
        
    Returns:
        torch.Tensor: Preprocessed image tensor in channel-first format (3,256,456)
    """
    # Load and convert to RGB float
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    # Resize if needed
    if img.shape != (256, 456, 3):
        img = cv2.resize(img, (456, 256))

    # Convert to channel first format
    img = img.transpose(2, 0, 1)

    # Convert to torch tensor
    img = torch.from_numpy(img).float()

    return img


def load_camera(camera_path):
    with open(camera_path, 'r') as f:
        camera_data = json.load(f)
    camera_params = camera_data['camera']['params']
    camera_matrix = np.float32([[camera_params[0], 0, camera_params[2]], [0, camera_params[1], camera_params[3]], [0, 0, 1]])
    dist_coeffs = np.float32(camera_params[4:])
    return camera_matrix, dist_coeffs


def rmat_to_euler(rot_mat, degrees=False):
    euler = R.from_matrix(rot_mat).as_euler('xyz', degrees=degrees)
    return euler

def euler_to_rmat(euler, degrees=False):
    return R.from_euler('xyz', euler, degrees=degrees).as_matrix()


def determine_which_hand_hamer(hand_info):
    if "left_hand" in hand_info and not "right_hand" in hand_info:
        return "left_hand"
    elif "right_hand" in hand_info and not "left_hand" in hand_info:
        return "right_hand"
    else: 
        # if both hands are detected, select the hand that has contact state of 3 or 4
        # -1: did not appear, 0: non contact, 1: self-contact, 2: other person, 3: mobile object, 4: immobile object (furniture)
        if 'contact_list' not in hand_info:
            return "Error"
        left_contact = hand_info['contact_list'][0]['left_hand']
        right_contact = hand_info['contact_list'][0]['right_hand']
        # print(f'Left contact: {left_contact}, Right contact: {right_contact}')
        if left_contact == 3 or left_contact == 4:
            if right_contact != 3 and right_contact != 4:
                return "left_hand"
        elif right_contact == 3 or right_contact == 4:
            if left_contact != 3 and left_contact != 4:
                return "right_hand"
        else:
            return "None"
        # prefer the hand with mobile object contact
        if left_contact < right_contact:
            return "left_hand"
        elif right_contact < left_contact:
            return "right_hand"
        
        # if both hands have the same contact, select the hand with the higher confidence
        if hand_info["right_hand"]["right_confidence"] >= hand_info["left_hand"]["left_confidence"]:
            return "right_hand" 
        else:
            return "left_hand"


def load_wrist_qpos(mocap_pred_path, pose_norm_limit=2.0):
    with open(mocap_pred_path, 'rb') as f:
        mocap_pred = pickle.load(f)
    hand = determine_which_hand_hamer(mocap_pred)

    camera_wrist_coords = mocap_pred[hand]['joints'][0].reshape(1, 3)

    # if camera wrist coords are out of the norm limit, return None
    if np.linalg.norm(camera_wrist_coords) > pose_norm_limit:
        print('Camera wrist coords are out of the norm limit', mocap_pred_path)
        return -1

    if 'global_orient' not in mocap_pred[hand]:
        print('No global orientation found in the mocap prediction', mocap_pred_path)
        return -2
    else:
        camera_wrist_ori = mocap_pred[hand]['global_orient'][0]

    # rotate the orientation by -90 degrees around the y-axis
    camera_wrist_ori = np.dot(camera_wrist_ori, np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]))
    camera_wrist_ori_euler = rmat_to_euler(camera_wrist_ori)

    return torch.from_numpy(np.concatenate([camera_wrist_coords[0], camera_wrist_ori_euler])).float()



def draw_point(image, current_pose, camera_matrix, dist_coeffs, future_depth, future_pose=None, current_ori=None, future_ori=None, thickness=2):
    # Make sure the image is in BGR format and uint8
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    height, width, _ = image.shape
    # Convert the position to the correct format
    position = np.array([[current_pose]], dtype=np.float32)
    # Project 3D points to 2D
    image_points, _ = cv2.projectPoints(position, (0, 0, 0), (0, 0, 0), camera_matrix, dist_coeffs)
    # Extract the 2D coordinates
    x, y = image_points[0][0]
    start = (int(x), int(y))
    # Draw the point
    # image = cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), thickness)
    arrow_length = 0.1

    # TODO Draw the current orientation as an arrow
    if current_ori is not None:
        # Convert the orientation to a rotation matrix
        # rotation_matrix = cv2.Rodrigues(current_ori)[0]
        # print(f'Rotation matrix: {rotation_matrix}')
        rotation_matrix = euler_to_rmat(current_ori)
        # Define the arrow end point
        end_point_x = np.dot(rotation_matrix, np.array([arrow_length, 0, 0])) + current_pose
        end_point_y = np.dot(rotation_matrix, np.array([0, arrow_length, 0])) + current_pose
        end_point_z = np.dot(rotation_matrix, np.array([0, 0, arrow_length])) + current_pose
        end_points = np.array([end_point_x, end_point_y, end_point_z])
        # end_points = np.array([end_point_z])
        # end_point = np.dot(rotation_matrix, np.array([-arrow_length, 0, 0])) + current_pose
        # Project the end point to 2D
        for i in range(end_points.shape[0]):
            end_point = end_points[i]
            end_point = np.expand_dims(end_point, axis=0)
            end_image_points, _ = cv2.projectPoints(end_point, (0, 0, 0), (0, 0, 0), camera_matrix, dist_coeffs)
            end_x, end_y = end_image_points[0][0]
            # clip the end point if it is out of the image
            end_x = np.clip(end_x, 0, 10000)
            end_y = np.clip(end_y, 0, 10000)
            end = (int(end_x), int(end_y))
            # # Draw the arrow
            if i == 0:
                color = (0, 0, 255)
            elif i == 1:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            # color = (0, 255, 0)
            image = cv2.arrowedLine(image, start, end, color, thickness)

    # Draw the future point
    if future_pose is not None:
        if len(future_pose.shape) == 2:
            future_pose = np.expand_dims(future_pose, axis=0)
            future_depth = np.expand_dims(future_depth, axis=0)
        for pose, depth in zip(future_pose, future_depth):
            non_zero_indices = np.nonzero(depth)
            future_image_points, _ = cv2.projectPoints(np.copy(pose), (0, 0, 0), (0, 0, 0), camera_matrix, dist_coeffs)
            future_image_points[..., 0] = np.clip(future_image_points[..., 0], 0, width).astype(int)
            future_image_points[..., 1] = np.clip(future_image_points[..., 1], 0, height).astype(int)
            future_image_points = future_image_points[non_zero_indices]

            # orientation
            if future_ori is not None:
                future_ori = future_ori[non_zero_indices]
                for i in range(future_ori.shape[0]):
                    future_x, future_y = future_image_points[i][0]
                    

                    rotation_matrix_future = euler_to_rmat(future_ori[i])
                    # the first arrow is the longest
                    new_arrow_length = arrow_length
                    # new_arrow_length = arrow_length * (1 - float(i + 1) / future_ori.shape[0])
                    end_point_future_x = np.dot(rotation_matrix_future, np.array([new_arrow_length, 0, 0])) + pose[i]
                    end_point_future_y = np.dot(rotation_matrix_future, np.array([0, new_arrow_length, 0])) + pose[i]
                    end_point_future_z = np.dot(rotation_matrix_future, np.array([0, 0, new_arrow_length])) + pose[i]
                    # end_point_future = np.dot(rotation_matrix_future, np.array([-new_arrow_length, 0, 0])) + pose[i]
                    end_point_futures = np.array([end_point_future_x, end_point_future_y, end_point_future_z])
                    # end_point_futures = np.array([end_point_future_z])
                    for j in range(end_point_futures.shape[0]):
                        end_point_future = end_point_futures[j]
                        end_point_future = np.expand_dims(end_point_future, axis=0)
                        end_image_point_future, _ = cv2.projectPoints(np.copy(end_point_future), (0, 0, 0), (0, 0, 0), camera_matrix, dist_coeffs)
                        future_x, future_y = future_image_points[i][0]
                        end_x_future, end_y_future = end_image_point_future[0][0]
                        # clip the end point if it is out of range of int()
                        end_x_future = np.clip(end_x_future, 0, 10000)
                        end_y_future = np.clip(end_y_future, 0, 10000)
                        start_future = (int(future_x), int(future_y))
                        end_future = (int(end_x_future), int(end_y_future))
                        if j == 0:
                            color = (0, 0, 255)
                        elif j == 1:
                            color = (0, 255, 0)
                        else:
                            color = (255, 0, 0)
                        image = cv2.arrowedLine(image, start_future, end_future, color, thickness)    
    
    return image


def draw_point_3d(image, current_pose, camera_matrix, dist_coeffs, current_depth, future_depth, future_pose=None, thickness=2):
    height, width, _ = image.shape
    # Convert the position to the correct format
    position = np.array([[current_pose]], dtype=np.float32)
    # Project 3D points to 2D
    image_points, _ = cv2.projectPoints(position, (0, 0, 0), (0, 0, 0), camera_matrix, dist_coeffs)
    # Extract the 2D coordinates
    x, y = image_points[0][0]
    start = (int(x), int(y))
    # Draw the point
    cv2.circle(image, (int(x), int(y)), 5, (0, 25500, 0), thickness)

    # Draw the future point
    if future_pose is not None:
        if len(future_pose.shape) == 2:
            future_pose = np.expand_dims(future_pose, axis=0)
            future_depth = np.expand_dims(future_depth, axis=0)
        for pose, depth in zip(future_pose, future_depth):
            non_zero_indices = np.nonzero(depth)
            future_image_points, _ = cv2.projectPoints(np.copy(pose), (0, 0, 0), (0, 0, 0), camera_matrix, dist_coeffs)
            future_image_points[..., 0] = np.clip(future_image_points[..., 0], 0, width).astype(int)
            future_image_points[..., 1] = np.clip(future_image_points[..., 1], 0, height).astype(int)
            future_image_points = future_image_points[non_zero_indices]
            for future_image_point in future_image_points:
                future_x, future_y = future_image_point[0]
                cv2.circle(image, (int(future_x), int(future_y)), 5, (255, 0, 0), thickness)

            image = draw_gradient_line(image, start, future_image_points, current_depth, depth, thickness=thickness)

    return image


def draw_gradient_line(img, pt1, pt2, pt1_depth, pt2_depth, channel_temporal=(0, 0, 255), channel_depth=(0, 255, 0), thickness=2, num_segments=50):
    max_depth = max(pt1_depth, np.max(pt2_depth))

    # Calculate color for pt1 based on depth
    depth_ratio_pt1 = pt1_depth / max_depth

    for i in range(len(pt2)):
        pt = pt2[i, 0]  # Extract pixel coordinates from the array
        depth = pt2_depth[i]  # Extract depth value

        # Calculate the color based on depth
        depth_ratio_pt2 = depth / max_depth

        # Calculate start and end points
        start_point = (int(pt1[0]), int(pt1[1]))
        end_point = (int(pt[0]), int(pt[1]))

        # Calculate color gradient for the line segment, including pt1
        color_gradient = []
        for j in range(num_segments + 1):  # Include pt1 in the color gradient
            ratio = j / num_segments
            color_depth_G = int(channel_depth[1] * (depth_ratio_pt1 + (depth_ratio_pt2 - depth_ratio_pt1) * ratio))
            color_depth_B = 255 - color_depth_G
            color_depth_R = 255 - color_depth_G
            color_depth = tuple((color_depth_B, color_depth_G, color_depth_R))
            # color_temporal = tuple(int(ch * (1 - ratio)) for ch in channel_temporal)
            # combined_color = tuple(np.add(color_temporal, color_depth_pt2).tolist())
            color_gradient.append(color_depth)

        # Draw the line segment with color gradient
        for j in range(num_segments):
            start = (
                int(start_point[0] * (1 - (j / num_segments)) + end_point[0] * (j / num_segments)),
                int(start_point[1] * (1 - (j / num_segments)) + end_point[1] * (j / num_segments))
            )
            end = (
                int(start_point[0] * (1 - ((j + 1) / num_segments)) + end_point[0] * ((j + 1) / num_segments)),
                int(start_point[1] * (1 - ((j + 1) / num_segments)) + end_point[1] * ((j + 1) / num_segments))
            )
            cv2.line(img, start, end, color_gradient[j], thickness)

        # Update pt1 for the next segment
        pt1 = pt
        pt1_depth = depth
        depth_ratio_pt1 = pt1_depth / max_depth

    return img


def generate_visualization_image(
        current_hand_pose, pred_hand_pose, 
        current_img_path, goal_img_path, 
        task_name, camera_matrix, dist_coeffs
    ):
    if current_img_path is not None:
        current_img = cv2.imread(current_img_path)
    else:
        current_img = np.zeros_like(current_img)
    if goal_img_path is not None:
        goal_img = cv2.imread(goal_img_path)
    else:
        goal_img = np.zeros_like(current_img)
    img_h = current_img.shape[0]

    current_pose = current_hand_pose[:3]
    if torch.is_tensor(pred_hand_pose):
        pred_pose = pred_hand_pose.detach().cpu().numpy()[:, :3]
    else:
        pred_pose = np.array(pred_hand_pose, dtype=np.float32)[:, :3]
    # camera_matrix = camera_matrix.detach().cpu().numpy()
    # dist_coeffs = dist_coeffs.detach().cpu().numpy()
    camera_matrix = camera_matrix
    dist_coeffs = dist_coeffs
    current_depth = np.linalg.norm(current_pose)
    pred_depth = np.linalg.norm(pred_pose, axis=-1)
       
    if current_hand_pose.shape[0] > 3:
        current_ori = current_hand_pose[3:]
        if torch.is_tensor(pred_hand_pose):
            pred_ori = pred_hand_pose.detach().cpu().numpy()[:, 3:]
        else:
            pred_ori = np.array(pred_hand_pose, dtype=np.float32)[:, 3:]
    else:
        current_ori = None
        pred_ori = None

    current_img_with_pred = draw_point(
        deepcopy(current_img), current_pose, 
        camera_matrix, dist_coeffs, 
        pred_depth, future_pose=pred_pose,
        current_ori=current_ori, future_ori=pred_ori
    )
    
    # current_img_with_pred = draw_point_3d(
    #     deepcopy(current_img), current_pose, 
    #     camera_matrix, dist_coeffs, 
    #     current_depth, pred_depth, future_pose=pred_pose
    # )
    
    vis_img_bgr = np.vstack([current_img_with_pred, goal_img]).astype(np.uint8)
    # initialize vis header
    white_top_height = 100
    white_left_width = 110
    next_line = 40
    desc_left_align = 20
    header_left_align = desc_left_align
    text_size = 0.6
    color = (0, 0, 0)
    thickness = 2
    # white_top_height += 50
    white_top = np.zeros((white_top_height, vis_img_bgr.shape[1], 3), np.uint8)
    white_top[:] = (255, 255, 255)
    vis_img_bgr = cv2.vconcat((white_top, vis_img_bgr))
    white_left = np.zeros((vis_img_bgr.shape[0], white_left_width, 3), np.uint8)
    white_left[:] = (255, 255, 255)
    vis_img_bgr = cv2.hconcat((white_left, vis_img_bgr))

    # add task string to vis header
    font = cv2.FONT_HERSHEY_COMPLEX
    task_str = "Task"
    task_desc_str = f'{task_str}: {task_name}'
    cv2.putText(vis_img_bgr, task_desc_str, (header_left_align, next_line), font, text_size, color, thickness, 0)
    next_line += 50

    # current_depth_str = f'Current Depth: {current_depth:.4f}'
    # cv2.putText(vis_img_bgr, current_depth_str, (30, next_line), font, 0.6, (0, 0, 0), 2, 0)
    # next_line += 50
    # if len(pred_depth.shape) == 1:
    #     # first 3 steps
    #     if len(pred_depth) >= 3:
    #         pred_depth_str = 'Pred first 3: ' + ", ".join([f'{depth:.4f}' for depth in pred_depth[:3]])
    #     else:
    #         pred_depth_str = f'Pred 1: {pred_depth[0]:.4f}'
    # else:
    #     pred_depth_str = 'Pred: ' + ", ".join([f'{depth[-1]:.4f}' for depth in pred_depth])
    # if len(pred_depth.shape) == 1:
    #     if len(pred_depth) > 6:
    #         pred_depth_str = 'Pred first 3: ' + ", ".join([f'{depth-current_depth:.3f}' for depth in pred_depth[:3]]) + \
    #                         ' | last 3: ' + ", ".join([f'{depth-current_depth:.3f}' for depth in pred_depth[-3:]])
    #     elif len(pred_depth) >= 3:
    #         pred_depth_str = 'Pred first 3: ' + ", ".join([f'{depth-current_depth:.3f}' for depth in pred_depth[:3]]) + \
    #                         ' | last: ' + ", ".join([f'{depth-current_depth:.3f}' for depth in pred_depth[-3:]])
    #     else:
    #         pred_depth_str = 'Pred values: ' + ", ".join([f'{depth-current_depth:.3f}' for depth in pred_depth])
    # else:
    #     pred_depth_str = 'Pred: ' + ", ".join([f'{depth[-1]-current_depth:.3f}' for depth in pred_depth])
    # cv2.putText(vis_img_bgr, pred_depth_str, (30, next_line), font, 0.45, (0, 0, 0), 2, 0)
    next_line += 125


    # add image description to blank space on the left
    cv2.putText(vis_img_bgr, 'pred', (desc_left_align, next_line), font, text_size, color, thickness, 0)
    next_line += img_h
    cv2.putText(vis_img_bgr, 'goal', (desc_left_align, next_line), font, text_size, color, thickness, 0)
    next_line += img_h

    return vis_img_bgr

import hydra
from hydra.utils import to_absolute_path
def load_data(cfg, task, example_id, device):
    ckpt_dir = os.path.join(cfg.debug_eval_path)
    ckpt_name = f'policy_epoch_{cfg.eval_epoch}_seed_{cfg.seed}.ckpt'
    norm_stats, policy = get_policy(cfg, ckpt_dir, ckpt_name)
 
    current_img_path = to_absolute_path(f'example_data/{task}/{example_id}/current_img.jpg')
    goal_img_path = to_absolute_path(f'example_data/{task}/{example_id}/goal_img.jpg')
    current_qpos_path = to_absolute_path(f'example_data/{task}/{example_id}/current_qpos.pkl')
    camera_path = to_absolute_path(f'example_data/{task}/{example_id}/camera.json')

    current_qpos = load_wrist_qpos(current_qpos_path).to(device).unsqueeze(0)
    current_qpos_normalized = normalize(current_qpos, norm_stats, pose='current_poses', policy_class=cfg.policy.policy_class)
    current_img = load_img(current_img_path).to(device).unsqueeze(0)
    goal_img = load_img(goal_img_path).to(device).unsqueeze(0)
    camera_matrix, dist_coeffs = load_camera(camera_path)

    return norm_stats, policy, current_qpos,current_qpos_normalized, current_img, goal_img, current_img_path, goal_img_path, camera_matrix, dist_coeffs

def video_set(task):
    video_output_path = f'vis_video_{task}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 encoding
    fps = 5  # Frames per second
    frame_size = (566, 612)  # Video frame size, adjust according to actual image size

    # Create VideoWriter object
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, frame_size)
    return video_writer, video_output_path, frame_size

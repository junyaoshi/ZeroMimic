import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import numpy as np

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
# import IPython
# e = IPython.embed
import logging

# Diffusion Policy
from collections import OrderedDict
from robomimic.models.base_nets import ResNet18Conv, ResNet50Conv, SpatialSoftmax
from robomimic.algo.diffusion_policy import replace_bn_with_gn, ConditionalUnet1D

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.parallel = args_override['parallel']
        if self.parallel == 'DataParallel':
            self.model = torch.nn.DataParallel(model).cuda()
        else:
            self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.loss = args_override['loss']
        logging.info(f'Loss: {self.loss}')
        if self.loss == 'cos_sim':
            self.cos_sim_weight = args_override['cos_sim_weight']
            self.mag_diff_weight = args_override['mag_diff_weight']
            logging.info(f'Cosine Similarity Weight: {self.cos_sim_weight}')
            logging.info(f'Magnitude Weight: {self.mag_diff_weight}')
        logging.info(f'KL Weight: {self.kl_weight}')

    def __call__(self, qpos, current_image, goal_image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # image = normalize(image)
        current_image = normalize(current_image)
        goal_image = normalize(goal_image)

        if actions is not None: # training time
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            actions = actions[:, :raw_model.num_queries]
            is_pad = is_pad[:, :raw_model.num_queries]
            
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, current_image, goal_image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            loss_dict['kl'] = total_kld[0]
            # print(f'actions: {actions.shape}, a_hat: {a_hat.shape}, is_pad: {is_pad.shape}, is_pad_hat: {is_pad_hat.shape}')
            if self.loss == 'l1':
                all_l1 = F.l1_loss(actions, a_hat, reduction='none')
                l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
                loss_dict['l1'] = l1
                loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            elif self.loss == 'l2':
                all_l2 = F.mse_loss(actions, a_hat, reduction='none')
                # print('all_l2:', all_l2.shape)
                l2 = (all_l2 * ~is_pad.unsqueeze(-1)).mean()
                loss_dict['l2'] = l2
                loss_dict['loss'] = loss_dict['l2'] + loss_dict['kl'] * self.kl_weight
            elif self.loss == 'cos_sim':
                cos_sim = F.cosine_similarity(actions, a_hat, dim=-1)
                mag_diff = (actions.norm(dim=-1) - a_hat.norm(dim=-1)).abs()
                cos_sim = (cos_sim * ~is_pad).mean()
                mag_diff = (mag_diff * ~is_pad).mean()
                loss_dict['cos_sim'] = cos_sim
                loss_dict['loss'] = (1-loss_dict['cos_sim']) * self.cos_sim_weight + mag_diff * self.mag_diff_weight + loss_dict['kl'] * self.kl_weight
            else:
                raise NotImplementedError

            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, current_image, goal_image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer
    
    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)

class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()

        self.camera_names = args_override['camera_names']
        self.goal_conditioned = args_override['goal_conditioned']
        self.backbone = args_override['backbone']

        self.observation_horizon = args_override['observation_horizon'] ### TODO TODO TODO DO THIS
        self.action_horizon = args_override['action_horizon'] # apply chunk size
        self.prediction_horizon = args_override['prediction_horizon'] # chunk size
        self.num_inference_timesteps = args_override['num_inference_timesteps']
        self.ema_power = args_override['ema_power']
        self.lr = args_override['lr']
        self.weight_decay = 0

        self.num_kp = 32
        self.feature_dimension = 64
        self.ac_dim = args_override['action_dim'] # 14 + 2
        self.obs_dim = self.feature_dimension * len(self.camera_names) + self.ac_dim # camera features and proprio

        backbones = []
        pools = []
        linears = []
        for _ in self.camera_names:
            if self.backbone == 'resnet18':
                backbones.append(ResNet18Conv(**{'input_channel': 3, 'pretrained': False, 'input_coord_conv': False}))
            elif self.backbone == 'resnet50':
                backbones.append(ResNet50Conv(**{'input_channel': 3, 'pretrained': False, 'input_coord_conv': False}))
            else:
                raise NotImplementedError
            pools.append(SpatialSoftmax(**{'input_shape': [512, 8, 15], 'num_kp': self.num_kp, 'temperature': 1.0, 'learnable_temperature': False, 'noise_std': 0.0}))
            linears.append(torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension))
        if self.goal_conditioned:
            if self.backbone == 'resnet18':
                backbones.append(ResNet18Conv(**{'input_channel': 3, 'pretrained': False, 'input_coord_conv': False}))
            elif self.backbone == 'resnet50':
                backbones.append(ResNet50Conv(**{'input_channel': 3, 'pretrained': False, 'input_coord_conv': False}))
            else:
                raise NotImplementedError
            pools.append(SpatialSoftmax(**{'input_shape': [512, 8, 15], 'num_kp': self.num_kp, 'temperature': 1.0, 'learnable_temperature': False, 'noise_std': 0.0}))
            linears.append(torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension))
            self.obs_dim += self.feature_dimension
        backbones = nn.ModuleList(backbones)
        pools = nn.ModuleList(pools)
        linears = nn.ModuleList(linears)
        
        backbones = replace_bn_with_gn(backbones) # TODO

        self.parallel = args_override['parallel']
        if self.parallel == 'DataParallel':
            for i in range(len(backbones)):
                backbones[i] = torch.nn.DataParallel(backbones[i]).cuda()
                pools[i] = torch.nn.DataParallel(pools[i]).cuda()
                linears[i] = torch.nn.DataParallel(linears[i]).cuda()


        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim*self.observation_horizon
        )
        
        # noise_pred_net = torch.nn.DataParallel(noise_pred_net).cuda()

        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'backbones': backbones,
                'pools': pools,
                'linears': linears,
                'noise_pred_net': noise_pred_net
            })
        })

        nets = nets.float().cuda()
        ENABLE_EMA = True
        if ENABLE_EMA:
            ema = EMAModel(model=nets, power=self.ema_power)
        else:
            ema = None
        self.nets = nets
        self.ema = ema

        # setup noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=50,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )

        n_parameters = sum(p.numel() for p in self.parameters())
        logging.info(f"number of parameters: {n_parameters/1e6:.2f}M")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


    def __call__(self, qpos, image, goal_image, actions=None, is_pad=None):
        B = qpos.shape[0]
        # print(f'qpos: {qpos.shape}, image: {image.shape}, goal_image: {goal_image.shape}')
        if actions is not None: # training time
            nets = self.nets
            all_features = []
            current_cam_features = nets['policy']['backbones'][0](image)
            current_pool_features = nets['policy']['pools'][0](current_cam_features)
            current_pool_features = torch.flatten(current_pool_features, start_dim=1)
            current_out_features = nets['policy']['linears'][0](current_pool_features)
            all_features.append(current_out_features)

            if self.goal_conditioned:
                goal_cam_features = nets['policy']['backbones'][1](goal_image)
                goal_pool_features = nets['policy']['pools'][1](goal_cam_features)
                goal_pool_features = torch.flatten(goal_pool_features, start_dim=1)
                goal_out_features = nets['policy']['linears'][1](goal_pool_features)
                all_features.append(goal_out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=obs_cond.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=obs_cond.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict the noise residual
            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)
            
            # L2 loss
            all_l2 = F.mse_loss(noise_pred, noise, reduction='none')
            loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {}
            loss_dict['l2_loss'] = loss
            loss_dict['loss'] = loss

            if self.training and self.ema is not None:
                self.ema.step(nets)
            return loss_dict
        else: # inference time
            To = self.observation_horizon
            Ta = self.action_horizon
            Tp = self.prediction_horizon
            action_dim = self.ac_dim
            
            nets = self.nets
            if self.ema is not None:
                nets = self.ema.averaged_model
            
            all_features = []
            current_cam_features = nets['policy']['backbones'][0](image)
            current_pool_features = nets['policy']['pools'][0](current_cam_features)
            current_pool_features = torch.flatten(current_pool_features, start_dim=1)
            current_out_features = nets['policy']['linears'][0](current_pool_features)
            all_features.append(current_out_features)

            if self.goal_conditioned:
                goal_cam_features = nets['policy']['backbones'][1](goal_image)
                goal_pool_features = nets['policy']['pools'][1](goal_cam_features)
                goal_pool_features = torch.flatten(goal_pool_features, start_dim=1)
                goal_out_features = nets['policy']['linears'][1](goal_pool_features)
                all_features.append(goal_out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, Tp, action_dim), device=obs_cond.device)
            naction = noisy_action
            
            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction, 
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            return naction

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        if "model.module.nets" in model_dict:
            nets_key = "model.module.nets"
            ema_key = "model.module.ema"
        elif "model.nets" in model_dict:
            nets_key = "model.nets"
            ema_key = "model.ema"
        else:
            nets_key = "nets"
            ema_key = "ema"

        status = self.nets.load_state_dict(model_dict[nets_key])
        print('Loaded model')
        if model_dict.get(ema_key, None) is not None:
            print('Loaded EMA')
            status_ema = self.ema.averaged_model.load_state_dict(model_dict[ema_key])
            status = [status, status_ema]
        return status


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=float, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    args = parser.parse_args()
    
    if args.policy_class == 'ACT':
        policy = ACTPolicy(vars(args))
    elif args.policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(vars(args))
    else:
        raise NotImplementedError
    
    import ipdb; ipdb.set_trace()
    print(f'Loaded {args.policy_class} policy.')

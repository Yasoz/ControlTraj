import torch
import torch.nn as nn
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
from einops import repeat, rearrange
import os
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from types import SimpleNamespace
from utils.config import args
from utils.EMA import EMAHelper
from utils.road_encoder import *
from utils.geounet import *
from utils.logger import Logger, log_info
from pathlib import Path
import shutil

# set the GPU enviroment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def resample_trajectory(x, length=200):
    """
    Resample a trajectory to a fixed length using linear interpolation
    :param x: trajectory to resample
    :param length: length of the resampled trajectory
    :return: resampled trajectory
    """
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    resampled_trajectory = np.zeros((2, length))
    for i in range(2):
        resampled_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return resampled_trajectory.T

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)
def compute_alpha(beta, t):
    """
    compute alpha for a given beta and t
    :param beta: tensor of shape (T,)
    :param t: tensor of shape (B,)
    :return: tensor of shape (B, 1, 1)
    """
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a

def p_xt(xt, noise, t, next_t, beta, eta=0):
    at = compute_alpha(beta.cuda(), t.long())
    at_next = compute_alpha(beta, next_t.long())
    x0_t = (xt - noise * (1 - at).sqrt()) / at.sqrt()
    c1 = (eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    eps = torch.randn(xt.shape, device=xt.device)
    xt_next = at_next.sqrt() * x0_t + c1 * eps + c2 * noise
    return xt_next


def setup_experiment_directories(config, Exp_name='ControlTraj', model_name="ControlTraj"):
    """
    setup the directories for the experiment
    :param config: configuration file
    :param Exp_name: Experiment name
    file_save: directory to save the files
    result_save: directory to save the results
    model_save: directory to save the models during training
    """
    root_dir = Path(__file__).resolve().parent
    result_name = f"{config.data.dataset}_bs={config.training.batch_size}"
    exp_dir = root_dir / Exp_name / result_name
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    exp_time_dir = exp_dir / timestamp
    files_save = exp_time_dir / 'Files'
    result_save = exp_time_dir / 'Results'
    model_save = exp_time_dir / 'models'

    # Creating directories
    for directory in [files_save, result_save, model_save]:
        directory.mkdir(parents=True, exist_ok=True)

    # Copying files
    for filename in os.listdir(root_dir / 'utils'):
        if filename.endswith('.py'):
            shutil.copy(root_dir / 'utils' / filename, files_save)
    # Copying the current file itself
    this_file = Path(__file__)
    shutil.copy(this_file, files_save)

    print("All files saved path ---->>", exp_time_dir)
    logger = Logger( __name__, log_path=exp_dir /  (timestamp + '/out.log'),colorize=True)
    return logger, files_save, result_save, model_save



def main(config, logger):
        # Modified to return the noise itself as well
    def q_xt_x0(x0, t):
        mean = gather(alpha_bar, t)**0.5 * x0
        var = 1 - gather(alpha_bar, t)
        eps = torch.randn_like(x0).to(x0.device)
        return mean + (var**0.5) * eps, eps  # also returns noise

    # initialize the model with the configuration
    unet = UNetModel(
        in_channels = config.model.in_channels,
        out_channels = config.model.out_channels,
        channels = config.model.channels,
        n_res_blocks = config.model.num_res_blocks,
        attention_levels = config.model.attention_levels,
        channel_multipliers = config.model.channel_multipliers,
        n_heads = config.model.n_heads,
        tf_layers = config.model.tf_layers,
        d_cond=128
    ).cuda()
    total_params = sum(p.numel() for p in unet.parameters())
    print(f'{total_params:,} total parameters.')
    # initialize the road encoder with RoadMAE
    autoencoder = MAE_ViT(image_size=200,
                 patch_size=5,
                 emb_dim=128,
                 encoder_layer=8,
                 encoder_head=4,
                 decoder_layer=4,
                 decoder_head=4,
                          mask_ratio=0.00).cuda()
    autoencoder.load_state_dict(torch.load('./models/road_encoder.pt'))
    # freeze the parameters of the road encoder
    for param in autoencoder.parameters():
        param.requires_grad = False

    # Load the data and create the dataloader
    roads = np.load('./data/porto_trajs.npy',allow_pickle=True)
    trajs = np.load('./data/porto_trajs.npy',allow_pickle=True)
    heads = np.load('./data/porto_heads.npy',allow_pickle=True)
    
    trajs = trajs.transpose(0,2,1)
    trajs = torch.from_numpy(trajs).float()
    roads = torch.from_numpy(roads).float()
    heads = torch.from_numpy(heads).float()
    dataset = TensorDataset(trajs, heads, roads)
    dataloader = DataLoader(dataset,
                            batch_size=config.training.batch_size,
                            shuffle=True,
                            num_workers=8)

    # Training params
    # Set up some parameters
    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end, n_steps).cuda()
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    lr = 2e-4  # Explore this - might want it lower when training on the full dataset

    losses = []  # Store losses for later plotting
    # optimizer
    optim = torch.optim.AdamW(unet.parameters(), lr=lr)  # Optimizer
    
    # EMA
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(unet)
    else:
        ema_helper = None
    # config.training.n_epochs = 1
    for epoch in range(0, config.training.n_epochs  + 1):
        losses = []  # Store losses for later plotting
        logger.info("<----Epoch-{}---->".format(epoch))
        for _, (x0, attr, road) in enumerate(dataloader):
            x0 = x0.cuda()
            attr = attr.cuda()
            new_roads = []
            for i in range(len(road)):
                new_roads.append(resample_trajectory(road[i]))
            new_roads = np.array(new_roads)
            new_roads = new_roads.transpose(0,2,1)
            # get the road embeddings by RoadMAE
            guide = torch.from_numpy(new_roads).float().cuda()
            with torch.no_grad():
                guide, _= autoencoder.encoder(guide)
                guide = guide[1:,:,:]
                guide = rearrange(guide, 't b c -> b t c')
            
            t = torch.randint(low=0, high=n_steps,
                              size=(len(x0) // 2 + 1, )).cuda()
            t = torch.cat([t, n_steps - t - 1], dim=0)[:len(x0)]
            # Get the noised images (xt) and the noise (our target)
            xt, noise = q_xt_x0(x0, t)
            pred_noise = unet(xt.float(), t,guide,attr)
            # Compare the predictions with the targets
            loss = F.mse_loss(noise.float(), pred_noise)
            # Store the loss for later viewing
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            if config.model.ema:
                ema_helper.update(unet)
        logger.info("<----Loss: {:.5f}---->".format(np.mean(losses)))
        
        
        if (epoch) % 10 == 0:
            m_path = model_save / f"unet_{epoch}.pt"
            torch.save(unet.state_dict(), m_path)
            # Start with random noise
            sample = torch.randn(config.training.batch_size, 2, config.data.traj_length).cuda()
            _, attr, road = next(iter(dataloader))
            attr = attr.cuda()
            new_roads = []
            for i in range(len(road)):
                new_roads.append(resample_trajectory(road[i]))
            new_roads = np.array(new_roads)
            new_roads = new_roads.transpose(0,2,1)
            guide = torch.from_numpy(new_roads).float().cuda()
            with torch.no_grad():
                guide, _ = autoencoder.encoder(guide)
                guide = guide[1:,:,:]
                guide = rearrange(guide, 't b c -> b t c')
                
            ims = []
            n = sample.size(0)
            eta=0.0
            timesteps=100
            skip = n_steps // timesteps
            seq = range(0, n_steps, skip)
            seq_next = [-1] + list(seq[:-1])
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).cuda()
                next_t = (torch.ones(n) * j).cuda()
                with torch.no_grad():
                    pred_noise = unet(sample, t, guide, attr)
                    # print(pred_noise.shape)
                    sample = p_xt(sample, pred_noise, t, next_t, beta, eta)
                    if i % 10 == 0:
                        ims.append(sample.squeeze(0))
            trajs = ims[-1].cpu().numpy()
       
            del ims
            plt.figure(figsize=(8,8))
            for i in range(len(trajs)):
                tj = trajs[i]
                plt.plot(tj[0,:],tj[1,:],color='#3f72af',alpha=0.1)
            plt.tight_layout()
            m_path = result_save / f"r_{epoch}.png"
            plt.savefig(m_path)
        
            
if __name__ == "__main__":
    # Load configuration
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    config = SimpleNamespace(**temp)
    
    logger,files_save, result_save, model_save = setup_experiment_directories(config, Exp_name='Control_Porto')
    
    log_info(config, logger)
    main(config, logger)

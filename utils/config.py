args = {
    'data': {
        'dataset': 'Chengdu',
        'traj_path': '/xxxxxx',
        'road_path': '/xxxxxxx',
        'attr_path': '/xxxxxxx',
        'traj_length': 256,
        'channels': 2,
        'num_workers': True,
    },
    'model': {
        'in_channels': 2,
        'out_channels': 2,
        'channels': 64,
        'channel_multipliers': [1, 2, 2, 2],
        'ae_ch_mult': [1, 2, 2, 2],
        'num_res_blocks': 2,
        'attention_levels': [0,1,2],
        'n_heads': 4,
        'tf_layers': 1,
        'd_cond' : 128,
        'ema_rate': 0.9999,
        'ema': True,
    },
    'diffusion': {
        'beta_schedule': 'linear',
        'beta_start': 0.0001,
        'beta_end': 0.05,
        'num_diffusion_timesteps': 500,
    },
    'training': {
        'batch_size': 768,
        'n_epochs': 200,
        'n_iters': 5000000,
        'snapshot_freq': 5000,
        'validation_freq': 2000,
    },
    'sampling': {
        'batch_size': 64,
        'last_only': True,
    }
}

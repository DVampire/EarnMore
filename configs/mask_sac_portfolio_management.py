# base parameters (do not modify)
root = None
workdir = "workdir"
tag = "mask_sac"
num_stocks = 28
num_envs = 1
num_features = 102 # num features name + num temporals name
temporal_dim = 3 # weekday, day, month
train_start_date = "2007-09-26"
val_start_date = "2019-07-22"
test_start_date = "2021-01-08"
test_end_date = None
if_use_per = False
if_use_rep = True
if_use_beta = True
if_norm = True
if_norm_temporal = False
save_freq = 20
repeat_times = 128
action_wrapper_method = "reweight"
T = 1.0
n_steps_per_episode = 1024

# train parameters (adjust mainly)
num_episodes = 1000
days = 10
batch_size = 64
buffer_size = 4096
horizon_len = 64
embed_dim = 128
decoder_embed_dim = 128
depth = 1 # 1 transformer
decoder_depth = 1
lr = 5e-5 # act_lr, cri_lr
act_lr = 5e-5
cri_lr = 5e-5
rep_lr = 5e-5
beta_lr = 5e-5
rep_loss_weight = 0.01
beta_loss_weight = 0.01
seed = 10

# size
feature_size = (days, num_features)
patch_size = (days, num_features)

transition = ["state", "action", "mask", "ids_restore", "reward", "done", "next_state"]
transition_shape = dict(
    state = dict(shape = (num_envs, num_stocks, days, num_features), type = "float32"),
    action = dict(shape = (num_envs, num_stocks + 1), type = "float32"),
    mask = dict(shape = (num_envs, num_stocks), type = "int32"),
    ids_restore = dict(shape = (num_envs, num_stocks), type = "int64"),
    reward = dict(shape = (num_envs, ), type = "float32"),
    done = dict(shape = (num_envs, ), type = "float32"),
    next_state  = dict(shape = (num_envs, num_stocks, days, num_features), type = "float32")
)

dataset = dict(
    type = "PortfolioManagementDataset",
    root = root,
    data_path= "datasets/dj30/features",
    stocks_path = "datasets/dj30/stocks.txt",
    aux_stocks_path = "datasets/dj30/aux_stocks_files",
    features_name = [
        'open',
        'high',
        'low',
        'close',
        'kmid2',
        'kup2',
        'klow',
        'klow2',
        'ksft2',
        'roc_5',
        'roc_10',
        'roc_20',
        'roc_30',
        'roc_60',
        'ma_5',
        'ma_10',
        'ma_20',
        'ma_30',
        'ma_60',
        'std_5',
        'std_10',
        'std_20',
        'std_30',
        'std_60',
        'beta_5',
        'beta_10',
        'beta_20',
        'beta_30',
        'beta_60',
        'max_5',
        'max_10',
        'max_20',
        'max_30',
        'max_60',
        'min_5',
        'min_10',
        'min_20',
        'min_30',
        'min_60',
        'qtlu_5',
        'qtlu_10',
        'qtlu_20',
        'qtlu_30',
        'qtlu_60',
        'qtld_5',
        'qtld_10',
        'qtld_20',
        'qtld_30',
        'qtld_60',
        'rank_5',
        'rank_10',
        'rank_20',
        'rank_30',
        'rank_60',
        'imax_5',
        'imax_10',
        'imax_20',
        'imax_30',
        'imax_60',
        'imin_5',
        'imin_10',
        'imin_20',
        'imin_30',
        'imin_60',
        'imxd_5',
        'imxd_10',
        'imxd_20',
        'imxd_30',
        'imxd_60',
        'cntp_5',
        'cntp_10',
        'cntp_20',
        'cntp_30',
        'cntp_60',
        'cntn_5',
        'cntn_10',
        'cntn_20',
        'cntn_30',
        'cntn_60',
        'cntd_5',
        'cntd_10',
        'cntd_20',
        'cntd_30',
        'cntd_60',
        'sump_5',
        'sump_10',
        'sump_20',
        'sump_30',
        'sump_60',
        'sumn_5',
        'sumn_10',
        'sumn_20',
        'sumn_30',
        'sumn_60',
        'sumd_5',
        'sumd_10',
        'sumd_20',
        'sumd_30',
        'sumd_60',
    ],
    temporals_name = [
        "weekday",
        "day",
        "month",
    ],
    labels_name = [
        'ret1',
        'mov1',
    ]
)

environment = dict(
    type = "EnvironmentPV",
    dataset = None,
    mode = "train",
    if_norm = if_norm,
    if_norm_temporal = if_norm_temporal,
    scaler = None,
    days = days,
    start_date = None,
    end_date = None,
    initial_amount = 1e3,
    transaction_cost_pct = 1e-3,
)

rep_net = dict(
    type = "MaskTimeState",
    embed_type = "TimesEmbed",
    feature_size = feature_size,
    patch_size = patch_size,
    t_patch_size = 1,
    num_stocks = num_stocks,
    pred_num_stocks = num_stocks,
    in_chans = 1,
    input_dim=num_features,
    temporal_dim=temporal_dim,
    embed_dim = embed_dim,
    depth = depth,
    num_heads = 4,
    decoder_embed_dim = decoder_embed_dim,
    decoder_depth = decoder_depth,
    decoder_num_heads = 8,
    mlp_ratio = 4.0,
    norm_pix_loss = False,
    cls_embed = True,
    sep_pos_embed = True,
    trunc_init = False,
    no_qkv_bias = False,
    mask_ratio_min = 0.6,
    mask_ratio_max = 0.8,
    mask_ratio_mu = 0.7,
    mask_ratio_std = 0.1,
)

act_net = dict(
        type = "ActorMaskSAC",
        embed_dim = decoder_embed_dim,
        depth = depth,
        cls_embed = True,
    )

cri_net = dict(
        type = "CriticMaskSAC",
        embed_dim = decoder_embed_dim,
        depth = depth,
        cls_embed = True,
    )

criterion = dict(type='MSELoss', reduction="none")
scheduler = dict(type='MultiStepLRScheduler',
                 multi_steps=[600 * n_steps_per_episode, 1000 * n_steps_per_episode, 1400 * n_steps_per_episode],
                 t_initial = num_episodes * n_steps_per_episode,
                 decay_t = 500 * n_steps_per_episode,
                 gamma = 0.1,
                 t_mul = 1.,
                 lr_min = 0.,
                 decay_rate = 1.,
                 warmup_t = 300 * n_steps_per_episode,
                 warmup_lr_init = 1e-8,
                 warmup_prefix = False,
                 cycle_limit = 0,
                 t_in_epochs = False,
                 noise_range_t = None,
                 noise_pct = 0.67,
                 noise_std = 1.0,
                 noise_seed = 42,
                 initialize = True)
optimizer = dict(type='AdamW',
                 params = None,
                 lr=lr)

agent = dict(
    type = "AgentMaskSAC",
    act_lr=act_lr,
    cri_lr=cri_lr,
    rep_lr=rep_lr,
    beta_lr=beta_lr,
    rep_net = rep_net,
    act_net = act_net,
    cri_net = cri_net,
    criterion = criterion,
    optimizer = optimizer,
    scheduler = scheduler,
    if_use_per=if_use_per,
    if_use_rep=if_use_rep,
    if_use_beta=if_use_beta,
    rep_loss_weight=rep_loss_weight,
    beta_loss_weight=beta_loss_weight,
    num_envs = num_envs,
    transition_shape = transition_shape,
    max_step = 1e4,
    gamma = 0.99,
    reward_scale = 2**0,
    repeat_times = repeat_times,
    batch_size = batch_size,
    clip_grad_norm = 3.0,
    soft_update_tau = 5e-3,
    state_value_tau = 0,
    device = None,
    action_wrapper_method = action_wrapper_method,
    T = T,
)
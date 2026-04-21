# Weight matrices: 1024x1024 (attn), 1024x4096 (MLP)

out_dir = 'out-shakespeare-large'
eval_interval = 200
eval_iters = 200
log_interval = 10

always_save_checkpoint = False
wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'large-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32        # reduced from 64 to fit larger model in V100 memory
block_size = 256

n_layer = 6
n_head = 16
n_embd = 1024
dropout = 0.2

learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
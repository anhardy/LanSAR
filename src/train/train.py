import gc
import glob
import re

import numpy as np
import torch
import torch.optim as optimizers
import tqdm
import transformers
from PIL import Image
from matplotlib import pyplot as plt
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_f1_score

from src.data_management.config import Config
from src.model_definition.lansar import LanSAR
from src.train.functions import lanseloss, plot_losses

import warnings
# import clip

"""
Training script for LanSAR. This went through well over a year of research 
with countless iterations of the training routine and model architecture. 
There is a lot of messy, poorly structured, and leftover code throughout this
entire project. It was cleaned up considerably for the first public commit, 
but needs additional refactoring.
"""


warnings.filterwarnings("ignore", category=UserWarning)

config = Config('../config.yaml')

model_path = f'../../models/{config.model_name}.pt'

space_mode = 'global'
pattern = r'global_p'

pattern_delta = r'global_d'

pattern_scale = r'unscaled'

pattern_input_only = r'input_only'
pattern_output_only = r'output_only'


# Relative or global positions
if re.search(pattern, config.model_name):
    space_mode = 'global'
else:
    space_mode = 'local'

if re.search(pattern_delta, config.model_name):
    deltas = False
else:
    deltas = True

if re.search(pattern_scale, config.model_name):
    scale = False
else:
    scale = True

if re.search(pattern_input_only, config.model_name):
    scale = True
    input_only = True
else:
    input_only = False

if re.search(pattern_output_only, config.model_name):
    scale = True
    input_only = False
    output_only = True
else:
    output_only = False


print(f"Using Deltas? {deltas}")
print(f"Scaling? {scale}")
print(f"Block Mode: {space_mode}")

train_ratio = 0
rollouts = 1
val_ratio = train_ratio

scale_embeddings = False
pad_token = -100
reactive = True
max_seq_context = 1
batch_size = 96  # 64 96
val_step = 25

train_paths = glob.glob(config.data_path + f'/train/calvin_train_{space_mode}.pt')
val_paths = glob.glob(config.data_path + f'/validation/calvin_validation_{space_mode}.pt')

all_batches = []
selected_batches = []

augment = False
window_size = 32
slide = 5


def load_model():
    model_info = torch.load(f'../../models/{config.load_path}')

    # Load model settings
    model = LanSAR(**model_info['model_args']).to(device)

    # Load weights
    model.load_state_dict(model_info['model_state_dict'])
    optimizer = optimizers.AdamW(model.parameters(), lr=0.0003, weight_decay=0.6)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    # scheduler = CyclicPlateau(optimizer, step_size_up=step_size)
    scheduler = None
    optimizer.load_state_dict(model_info['optimizer_state_dict'])
    # scheduler.load_state_dict(model_info['scheduler_state_dict'])
    start_epoch = model_info['epoch']
    best_val_loss = model_info['best_val_loss']
    model_args = model_info['model_args']
    current_phase = model_info['current_phase']
    current_step_teacher = model_info['current_step_teacher']
    current_step_pred = model_info['current_step_pred']

    # print('Current phase: ' + str(current_phase))
    # if current_phase == 1:
    #     print('Current Percent' + str(current_step_teacher))
    # elif current_phase == 2:
    #     print('Current Step' + str(current_step_pred))

    return model, optimizer, scheduler, start_epoch, best_val_loss, model_args, current_phase, \
        current_step_teacher, current_step_pred


datasets = []
for i, path in enumerate(train_paths):
    dataset = torch.load(path)
    dataset.gripper_states = dataset.agent_states[:, :, -1]
    dataset.relative_vectors_unscaled = dataset.relative_vectors
    dataset.scene_unscaled = dataset.scene
    dataset.agent_positions_unscaled = dataset.agent_positions
    dataset.agent_states_unscaled = dataset.agent_states
    dataset.agent_deltas_unscaled = dataset.agent_deltas
    if scale:
        dataset.scale_data()
    if augment:
        dataset.augment(window_size, slide)
    datasets.append(dataset)

val_datasets = []
for i, path in enumerate(val_paths):
    val_dataset = torch.load(path)
    val_dataset.gripper_states = val_dataset.agent_states[:, :, -1]
    val_dataset.relative_vectors_unscaled = val_dataset.relative_vectors
    val_dataset.scene_unscaled = val_dataset.scene
    val_dataset.agent_positions_unscaled = val_dataset.agent_positions
    val_dataset.agent_states_unscaled = val_dataset.agent_states
    val_dataset.agent_deltas_unscaled = val_dataset.agent_deltas
    if scale:
        val_dataset.scale_data()
    if augment:
        val_dataset.augment(window_size, slide)
    val_datasets.append(val_dataset)


def replace_inputs(model, current_batch_gpu, ratio):
    out, action = model(current_batch_gpu)
    out = out[:, :-1]
    action = action[:, :-1]
    if not input_only:
        y = current_batch_gpu['agent_positions'][:, 1:]
        # y_delta = current_batch_gpu['agent_deltas']
    else:
        y = current_batch_gpu['agent_positions_unscaled'][:, 1:]
        # y_delta = current_batch_gpu['agent_deltas_unscaled']
    y_action = current_batch_gpu['gripper_states'][:, 1:]

    pad_mask = torch.any(y == pad_token, dim=2)
    # If using deltas
    if deltas:
        # Get previous states
        prev = current_batch_gpu['agent_positions_unscaled'][:, :current_batch_gpu['agent_positions_unscaled'].size(1)-1][~pad_mask]
        delta = out[~pad_mask]
        if scale:
            # Unscale back to original size
            # prev = torch.tensor(dataset.position_scaler.inverse_transform(prev.cpu().detach())).to(device)
            if not input_only:
                delta = torch.tensor(dataset.delta_scaler.inverse_transform(delta.clone().cpu().detach())).to(device).float()
        # Add delta to previous states
        out[~pad_mask] = torch.add(prev, delta)
        if scale:  # Scale back down
            out[~pad_mask] = torch.tensor(dataset.position_scaler.transform(out[~pad_mask].clone().cpu().detach())).to(device).float()
    else:
        out[~pad_mask] = y[~pad_mask]
    action[~pad_mask] = y_action.unsqueeze(2)[~pad_mask]

    # mask_count = max(1, int(ratio * out.shape[1]))

    replacement_mask = torch.bernoulli(
        ratio * torch.ones((out.size(0), out.size(1)))).bool().to(
        current_batch_gpu['agent_positions'].device)
    mixed_input_states = torch.where(replacement_mask.unsqueeze(-1), out,
                                     current_batch_gpu['agent_states'][:, 1:, :out.size(2)])
    mixed_input_positions = torch.where(replacement_mask.unsqueeze(-1), out,
                                        current_batch_gpu['agent_positions'][:, 1:, :out.size(2)])
    current_batch_gpu['agent_states'][:, 1:, :out.size(2)] = mixed_input_states
    current_batch_gpu['agent_positions'][:, 1:, :out.size(2)] = mixed_input_positions

    return current_batch_gpu


def train(current_batch, train_model, optim):
    train_model.train()
    # clip_model.train()
    total_loss = 0
    total_mae_loss = 0
    # gc.collect()
    skip_count = 0
    total_f1 = 0
    # for i, current_batch in tqdm.tqdm(enumerate(batches), desc="Batch", total=len(batches), leave=False):
    optim.zero_grad()
    current_batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in current_batch.items()}
    with torch.autograd.set_detect_anomaly(True):
        # out, space = train_model(current_batch_gpu)

        if train_ratio > 0:
            model.eval()
            with torch.no_grad():
                for j in range(0, rollouts):
                    current_batch_gpu = replace_inputs(train_model, current_batch_gpu, train_ratio)
            model.train()

        # commands = clip.tokenize(current_batch_gpu['command']).to(device)
        # commands = clip_model.encode_text(commands)
        # current_batch_gpu['text_embeddings'] = commands

        out, action = train_model(current_batch_gpu)
        out = out[:, :-1]
        action = action[:, :-1]
        if not input_only:
            if not deltas:  # If not computing deltas, truth is next states
                y = current_batch_gpu['agent_positions'][:, 1:]
                y_unscaled = current_batch_gpu['agent_positions_unscaled'][:, 1:]
            else:  # Else, truth is delta sequence
                y = current_batch_gpu['agent_deltas']
                y_unscaled = current_batch_gpu['agent_deltas_unscaled']
        else:
            if not deltas:
                y = current_batch_gpu['agent_positions_unscaled'][:, 1:]
                y_unscaled = y
            else:
                y = current_batch_gpu['agent_deltas_unscaled']
                y_unscaled = y
        y_action = current_batch_gpu['gripper_states'][:, 1:]
        pad_mask = torch.any(y == pad_token, dim=2)
        y = y[~pad_mask]
        y_unscaled = y_unscaled[~pad_mask]
        y_action = y_action[~pad_mask]
        y_action = (y_action + 1) / 2
        # if scale:
        #     y_action = (y_action + 1) / 2
        out = out[~pad_mask]
        action = action[~pad_mask]
        loss, pos_loss, _ = lanseloss(out, action, y, y_action)  # ,
        if scale:
            out = torch.tensor(dataset.delta_scaler.inverse_transform(out.cpu().detach().numpy())).to(device)
        _, _, pos_loss_mae = lanseloss(out, action, y_unscaled, y_action)

        loss.backward()
        total_loss += pos_loss.item()
        total_mae_loss += pos_loss_mae.item()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

        optim.step()
        scheduler.step()
        f1 = binary_f1_score(torch.flatten(action), torch.flatten(y_action))
        total_f1 += f1
    del loss, out
    del current_batch_gpu
    try:
        torch.cuda.empty_cache()
    except Exception as e:
        print('Empty Cache Error')

    return total_loss + (1 - total_f1), total_f1, total_mae_loss  # total_loss / (len(batches) - skip_count)


def validate(batches, val_model):
    val_model.eval()
    # clip_model.eval()
    total_loss = 0
    total_f1 = 0
    with torch.no_grad():
        for i, current_batch in tqdm.tqdm(enumerate(batches), desc="Validation Batch", total=len(batches), leave=False):
            current_batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in
                                 current_batch.items()}

            if val_ratio > 0:
                with torch.no_grad():
                    for j in range(0, rollouts):
                        current_batch_gpu = replace_inputs(val_model, current_batch_gpu, val_ratio)

            # commands = clip.tokenize(current_batch_gpu['command']).to(device)
            # commands = clip_model.encode_text(commands)
            # current_batch_gpu['text_embeddings'] = commands
            out, action = val_model(current_batch_gpu)
            # if forcing_ratio == 0:
            out = out[:, :-1]
            action = action[:, :-1]
            if not input_only:
                if not deltas:  # If not computing deltas, truth is next states
                    y = current_batch_gpu['agent_positions'][:, 1:]
                else:  # Else, truth is delta sequence
                    y = current_batch_gpu['agent_deltas']
            else:
                if not deltas:
                    y = current_batch_gpu['agent_positions_unscaled'][:, 1:]
                else:
                    y = current_batch_gpu['agent_deltas_unscaled']
            y = current_batch_gpu['agent_deltas_unscaled']
            # y = current_batch_gpu['agent_deltas']
            y_action = current_batch_gpu['gripper_states'][:, 1:]
            pad_mask = torch.any(y == pad_token, dim=2)
            y = y[~pad_mask]
            y_action = y_action[~pad_mask]
            y_action = (y_action + 1) / 2
            out = out[~pad_mask]
            action = action[~pad_mask]

            # Validate on original space
            if scale:
                out = torch.tensor(dataset.delta_scaler.inverse_transform(out.cpu().detach().numpy())).to(device)
            loss, _, pos_loss = lanseloss(out, action, y, y_action)

            total_loss += pos_loss.item()
            f1 = binary_f1_score(torch.flatten(action), torch.flatten(y_action))
            total_f1 += f1
            del loss, out
            del current_batch_gpu
            # gc.collect()
            torch.cuda.empty_cache()
    total_f1 = total_f1 / len(batches)
    return (total_loss / (len(batches))) + (1 - total_f1), (
                total_loss / (len(batches))), total_f1  # total_loss / len(batches)


train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)
# clip_model = clip_model.float()

# TODO model config, list of model settings (e.g. length is layers, containing hidden size)
if not config.load:
    model_args = {
        'agent_state_dim': dataset.agent_states.size(2),  # + batch['text_embeddings'].shape[1],
        'agent_pos_dim': dataset.agent_positions.size(2),
        'relative_dim': dataset.relative_vectors.size(3),
        'scene_dim': dataset.scene.size(2),
        'img_dim': dataset.img_embeddings.size(2),
        'hidden_dim': 512,
        'dropout': 0.1,
        'enc_dim': 512,
        'space_enc_dim': 512,
        'n_heads_temp': 8,
        'n_heads_space': 16,
        'n_layers_temp': 6,
        'n_layers_space': 6,
        'space_queries': 92,
        'max_seq_len': 300,
        'text_dim': dataset.text_embeddings.size(1),
        'space_sequence_len': dataset.relative_vectors.size(2) + 3 + 1 + 1 + 1# Blocks + agent + scene state + images + cls
    }

    model = LanSAR(**model_args).to(device)
    model._initialize_weights()
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(opti0mizer, 'min', factor=0.5, patience=5)
    # scheduler = CyclicPlateau(optimizer, step_size_up=step_size)
    parameters = sum(p.numel() for p in model.parameters())  # if p.requires_grad)
    print(f'Model created with {parameters:,} parameters')
    current_step_pred = 1
    current_step_teacher = 0.0
    current_phase = 0
    best_val_loss = float('inf')
    best_train_loss = float('inf')
else:
    model, optimizer, scheduler, start_epoch, best_val_loss, model_args, current_phase, \
        current_step_teacher, current_step_pred = load_model()
# optimizer = optimizers.AdamW(list(clip_model.parameters()) + list(model.parameters()), lr=0.0003, weight_decay=0.01)
best_val_loss = float('inf')
optimizer = optimizers.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
best_train_loss = float('inf')


# TODO load this from saved model
epochs_without_improvement = 0

num_epochs = 165
num_warmup_steps = 5
train_step_multiplier = 1  # For fewer epochs but higher learning rates
# num_warmup_steps = len(train_dataloader) * num_warmup_steps
# num_total_steps = 1200  # Total number of training steps
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps * len(train_dataloader),
                                                         num_epochs * len(train_dataloader) * train_step_multiplier)
desired_restarts = 1
restart_step = int(round((num_epochs * len(train_dataloader) / desired_restarts)))
# scheduler = CosineAnnealingWarmRestarts(optimizer, restart_step)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
# scheduler = NoamLR(optimizer, 512, num_warmup_steps * len(train_batches))


epochs_without_improvement = 0
total_train_loss = float('inf')
val_loss = float('inf')
pos_loss = float('inf')
f1 = 0
train_losses = []
val_losses = []
with tqdm.tqdm(range(num_epochs), desc="Epoch", leave=False) as pbar:
    step_count = 0
    for epoch in pbar:
        gc.collect()
        torch.cuda.empty_cache()
        patience = 200000
        # TODO seed for replication
        lr = optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate: {lr}')

        total_train_loss = 0
        total_train_f1 = 0
        pbar.set_description(
            f"Epoch {epoch + 1}, Train Loss: {total_train_loss / 1:.4f}, Val Loss: "
            f"{val_loss:.4f}, Pos Loss: {pos_loss:.4f}, F1: {f1:.4f}, Best Val Loss: {best_val_loss:.4f}, "
            f"Patience Counter: {epochs_without_improvement}",
            refresh=True)

        # sanity_loss = validate(train_batches, model, current_step_pred, current_step_teacher)
        for i, current_batch in tqdm.tqdm(enumerate(train_dataloader), desc="Batch", total=len(train_dataloader),
                                          leave=False):
            train_loss, train_f1, train_mae = train(current_batch, model, optimizer)

            total_train_loss += train_mae
            total_train_f1 += train_f1
            step_count += 1
            if step_count % val_step == 0:
                val_loss, pos_loss, f1 = validate(val_dataloader, model)
                val_loss = pos_loss  # TODO remove
                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0

                    # TODO save model architecture/arguments
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        # 'scheduler_state_dict': scheduler.state_dict(),
                        'current_phase': current_phase,
                        'current_step_teacher': current_step_teacher,
                        'current_step_pred': current_step_pred,
                        'best_val_loss': best_val_loss,
                        'model_args': model_args
                    }, f'../../models/{config.model_name}.pt')
                    # torch.save(clip_model.state_dict(), f'../../models/{config.model_name}_clip.pt')
                else:
                    epochs_without_improvement += 1
                    # print(f'Patience Counter: {epochs_without_improvement}')
            # train_loss = train_loss / len(train_dataloader)
            pbar.set_description(
                f"Epoch {epoch}, Train Loss: {total_train_loss / (i + 1):.4f}, Val Loss: "
                f"{val_loss:.4f}, Pos Loss: {pos_loss:.4f}, F1: {f1:.4f}, Best Val Loss: {best_val_loss:.4f}, "
                f"Patience Counter: {epochs_without_improvement}",
                refresh=True)

            # print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_train_loss / i + 1}, Val Loss: {val_loss}')
            # print(f'Val Position Loss: {pos_loss}, Val F1: {f1}')

        train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if train_loss < best_train_loss:
            # TODO save model architecture/arguments
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
                'current_phase': current_phase,
                'current_step_teacher': current_step_teacher,
                'current_step_pred': current_step_pred,
                'best_val_loss': best_val_loss,
                'model_args': model_args
            }, f'../../models/{config.model_name}_train.pt')
            # torch.save(clip_model.state_dict(), f'../../models/{config.model_name}_clip_train.pt')
        if epoch > 0:
            plot_losses(train_losses, val_losses, config)

print(best_val_loss)

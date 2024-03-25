import glob
import os
from random import sample

import numpy as np
import torch
from matplotlib import pyplot as plt
# from torch_geometric.data import Data, ClusterData, Batch
from torch.utils.data import Dataset

# from torch_geometric.loader import ClusterLoader
from tqdm import tqdm


def pad_sequence(sequence, max_len):
    # Pad the sequence with zeros up to the max_len
    remaining_shape = list(sequence.shape[1:])
    padded_sequence = np.ones([max_len] + remaining_shape) * -100
    indices = np.arange(0, sequence.shape[0])
    padded_sequence[indices] = sequence
    return padded_sequence


class SpatialTransformerData(Dataset):
    def __init__(self, tasks, position_scaler, state_scaler, spatial_scaler, scene_scaler, delta_scaler):
        self.agent_deltas_unscaled = None
        self.relative_vectors_unscaled = None
        self.scene_unscaled = None
        self.agent_positions_unscaled = None
        self.agent_states_unscaled = None
        self.gripper_states = None
        self.position_scaler = position_scaler
        self.spatial_scaler = spatial_scaler
        self.state_scaler = state_scaler
        self.scene_scaler = scene_scaler
        self.delta_scaler = delta_scaler

        (self.agent_positions, self.agent_states, self.agent_deltas, self.scene, self.relative_vectors,
         self.text_embeddings, self.img_embeddings, self.gripper_embeddings,
         self.command, self.task_desc) = self.preprocess_tasks(tasks)

    def preprocess_tasks(self, tasks):
        # Find the maximum sequence length across all tasks
        max_seq_len = max(len(task['robot_states_scaled']) for task in tasks)

        # Initialize lists to store processed data
        (agent_states, agent_positions, agent_deltas, scenes,
         relative_vectors, text_embeddings, img_embeddings, gripper_embeddings, command, task_desc) = ([], [], [], [],
                                                                                                       [], [], [], [],
                                                                                                       [], [])

        for i, task in tqdm(enumerate(tasks), desc="Creating Dataset", total=len(tasks), leave=False):
            # Pad each task's data to the maximum sequence length
            agent_states.append(pad_sequence(task['robot_states_scaled'], max_seq_len))
            agent_positions.append(pad_sequence(task['robot_positions_scaled'], max_seq_len))
            agent_deltas.append(pad_sequence(task['state_deltas'], max_seq_len - 1))
            scenes.append(pad_sequence(task['scene'], max_seq_len))
            relative_vectors.append(pad_sequence(task['relative_scaled'], max_seq_len))
            text_embeddings.append(task['text_embedding'])
            img_embeddings.append(pad_sequence(task['img_embedding'], max_seq_len)),
            gripper_embeddings.append(pad_sequence(task['gripper_embedding'], max_seq_len))
            command.append(task['annotation'])
            task_desc.append(task['task_desc'])

        return (torch.tensor(agent_positions), torch.tensor(agent_states), torch.tensor(agent_deltas),
                torch.tensor(scenes),
                torch.tensor(relative_vectors), torch.tensor(text_embeddings), torch.tensor(img_embeddings),
                torch.tensor(gripper_embeddings),
                command, task_desc)

    def __len__(self):
        return len(self.agent_states)

    def __getitem__(self, idx):
        return {
            'agent_states': torch.tensor(self.agent_states[idx], dtype=torch.float),
            'agent_states_unscaled': torch.tensor(self.agent_states_unscaled[idx], dtype=torch.float),
            'gripper_states': torch.tensor(self.gripper_states[idx], dtype=torch.float),
            'agent_positions': torch.tensor(self.agent_positions[idx], dtype=torch.float),
            'agent_positions_unscaled': torch.tensor(self.agent_positions_unscaled[idx], dtype=torch.float),
            'agent_deltas': torch.tensor(self.agent_deltas[idx], dtype=torch.float),
            'agent_deltas_unscaled': torch.tensor(self.agent_deltas_unscaled[idx], dtype=torch.float),
            'scene': torch.tensor(self.scene[idx], dtype=torch.float),
            'scene_unscaled': torch.tensor(self.scene_unscaled[idx], dtype=torch.float),
            'relative_vectors': torch.tensor(self.relative_vectors[idx], dtype=torch.float),
            'relative_vectors_unscaled': torch.tensor(self.relative_vectors_unscaled[idx], dtype=torch.float),
            'text_embeddings': torch.tensor(self.text_embeddings[idx], dtype=torch.float),
            'img_embeddings': torch.tensor(self.img_embeddings[idx], dtype=torch.float),
            'gripper_embeddings': torch.tensor(self.gripper_embeddings[idx], dtype=torch.float),
            'command': self.command[idx],
            'task_desc': self.task_desc[idx]
        }

    def augment(self, window_size, stride):
        num_samples = self.agent_states.shape[0]

        augmented_data = {}
        for key in ['agent_states', 'agent_states_unscaled', 'gripper_states', 'agent_positions',
                    'agent_positions_unscaled', 'agent_deltas', 'agent_deltas_unscaled', 'scene',
                    'scene_unscaled',
                    'img_embeddings', 'gripper_embeddings']:
            data = getattr(self, key)
            if key == 'gripper_states':
                data = data.unsqueeze(2)
            if key == 'agent_deltas' or key == 'agent_deltas_unscaled':
                last_values = data[:, -1:, :]
                data = torch.cat((data, last_values), dim=1)

            sequence_length = data.shape[1]

            output_size = ((sequence_length - window_size) // stride + 1, window_size)
            num_augmented_per_sequence = output_size[0]
            total_augmented_samples = num_samples * num_augmented_per_sequence
            channels = data.shape[2]
            indices = torch.arange(0, sequence_length - window_size + 1, stride)
            windows_indices = indices[:, None] + torch.arange(window_size)
            windows_indices = windows_indices.flatten()

            data = data[:, windows_indices, :].reshape(num_samples, num_augmented_per_sequence,
                                                       window_size, channels).permute(0, 1, 2, 3).reshape(
                total_augmented_samples, window_size, channels)
            if key == 'gripper_states':
                data = data.squeeze(2)
            setattr(self, key, data)

        for key in ['relative_vectors', 'relative_vectors_unscaled']:
            data = getattr(self, key)
            _, _, spatial_seq, channels = data.shape
            windows_indices = indices[:, None] + torch.arange(window_size)
            windows_indices = windows_indices.flatten()

            augmented_data[key] = (data[:, windows_indices, :, :].
                                   reshape(num_samples, num_augmented_per_sequence, window_size,
                                           spatial_seq, channels).permute(0, 1, 2, 3, 4).
                                   reshape(total_augmented_samples, window_size, spatial_seq, channels))

        augmented_data['command'] = np.array(self.command).repeat(num_augmented_per_sequence, axis=0)
        augmented_data['task_desc'] = np.array(self.task_desc).repeat(num_augmented_per_sequence, axis=0)
        augmented_data['text_embeddings'] = torch.tensor(np.array(self.text_embeddings).
                                                         repeat(num_augmented_per_sequence, axis=0), dtype=torch.float)

        for key in augmented_data:
            setattr(self, key, augmented_data[key])

        self.agent_deltas = self.agent_deltas[:, :-1]
        self.agent_deltas_unscaled = self.agent_deltas_unscaled[:, :-1]

        print(f"Augmentation complete with {self.agent_deltas.size(0)} samples")

    def fit_data(self, position_scaler, state_scaler, spatial_scaler, scene_scaler, delta_scaler, pad_token=-100):
        # Helper function to reshape, filter, and fit
        def reshape_and_fit(data, scaler):
            data_reshaped = data.view(-1, data.shape[-1])

            mask = ~(data_reshaped == pad_token).any(dim=1)
            filtered_data = data_reshaped[mask]

            filtered_data_np = filtered_data.numpy()

            scaler.fit(filtered_data_np)

            return scaler

        original_shape = self.agent_positions.shape
        self.position_scaler = reshape_and_fit(self.agent_positions, position_scaler)

        original_shape = self.agent_states.shape
        self.state_scaler = reshape_and_fit(self.agent_states, state_scaler)

        original_shape = self.relative_vectors.shape
        self.spatial_scaler = reshape_and_fit(self.relative_vectors, spatial_scaler)

        original_shape = self.scene.shape
        self.scene_scaler = reshape_and_fit(self.scene, scene_scaler)

        self.delta_scaler = reshape_and_fit(self.agent_deltas, delta_scaler)

    def scale_data(self, pad_token=-100, output_only=False):
        # Helper function to reshape, scale, and handle pad_token
        def reshape_and_scale(data, scaler, original_shape):
            data_reshaped = data.reshape(-1, data.shape[-1])

            # Mask pad_token values
            mask = (data_reshaped != pad_token)
            masked_data = np.where(mask, data_reshaped, np.nan)

            # Scale the data
            scaled_data = scaler.transform(masked_data)

            # Restore pad_token values
            scaled_data = np.where(np.isnan(scaled_data), pad_token, scaled_data)

            return torch.tensor(scaled_data.reshape(original_shape))

        if not output_only:
            original_shape = self.agent_positions.shape
            self.agent_positions = reshape_and_scale(self.agent_positions, self.position_scaler, original_shape)

            original_shape = self.agent_states.shape
            self.agent_states = reshape_and_scale(self.agent_states, self.state_scaler, original_shape)

            original_shape = self.relative_vectors.shape
            self.relative_vectors = reshape_and_scale(self.relative_vectors, self.spatial_scaler, original_shape)

            original_shape = self.scene.shape
            self.scene = reshape_and_scale(self.scene, self.scene_scaler, original_shape)

        original_shape = self.agent_deltas.shape
        self.agent_deltas = reshape_and_scale(self.agent_deltas, self.delta_scaler, original_shape)

    def unscale_data(self):
        def reshape_and_unscale(data, scaler, original_shape):
            # Reshape data for the scaler (2D)
            data_reshaped = data.reshape(-1, data.shape[-1])
            # t1 = data.cpu().numpy()
            # t2 = data_reshaped.cpu().numpy()
            # Unscale data
            unscaled_data = scaler.inverse_transform(data_reshaped)
            # Reshape back to the original shape
            return unscaled_data.reshape(original_shape)

        # Unscale agent_positions
        original_shape = self.agent_positions.shape
        self.agent_positions = reshape_and_unscale(self.agent_positions, self.position_scaler, original_shape)

        # Unscale agent_states
        original_shape = self.agent_states.shape
        self.agent_states = reshape_and_unscale(self.agent_states, self.state_scaler, original_shape)

        # Unscale relative_vectors
        original_shape = self.relative_vectors.shape
        self.relative_vectors = reshape_and_unscale(self.relative_vectors, self.spatial_scaler, original_shape)

        # Unscale scene
        original_shape = self.scene.shape
        self.scene = reshape_and_unscale(self.scene, self.scene_scaler, original_shape)

    def delete_attributes(self):
        del self.agent_positions
        del self.agent_states
        del self.agent_states
        del self.scene
        del self.relative_vectors
        del self.text_embeddings
        del self.img_embeddings
        del self.command
        del self.task_desc
        del self

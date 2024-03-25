# import gc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# Generate staircased mask
def generate_causal_mask(seq_len, num_agents, device):
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).to(device)

    mask = mask.view(seq_len, 1, seq_len, 1)

    mask = mask.repeat(1, num_agents, 1, num_agents)

    mask = mask.view(seq_len * num_agents, seq_len * num_agents)

    return mask


def generate_cross_attention_mask(target_seq_len, input_seq_len, device):
    return torch.ones(target_seq_len, input_seq_len, dtype=torch.bool).to(device)


# Output head networks
class OutputNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5):
        super(OutputNetwork, self).__init__()
        hidden_dim_1 = max(in_dim - 256, 64)
        hidden_dim_2 = max(hidden_dim_1 - 256, 64)

        self.fc1 = nn.Linear(in_dim, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, out_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.view(-1, original_shape[-1])

        x = F.leaky_relu(self.dropout(self.bn1(self.fc1(x))))
        x = F.leaky_relu(self.dropout(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        if len(original_shape) > 2:
            x = x.view(original_shape[0], original_shape[1], x.shape[1])

        return x


# Input projection networks
class InputNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.5):
        super(InputNetwork, self).__init__()
        hidden_dim_1 = hidden_dim // 3
        hidden_dim_2 = hidden_dim * 2 // 3

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim_2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.model(x)


class LanSAR(nn.Module):
    def __init__(self, agent_pos_dim, agent_state_dim, scene_dim, relative_dim, hidden_dim, text_dim, img_dim,
                 enc_dim=512, space_enc_dim=128, n_heads_temp=8, n_heads_space=8, n_layers_temp=6, n_layers_space=6,
                 space_queries=32, space_sequence_len=10,
                 dropout=0.8, max_seq_len=10000, ignore_tokens=0):
        super(LanSAR, self).__init__()
        self.agent_embedding_target = nn.Embedding(agent_state_dim, enc_dim)
        self.space_embedding = nn.Embedding(space_sequence_len, hidden_dim)

        self.command_projection_input = InputNetwork(text_dim, hidden_dim)
        self.command_projection_target = InputNetwork(text_dim, hidden_dim)
        self.img_projection = InputNetwork(img_dim, hidden_dim)
        self.gripper_projection = InputNetwork(img_dim, hidden_dim)
        self.agent_projection = InputNetwork(agent_state_dim, hidden_dim)
        self.scene_projection = InputNetwork(scene_dim, hidden_dim)
        self.agent_space_projection = InputNetwork(agent_state_dim, hidden_dim)
        self.relative_space_projection = InputNetwork(relative_dim, hidden_dim)

        self.pos_emb_input = nn.Embedding(max_seq_len, enc_dim)
        self.pos_emb_target = nn.Embedding(max_seq_len, enc_dim)
        self.relative_dim = relative_dim
        self.drop_p = dropout
        self.d_model = enc_dim

        self.max_seq_len = max_seq_len

        self.space_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=enc_dim, nhead=n_heads_space,
                                                                              batch_first=True, dropout=dropout,
                                                                              norm_first=False),
                                                   num_layers=n_layers_space)

        self.space_token = nn.Embedding(1, enc_dim)

        self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=enc_dim, nhead=n_heads_temp,
                                                                                    batch_first=True, dropout=dropout,
                                                                                    norm_first=False),
                                                         num_layers=n_layers_temp)

        out_dim = enc_dim
        self.agent_out_dim = agent_pos_dim

        self.agent_out = OutputNetwork(out_dim, agent_pos_dim, dropout)
        self.action_out = OutputNetwork(out_dim, 1, dropout)


    # Xavier Uniform Initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.TransformerEncoder, nn.TransformerDecoder)):
                for submodule in m.modules():
                    if isinstance(submodule, nn.Linear):
                        init.xavier_uniform_(submodule.weight)
                        if submodule.bias is not None:
                            init.constant_(submodule.bias, 0)



    def context_encoder(self, agent_positions_expanded, relative_vectors, scene_states_expanded,
                        img_expanded, gripper_expanded, pad_token):

        # We avoid dealing with learning padded values or batch normalization issues by simply not encoding them
        pad_mask = (agent_positions_expanded == pad_token).any(dim=-1).any(dim=-1)

        states_actual = agent_positions_expanded[~pad_mask]
        relative_actual = relative_vectors[~pad_mask]
        scene_states_actual = scene_states_expanded[~pad_mask]
        img_actual = img_expanded[~pad_mask]
        gripper_actual = gripper_expanded[~pad_mask]
        # Ensure 2D for linear layers and 1d batchnorm
        if len(list(states_actual.size())) > 2:
            states_actual = states_actual.reshape(-1, agent_positions_expanded.size(-1))
            relative_actual = relative_actual.reshape(-1, relative_vectors.size(-1))
            scene_states_actual = scene_states_actual.reshape(-1, scene_states_expanded.size(-1))
            img_actual = img_actual.reshape(-1, img_expanded.size(-1))
            gripper_actual = gripper_actual.reshape(-1, img_expanded.size(-1))

        # Get original dimensions
        agent_size = agent_positions_expanded.size()

        batch_size, temporal_length, _, _ = agent_size

        # Encode and reshape back
        encoded_agents = self.agent_space_projection(states_actual)
        encoded_relative = self.relative_space_projection(relative_actual)
        encoded_scene_states = self.scene_projection(scene_states_actual)
        encoded_image = self.img_projection(img_actual)
        encoded_gripper = self.gripper_projection(gripper_actual)

        encoded_agents = encoded_agents.unsqueeze(1)
        encoded_relative = encoded_relative.reshape(-1, relative_vectors.size(2), encoded_relative.size(-1))
        encoded_scene_states = encoded_scene_states.unsqueeze(1)
        encoded_image = encoded_image.unsqueeze(1)
        encoded_gripper = encoded_gripper.unsqueeze(1)


        # Create context sequence
        context = torch.cat([encoded_agents, encoded_relative, encoded_scene_states, encoded_image,
                           encoded_gripper], dim=1)


        # Get CLS token
        space_token_indices = torch.zeros(context.size(0), 1, dtype=torch.long, device=context.device)
        space_token_embeddings = self.space_token(space_token_indices)

        # Prepend
        context = torch.cat([space_token_embeddings, context], dim=1)
        # Modality positional encodings
        space_positions = torch.arange(0, context.size(1), dtype=torch.long,
                                       device=context.device)
        space_encoding = self.space_embedding(space_positions)
        context = torch.add(context, space_encoding.unsqueeze(0))
        context = F.dropout(context, p=self.drop_p, training=self.training)

        # scene = self.space_encoder(scene).squeeze(1) # FOR SET TRANSFORMER
        context = self.space_encoder(context)[:, 0]

        # If there is padding, need to reshape and fill in the non-padded values
        if pad_mask.any():
            # Flatten the pad_mask to match the first dimension of true_scene
            flat_pad_mask = pad_mask.view(batch_size * temporal_length, -1).squeeze()
            # Initialize true context
            true_context = torch.ones(batch_size * temporal_length, context.size(1)).to(context.device) * -1
            true_context[~flat_pad_mask] = context
            context = true_context
        context = context.reshape(batch_size, temporal_length, -1)

        return context, pad_mask

    def forward(self, batch, autoencode=False, max_seq_context=1, pad_token=-100):

        # Get input tensors
        agent_states = batch['agent_states']
        # agent_positions = batch['agent_positions']
        text_embeddings = batch['text_embeddings']
        img_embeddings = batch['img_embeddings']
        gripper_embeddings = batch['gripper_embeddings']
        scene_input = batch['scene']

        # img_embeddings = self.vit_encoder(img_embeddings)

        max_seq_context = agent_states.size(1)

        relative_vectors = batch['relative_vectors'][:, :max_seq_context, :, :]

        agent_positions_expanded = agent_states.unsqueeze(2)[:, :max_seq_context, :, :]
        scene_expanded = scene_input.unsqueeze(2)[:, :max_seq_context, :, :]
        img_expanded = img_embeddings.unsqueeze(2)[:, :max_seq_context, :, :]
        gripper_expanded = gripper_embeddings.unsqueeze(2)[:, :max_seq_context, :, :]
        # Encode spatial information into each agent at each timestep with context encoder.
        # Temporal order is along batch dimension
        scene, pad_mask_input = self.context_encoder(agent_positions_expanded, relative_vectors, scene_expanded,
                                                     img_expanded, gripper_expanded, pad_token)

        agent_target = agent_states[:, :max_seq_context]

        # Get where sequences contain padded values values
        pad_mask = torch.any(agent_target == pad_token, dim=2)

        if len(list(scene.size())) < 3:
            scene = scene.unsqueeze(1)

        # Project command to same dimensionality
        embedding_proj_input = self.command_projection_input(text_embeddings.squeeze(1))
        embedding_proj_input = embedding_proj_input.unsqueeze(1)
        embedding_proj_target = self.command_projection_target(text_embeddings.squeeze(1))
        embedding_proj_target = embedding_proj_target.unsqueeze(1)
        embedding_proj_input = embedding_proj_input.repeat(1, scene.size(1), 1)
        embedding_proj_target = embedding_proj_target.repeat(1, scene.size(1), 1)

        # scene = torch.torch.cat([embedding_proj_input, scene], dim=1)
        # Interleave command
        scene = torch.stack((embedding_proj_input, scene), dim=2).view(scene.size(0), -1, scene.size(-1))

        # Again only encode states which are not padded
        states_actual = agent_target[~pad_mask]
        batch_size, temporal_length, _ = agent_target.size()
        if len(list(states_actual.size())) > 2:
            states_actual = states_actual.reshape(-1, agent_target.size(-1))

        encoded_agents = self.agent_projection(states_actual)
        # encoded_agents = scene[:, 1:]

        # Recreate sequences with padding at new dimensionality
        if torch.any(pad_mask):
            padded_agents = torch.ones(batch_size, temporal_length, encoded_agents.size(-1)).to(scene.device) * -1
            padded_agents[~pad_mask] = encoded_agents
            encoded_agents = padded_agents

        else:
            encoded_agents = encoded_agents.reshape(-1, temporal_length, encoded_agents.size(-1))


        # Get padded values
        pad_mask_target = pad_mask

        # Create pad mask for transformer
        encoded_agents[pad_mask_target] = 0

        # encoded_agents = torch.cat([embedding_proj_target, encoded_agents], dim=1)
        encoded_agents = torch.stack((embedding_proj_target, encoded_agents), dim=2).view(scene.size(0), -1, scene.size(-1))
        # encoded_agents = torch.cat([embedding_proj_target, encoded_agents], dim=1)
        interleaved_mask = torch.zeros_like(pad_mask).to(pad_mask.device)

        pad_mask = torch.stack((interleaved_mask, pad_mask), dim=2).view(scene.size(0), -1)

        # pad_mask = torch.cat([torch.zeros(pad_mask.size(0), 1, dtype=bool, device=encoded_agents.device), pad_mask],
        #                      dim=1)
        pad_mask_target = pad_mask

        # Temporal positional encodings for target
        positions = torch.arange(0, encoded_agents.size(1), dtype=torch.long,
                                 device=encoded_agents.device)
        positions_enc = self.pos_emb_target(positions).unsqueeze(0)
        target_enc = torch.add(encoded_agents, positions_enc)
        target_enc = F.dropout(target_enc, p=self.drop_p, training=self.training)

        # Temporal positional encodings for context
        positions = torch.arange(0, scene.size(1), dtype=torch.long,
                                 device=encoded_agents.device)
        positions_enc_scene = self.pos_emb_input(positions)

        scene = torch.add(scene, positions_enc_scene)
        scene = F.dropout(scene, p=self.drop_p, training=self.training)

        # Causally mask
        causal_mask_target = generate_causal_mask(encoded_agents.size(1), 1, encoded_agents.device)
        pad_mask_target = pad_mask_target.reshape(scene.size(0), -1)

        memory_key_padding_mask = pad_mask_target
        memory_mask = causal_mask_target
        memory_is_causal = True

        # scene = torch.zeros_like(scene).to(scene.device)

        # Predict with decoder
        pred = self.transformer_decoder(tgt=target_enc, memory=scene, tgt_mask=causal_mask_target, tgt_is_causal=True,
                                        tgt_key_padding_mask=pad_mask_target, memory_mask=memory_mask,
                                        memory_key_padding_mask=memory_key_padding_mask,
                                        memory_is_causal=memory_is_causal)

        # Decode back to motion and gripper action
        out = self.agent_out(pred)
        action = F.sigmoid(self.action_out(pred))

        # Ignore prediction for command
        out = out[:, 1::2]
        action = action[:, 1::2]


        return out, action

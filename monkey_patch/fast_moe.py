"""
Adapted from https://github.com/shawntan/scattermoe/blob/63b76a2f5f28c052fb4cd7c34479a54158354052/examples/mixtral/modeling_mixtral.py#L667
"""
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.qwen3_moe.modular_qwen3_moe import Qwen3MoeMLP
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from monkey_patch.scattermoe.ops import padded_block_indices, scattered_experts

    
class ScatterMoeQwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        ### PATCH START ###
        sorted_expert_idxs, sorted_scattered_idxs = torch.sort(selected_experts.flatten())
        padded_block_idxs, expert_offsets = padded_block_indices(sorted_expert_idxs, self.num_experts)

        # stack all experts with shape (num_experts, in_dim, out_dim)
        parallel_gate_proj = torch.stack(
            [expert.gate_proj.weight for expert in self.experts], dim=0
        ).permute(0, 2, 1)
        parallel_up_proj = torch.stack(
            [expert.up_proj.weight for expert in self.experts], dim=0
        ).permute(0, 2, 1)
        parallel_down_proj = torch.stack(
            [expert.down_proj.weight for expert in self.experts], dim=0
        ).permute(0, 2, 1)

        # conduct the parallel matmul by gate_proj and up_proj respectively
        # self.gate_proj(x) and self.up_proj(x)
        parallel_gate = scattered_experts(
            hidden_states,
            parallel_gate_proj,
            self.top_k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            grouped_out=True
        )
        parallel_up = scattered_experts(
            hidden_states,
            parallel_up_proj,
            self.top_k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            grouped_out=True
        )
        final_hidden_states = scattered_experts(
            LigerSiLUMulFunction.apply(parallel_gate, parallel_up),
            parallel_down_proj,
            1,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates=routing_weights,
            grouped_in=True
        ).view(batch_size, sequence_length, hidden_dim).to(hidden_states.dtype)
        ### PATCH END ###

        return final_hidden_states, router_logits
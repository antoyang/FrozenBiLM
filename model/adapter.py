import torch.nn as nn
import torch


class Adapter(nn.Module):
    def __init__(
        self, ds_factor, hidden_dim, ln_after=False, ln_before=False, dropout=0.1
    ):
        super().__init__()
        assert not hidden_dim % ds_factor
        self.down = nn.Linear(hidden_dim, hidden_dim // ds_factor)
        self.act = nn.ReLU()
        self.up = nn.Linear(hidden_dim // ds_factor, hidden_dim)
        self.apply(self.init_weights)
        self.ln_after = ln_after
        self.ln_before = ln_before
        self.dropout = dropout
        if ln_after or ln_before:
            self.ln = nn.LayerNorm(hidden_dim)
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def init_weights(self, m: nn.Module, std=1e-3):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=std)
            torch.nn.init.normal_(m.bias, std=std)
            m.weight.data = torch.clamp(m.weight.data, min=-2 * std, max=2 * std)
            m.bias.data = torch.clamp(m.bias.data, min=-2 * std, max=2 * std)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)

    def forward(self, hidden_states):
        if self.ln_before:
            residual = self.ln(hidden_states)
            residual = self.down(residual)
        else:
            residual = self.down(hidden_states)
        residual = self.act(residual)
        if self.dropout:
            residual = self.dropout(residual)
        residual = self.up(residual)
        if self.ln_after:
            residual = self.ln(hidden_states)
        return hidden_states + residual

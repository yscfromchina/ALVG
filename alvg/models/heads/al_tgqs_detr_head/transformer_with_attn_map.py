import copy
import torch.nn.functional as F
from torch import nn
import torch


class TransformerDecoder(nn.Module):

    def __init__(self, num_layers=3, return_intermediate=True):
        super().__init__()
        self.layers = _get_clones(TransformerDecoderLayer(), num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, query, key, value, key_pos, query_pos, key_padding_mask):
        output = query
        attn_map = None

        intermediate = []
        intermediate_attn_map = []

        for layer in self.layers:
            output, attn_map = layer(query, key, value, key_pos, query_pos, key_padding_mask)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_attn_map.append(attn_map)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_attn_map)

        return output.unsqueeze(0), attn_map.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim=256, num_heads=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu", grec=False):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, query, key, value, key_pos, query_pos, key_padding_mask):

        q = k = self.with_pos_embed(query, query_pos)
        query2 = self.self_attn(q, k, value=query)[0]
        query = query + self.dropout(query2)
        query = self.norm1(query)

        q = self.with_pos_embed(query, query_pos)
        k = self.with_pos_embed(key, key_pos)
        q2, attn_map = self.cross_attn(q, k, value, key_padding_mask)
        q = q + self.dropout1(q2)
        q = self.norm3(q)

        # ffn
        q = self.forward_ffn(q)

        return q, attn_map


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
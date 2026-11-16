"""All neural models. Every model exposes:
    forward(x_num, x_cat)                -> logits (B, 2)
    cat_embed(x_cat)                      -> (B, n_cat, e) or None
    forward_embedded(x_num, cat_e)        -> logits   (used by SHAP so categorical inputs become differentiable)
'ordinal' models receive X_ord as x_num and no categorical input.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NumericTokenizer(nn.Module):
    """shared=True  : one Linear(1->d) for every feature (thesis model)
       shared=False : a separate (w_j, b_j) per feature (FT-Transformer style)"""
    def __init__(self, n, d, shared):
        super().__init__()
        self.shared = shared
        if shared:
            self.lin = nn.Linear(1, d)
        else:
            self.weight = nn.Parameter(torch.empty(n, d))
            self.bias = nn.Parameter(torch.empty(n, d))
            bound = 1 / math.sqrt(d)
            nn.init.uniform_(self.weight, -bound, bound)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.shared:
            return self.lin(x.unsqueeze(-1))
        return x.unsqueeze(-1) * self.weight + self.bias


class CatEmbedding(nn.Module):
    def __init__(self, cards, d):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c in cards])
        for e in self.embs:
            nn.init.uniform_(e.weight, -1 / math.sqrt(d), 1 / math.sqrt(d))

    def forward(self, x_cat):
        return torch.stack([e(x_cat[:, j]) for j, e in enumerate(self.embs)], dim=1)


class _Base(nn.Module):
    n_cat = 0

    def cat_embed(self, x_cat):
        return None

    def forward(self, x_num, x_cat=None):
        return self.forward_embedded(x_num, self.cat_embed(x_cat) if self.n_cat else None)


class FeatureTokenTransformer(_Base):
    """Feature-Token Transformer (thesis model and its variants).

    num_embedding : 'shared' (thesis) | 'per_feature'
    cat_mode      : 'ordinal' (categoricals arrive scaled inside x_num, thesis) | 'embed'
    identity      : learnable feature-identity embedding added to every token
    single_token  : the thesis Exp01 control: all features -> one token (sequence length 1)
    """
    def __init__(self, n_num, cat_cards=(), d_model=64, nhead=4, num_layers=2, dim_ff=2048,
                 dropout=0.1, num_embedding="shared", cat_mode="ordinal", identity=True,
                 pooling="mean", single_token=False):
        super().__init__()
        self.single_token = single_token
        self.cat_mode = cat_mode
        self.n_cat = len(cat_cards) if cat_mode == "embed" else 0
        self.n_tokens = n_num + self.n_cat
        self.pooling = pooling
        if single_token:
            self.in_proj = nn.Linear(n_num, d_model)
        else:
            self.num_tok = NumericTokenizer(n_num, d_model, shared=(num_embedding == "shared"))
            if self.n_cat:
                self.cat_tok = CatEmbedding(cat_cards, d_model)
            self.identity = nn.Parameter(torch.zeros(1, self.n_tokens, d_model)) if identity else None
            if pooling == "cls":
                self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 2)

    def cat_embed(self, x_cat):
        return self.cat_tok(x_cat) if self.n_cat else None

    def forward_embedded(self, x_num, cat_e=None):
        if self.single_token:
            h = self.encoder(self.in_proj(x_num).unsqueeze(1)).squeeze(1)
            return self.head(h)
        t = self.num_tok(x_num)
        if cat_e is not None:
            t = torch.cat([t, cat_e], dim=1)
        if self.identity is not None:
            t = t + self.identity
        if self.pooling == "cls":
            t = torch.cat([self.cls.expand(t.size(0), -1, -1), t], dim=1)
        h = self.encoder(t)
        h = h[:, 0] if self.pooling == "cls" else h.mean(dim=1)
        return self.head(h)


class MLP(_Base):
    """MLP; categorical features through small embeddings when cat_mode='embed'."""
    def __init__(self, n_num, cat_cards=(), cat_mode="embed", emb_dim=8, hidden=(256, 256, 128), dropout=0.1):
        super().__init__()
        self.n_cat = len(cat_cards) if cat_mode == "embed" else 0
        if self.n_cat:
            self.cat_tok = CatEmbedding(cat_cards, emb_dim)
        layers, d_in = [], n_num + self.n_cat * emb_dim
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d_in = h
        layers.append(nn.Linear(d_in, 2))
        self.net = nn.Sequential(*layers)

    def cat_embed(self, x_cat):
        return self.cat_tok(x_cat) if self.n_cat else None

    def forward_embedded(self, x_num, cat_e=None):
        x = x_num if cat_e is None else torch.cat([x_num, cat_e.flatten(1)], dim=1)
        return self.net(x)


class CNN1D(_Base):
    """1D-CNN over the (ordinal, scaled) feature vector, a common NIDS deep baseline."""
    def __init__(self, n_num, cat_cards=(), channels=(32, 64), dropout=0.2, **_):
        super().__init__()
        c1, c2 = channels
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, 3, padding=1), nn.BatchNorm1d(c1), nn.ReLU(),
            nn.Conv1d(c1, c2, 3, padding=1), nn.BatchNorm1d(c2), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c2 * (n_num // 2), 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, 2))

    def forward_embedded(self, x_num, cat_e=None):
        return self.head(self.conv(x_num.unsqueeze(1)))


class _ReGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class _FTBlock(nn.Module):
    def __init__(self, d, heads, attn_drop, ffn_hidden, ffn_drop, res_drop, first):
        super().__init__()
        self.ln1 = nn.Identity() if first else nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=attn_drop, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * ffn_hidden), _ReGLU(), nn.Dropout(ffn_drop), nn.Linear(ffn_hidden, d))
        self.drop = nn.Dropout(res_drop)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, h, h, need_weights=False)[0])
        return x + self.drop(self.ffn(self.ln2(x)))


class FTTransformer(_Base):
    """FT-Transformer (Gorishniy et al., NeurIPS 2021) with the authors' default configuration."""
    def __init__(self, n_num, cat_cards=(), d_model=192, n_layers=3, heads=8, attn_drop=0.2,
                 ffn_drop=0.1, res_drop=0.0, **_):
        super().__init__()
        self.n_cat = len(cat_cards)
        self.num_tok = NumericTokenizer(n_num, d_model, shared=False)
        if self.n_cat:
            self.cat_tok = CatEmbedding(cat_cards, d_model)
            self.cat_bias = nn.Parameter(torch.zeros(self.n_cat, d_model))
        self.cls = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.uniform_(self.cls, -1 / math.sqrt(d_model), 1 / math.sqrt(d_model))
        hidden = int(d_model * 4 / 3)
        self.blocks = nn.ModuleList([_FTBlock(d_model, heads, attn_drop, hidden, ffn_drop, res_drop, i == 0)
                                     for i in range(n_layers)])
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.ReLU(), nn.Linear(d_model, 2))

    def cat_embed(self, x_cat):
        return self.cat_tok(x_cat) + self.cat_bias if self.n_cat else None

    def forward_embedded(self, x_num, cat_e=None):
        t = self.num_tok(x_num)
        if cat_e is not None:
            t = torch.cat([t, cat_e], dim=1)
        t = torch.cat([t, self.cls.expand(t.size(0), -1, -1)], dim=1)   # CLS appended last
        for blk in self.blocks:
            t = blk(t)
        return self.out(t[:, -1])


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

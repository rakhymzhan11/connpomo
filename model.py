import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, dim=64, heads=4, ff_mult=2):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * ff_mult), nn.ReLU(),
                                 nn.Linear(dim * ff_mult, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, h, adj):
        B, N, _ = h.shape
        eye = torch.eye(N, device=h.device, dtype=adj.dtype).unsqueeze(0)
        adj_full = (adj + eye).clamp(max=1)
        qkv = self.qkv(h).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(adj_full.unsqueeze(1) == 0, float("-inf"))
        attn = torch.nan_to_num(attn.softmax(dim=-1), nan=0.0)
        out = self.proj((attn @ v).transpose(1, 2).reshape(B, N, self.dim))
        h = self.norm1(h + out)
        return self.norm2(h + self.ffn(h))


class PointerDecoder(nn.Module):
    def __init__(self, dim=64, heads=4, clip=10.0, n_dyn=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.clip = clip
        self.ctx_proj = nn.Linear(2 * dim, dim, bias=False)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.final_q = nn.Linear(dim, dim, bias=False)
        self.final_k = nn.Linear(dim, dim, bias=False)
        self.dyn_proj = nn.Sequential(nn.Linear(n_dyn, dim), nn.ReLU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.dyn_proj[-1].weight)
        nn.init.zeros_(self.dyn_proj[-1].bias)

    def forward(self, h, cur, valid, dyn=None):
        if dyn is not None:
            h = h + self.dyn_proj(dyn)
        B, N, D = h.shape
        H, hd = self.heads, D // self.heads
        cur_emb = h.gather(1, cur.view(B, 1, 1).expand(-1, 1, D)).squeeze(1)
        ctx = self.ctx_proj(torch.cat([cur_emb, h.mean(dim=1)], dim=-1))
        q = self.q_proj(ctx).view(B, 1, H, hd).transpose(1, 2)
        k = self.k_proj(h).view(B, N, H, hd).transpose(1, 2)
        v = self.v_proj(h).view(B, N, H, hd).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        attn = attn.masked_fill(~valid.view(B, 1, 1, N), float("-inf"))
        attn = torch.nan_to_num(attn.softmax(dim=-1), nan=0.0)
        ctx2 = (attn @ v).transpose(1, 2).reshape(B, D)
        logits = (self.final_q(ctx2).unsqueeze(1) @ self.final_k(h).transpose(-2, -1)).squeeze(1)
        logits = self.clip * torch.tanh(logits / math.sqrt(D))
        return logits.masked_fill(~valid, float("-inf"))


class ConnPOMO(nn.Module):
    def __init__(self, dim=64, layers=3, heads=4, use_dyn=True):
        super().__init__()
        self.use_dyn = use_dyn
        self.embed = nn.Linear(2, dim)
        self.layers = nn.ModuleList([GATLayer(dim, heads) for _ in range(layers)])
        self.decoder = PointerDecoder(dim, heads)

    def encode(self, xy, conn):
        h = self.embed(xy)
        for layer in self.layers:
            h = layer(h, conn)
        return h

    def forward(self, xy, conn, start, decode_type="sample", use_pi=False):
        B, N, _ = xy.shape
        dev, fdt = xy.device, conn.dtype
        h = self.encode(xy, conn)
        need_deg = use_pi or self.use_dyn
        conn_cat = torch.cat([conn, conn.transpose(1, 2)], dim=1) if need_deg else None
        unvis = torch.ones(B, N, dtype=torch.bool, device=dev)
        cur = start.clone()
        unvis.scatter_(1, cur.unsqueeze(-1), False)
        cols = [cur]
        log_prob = torch.zeros(B, device=dev)
        feasible = torch.ones(B, dtype=torch.bool, device=dev)
        for step in range(1, N):
            n_left = float(N - step)
            cur_pos = conn.gather(1, cur.view(B, 1, 1).expand(B, 1, N)).squeeze(1) > 0
            valid = unvis & cur_pos
            unvis_f = odeg = ideg = None
            if need_deg:
                unvis_f = unvis.to(fdt)
                deg = torch.bmm(conn_cat, unvis_f.unsqueeze(-1)).view(B, 2, N)
                odeg, ideg = deg[:, 0], deg[:, 1]
            if use_pi:
                out = valid if n_left <= 1.0 else (valid & (odeg >= 1))
                crit = unvis & (ideg == 0) & cur_pos
                n_crit = crit.sum(dim=1, keepdim=True)
                forced = out & crit
                pruned = torch.where((n_crit == 1) & forced.any(dim=1, keepdim=True), forced, out)
                valid = torch.where(pruned.any(dim=-1, keepdim=True), pruned, valid)
            row_ok = valid.any(dim=-1)
            feasible = feasible & row_ok
            valid = valid | (unvis & (~row_ok).unsqueeze(-1))
            dyn = None
            if self.use_dyn:
                dyn = torch.stack([odeg / n_left, ideg / n_left,
                                   (odeg == 1).to(fdt), (ideg == 1).to(fdt)], dim=-1)
            logits = self.decoder(h, cur, valid, dyn)
            logits = torch.nan_to_num(logits, nan=-1e9, neginf=-1e9, posinf=1e9)
            logp = F.log_softmax(logits, dim=-1)
            if decode_type == "greedy":
                nxt = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                bad = probs.sum(dim=-1) < 1e-6
                if unvis_f is None:
                    unvis_f = unvis.to(fdt)
                probs = torch.where(bad.unsqueeze(-1), unvis_f / n_left, probs)
                nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
            picked = logp.gather(1, nxt.unsqueeze(-1)).squeeze(-1)
            log_prob = log_prob + torch.where(row_ok, picked, torch.zeros_like(picked))
            cols.append(nxt)
            unvis.scatter_(1, nxt.unsqueeze(-1), False)
            cur = nxt
        return torch.stack(cols, dim=1), log_prob, feasible


def tour_length(tour, xy):
    coords = xy.gather(1, tour.unsqueeze(-1).expand(-1, -1, 2))
    return (coords[:, 1:] - coords[:, :-1]).pow(2).sum(-1).sqrt().sum(-1)


def arc_violations(tour, conn):
    N = tour.shape[1]
    a, b = tour[:, :-1], tour[:, 1:]
    ok = conn.gather(1, a.unsqueeze(-1).expand(-1, -1, N)).gather(2, b.unsqueeze(-1)).squeeze(-1)
    return (ok <= 0).float().sum(dim=-1)


def load_model(path, device="cpu"):
    model = ConnPOMO()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()

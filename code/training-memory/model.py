import torch 
import torch.nn as nn
from dataclasses import dataclass
from transformers import GPT2LMHeadModel
import torch.nn.functional as F

@dataclass(frozen = True)
class GPTconfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768





class MLP(nn.Module):
    def __init__(self, cfg: GPTconfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, cfg.n_embd*4 )
        self.activation = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(cfg.n_embd*4, cfg.n_embd)

    def forward(self, x):
        x_fc = self.c_fc(x)
        x_a = self.activation(x_fc)
        out = self.c_proj(x_a)
        return out




class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd// config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("mask", torch.tril(torch.ones(1,1,config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.size()
        c_x = self.c_attn(x)
        q,k,v = c_x.split(self.n_embd, dim = 2)
        q = q.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.n_head,self.head_dim).transpose(1,2)

       
       

        
        

        score = q @ k.transpose(-1, -2)/ (self.head_dim)**0.5

        masked_score = score.masked_fill(self.mask[:,:,:T, :T] == 0, float('-inf'))
        smax_score = torch.softmax(masked_score, dim = -1)
        final_score = smax_score @ v
        tr_score = final_score.transpose(1,2).contiguous()
        out_x = tr_score.view(B,T,self.n_embd)


        out = self.c_proj(out_x)
        return out

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        a = self.attn(self.ln_1(x))
        x = x + a
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.block_size = config.block_size
        self.transformer = nn.ModuleDict(dict(
                                wte = nn.Embedding(config.vocab_size, config.n_embd),
                                wpe = nn.Embedding(config.block_size, config.n_embd),
                                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                                ln_f = nn.LayerNorm(config.n_embd) ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
    def forward(self, idx, targets = None):
        B, T= idx.size()
       

        assert T <= self.block_size

        pos = torch.arange( T, device = idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss  = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss







def load_hf_weights(model):
    hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd_hf = hf.state_dict()
    sd = model.state_dict()
    kept_hf = [k for k in sd_hf if not k.endswith((".attn.bias", ".attn.masked_bias"))]
    kept =  [k for k in sd if not k.endswith(".attn.mask")]
    assert len(kept_hf) == len(kept), f"Mismatched Keys"

    transposed = (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight"
    )

    with torch.no_grad():
        for k in kept_hf:
            if k.endswith(transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, "transpose is not the same shape"
                sd[k].copy_(sd_hf[k].t()) 
            else:
                assert sd_hf[k].shape == sd[k].shape, "direct is not the same shape"
                sd[k].copy_(sd_hf[k]) 


    return model
    




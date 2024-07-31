import torch 
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict

device = 'cpu'

# multihead self-attention implementation
class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, use_enc_embed=False):
        super().__init__()
        self.encode = 0
        self.h = h
        self.W_O = nn.Parameter(torch.rand(h*d_v, d_model, device=device))
        self.attn = nn.ModuleList([ScaledDotProductAttention(d_model, d_k, d_v, use_enc_embed=use_enc_embed) for _ in range(h)])
        self.d_v = d_v

    def forward(self, x, enc_embed=None, use_mask=False, mask=None):
        attn_out = torch.zeros((x.shape[0], x.shape[1], self.d_v*self.h), device=device)

        for i in range(self.h):
            attn_out[:, :, i*self.d_v:(i+1)*self.d_v] = self.attn[i](x, enc_embed, use_mask=use_mask, mask=mask)

        return attn_out @ self.W_O
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, use_enc_embed=False):
        super().__init__()
        self.W_Q = nn.Parameter(torch.rand(d_model, d_k,  device=device))
        self.W_K = nn.Parameter(torch.rand(d_model, d_k,  device=device))
        self.W_V = nn.Parameter(torch.rand(d_model, d_v,  device=device))

        nn.init.xavier_normal_(self.W_Q)
        nn.init.xavier_normal_(self.W_K)
        nn.init.xavier_normal_(self.W_V)

        self.d_k = d_k
        self.use_enc_embed = use_enc_embed

    def forward(self, x, enc_embed, use_mask=False, mask=None):
        if self.use_enc_embed: 
            v = torch.matmul(enc_embed, self.W_V)
            k = torch.matmul(enc_embed, self.W_K)
        else:
            k = torch.matmul(x, self.W_K)
            v = torch.matmul(x, self.W_V)
        
        q = torch.matmul(x, self.W_Q)

        if use_mask:
            weights = F.softmax(mask + torch.matmul(q, torch.permute(k, (0, 2, 1)))/torch.sqrt(torch.tensor(self.d_k)), dim=-1)
            return weights @ v
        else:
            return F.softmax(torch.matmul(q, torch.permute(k, (0, 2, 1)))/torch.sqrt(torch.tensor(self.d_k)), dim=-1) @ v
        

# positional encoder implementation
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, input_len):
        super().__init__()
        self.input_len = input_len
        self.d_model = d_model
        
    def forward(self):
        pos = torch.zeros(self.d_model, self.input_len)

        for i in range(int(self.input_len/2)):
            pos[:, 2*i] = torch.sin(torch.range(start=0, end=self.d_model-1, step=1)/(10000 ** (2*i/self.input_len)))
            pos[:, 2*i+1] = torch.cos(torch.range(start=1, end=self.d_model, step=1)/(10000 ** (2*i/self.input_len)))
        return pos
    

# transformer encoder layer implementation
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k=4, d_v=4, n_heads=4, d_ff = 128):
        super().__init__()
        self.d_model = d_model
        self.mha = MultiHeadedAttention(d_model, d_k, d_v, n_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x, use_mask=False, mask=None):
        x = self.layer_norm1(x + self.mha(x, use_mask=use_mask, mask=mask))
        x = self.layer_norm2(x + self.linear2(self.relu(self.linear1(x))))
        return x
    

# transformer decoder layer implementation
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_k=4, d_v=4, h=4, d_ff = 128):
        super().__init__()
        self.d_model = d_model
        self.mmha = MultiHeadedAttention(d_model, d_k, d_v, h)
        self.mha = MultiHeadedAttention(d_model, d_k, d_v, h, use_enc_embed=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, target, memory, use_mask=False, mask=None):
        x = self.layer_norm1(target + self.mmha(target, use_mask=use_mask, mask=mask))
        x = self.layer_norm2(x + self.mha(x, memory, use_mask=False, mask=None))
        x = self.layer_norm3(x + self.linear2(self.relu(self.linear1(x))))
        return x
    
# put all the separate components together to make the transformer
class Transformer(nn.Module):
    def __init__(self, n_encoder = 6, n_decoder = 6, d_model = 16, n_heads = 2, seq_len = 10, dim_feedforward=2048, use_pos_enc=True, batch_first=True):
        super().__init__()
        self.encoder = nn.ModuleList([EncoderLayer(d_model, d_v=int(d_model/n_heads), d_k=int(d_model/n_heads), d_ff=dim_feedforward, n_heads=n_heads) for i in range(n_encoder)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, d_v=int(d_model/n_heads), d_k=int(d_model/n_heads), d_ff=dim_feedforward, h=n_heads) for i in range(n_decoder)])
        self.pos_encoder = PositionalEncoder(seq_len, d_model)
        self.seq_len = seq_len
        self.n_decoder = n_decoder
        self.n_encoder = n_encoder
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.use_pos_enc = use_pos_enc
        self.batch_first = batch_first

    def forward(self, input, target, use_enc_mask = False, use_dec_mask = False, enc_mask=1, dec_mask=1):
        if not self.batch_first:
            input = torch.permute(input, (1, 0, 2))
            target = torch.permute(target, (1, 0, 2))

        if self.use_pos_enc:
            memory = input + self.pos_encoder()
        else:
            memory = input

        for i in range(self.n_encoder):
            memory = self.encoder[i](memory, use_mask=use_enc_mask, mask=enc_mask)

        memory = self.layer_norm1(memory)
        
        if self.use_pos_enc:
            x = target + self.pos_encoder()
        else:
            x = target

        for i in range(self.n_decoder):
            x = self.decoder[i](x, memory, use_mask=use_dec_mask, mask=dec_mask)
        x = self.layer_norm2(x)

        if not self.batch_first:
            x = torch.permute(x, (1, 0, 2))

        return x

    def encode(self, input, use_enc_mask = False, enc_mask=1):
        if self.use_pos_enc:
            memory = input + self.pos_encoder()
        else:
            memory = input

        for i in range(self.n_encoder):
            memory = self.encoder[i](memory, use_mask=use_enc_mask, mask=enc_mask)

        memory = self.layer_norm1(memory)
        return memory 
    
    def decode(self, target, memory, use_dec_mask=True, dec_mask=1):        
        if self.use_pos_enc:
            x = target + self.pos_encoder()
        else:
            x = target

        for i in range(self.n_decoder):
            x = self.decoder[i](x, memory, use_mask=use_dec_mask, mask=dec_mask)
        x = self.layer_norm2(x)

        return x
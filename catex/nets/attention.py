import torch

'''
Relative positional multi-head attention implementation of Transformer-XL
from huggingface
https://github.com/huggingface/transformers/blob/1bdf42409c452a767ac8e2119bceb8f5c704c8f1/src/transformers/models/transfo_xl/modeling_transfo_xl.py
And part of the codes are modified according to TensorflowASR impl
https://github.com/TensorSpeech/TensorFlowASR/blob/5fbd6a89b93b703888662f5c47d05bae256e98b0/tensorflow_asr/models/layers/multihead_attention.py
'''

class RelPositionMultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        idim: int,
        n_head: int,
        d_head: int,
        dropatt: float = 0.
    ):

        super().__init__()

        # Define positional encoding in SuperNet level for efficient memory
        self.pe = None

        self.n_head = n_head
        self.d_model = idim
        self.d_head = d_head

        self.call_qkv = torch.nn.Linear(idim, 3 * n_head * d_head, bias=False)

        self.dropoutatt = torch.nn.Dropout(dropatt)
        self.linearout = torch.nn.Linear(n_head * d_head, idim, bias=False)

        self.scale = 1 / (d_head ** 0.5)

        self.r_r_bias = torch.nn.Parameter(
            torch.FloatTensor(self.n_head, self.d_head))
        self.r_w_bias = torch.nn.Parameter(
            torch.FloatTensor(self.n_head, self.d_head))

        torch.nn.init.xavier_uniform_(self.r_r_bias)
        torch.nn.init.xavier_uniform_(self.r_w_bias)

        self.linearpos = torch.nn.Linear(
            self.d_model, self.n_head * self.d_head, bias=False)

    @staticmethod
    def _rel_shift(x: torch.Tensor):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = x.new_zeros(zero_pad_shape)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)

        x = x_padded[1:].view_as(x)

        return x

    def forward(self, word_embedding: torch.Tensor, lens: torch.Tensor, mems=None):
        """Relative positional embedding multihead attention

        B: batch size
        T: (maximum) sequence length
        E: word embedding size (also the postional embedding size)
        C: memory context length, K=C+T

        Args:
            word_embedding (torch.Tensor): input segment (word) embedding, size [B, T, E]
            lens (torch.Tensor): input sequence lengths, used to generate attention mask, size [B]
            mems (torch.Tensor, None): if not None, memory tensors, size [B, C, E]
        
        Returns:
            torch.Tensor: attention out

        """
        '''
        word_embedding: [B, T, E] -> [T, B, E]
        mems: [B, C, E] -> [C, B, E] (if not None)
        '''
        word_embedding = word_embedding.transpose(0, 1).contiguous()
        len_seq, batchsize = word_embedding.size()[:2]
        len_k = len_seq
        if all(le == lens[0] for le in lens):
            # set mask to None
            attn_mask = None
            if mems is not None:
                mems = mems.transpose(0, 1).contiguous()
                len_k += mems.size(0)
        else:
            if mems is not None:
                mems = mems.transpose(0, 1).contiguous()
                lens = lens + mems.size(0)
                len_k += mems.size(0)
            # attn_mask: [K, B]_{1,0}
            attn_mask = torch.arange(len_k, device=word_embedding.device)[
                :, None] >= lens[None, :].to(word_embedding.device)

        # pos_embedding: [K, E]
        pos_enc = self.pe(torch.empty((1, len_k)))

        if mems is not None:
            # embed_with_mem: cat([C, B, E], [T, B, E]) -> [C+T, B, E] = [K, B, E]
            embed_with_mem = torch.cat([mems, word_embedding], dim=0)
            '''
            W_heads: f([K, B, E]) -> [K, B, 3*H*D]
                H: n_heads
                D: d_heads
            '''
            W_heads = self.call_qkv(embed_with_mem)

            # R_head_k: f([K, E]) -> [K, H*D]
            R_head_k = self.linearpos(pos_enc)

            # W_head_q/W_head_k/W_head_v: f([K, B, 3HD]) -> [K, B, HD]
            W_head_q, W_head_k, W_head_v = torch.chunk(W_heads, 3, dim=-1)

            # W_head_q: f([K, B, HD]) -> [T, B, HD]
            W_head_q = W_head_q[-len_seq:]

        else:
            # W_heads: f([T, B, E]) -> [T, B, 3*H*D]
            W_heads = self.call_qkv(word_embedding)

            # R_head_k: f([T, E]) -> [T, H*D]
            R_head_k = self.linearpos(pos_enc)

            # W_head_q/W_head_k/W_head_v: f([T, B, 3HD]) -> [T, B, HD]
            W_head_q, W_head_k, W_head_v = torch.chunk(W_heads, 3, dim=-1)

        # W_head_q: [T, B, HD] -> [T, B, H, D]
        W_head_q = W_head_q.view(len_seq, batchsize, self.n_head, self.d_head)

        # W_head_k/W_head_v: [K, B, HD] -> [K, B, H, D]
        W_head_k = W_head_k.view(len_k, batchsize, self.n_head, self.d_head)
        W_head_v = W_head_v.view_as(W_head_k)

        # R_head_k: [K, H*D] -> [K, H, D]
        R_head_k = R_head_k.view(len_k, self.n_head, self.d_head)

        # compute attention score

        # RW_head_q/RR_head_q: [T, B, H, D]
        RW_head_q = W_head_q + self.r_w_bias
        RR_head_q = W_head_q + self.r_r_bias

        # FIXME: torch.einsum is not optimized, which might cause slow computation
        # AC: f([T, B, H, D], [K, B, H, D]) -> [T, K, B, H]
        AC = torch.einsum("ibnd,jbnd->ijbn", (RW_head_q, W_head_k))

        # BD: f([T, B, H, D], [K, H, D]) -> [T, K, B, H]
        BD = torch.einsum("ibnd,jnd->ijbn", (RR_head_q, R_head_k))

        # BD: [T, K, B, H] -> [T, K, B, H]
        BD = self._rel_shift(BD)

        # attn_score: [T, K, B, H]
        attn_score = AC + BD
        attn_score *= self.scale

        # compute attention probability
        if attn_mask is not None:
            # use in-plcae fill
            # FIXME(wangqq): provides autocast compatibility to prevent overflows
            attn_score = attn_score.masked_fill_(
                attn_mask[None, :, :, None], -1e4)

        # attn_prob: f([T, K, B, H]) -> [T, K, B, H]
        attn_prob = torch.softmax(attn_score, dim=1)
        attn_prob = self.dropoutatt(attn_prob)

        # compute attention vector

        # attn_vec: f([T, K, B, H], [K, B, H, D]) -> [T, B, H, D]
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, W_head_v))

        # attn_vec: [T, B, H, D] -> [T, B, HD]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1),
            self.n_head * self.d_head)

        # attn_out: f([T, B, HD]) -> [T, B, E]
        attn_out = self.linearout(attn_vec)

        # attn_out: [T, B, E] -> [B, T, E]
        attn_out = attn_out.transpose(0, 1).contiguous()

        return attn_out


class MHSAModule(torch.nn.Module):
    def __init__(self, idim, d_head: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        self.ln = torch.nn.LayerNorm(idim)
        self.mha = RelPositionMultiHeadAttention(
            idim, num_heads, d_head)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lens: torch.Tensor, mems=None):
        x_norm = self.ln(x)
        attn_out = self.mha(x_norm, lens, mems)
        attn_out = self.dropout(attn_out)
        return x + attn_out, lens


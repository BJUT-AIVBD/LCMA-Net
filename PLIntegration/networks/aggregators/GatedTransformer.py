#!usr/bin/env python
# -*- coding:utf-8 _*-
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter


def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    device = tensor.get_device()
    buf_name = f'range_buf_{device}'
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, tensor.new())
    setattr(make_positions, buf_name, getattr(make_positions, buf_name).type_as(tensor))
    if getattr(make_positions, buf_name).numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=getattr(make_positions, buf_name))
    mask = tensor.ne(padding_idx)
    positions = getattr(make_positions, buf_name)[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    new_tensor = tensor.clone()
    return new_tensor.masked_scatter_(mask, positions[mask]).long()


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1 + abs(dim2 - dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, add_mask=None, mul_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)
        orig_len = src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        # project (batch_size, seq_len, seq_len)
        add_mask = add_mask.unsqueeze(1).expand(add_mask.size(0), self.num_heads, orig_len, src_len).contiguous().view(
            -1, src_len, src_len)
        mul_mask = mul_mask.unsqueeze(1).expand(mul_mask.size(0), self.num_heads, orig_len, src_len).contiguous().view(
            -1, src_len, src_len)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if add_mask is not None:
            try:
                attn_weights = attn_weights + add_mask
            except:
                print(attn_weights.shape)
                print(add_mask.shape)
                assert False

        # if attention from language to other modal, then maybe align to a void space (after the end of a sentence)
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)

        # attn_weights_mask = attn_weights.isnan()
        # attn_weights = torch.masked_fill(attn_weights, attn_weights_mask, 0.0)

        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)
        attn_weights = attn_weights * mul_mask

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = dict()  # device --> actual weight; due to nn.DataParallel :-(
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb_c1 = math.log(10000) / (half_dim - 1)

        emb_c2 = torch.arange(embedding_dim, dtype=torch.int32)

        emb = torch.exp((emb_c2 // 2).to(torch.float) * -emb_c1)  # (embedding_dim,)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(
            0)  # (num_emb, embedding_dim)

        # assign sinusoidal positional embedding to correct positions
        emb[:, emb_c2 % 2 == 0] = torch.sin(emb[:, emb_c2 % 2 == 0])
        emb[:, emb_c2 % 2 == 1] = torch.cos(emb[:, emb_c2 % 2 == 1])

        # emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1) # (num_emb, half_dim*2)

        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.get_device()
        if device not in self.weights or max_pos > self.weights[device].size(0):
            # recompute/expand embeddings if needed
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights[device] = self.weights[device].type_as(self._float_tensor)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        return self.weights[device].index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        # Memory and Compound control
        self.mem_proj = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.att_proj = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid()
        )

        # Dense Layer
        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)  # The "Add & Norm" part in the paper
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None, ctr_vec=None, lengths=None, mode='l2o'):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
            ctc_vec (Tensor): The control vector generated from DIV encoder
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)

        def get_mask(batch_size, len1, len2, lengths=None, lang_proj=False):
            """l2o means x is other modal
            Returns:
                mask (Tensor): An attention mask of size (len1, len2)
            """
            assert len1 == len2

            bool_mask1 = torch.cuda.BoolTensor(batch_size, len1, len2)
            bool_mask2 = torch.cuda.BoolTensor(batch_size, len1, len2)
            for i, j in enumerate(lengths):
                bool_mask2[i, :, :] = False
                if j < len1:
                    # bool_mask[i,j:,j:] is TOTALLY WRONG
                    bool_mask1[i, j:, :] = True
                    bool_mask1[i, :, j:] = True
                    bool_mask2[i, :,
                    j:] = True  # only add minus infinity to the region of exceeded lengths in valid inputs
                bool_mask1[i, :j, :j] = False

            add_mask = torch.masked_fill(torch.zeros(bool_mask2.size()), bool_mask2, float('-inf'))
            mul_mask = torch.masked_fill(torch.ones(bool_mask1.size()), bool_mask1, 0.0)
            add_mask.detach_()
            mul_mask.detach_()

            # if projection to lengths, then some positions are projected to invalid space
            return add_mask, mul_mask

        # add heterogeneous mask, l2o means attention projects to other modal
        if mode == 'l2o':
            add_mask, mul_mask = get_mask(x.size(1), x.size(0), x_v.size(0), lengths=lengths, lang_proj=False)
        elif mode == 'o2l':
            add_mask, mul_mask = get_mask(x.size(1), x_v.size(0), x.size(0), lengths=lengths, lang_proj=True)

        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, add_mask=add_mask, mul_mask=mul_mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, add_mask=add_mask, mul_mask=mul_mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)

        mem_gate = self.mem_proj(ctr_vec)
        fuse_gate = self.att_proj(ctr_vec)

        if ctr_vec is not None:
            x = x * fuse_gate + residual * mem_gate
        else:
            x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class DIVEncoder(nn.Module):
    """Construct a domain-invariant encoder for all modalities. Forward and return domain-invariant
    encodings for these modality with similarity and reconstruction (optional) loss.
    Args:
        in_size (int): hidden size of input vector(s), of which is a representation for each modality
        out_size (int): hidden_size
    """

    def __init__(self, in_size, out_size, prj_type='linear', use_disc=False,
                 rnn_type=None, rdc_type=None, p_l=0.0, p_o=0.0):
        super(DIVEncoder, self).__init__()
        self.prj_type = prj_type
        self.reduce = rdc_type
        self.use_disc = use_disc

        self.in_size = in_size
        self.out_size = out_size

        if prj_type == 'linear':
            self.encode_l = nn.Linear(in_size, out_size)
            self.encode_o = nn.Linear(in_size, out_size)

        elif prj_type == 'rnn':
            self.rnn_type = rnn_type.upper()
            rnn = getattr(nn, self.rnn_type)

            self.encode_l = rnn(input_size=in_size,
                                hidden_size=out_size,
                                num_layers=1,
                                dropout=p_l,
                                bidirectional=True)
            self.encode_o = rnn(input_size=in_size,
                                hidden_size=out_size,
                                num_layers=1,
                                dropout=p_o,
                                bidirectional=True)

        if use_disc:
            self.discriminator = nn.Sequential(
                nn.Linear(out_size, 4 * out_size),
                nn.ReLU(),
                nn.Linear(4 * out_size, 1),
                nn.Sigmoid()
            )

        self.dropout_l = nn.Dropout(p_l)
        self.dropout_o = nn.Dropout(p_o)

    def _masked_avg_pool(self, lengths, mask, *inputs):
        """Perform a masked average pooling operation
        Args:
            lengths (Tensor): A tensor represents the lengths of input sequence with size (batch_size,)
            mask (Tensor):
            inputs (Tuple[Tensor]): Hidden representations of input sequence with shape of (max_seq_len, batch_size, embedding)
        """
        res = []

        # bert mask only has 2 dimensions
        if len(mask.size()) == 2:
            mask = mask.unsqueeze(-1)

        for t in inputs:
            masked_mul = t.permute(1, 0, 2) * mask  # batch_size, seq_len, emb_size
            res.append(masked_mul.sum(1) / lengths.unsqueeze(-1))  # batch_size, emb_size
        return res

    def _forward_rnn(self, rnn, input, lengths):
        packed_sequence = pack_padded_sequence(input, lengths.cpu())
        packed_h, h_out = rnn(packed_sequence)
        padded_h, _ = pad_packed_sequence(packed_h)
        return padded_h, h_out

    def forward(self, input_l, input_o, lengths, mask):
        if self.prj_type == 'linear':
            if self.reduce == 'avg':
                avg_l, avg_o = self._masked_avg_pool(lengths, mask, input_l, input_o)
            elif self.reduce is None:
                avg_l, avg_o = input_l, input_o
            else:
                raise ValueError("Reduce method can be either average or none if projection type is linear")
            enc_l = self.encode_l(avg_l)
            enc_o = self.encode_o(avg_o)

        elif self.prj_type == 'rnn':
            out_l, h_l = self._forward_rnn(self.encode_l, input_l, lengths)
            out_o, h_o = self._forward_rnn(self.encode_o, input_o, lengths)
            if self.reduce == 'last':
                h_l_last = h_l[0] if isinstance(h_l, tuple) else h_l
                h_o_last = h_o[0] if isinstance(h_o, tuple) else h_o
                enc_l = (h_l_last[0] + h_l_last[1]) / 2
                enc_o = (h_o_last[0] + h_o_last[1]) / 2
            elif self.reduce == 'avg':
                enc_l, enc_o = self._masked_avg_pool(lengths, mask, out_l, out_o)
                enc_l = (enc_l[:, :enc_l.size(1) // 2] + enc_l[:, enc_l.size(1) // 2:]) / 2
                enc_o = (enc_o[:, :enc_o.size(1) // 2] + enc_o[:, enc_o.size(1) // 2:]) / 2
            else:
                raise ValueError("Reduce method can be either last or average if projection type is linear")

        enc_l, enc_o = self.dropout_l(enc_l), self.dropout_o(enc_o)

        if self.use_disc:
            # generate discriminator output together with its labels
            disc_out = self.discriminator(torch.cat((enc_l, enc_o), dim=0)).squeeze()  # (2 * batch_size, 1)
            batch_size = enc_l.size(0)
            disc_labels = torch.cat([torch.Tensor([0]).expand(size=(batch_size,)), \
                                     torch.Tensor([1]).expand(size=(batch_size,))], dim=0).squeeze()

        return enc_l, enc_o, disc_out, disc_labels


class GatedTransformer(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers=5, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, div_dropout=0.0, attn_mask=False, use_disc=True):
        super().__init__()
        self.dropout = embed_dropout  # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)

        self.attn_mask = attn_mask

        # a pair of transformers plus a domain-invariant encoder
        self.l2other_layers = nn.ModuleList([])
        self.other2l_layers = nn.ModuleList([])
        self.div_encoders = nn.ModuleList([])

        for layer in range(layers):
            l2other_new = TransformerEncoderLayer(embed_dim,
                                                  num_heads=num_heads,
                                                  attn_dropout=attn_dropout,
                                                  relu_dropout=relu_dropout,
                                                  res_dropout=res_dropout,
                                                  attn_mask=attn_mask)
            other2l_new = TransformerEncoderLayer(embed_dim,
                                                  num_heads=num_heads,
                                                  attn_dropout=attn_dropout,
                                                  relu_dropout=relu_dropout,
                                                  res_dropout=res_dropout,
                                                  attn_mask=attn_mask)

            if layer == 0:
                new_div_layer = DIVEncoder(embed_dim, embed_dim, prj_type='linear', use_disc=use_disc)
            else:
                # TODO: Change dropout rate here
                # new_div_layer = DIVEncoder(embed_dim, embed_dim, prj_type='rnn', rnn_type='lstm', rdc_type='avg', use_disc=use_disc)
                new_div_layer = DIVEncoder(embed_dim, embed_dim, prj_type='rnn', rnn_type='gru', rdc_type='avg',
                                           use_disc=use_disc)

            self.l2other_layers.append(l2other_new)
            self.other2l_layers.append(other2l_new)
            self.div_encoders.append(new_div_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward_transformer(self, x_in, x_in_k=None, x_in_v=None):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x

    def forward(self, seq_l, seq_other, h_l, h_other, lengths=None, mask=None):
        """Forward 2 input modals thorugh the DIVencoder and Trnasformer
        Args:
            input_l (FloatTensor): Representative tensor of the language modal
            input_other (FloatTensor): Representative tensor of the other modal
        """
        sim_loss_total = 0.0
        recon_loss_total = 0.0

        assert lengths is not None or mask is not None

        if mask is None:
            batch_size = lengths.size(0)
            mask = torch.arange(lengths.max()).repeat(batch_size, 1) < lengths.unsqueeze(-1)
            mask = mask.unsqueeze(-1).to(torch.float)
        elif lengths is None:
            lengths = mask.squeeze().sum(1)

        # output all shared encoding to train the discriminator
        # enc_l_all = []
        # enc_other_all = []

        # outputs of all discriminators in every layer
        disc_out_all = []
        disc_labels_all = []

        input_l, input_other = seq_l, seq_other

        # add residual connection
        # resl_all = []
        # resother_all = []

        for layer_i, (div_encoder, trans_l2other, trans_other2l) in enumerate(
                zip(self.div_encoders, self.l2other_layers,
                    self.other2l_layers)):
            enc_l, enc_other, disc_out, disc_labels = div_encoder(h_l, h_other, lengths, mask)  # batch_size, emb_size

            ctr_vec = torch.cat([enc_l, enc_other], dim=-1)  # seq_len x bs x (2 * emb_size)

            disc_out_all.append(disc_out)
            disc_labels_all.append(disc_labels)

            # project language to other modals
            l2other = trans_other2l(input_other, x_k=input_l, x_v=input_l, ctr_vec=ctr_vec, lengths=lengths, mode='l2o')

            # project other modals to language
            other2l = trans_l2other(input_l, x_k=input_other, x_v=input_other, ctr_vec=ctr_vec, lengths=lengths,
                                    mode='o2l')

            # resl_all.append(other2l)
            # resother_all.append(l2other)

            # if layer_i > 0:
            #     for res_l in resl_all[:-1]:
            #         other2l += res_l
            #     for res_other in resother_all[:-1]:
            #         l2other += res_other

            input_l, input_other = other2l, l2other
            h_l, h_other = other2l, l2other

        disc_out_all = torch.cat(disc_out_all)
        disc_labels_all = torch.cat(disc_labels_all)

        return other2l, l2other, disc_out_all, disc_labels_all  # placeholder for DIV output

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


if __name__ == '__main__':
    encoder = GatedTransformer(300, 4, 2)
    x = torch.tensor(torch.rand(20, 2, 300))
    print(encoder(x).shape)

# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import EmbLoss, BPRLoss
import math
import numpy as np
import common.utils.tool as tool
import torch.fft as fft
from kan import KANLinear, NaiveFourierKANLayer
# from info_nce import InfoNCE, info_nce
from infonce import InfoNCE

# class GNN(nn.Module):
    # r"""Graph neural networks are well-suited for session-based recommendation,
    # because it can automatically extract features of session graphs with considerations of rich node connections.
    # """

    # def __init__(self, embedding_size, step=1):
    #     super(GNN, self).__init__()
    #     self.step = step
    #     self.embedding_size = embedding_size
    #     self.input_size = embedding_size * 2
    #     self.gate_size = embedding_size * 3
    #     self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
    #     self.w_hh = Parameter(torch.Tensor(self.gate_size, self.embedding_size))
    #     self.b_ih = Parameter(torch.Tensor(self.gate_size))
    #     self.b_hh = Parameter(torch.Tensor(self.gate_size))
    #     self.b_iah = Parameter(torch.Tensor(self.embedding_size))
    #     self.b_ioh = Parameter(torch.Tensor(self.embedding_size))

    #     self.linear_edge_in = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
    #     self.linear_edge_out = nn.Linear(self.embedding_size * (self.step + 1), self.embedding_size, bias=True)

    #     self.kan = KANLinear(self.embedding_size, self.embedding_size)

    #     self.linear = nn.Linear(self.embedding_size, self.embedding_size, bias=True)

    #     # parameters initialization
    #     self._reset_parameters()

    # def _reset_parameters(self):
    #     # Inside the function, a standard deviation (stdv) is first calculated, which is based on the inverse square of embedding_size.
    #     # Then, all the parameters of the model are iterated over, initializing the data (i.e., the weights) for each parameter to a value drawn from the uniform distribution, ranging from -stdv to stdv.
    #     # This initialization helps to set the parameters to small random values at the beginning of training for better model training and optimization.
    #     stdv = 1.0 / math.sqrt(self.embedding_size)
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stdv, stdv)


    # def GNNCell(self, A, hidden):
    #     r"""Obtain latent vectors of nodes via graph neural networks.

    #     Args:
    #         A(torch.FloatTensor):The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]

    #         hidden(torch.FloatTensor):The item node embedding matrix, shape of
    #             [batch_size, max_session_len, embedding_size]

    #     Returns:
    #         torch.FloatTensor: Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]

    #     """

    #     input_in = torch.matmul(A[:, :, :A.size(1)], hidden)
    #     input_out = torch.matmul(A[:, :, A.size(1):2 * A.size(1)], input_in)
    #     # # [batch_size, max_session_len, embedding_size * 2]
    #     # inputs = torch.cat([input_in, input_out], 2)

    #     # inputs = self.linear(inputs)
    #     return input_out

    # def forward(self, A, hidden):
    #     x = self.linear_edge_in(hidden)
    #     # embs_list = [x]
    #     for i in range(self.step):
    #         x = torch.matmul(A[:, :, :A.size(1)], x)
    #         x = torch.matmul(A[:, :, A.size(1):2 * A.size(1)], x)
    #         x = self.kan(x)
        

    #     # lightgcn_embeddings = torch.stack(embs_list, dim=-1)
    #     # lightgcn_embeddings = torch.sum(lightgcn_embeddings, dim=-1)
    #     # lightgcn_embeddings = self.linear_edge_out(lightgcn_embeddings)
    #     lightgcn_embeddings = self.linear(x)
    #     return lightgcn_embeddings


class GNN(nn.Module):
    r"""Graph neural networks are well-suited for session-based recommendation,
    because it can automatically extract features of session graphs with considerations of rich node connections.
    """

    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.embedding_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.embedding_size))
        self.b_ioh = Parameter(torch.Tensor(self.embedding_size))

        self.linear_edge_in = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_edge_out = nn.Linear(self.embedding_size, self.embedding_size, bias=True)

        # parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        # Inside the function, a standard deviation (stdv) is first calculated, which is based on the inverse square of embedding_size.
        # Then, all the parameters of the model are iterated over, initializing the data (i.e., the weights) for each parameter to a value drawn from the uniform distribution, ranging from -stdv to stdv.
        # This initialization helps to set the parameters to small random values at the beginning of training for better model training and optimization.
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def GNNCell(self, A, hidden):
        r"""Obtain latent vectors of nodes via graph neural networks.

        Args:
            A(torch.FloatTensor):The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]

            hidden(torch.FloatTensor):The item node embedding matrix, shape of
                [batch_size, max_session_len, embedding_size]

        Returns:
            torch.FloatTensor: Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]

        """

        input_in = torch.matmul(A[:, :, :A.size(1)], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.size(1):2 * A.size(1)], self.linear_edge_out(hidden)) + self.b_ioh
        # [batch_size, max_session_len, embedding_size * 2]
        inputs = torch.cat([input_in, input_out], 2)

        # gi.size equals to gh.size, shape of [batch_size, max_session_len, embedding_size * 3]
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        # (batch_size, max_session_len, embedding_size)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
            if i == self.step - 1:
                final_hidden = hidden
        return hidden, final_hidden


class BandedFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, band, num_bands, length=201):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  # zero indexed
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (
            self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs

        # case: from other frequencies
        self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))
        self.reset_parameters()

    def forward(self, input):
        # input - b t d
        b, t, _ = input.shape
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = self._forward(input_fft)
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _forward(self, input):
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
        return output + self.bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)



class MyModel(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    # 1
    def __init__(self, config, dataset):
        super(MyModel, self).__init__(config, dataset)

        # load parameters info
        ## tf
        # Get the values of "n_layers" and "n_heads" from "config", representing the number of layers and attention heads in the model, respectively
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        # Get "hidden_size", which represents the dimensions of the hidden state in the model, also the same as the dimensions of the embedded vector, and "inner_size", which represents the internal dimensions in the feedforward neural network
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        # Obtain hyperparameters related to the model's hidden state, attention, activation function, and layer normalization
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        # Gets hyperparameters for model initialization range and loss type
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        # Gets the batch size at training time
        self.batch_size = config['train_batch_size']

        self.tau = config['tau']
        self.sim = config['sim']
        self.lmd = config['lmd']
        self.lmd_tf = config['lmd_tf']

        self.aug_item_field_1 = config['AUG_ITEM_LIST_1']
        self.aug_item_len_field_1 = config['AUG_ITEM_LENGTH_LIST_1']
        self.aug_item_field_2 = config['AUG_ITEM_LIST_2']
        self.aug_item_len_field_2 = config['AUG_ITEM_LENGTH_LIST_2']

        self.config = config

        # define layers and loss
        # It is a simple lookup table that stores embedded vectors of fixed size dictionaries,
        # means that, given a number, the embedding layer can return the embedding vector corresponding to the number, the embedding vector reflects the semantic relationship between the symbols represented by each number
        # The input is a numbered list, and the output is a list of corresponding symbolic embedding vectors
        # This code creates an embedding layer that maps a discrete integer number to a continuous embedding vector. The specific explanation is as follows:
        # nn.Embedding: This is a class in PyTorch that is used to create the embedding layer.
        # self.n_items: This is the number of items to be represented by the embedding layer, which indicates how many different embedding vectors to be created in the embedding layer, usually corresponding to the total number of items or the number of categories.
        # self.hidden_size: This is the dimension of the embedding vector, which can also be called the dimension of the embedding space. It specifies the length of each embedded vector.
        # padding_idx=0: This is an optional parameter that specifies a special number for "fill".
        # In sequence data, you can use a number (usually 0) as a fill item, and the embedding layer creates an all-zero embedding vector for that number.
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.fft_layer = BandedFourierLayer(self.hidden_size, self.hidden_size, 0, 1, length=self.max_seq_length)

        self.kan = KANLinear(self.hidden_size, self.hidden_size)

        # self.fft_layer = NaiveFourierKANLayer(self.hidden_size, self.hidden_size, 3)
        
        # LayerNorm,BatchNorm
        # These two lines of code initialize a LayerNorm layer and a Dropout layer for standardization and random deactivation operations, respectively
        # LayerNorm is used to standardize the input
        # The dropout is used to randomly zero neurons with a certain probability during training to reduce overfitting
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        # Initialized a loss function "loss_fct" based on the set loss type
        # 'BPR', "BPRLoss" class
        # 'CE', then use the cross entropy loss in PyTorch "nn.CrossEntropyLoss()"
        # If the loss type is not one of these two, an unrealized error is thrown
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # dim: 0: cos similarity is calculated between column vectors of corresponding columns
        # 1(default): calculates the similarity between row vectors
        # -1: is to calculate the similarity between row vectors
        self.simf1 = nn.CosineSimilarity(dim=1)

        self.nce_fct = nn.CrossEntropyLoss()

        self.InfoNCE = InfoNCE(temperature=self.tau)

        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)

        ## gnn
        self.embedding_size = config['embedding_size']
        self.step = config['step']
        # define layers and loss
        self.gnn = GNN(self.embedding_size, self.step)
        self.linear_1 = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_2 = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_3 = nn.Linear(self.embedding_size, 1, bias=False)
        self.linear_out = nn.Linear(self.embedding_size * 2, self.embedding_size, bias=True)

        ## cl
        self.count_dict = {'idx': 0}
        # Initializes a cross entropy loss "loss_fct_ce" to calculate the cross entropy loss
        self.loss_fct_ce = nn.CrossEntropyLoss()


        # parameters initialization
        self.apply(self._init_weights)


    def _get_slice(self, item_seq):
        # Mask matrix, shape of [batch_size, max_session_len]
        mask = item_seq.gt(0)
        items, n_node, A, alias_inputs = [], [], [], []
        max_n_node = item_seq.size(1)
        item_seq = item_seq.cpu().numpy()
        for u_input in item_seq:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))

            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break

                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1

            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)

            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        # The relative coordinates of the item node, shape of [batch_size, max_session_len]
        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        # The connecting matrix, shape of [batch_size, max_session_len, 2 * max_session_len]
        A = torch.FloatTensor(np.array(A)).to(self.device)
        # The unique item nodes, shape of [batch_size, max_session_len]
        items = torch.LongTensor(items).to(self.device)

        return alias_inputs, A, items, mask

    # 2
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # 4
    def forward(self, item_seq, item_seq_len, generate_layer=None):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)   
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output_t = trm_output[-1]
        seq_output_t = self.gather_indexes(output_t, item_seq_len - 1)
        # seq_output_t = self.kan(seq_output_t)

        if generate_layer is not None:
            
            # alias_inputs, A, items, _ = self._get_slice(item_seq)

            # alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.embedding_size)

            # # noise = tool.gaussian_noise(input_emb, self.config['noise_base'])
            # # mask1 = item_seq.gt(0).unsqueeze(dim=2)
            # # noise1 = noise * mask1
            # # input_emb_noise = input_emb + noise1
            # hidden = self.item_embedding(items)

            # output_aug = self.gnn(A, hidden)
            
            # seq_output_aug = torch.gather(output_aug, dim=1, index=alias_inputs)
            # # fetch the last hidden state of last timestamp
            # seq_output_aug = self.gather_indexes(seq_output_aug, item_seq_len - 1)

            output_fft = generate_layer(input_emb)
            trm_output_f = self.trm_encoder(output_fft, extended_attention_mask, output_all_encoded_layers=True)

            output_f = trm_output_f[-1]
            # output_f = self.kan(output_f)
            seq_output_f = self.gather_indexes(output_f, item_seq_len - 1)
            # seq_output_f = self.linear_1(seq_output_f)
            seq_output_f = self.kan(seq_output_f)
            # seq_output_f = self.linear_2(seq_output_f)
            # return seq_output_t, seq_output_f, seq_output_aug
            return seq_output_t, seq_output_f
        
        return seq_output_t  # [B H]


    def forward_gcn(self, item_seq, item_seq_len):
        alias_inputs, A, items, _ = self._get_slice(item_seq)

        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.embedding_size)

        hidden = self.item_embedding(items)
        # hidden_t = self.linear_1(hidden)
        hidden, final_hidden = self.gnn(A, hidden)
        
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)
        # fetch the last hidden state of last timestamp
        seq_output = self.gather_indexes(seq_hidden, item_seq_len - 1)

        seq_final_hidden = torch.gather(final_hidden, dim=1, index=alias_inputs)
        # fetch the last hidden state of last timestamp
        seq_final_output = self.gather_indexes(seq_final_hidden, item_seq_len - 1)
        # seq_output_t = self.kan(seq_output_t)
        
        return seq_output, seq_final_output  # [B H]

    # 3
    def calculate_loss(self, interaction):
        # cal loss
        loss = torch.tensor(0.0).to(self.device)
        
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        aug_item_seq_1 = interaction[self.aug_item_field_1]
        aug_len_1 = interaction[self.aug_item_len_field_1]

        loss += self.tff_loss(interaction, item_seq, item_seq_len, aug_item_seq_1, aug_len_1, self.config['t_weight'], self.forward) * 0.5
        loss += self.gcl_loss(interaction, item_seq, item_seq_len, self.config['g_weight'], self.forward_gcn) * 0.5
        return loss
    

    def tff_loss(self, interaction, item_seq, item_seq_len, aug_item_seq, aug_item_seq_len, cff, forward):
        loss = torch.tensor(0.0).to(self.device)

        nce_loss_t = 0.0
        nce_loss_t_f = 0.0
        nce_loss_amp = 0.0
        nce_loss_phase = 0.0

        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]


        seq_output_t, seq_output_f = forward(item_seq, item_seq_len, self.fft_layer)
        seq_output_t_aug = forward(aug_item_seq, aug_item_seq_len)

        loss += cff * self.rec_loss(interaction, seq_output_t)

        # Time ssl
        nce_loss_t = self.InfoNCE(seq_output_t_aug, seq_output_t)


         # Time-Frequency ss
        # nce_loss_t_f = self.InfoNCE(seq_output_f_aug, seq_output_t_aug)

        # Frequency ssl
        f_aug_seq_output_amp, f_aug_seq_output_phase = self.my_fft(seq_output_t)
        f_seq_output_amp, f_seq_output_phase = self.my_fft(seq_output_f)

        nce_loss_amp = self.InfoNCE(f_aug_seq_output_amp, f_seq_output_amp)
        nce_loss_phase = self.InfoNCE(f_aug_seq_output_phase, f_seq_output_phase)

        loss += self.lmd/2 * (self.lmd_tf * nce_loss_t + (1 - self.lmd_tf)/2 * (nce_loss_t_f + nce_loss_phase + nce_loss_amp))
        return loss
    

    def gcl_loss(self, interaction,item_seq, item_seq_len, cff, forward):
        loss = torch.tensor(0.0).to(self.device)
        
        seq_output, seq_output_aug = forward(item_seq, item_seq_len)

        loss += cff * self.rec_loss(interaction, seq_output)
        loss += self.InfoNCE(seq_output_aug, seq_output)

        return loss

    # 6
    def predict(self, interaction):
        # print("================= predict(self, interaction) ===================")
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        # print("================= full_sort_predict(self, interaction) ===================")
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

    def fast_predict(self, interaction):
        # print('fast_predict: ***********')
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction["item_id_with_negs"]
        seq_output = self.forward(item_seq, item_seq_len)

        test_item_emb = self.item_embedding(test_item)  # [B, num, H]
        scores = torch.matmul(seq_output.unsqueeze(
            1), test_item_emb.transpose(1, 2)).squeeze()
        return scores
    
    def my_fft(self, seq):
        f = torch.fft.rfft(seq, dim=1)
        amp = torch.absolute(f)
        phase = torch.angle(f)
        return amp, phase

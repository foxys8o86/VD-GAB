import torch
import torch.nn as nn

from models.layers import GraphConv
from parser import parameter_parser
from models.attentionLSTM import BiAttentionLSTM

from torch.nn import LSTM

args = parameter_parser()


class GCN_BiLstm(nn.Module):
    """
    Baseline Graph Convolutional Network with a stack of Graph Convolution Layers and global pooling over nodes.
    """

    def __init__(self, in_features, out_features, filters=args.filters,
                 n_hidden=args.n_hidden, dropout=args.dropout, adj_sq=False, scale_identity=False):
        super(GCN_BiLstm, self).__init__()

        # Graph convolution layers
        # module 1
        # GCN Start
        self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                                out_features=f, activation=nn.ReLU(inplace=True),
                                                adj_sq=adj_sq, scale_identity=scale_identity) for layer, f in
                                      enumerate(filters)]))
        # GCN End

        # Bidirectional LSTM layer

        # self.attention = Attention(n_hidden * 2, 1)

        # self.bilstm = AttentionEnhancedBiLSTM(input_size=filters[-1], hidden_size=n_hidden, num_layers=2, batch_first=True,
        #                       bidirectional=True)

        self.bilstm_layers = nn.ModuleList([
            BiAttentionLSTM(input_size=filters[-1], hidden_size=n_hidden, attention_dim=args.multi_head)
            for _ in range(1)
        ])

        self.bilstm = BiAttentionLSTM(input_size=filters[-1], hidden_size=n_hidden, attention_dim=args.multi_head)

        # Fully connected layers Modify with LSTM
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(n_hidden * 2, out_features))  # Multiply by 2 for bidirectional LSTM
            # fc.append(nn.Linear(n_hidden, out_features))  #  LSTM
        else:
            fc.append(nn.Linear(filters[-1] * 2, out_features))  # Multiply by 2 for bidirectional LSTM
            # fc.append(nn.Linear(filters[-1], out_features))  #  LSTM
        self.fc = nn.Sequential(*fc)

        # Fully connected layers Origin
        # fc = []
        # if dropout > 0:
        #     fc.append(nn.Dropout(p=dropout))
        # if n_hidden > 0:
        #     fc.append(nn.Linear(filters[-1], n_hidden))
        #     if dropout > 0:
        #         fc.append(nn.Dropout(p=dropout))
        #     n_last = n_hidden
        # else:
        #     n_last = filters[-1]
        # fc.append(nn.Linear(n_last, out_features))
        # self.fc = nn.Sequential(*fc)

    def forward(self, data):
        # gcn with bilstm

        x = self.gconv(data)[0]
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes (usually performs better than average)

        # Pass the pooled features through the bidirectional LSTM
        # self.bilstm.flatten_parameters()
        # x, _ = self.bilstm(x.unsqueeze(0))  # Add an extra dimension for the batch

        x, _ = self.bilstm(x)

        # x = self.attention(x)
        #
        x = x.squeeze(0)  # Remove the extra dimension

        # fullconnect layer bilstm
        x = self.fc(x)
        return x

        # gcn with bilstm end

        # # gcn_modify origin
        # x = self.gconv(data)[0]
        # x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes (usually performs better than average)
        # x = self.fc(x)
        # return x
        # # gcn_modify origin end


class Attention(nn.Module):
    """
    Attention layer to compute weights for nodes.
    """

    def __init__(self, in_features, hidden_features):
        super(Attention, self).__init__()

        self.query = nn.Linear(in_features, hidden_features)
        self.key = nn.Linear(in_features, hidden_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        energy = torch.matmul(query, key.transpose(1, 2))
        attention_weights = self.softmax(energy)
        weighted_value = torch.matmul(attention_weights, x)
        return weighted_value

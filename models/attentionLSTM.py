import torch
import torch.nn as nn
from parser import parameter_parser

args = parameter_parser()


class BiAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, attention_dim):
        super(BiAttentionLSTM, self).__init__()

        self.forward_lstm = AttentionLSTMCell(input_size, hidden_size, attention_dim)
        self.backward_lstm = AttentionLSTMCell(input_size, hidden_size, attention_dim)

    def forward(self, inputs, initial_states=None):
        if initial_states is None:
            # 假设AttentionLSTMCell可以处理None初始状态
            forward_initial_state = None
            backward_initial_state = None
        else:
            forward_initial_state = initial_states[0]
            backward_initial_state = initial_states[1]

        # 正向处理
        forward_final_hidden_state, forward_final_cell_state = self.forward_lstm(
            inputs, forward_initial_state
        )

        # 反向处理（注意需要对序列进行反转）
        reversed_inputs = inputs.flip(dims=[1])
        backward_final_hidden_state, backward_final_cell_state = self.backward_lstm(
            reversed_inputs, backward_initial_state
        )
        # 反转回原始顺序
        backward_final_hidden_state = backward_final_hidden_state.flip(dims=[1])

        # 拼接正向和反向的最终隐藏状态
        final_hidden_state = torch.cat((forward_final_hidden_state, backward_final_hidden_state), dim=-1)

        return final_hidden_state, (forward_final_hidden_state, backward_final_cell_state)


class AttentionEnhancedBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_dim, device=None):
        super(AttentionEnhancedBiLSTM, self).__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.layers = nn.ModuleList([
            BiAttentionLSTM(input_size if layer == 0 else hidden_size * 2, hidden_size, attention_dim)
            for layer in range(num_layers)
        ])

        # 添加默认初始状态生成器
        self.default_initial_states = []
        for _ in range(num_layers):
            forward_init_hidden = torch.zeros(1, 1, hidden_size).to(device)
            forward_init_cell = torch.zeros(1, 1, hidden_size).to(device)
            backward_init_hidden = torch.zeros(1, 1, hidden_size).to(device)
            backward_init_cell = torch.zeros(1, 1, hidden_size).to(device)

            self.default_initial_states.append(
                ((forward_init_hidden, forward_init_cell), (backward_init_hidden, backward_init_cell)))

    def forward(self, inputs, initial_states=None):
        if initial_states is None:
            initial_states = self.default_initial_states

        forward_hidden_states = []
        forward_cell_states = []
        backward_hidden_states = []
        backward_cell_states = []

        for i in range(len(initial_states)):
            forward_hidden_states.append(initial_states[i][0][0].unsqueeze(0))
            forward_cell_states.append(initial_states[i][0][1].unsqueeze(0))
            backward_hidden_states.append(initial_states[i][1][0].unsqueeze(0))
            backward_cell_states.append(initial_states[i][1][1].unsqueeze(0))

        # 对每个时间步应用双向自注意力LSTM单元
        for t in range(inputs.size(1)):
            for layer in range(len(self.layers)):
                forward_hs, (forward_h, backward_c) = self.layers[layer](
                    inputs[:, t, :],
                    (forward_hidden_states[layer], (backward_hidden_states[layer], backward_cell_states[layer]))
                )
                forward_hidden_states[layer] = forward_h.to(self.device)
                forward_cell_states[layer] = forward_hs.to(self.device)
                backward_hidden_states[layer] = backward_c.to(self.device)

        # 返回最后一层的正向和反向隐藏状态
        final_forward_hidden_state = forward_hidden_states[-1].squeeze(0)
        final_backward_hidden_state = backward_hidden_states[-1].squeeze(0)

        # 拼接正向和反向的最终隐藏状态
        final_hidden_state = torch.cat((final_forward_hidden_state, final_backward_hidden_state), dim=-1)

        return final_hidden_state


class AttentionLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, attention_dim, device=None):
        super(AttentionLSTMCell, self).__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

        self.hidden_size = hidden_size  # 添加这行，定义hidden_size为实例属性

        # 自注意力部分
        self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=1)

        # LSTM单元
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

    def init_hidden(self, batch_size=args.batch_size):
        return (
            torch.zeros(batch_size, self.hidden_size),
            torch.zeros(batch_size, self.hidden_size),
        )

    def forward(self, input, hidden_state=None, cell_state=None, sequence_mask=None):
        if hidden_state is None or cell_state is None:
            hidden_state, cell_state = self.init_hidden(input.size(0))

        hidden_state = hidden_state.to(self.device)
        cell_state = cell_state.to(self.device)
        # 应用自注意力机制
        attended_input, _ = self.attention(input, input, input, key_padding_mask=sequence_mask)
        # 取加权后的输入作为LSTM单元的输入

        assert isinstance(attended_input, torch.Tensor) and isinstance(input, torch.Tensor)

        lstm_input = attended_input + input
        # lstm_input = attended_input

        if self.device != lstm_input.device:
            lstm_input = lstm_input.to(self.device)

        # 将自注意力处理后的结果输入到LSTM单元
        new_hidden_state, new_cell_state = self.lstm_cell(lstm_input, (hidden_state, cell_state))

        return new_hidden_state, new_cell_state

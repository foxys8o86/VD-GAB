# VD-GAB
This repo is a python implementation of the model proposed in the paper "VD-GAB: A Smart Contract Reentrancy Vulnerability Detection Method based on Graph Convolutional Networks and Improved Bidirectional Long Short-term Memory Networks".

The project code is in the other branch, `master` branch.

## Code structure
The main implementation of the model is in the folder `models`.

`attentionLSTM.py` implementation the LSTM cells enhanced with attention mechanisms.

`gcn_bilstm.py` uses the enhanced LSTM cells to build bidirectional LSTM network after GCN. It is the main contribution of the model.

We also uploaded the simple version of `GAT` and `GCN`.

## Running project
- Please use the command `python VD_GAB.py` to run the program.
- You need to add command like `--deivce 0 1 2 3` to assign the gpu devices used.
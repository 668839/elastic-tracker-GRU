#!/usr/bin/env python3
"""
Teacher GRU模型（大容量）
用于知识蒸馏的教师模型
"""

import torch
import torch.nn as nn

class TeacherGRU(nn.Module):
    """大容量教师模型"""
    
    def __init__(self, 
                 input_dim=4,
                 hidden_dim=256,      # 大容量
                 num_layers=3,        # 深层网络
                 output_dim=2,
                 pred_horizon=60,
                 dropout=0.5):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.pred_horizon = pred_horizon
        
        # 编码器：3层GRU + Dropout
        self.encoder = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # 注意力机制（可选增强）
        self.attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 解码器
        self.decoder_gru = nn.GRU(
            output_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        """
        x: (batch, obs_len=30, input_dim=4)
        返回: (batch, pred_len=60, output_dim=2)
        """
        batch_size = x.size(0)
        
        # 编码
        enc_out, hidden = self.encoder(x)
        
        # 自注意力（增强历史理解）
        attn_out, _ = self.attention(enc_out, enc_out, enc_out)
        
        # 自回归解码
        decoder_input = x[:, -1, :2].unsqueeze(1)  # 最后位置
        predictions = []
        
        # 初始化解码器隐藏状态
        decoder_hidden = hidden[-2:, :, :]  # 取最后2层
        
        for t in range(self.pred_horizon):
            dec_out, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)
            pred_t = self.fc_out(dec_out)
            predictions.append(pred_t)
            decoder_input = pred_t
        
        predictions = torch.cat(predictions, dim=1)
        return predictions

# 测试
if __name__ == '__main__':
    model = TeacherGRU()
    x = torch.randn(4, 30, 4)
    pred = model(x)
    print(f"Teacher GRU - Input: {x.shape}, Output: {pred.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
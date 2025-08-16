#!/usr/bin/env python3
# =============================
# ノイズ追加分類器用モデルクラス
# - Wav2VecClassifier
# =============================

import torch
import torch.nn as nn

class Wav2VecClassifier(nn.Module):
    """wav2vec特徴量用Transformer Encoder分類器（2値分類版）"""
    
    def __init__(self, classification_config):
        super().__init__()
        
        # 設定ファイルからハイパーパラメータを取得
        model_config = classification_config['model']
        
        self.wav2vec_dim = model_config['wav2vec_dim'] 
        self.hidden_dim = model_config['hidden_size']  
        self.num_classes = model_config['num_classes'] 
        
        # 特徴量投影層（wav2vec特徴量をTransformerの入力次元に投影）
        self.feature_projection = nn.Linear(self.wav2vec_dim, self.hidden_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=model_config['n_heads'],
            dim_feedforward=model_config['intermediate_size'], 
            dropout=model_config['dropout'],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_config['n_layers']
        )
        
        # プーリング戦略（設定ファイルから取得）
        self.pooling = model_config['pooling']
        
        # 分類器（2値分類用、出力次元1）
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(model_config['dropout']),
            nn.Linear(self.hidden_dim, model_config['hidden_mlp_size']),
            nn.ReLU(),
            nn.Linear(model_config['hidden_mlp_size'], 1)  # 出力次元1
        )
    
    def forward(self, features, mask=None):
        """
        wav2vec特徴量用Transformer分類器の順伝播

        Args:
            features (torch.Tensor): wav2vec特徴量 (batch_size, time_steps, feature_dim)
            mask (torch.BoolTensor, optional): パディングマスク
        Returns:
            torch.Tensor: 分類器の出力（ロジット）
        """

        # 特徴量投影
        features = self.feature_projection(features)  # (batch_size, time_steps, hidden_dim)
        
        # パディングマスクの作成（ゼロパディング部分をマスク）
        if mask is None:
            # ゼロパディング部分を検出してマスクを作成
            mask = (features.sum(dim=-1) == 0)  # (batch_size, time_steps)
        
        # TransformerEncoderで特徴量を処理
        features = self.transformer_encoder(features, src_key_padding_mask=mask)
                
        # プーリング戦略の適用
        if self.pooling == 'mean':
            # マスクされた部分を除外して平均を計算
            if mask is not None:
                features = features.masked_fill(mask.unsqueeze(-1), 0)
                lengths = (~mask).sum(dim=1, keepdim=True).float()
                features = features.sum(dim=1) / lengths.clamp(min=1)
            else:
                features = features.mean(dim=1)  # 時間次元で平均
        elif self.pooling == 'cls':
            features = features[:, 0, :]  # 最初のトークン（CLSトークン）を使用
        
        # 最終分類器でクラスロジットを出力
        return self.classifier(features)

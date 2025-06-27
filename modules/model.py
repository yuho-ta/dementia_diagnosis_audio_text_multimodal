# =============================
# マルチモーダル融合モデル群
# - 音声・テキスト埋め込みの融合手法
# - クロスアテンション、双方向融合、要素ごと融合
# - アテンションプーリング、ゲート機構
# - ResNet音声特徴抽出器
# =============================

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPooling(nn.Module):
    """
    アテンションプーリング層
    シーケンスの各要素に重みを付けて平均を取る
    """
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)  # アテンションスコアを計算する線形層

    def forward(self, x, mask=None):
        """
        Args:
            x: 形状 (B, T, D) のテンソル（バッチサイズ、時間長、次元数）
            mask: (オプション) 形状 (B, T) のブールテンソル、True=保持、False=無視
        Returns:
            形状 (B, D) のプールされたテンソル
        """
        # 生のアテンションスコアを計算
        scores = self.attn(x).squeeze(-1)  # (B, T)

        if mask is not None:
            # パディングトークンをマスクアウト
            scores = scores.masked_fill(~mask, float('-inf'))

        # ソフトマックスで重みを正規化
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        # 重み付き平均を計算
        pooled = torch.sum(weights * x, dim=1)  # (B, D)
        return pooled
    

class GatedAttnPooling(nn.Module):
    """
    ゲート付きアテンションプーリング層
    アテンションスコアにゲート機構を組み合わせたプーリング
    """
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)  # アテンションスコア
        self.gate = nn.Linear(dim, 1)  # ゲートスコア

    def forward(self, x, mask=None):
        attn_score = self.attn(x).squeeze(-1)  # アテンションスコア
        gate_score = torch.sigmoid(self.gate(x)).squeeze(-1)  # ゲートスコア（0-1）

        # アテンションスコアとゲートスコアを組み合わせ
        scores = attn_score * gate_score  # (B, T)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(weights * x, dim=1)


class ResidualBlock(nn.Module):
    """
    ResNetの残差ブロック
    2つの畳み込み層とスキップ接続を持つ
    """
    def __init__(self, in_c, out_c, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        # スキップ接続（入力と出力のチャンネル数が異なる場合は1x1畳み込み）
        self.shortcut = (
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
            if in_c != out_c else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)  # スキップ接続
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity  # 残差接続
        return self.relu(out)

class ResNetAudio(nn.Module):
    """
    音声特徴抽出用のResNet
    メルスペクトログラムやeGeMAPS特徴量を処理
    """
    def __init__(self, in_channels=1, out_channels=768, dropout=0.1):
        super().__init__()

        # 初期特徴抽出層
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 残差ブロック層
        self.layer1 = ResidualBlock(32, 64, dropout)
        self.layer2 = ResidualBlock(64, 128, dropout)
        self.layer3 = ResidualBlock(128, out_channels, dropout)

        # グローバルプーリング（時間次元は保持、周波数次元をプール）
        self.global_pool = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, x):
        """
        Args:
            x: 形状 (B, T, F) のテンソル — 時間、周波数（例：メルスペクトログラム）
        Returns:
            形状 (B, T, out_channels) のテンソル
        """
        x = x.unsqueeze(1)  # (B, 1, T, F) - チャンネル次元を追加
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)  # (B, C, T, 1)
        x = x.squeeze(-1).transpose(1, 2)  # (B, T, C) - 形状を調整
        return x


class CrossAttentionEncoderLayer(nn.Module):
    """
    クロスアテンションTransformerエンコーダー層
    2つのモーダル間でクロスアテンションを実行
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU()):
        super(CrossAttentionEncoderLayer, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # マルチヘッドクロスアテンション
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # フィードフォワードネットワーク
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, memory, src_mask=None, src_key_padding_mask=None):
        """
        クロスアテンション層の順伝播
        Args:
            src: クエリ（例：音声）
            memory: キー/バリュー（例：テキスト）
        """

        # Pre-Normalization
        src = self.norm1(src)
        memory = self.norm1(memory)

        # クロスアテンション（クエリ: src, キー/バリュー: memory）
        attn_output, _ = self.cross_attention(src, memory, memory, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)  # 残差接続

        # フィードフォワードネットワーク
        src = self.norm2(src)
        src = src + self.dropout2(self.feedforward(src))  # 残差接続

        return src
    


class GatedCrossAttentionFusion(nn.Module):
    """
    ゲート付き残差クロスアテンション融合層
    クロスアテンションの出力にゲート機構を適用
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU()):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # ゲート機構（入力とアテンション出力を組み合わせてゲートを計算）
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, memory, src_mask=None, src_key_padding_mask=None):
        """
        クロスアテンション: srcがmemoryにクエリを送る
        Args:
            src: テンソル (B, T, d_model) — クエリ（例：音声）
            memory: テンソル (B, T, d_model) — キー/バリュー（例：テキスト）
            src_mask: オプションのアテンションマスク
            src_key_padding_mask: オプションのパディングマスク
        Returns:
            テンソル (B, T, d_model): 融合された出力
        """
        # Pre-norm付きクロスアテンション
        src_norm = self.norm1(src)
        memory_norm = self.norm1(memory)

        attn_output, _ = self.cross_attention(
            query=src_norm,
            key=memory_norm,
            value=memory_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )

        attn_output = self.dropout1(attn_output)

        # ゲート融合
        gate_input = torch.cat([src, attn_output], dim=-1)  # (B, T, 2*d_model)
        gate = self.gate(gate_input)  # (B, T, d_model)
        fused = src + gate * attn_output  # ゲート付き残差接続

        # Post-norm付きフィードフォワード
        fused = self.norm2(fused)
        fused = fused + self.dropout2(self.feedforward(fused))  # 残差接続

        return fused

class CrossAttentionTransformerEncoder(nn.Module):
    """
    クロスアテンションTransformerエンコーダー
    音声とテキストのクロスアテンション融合
    """
    def __init__(self, config):
        super(CrossAttentionTransformerEncoder, self).__init__()

        self.num_layers = config.n_layers
        self.model_name = config.model_name
        self.config = config

        # メルスペクトログラムやeGeMAPS用の特徴抽出器
        if 'mel' in self.model_name or 'egemaps' in self.model_name:
            self.mel_extractor = ResNetAudio(in_channels=1, out_channels=config.hidden_size, dropout=config.dropout)

        # 融合方法に応じてレイヤーを選択
        if 'gated' in config.fusion:
            # ゲート付きクロスアテンション層
            self.layers = nn.ModuleList([
                GatedCrossAttentionFusion(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])
        else:
            # 通常のクロスアテンション層
            self.layers = nn.ModuleList([
                CrossAttentionEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])

        # レイヤー間のLayerNorm
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(config.hidden_size) for _ in range(config.n_layers - 1)
        ])

        self.dropout = nn.Dropout(config.dropout)
        self.pooling = config.pooling

        # プーリング戦略の設定
        if config.pooling == 'attn':
            self.attn_pooling = AttnPooling(config.hidden_size)
        elif config.pooling == 'gatedattn':
            self.attn_pooling = GatedAttnPooling(config.hidden_size)

        # 分類器
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_mlp_size),
            nn.ReLU(),
            nn.Linear(config.hidden_mlp_size, config.num_classes)
        )

    def forward(self, features, mask=None, key_padding_mask=None):
        """
        マルチレイヤークロスアテンションTransformerエンコーダーの順伝播
        """
        
        src, memory = features

        audio_model = self.model_name.split('_')[1] if self.config.audio_model != '' else ''

        # 音声特徴抽出（必要に応じて）
        if 'mel' == audio_model or 'egemaps' == audio_model:
            src = self.mel_extractor(src)
        elif 'mel' == audio_model or 'egemaps' == audio_model:
            memory = self.mel_extractor(memory)

        # レイヤーを順次適用（レイヤー間で正規化）
        for i, layer in enumerate(self.layers):
            src = layer(src, memory, src_mask=mask, src_key_padding_mask=key_padding_mask)
            if i < len(self.norm_layers):  # レイヤー間で正規化を適用
                src = self.norm_layers[i](src)
                src = self.dropout(src)

        # プーリング戦略
        if self.pooling == 'mean':
            src = src.mean(dim=1)  # 時間次元で平均
        elif self.pooling == 'cls':
            src = src[:, 0, :]  # 最初のトークン（CLS）
        elif 'attn' in self.pooling:
            src = self.attn_pooling(src, mask=mask)  # アテンションプーリング

        return self.classifier(src)
    

class BidirectionalCrossAttentionTransformerEncoder(nn.Module):
    """
    双方向クロスアテンションTransformerエンコーダー
    音声→テキスト、テキスト→音声の両方向でクロスアテンション
    """
    def __init__(self, config):
        super(BidirectionalCrossAttentionTransformerEncoder, self).__init__()

        self.num_layers = config.n_layers
        self.model_name = config.model_name
        self.fusion = config.fusion
        self.config = config

        # メルスペクトログラムやeGeMAPS用の特徴抽出器
        if 'mel' in self.model_name or 'egemaps' in self.model_name:
            self.mel_extractor = ResNetAudio(in_channels=1, out_channels=config.hidden_size, dropout=config.dropout)
        
        # 双方向の融合方法に応じてレイヤーを選択
        if 'gated' in config.fusion:
            # ゲート付きクロスアテンション層（音声→テキスト）
            self.layers_1 = nn.ModuleList([
                GatedCrossAttentionFusion(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])

            # ゲート付きクロスアテンション層（テキスト→音声）
            self.layers_2 = nn.ModuleList([
                GatedCrossAttentionFusion(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])
        else:
            # 通常のクロスアテンション層（音声→テキスト）
            self.layers_1 = nn.ModuleList([
                CrossAttentionEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])

            # 通常のクロスアテンション層（テキスト→音声）
            self.layers_2 = nn.ModuleList([
                CrossAttentionEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])

        # レイヤー間のLayerNorm（両方向）
        self.norm_layers_1 = nn.ModuleList([
            nn.LayerNorm(config.hidden_size) for _ in range(config.n_layers - 1)
        ])

        self.norm_layers_2 = nn.ModuleList([
            nn.LayerNorm(config.hidden_size) for _ in range(config.n_layers - 1)
        ])

        self.dropout = nn.Dropout(config.dropout)
        self.pooling = config.pooling

        # 融合方法に応じて分類器の入力サイズを調整
        init_mlp_size = config.hidden_size * 2 if 'concat' in self.fusion else config.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(init_mlp_size),
            nn.Dropout(config.dropout),
            nn.Linear(init_mlp_size, config.hidden_mlp_size),
            nn.ReLU(),
            nn.Linear(config.hidden_mlp_size, config.num_classes)
        )

    def forward(self, features, mask=None, key_padding_mask=None):
        """
        双方向クロスアテンションTransformerエンコーダーの順伝播
        """
        
        src, memory = features

        audio_model = self.model_name.split('_')[1] if self.config.audio_model != '' else ''

        # 音声特徴抽出（必要に応じて）
        if 'mel' == audio_model or 'egemaps' == audio_model:
            src = self.mel_extractor(src)
        elif 'mel' == audio_model or 'egemaps' == audio_model:
            memory = self.mel_extractor(memory)

        # 音声→テキスト方向の処理
        src1 = src.clone()
        memory1 = memory.clone()

        # 第1方向の埋め込み（音声がテキストに注意）
        for i, layer in enumerate(self.layers_1):
            src1 = layer(src1, memory1, src_mask=mask, src_key_padding_mask=key_padding_mask)
            if i < len(self.norm_layers_1):  # レイヤー間で正規化を適用
                src1 = self.norm_layers_1[i](src1)
                src1 = self.dropout(src1)
        
        # テキスト→音声方向の処理
        src2 = memory.clone()
        memory2 = src.clone()

        # 第2方向の埋め込み（テキストが音声に注意）
        for i, layer in enumerate(self.layers_2):
            src2 = layer(src2, memory2, src_mask=mask, src_key_padding_mask=key_padding_mask)
            if i < len(self.norm_layers_2):  # レイヤー間で正規化を適用
                src2 = self.norm_layers_2[i](src2)
                src2 = self.dropout(src2)

        # src1とsrc2を融合

        if 'concat' in self.fusion:
            src = torch.cat((src1, src2), dim=2)  # 連結
        elif 'sum' in self.fusion:
            src = src1 + src2  # 加算
        elif 'mul' in self.fusion:
            src = src1 * src2  # 要素ごと乗算
        elif 'mean' in self.fusion:
            src = (src1 + src2) / 2  # 平均
        else:
            src = src1 + src2  # デフォルトは加算
            
        # プーリング戦略
        if self.pooling == 'mean':
            src = src.mean(dim=1)
        elif self.pooling == 'cls':
            src = src[:, 0, :]

        return self.classifier(src)


class ElementWiseFusionEncoder(nn.Module):
    """
    要素ごと融合エンコーダー
    音声とテキストを要素ごとに融合してからTransformerで処理
    """
    def __init__(self, config):
        super(ElementWiseFusionEncoder, self).__init__()

        self.model_name = config.model_name
        self.fusion = config.fusion
        self.config = config

        # 融合方法に応じて隠れ層サイズを調整
        hidden_size = config.hidden_size * 2 if self.fusion == 'concat' else config.hidden_size

        # メルスペクトログラムやeGeMAPS用の特徴抽出器
        if 'mel' in self.model_name or 'egemaps' in self.model_name:
            self.mel_extractor = ResNetAudio(in_channels=1, out_channels=config.hidden_size, dropout=config.dropout)
        
        # Transformerエンコーダー
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size, 
                nhead=config.n_heads, 
                dim_feedforward=config.intermediate_size, 
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.n_layers
        )

        self.pooling = config.pooling
        
        # 分類器
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, config.hidden_mlp_size),
            nn.ReLU(),
            nn.Linear(config.hidden_mlp_size, config.num_classes)
        )
        
    def forward(self, features, mask=None, key_padding_mask=None):
        """
        要素ごと融合エンコーダーの順伝播
        """

        src, memory = features

        audio_model = self.model_name.split('_')[1] if self.config.audio_model != '' else ''

        # 音声特徴抽出（必要に応じて）
        if 'mel' == audio_model or 'egemaps' == audio_model:
            src = self.mel_extractor(src)
        elif 'mel' == audio_model or 'egemaps' == audio_model:
            memory = self.mel_extractor(memory)

        # 要素ごとの融合方法
        if self.fusion == 'concat':
            features = torch.cat((src, memory), dim=2)  # 連結
        elif self.fusion == 'selfattn':
            src = src.mean(dim=1)  # 時間次元で平均
            memory = memory.mean(dim=1)
            features = torch.stack((src, memory), dim=1)  # スタック
        elif self.fusion == 'mean':
            features = (src + memory) / 2  # 平均
        elif self.fusion == 'sum':
            features = src + memory  # 加算
        elif self.fusion == 'mul':
            features = src * memory  # 要素ごと乗算
        
        # Transformerエンコーダーで処理
        features = self.encoder(features, src_key_padding_mask=key_padding_mask)
                
        # プーリング戦略
        if self.pooling == 'mean':
            features = features.mean(dim=1)
        elif self.pooling == 'cls':
            features = features[:, 0, :]

        return self.classifier(features)



class MyTransformerEncoder(nn.Module):
    """
    単一モーダル用Transformerエンコーダー
    音声またはテキストの単一モーダルを処理
    """
    def __init__(self, config):
        super(MyTransformerEncoder, self).__init__()

        self.model_name = config.model_name

        # メルスペクトログラムやeGeMAPS用の特徴抽出器
        if 'mel' in self.model_name or 'egemaps' in self.model_name:
            self.mel_extractor = ResNetAudio(in_channels=1, out_channels=config.hidden_size)
        
        # Transformerエンコーダー
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size, 
                nhead=config.n_heads, 
                dim_feedforward=config.intermediate_size, 
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.n_layers
        )

        self.pooling = config.pooling
        
        # 分類器
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_mlp_size),
            nn.ReLU(),
            nn.Linear(config.hidden_mlp_size, config.num_classes)
        )
        
    def forward(self, features, mask=None, key_padding_mask=None):
        """
        単一モーダルTransformerエンコーダーの順伝播
        """

        # 音声特徴抽出（必要に応じて）
        if 'mel' in self.model_name or 'egemaps' in self.model_name:
            features = self.mel_extractor(features)
        
        # Transformerエンコーダーで処理
        features = self.encoder(features, src_key_padding_mask=key_padding_mask)
                
        # プーリング戦略
        if self.pooling == 'mean':
            features = features.mean(dim=1)  # 時間次元で平均
        elif self.pooling == 'cls':
            features = features[:, 0, :]  # 最初のトークン
        
        return self.classifier(features)
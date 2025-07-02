import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPooling(nn.Module):
    """
    アテンションプーリング層

    シーケンスの各要素に重みを付けて平均を取ることで、
    シーケンス全体の重要な情報を凝縮した固定長のベクトル表現を生成します。
    """
    def __init__(self, dim):
        """
        Args:
            dim (int): 入力特徴量の次元数。
        """
        super().__init__()
        # 各シーケンス要素のアテンションスコアを計算するための線形層。
        # 入力次元 'dim' から単一のスカラー値（スコア）を出力します。
        self.attn = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        """
        順伝播処理。

        Args:
            x (torch.Tensor): 形状 `(B, T, D)` のテンソル。
                              `B` はバッチサイズ、`T` はシーケンス長、`D` は特徴量次元数です。
            mask (torch.BoolTensor, optional): 形状 `(B, T)` のブールテンソル。
                                               `True` の位置は保持され、`False` の位置は無視（パディングなど）されます。
                                               デフォルトは None です。
        Returns:
            torch.Tensor: 形状 `(B, D)` のプールされたテンソル。
                          シーケンス長 `T` がアテンションによって集約され、固定長の特徴量となります。
        """
        # 1. 生のアテンションスコアを計算
        # x (B, T, D) -> self.attn(x) (B, T, 1) -> squeeze(-1) -> scores (B, T)
        scores = self.attn(x).squeeze(-1)

        # 2. マスクの適用（パディングトークン:バッチ処理で複数の文を同時に処理する際、長さを揃えるために短い文の末尾に追加される特殊トークンの影響を排除）
        if mask is not None:
            # マスクがFalse（無視する要素）の位置のスコアを負の無限大に設定します。
            # これにより、softmaxを適用したときにこれらの位置の重みがゼロになります。
            scores = scores.masked_fill(~mask, float('-inf'))

        # 3. ソフトマックスで重みを正規化
        # scores (B, T) -> F.softmax(scores, dim=1) (B, T) -> unsqueeze(-1) -> weights (B, T, 1)
        # dim=1 でソフトマックスを取ることで、各シーケンス内で重みの合計が1になります。
        weights = F.softmax(scores, dim=1).unsqueeze(-1)

        # 4. 重み付き平均を計算
        # (weights * x) (B, T, D) となり、時間次元 (dim=1) で合計することで (B, D) のプールされた特徴量を得ます。
        pooled = torch.sum(weights * x, dim=1)
        return pooled


class GatedAttnPooling(nn.Module):
    """
    ゲート付きアテンションプーリング層

    アテンションスコアにゲート機構を組み合わせたプーリング手法です。
    ゲート機構により、どの情報をどれだけアテンションに反映させるかを動的に制御できます。
    """
    def __init__(self, dim):
        """
        Args:
            dim (int): 入力特徴量の次元数。
        """
        super().__init__()
        # アテンションスコアを計算する線形層
        self.attn = nn.Linear(dim, 1)
        # ゲートスコア（0-1の範囲）を計算する線形層
        self.gate = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        """
        順伝播処理。

        Args:
            x (torch.Tensor): 形状 `(B, T, D)` のテンソル。
            mask (torch.BoolTensor, optional): 形状 `(B, T)` のブールテンソル。デフォルトは None です。
        Returns:
            torch.Tensor: 形状 `(B, D)` のプールされたテンソル。
        """
        # 1. 生のアテンションスコアを計算
        attn_score = self.attn(x).squeeze(-1)  # (B, T)
        # 2. ゲートスコアを計算し、シグモイドで0-1に正規化
        gate_score = torch.sigmoid(self.gate(x)).squeeze(-1)  # (B, T)

        # 3. アテンションスコアとゲートスコアを要素ごとに乗算して結合スコアを生成
        # これにより、ゲートが低い位置のアテンションスコアは抑制されます。
        scores = attn_score * gate_score  # (B, T)

        # 4. マスクの適用（パディングトークンの影響を排除）
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        # 5. ソフトマックスで重みを正規化し、重み付き平均を計算
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return torch.sum(weights * x, dim=1) # (B, D)


class ResidualBlock(nn.Module):
    """
    ResNetの残差ブロック

    2つの畳み込み層とスキップ接続（残差接続）を持ち、
    深いネットワークでの勾配消失問題を軽減し、学習を安定させます。
    """
    def __init__(self, in_c, out_c, dropout=0.1):
        """
        Args:
            in_c (int): 入力チャンネル数。
            out_c (int): 出力チャンネル数。
            dropout (float): ドロップアウト率。デフォルトは 0.1 です。
        """
        super().__init__()
        # 最初の畳み込み層とバッチ正規化、ReLU活性化
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        # ドロップアウト層 (2Dデータ用)
        self.dropout = nn.Dropout2d(dropout)

        # 2番目の畳み込み層とバッチ正規化
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        # スキップ接続: 入力と出力のチャンネル数が異なる場合、1x1畳み込みでチャンネル数を合わせます。
        # 同じ場合は Identity（何もしない）を使います。
        self.shortcut = (
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
            if in_c != out_c else nn.Identity()
        )

    def forward(self, x):
        """
        順伝播処理。

        Args:
            x (torch.Tensor): 入力テンソル。通常、形状 `(B, C, H, W)`。
        Returns:
            torch.Tensor: 残差接続と活性化関数が適用された出力テンソル。
        """
        # スキップ接続のパス（identity）を計算
        identity = self.shortcut(x)
        
        # メインパスの処理
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        # 残差接続: メインパスの出力にスキップ接続の出力を加算
        out += identity
        
        # 最終的なReLU活性化
        return self.relu(out)

class ResNetAudio(nn.Module):
    """
    音声特徴抽出用のResNet

    メルスペクトログラムやeGeMAPSのような2次元音声特徴量を処理し、
    時間方向に沿った高レベルな特徴シーケンスを抽出します。
    """
    def __init__(self, in_channels=1, out_channels=768, dropout=0.1):
        """
        Args:
            in_channels (int): 入力チャンネル数（例: メルスペクトログラムの場合1）。デフォルトは 1 です。
            out_channels (int): 出力特徴量の次元数。デフォルトは 768 です。
            dropout (float): 残差ブロック内のドロップアウト率。デフォルトは 0.1 です。
        """
        super().__init__()

        # 初期特徴抽出層: 最初の畳み込み、バッチ正規化、ReLU
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 残差ブロック層: 複数のResidualBlockを積み重ねて深い特徴を抽出
        self.layer1 = ResidualBlock(32, 64, dropout)
        self.layer2 = ResidualBlock(64, 128, dropout)
        self.layer3 = ResidualBlock(128, out_channels, dropout)

        # グローバルプーリング層: 時間次元は保持し、周波数次元を平均プーリングして次元を削減します。
        # (None, 1) は、時間次元はそのままに、最後の次元を1にプーリングすることを示します。
        self.global_pool = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, x):
        """
        順伝播処理。

        Args:
            x (torch.Tensor): 形状 `(B, T, F)` の入力テンソル。
                              `B` はバッチサイズ、`T` は時間長、`F` は周波数次元（例: メルスペクトログラムの周波数ビン数）です。
        Returns:
            torch.Tensor: 形状 `(B, T, out_channels)` の出力テンソル。
                          各時間ステップにおける音声の高レベル特徴量シーケンス。
        """
        # 畳み込み処理のためにチャンネル次元を追加 (B, T, F) -> (B, 1, T, F)
        x = x.unsqueeze(1)
        
        # ステム層で初期特徴抽出
        x = self.stem(x)
        # 残差ブロック層を順次適用
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # グローバルプーリング (B, C, T, F) -> (B, C, T, 1)
        x = self.global_pool(x)
        
        # 形状を整形: 最後の次元を削除し、時間次元とチャンネル次元を入れ替える
        # (B, C, T, 1) -> (B, C, T) -> transpose(1, 2) -> (B, T, C)
        x = x.squeeze(-1).transpose(1, 2)
        return x


class CrossAttentionEncoderLayer(nn.Module):
    """
    クロスアテンションTransformerエンコーダー層

    2つの異なるモーダル（例: 音声とテキスト）間でクロスアテンションを実行し、
    一方のモーダルがもう一方のモーダルから関連情報を抽出・統合できるようにします。
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU()):
        """
        Args:
            d_model (int): モデルの埋め込み次元数。
            nhead (int): マルチヘッドアテンションのヘッド数。
            dim_feedforward (int): フィードフォワードネットワークの中間層の次元数。デフォルトは 2048 です。
            dropout (float): ドロップアウト率。デフォルトは 0.1 です。
            activation (nn.Module): フィードフォワードネットワークで使用する活性化関数。デフォルトは ReLU です。
        """
        super(CrossAttentionEncoderLayer, self).__init__()

        # Layer Normalization 層
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # マルチヘッドクロスアテンション層
        # query (src), key (memory), value (memory) を取ることで、srcがmemoryにアテンションします。
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # ポジションワイズフィードフォワードネットワーク (FFN)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # ドロップアウト層
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, memory, src_mask=None, src_key_padding_mask=None):
        """
        クロスアテンション層の順伝播。

        Args:
            src (torch.Tensor): クエリとなるテンソル。形状 `(B, T_src, d_model)`。（例: 音声特徴）
            memory (torch.Tensor): キー/バリューとなるテンソル。形状 `(B, T_mem, d_model)`。（例: テキスト特徴）
            src_mask (torch.Tensor, optional): クエリのアテンションマスク。Transformerの `attn_mask` 引数に渡されます。
                                               デフォルトは None です。
            src_key_padding_mask (torch.BoolTensor, optional): クエリのキーパディングマスク。
                                                                `True` は無視する位置を示します。
                                                                デフォルトは None です。
        Returns:
            torch.Tensor: クロスアテンションとFFNを通過した `src` の出力テンソル。形状は `(B, T_src, d_model)`。
        """
        # Pre-Normalization: アテンション計算前に正規化を適用
        src_norm = self.norm1(src)
        memory_norm = self.norm1(memory)

        # クロスアテンション: src (クエリ) が memory (キー/バリュー) にアテンション
        # attn_output (B, T_src, d_model)
        attn_output, _ = self.cross_attention(
            query=src_norm, 
            key=memory_norm, 
            value=memory_norm, 
            attn_mask=src_mask, 
            key_padding_mask=src_key_padding_mask
        )
        
        # 残差接続とドロップアウト
        src = src + self.dropout1(attn_output)

        # Post-Normalizationとフィードフォワードネットワーク
        src_norm = self.norm2(src)
        src = src + self.dropout2(self.feedforward(src_norm))

        return src


class GatedCrossAttentionFusion(nn.Module):
    """
    ゲート付き残差クロスアテンション融合層

    CrossAttentionEncoderLayerに加えて、アテンションの出力にゲート機構を適用します。
    これにより、アテンションによって抽出された情報が元の情報にどれだけ寄与するかを
    動的に調整できます。
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU()):
        """
        Args:
            d_model (int): モデルの埋め込み次元数。
            nhead (int): マルチヘッドアテンションのヘッド数。
            dim_feedforward (int): フィードフォワードネットワークの中間層の次元数。デフォルトは 2048 です。
            dropout (float): ドロップアウト率。デフォルトは 0.1 です。
            activation (nn.Module): フィードフォワードネットワークで使用する活性化関数。デフォルトは ReLU です。
        """
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

        # ゲート機構: srcとattn_outputを連結し、そこからシグモイド活性化を持つゲートを生成します。
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), # srcとattn_outputを連結するため、入力次元は d_model * 2
            nn.Sigmoid() # ゲート値を0から1の間に制限
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, memory, src_mask=None, src_key_padding_mask=None):
        """
        順伝播処理。

        Args:
            src (torch.Tensor): クエリとなるテンソル。形状 `(B, T, d_model)`。（例: 音声特徴）
            memory (torch.Tensor): キー/バリューとなるテンソル。形状 `(B, T, d_model)`。（例: テキスト特徴）
            src_mask (torch.Tensor, optional): クエリのアテンションマスク。
            src_key_padding_mask (torch.BoolTensor, optional): クエリのキーパディングマスク。
        Returns:
            torch.Tensor: ゲート付き融合とFFNを通過した `src` の出力テンソル。形状は `(B, T, d_model)`。
        """
        # Pre-normalization
        src_norm = self.norm1(src)
        memory_norm = self.norm1(memory)

        # クロスアテンション
        attn_output, _ = self.cross_attention(
            query=src_norm,
            key=memory_norm,
            value=memory_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        attn_output = self.dropout1(attn_output)

        # ゲート融合
        # 元のsrcとアテンション出力attn_outputを連結してゲートを計算
        gate_input = torch.cat([src, attn_output], dim=-1)  # (B, T, 2*d_model)
        gate = self.gate(gate_input)  # (B, T, d_model)

        # ゲート付き残差接続: ゲート値によってアテンション出力の寄与度を調整
        fused = src + gate * attn_output  # (B, T, d_model)

        # Post-normalizationとフィードフォワードネットワーク
        fused_norm = self.norm2(fused)
        fused = fused + self.dropout2(self.feedforward(fused_norm))

        return fused

class CrossAttentionTransformerEncoder(nn.Module):
    """
    クロスアテンションTransformerエンコーダー

    音声とテキストの単方向クロスアテンション融合モデルを構築します。
    設定に応じて、通常のクロスアテンションまたはゲート付きクロスアテンションを使用し、
    ResNetを音声特徴抽出器として組み込むことができます。
    """
    def __init__(self, config):
        """
        Args:
            config (object): モデル設定を含むオブジェクト。
                             以下の属性を持つことを想定:
                             - n_layers (int): Transformer層の数
                             - model_name (str): モデル名（音声特徴抽出器の選択に影響）
                             - hidden_size (int): 隠れ層の次元数
                             - n_heads (int): アテンションヘッド数
                             - intermediate_size (int): FFNの中間層の次元数
                             - dropout (float): ドロップアウト率
                             - fusion (str): 融合戦略（例: 'gated' を含むか否か）
                             - pooling (str): プーリング戦略（'mean', 'cls', 'attn', 'gatedattn'）
                             - audio_model (str): 使用する音声モデルのタイプ（例: 'mel', 'egemaps'）
                             - hidden_mlp_size (int): 分類器の中間層の次元数
                             - num_classes (int): 分類タスクのクラス数
        """
        super(CrossAttentionTransformerEncoder, self).__init__()

        self.num_layers = config.n_layers
        self.model_name = config.model_name
        self.config = config

        # 音声特徴抽出器の初期化
        # 'mel' または 'egemaps' がモデル名に含まれる場合、ResNetAudioを使用
        # config.audio_model が設定されている場合、そのタイプに基づいて決定
        audio_model_type = self.model_name.split('_')[1] if hasattr(self.config, 'audio_model') and self.config.audio_model != '' else ''
        
        if 'mel' == audio_model_type or 'egemaps' == audio_model_type:
            self.mel_extractor = ResNetAudio(in_channels=1, out_channels=config.hidden_size, dropout=config.dropout)
        else:
            self.mel_extractor = None # 音声特徴抽出が不要な場合

        # 融合方法に応じたTransformerエンコーダー層の選択
        # 'gated' が fusion 設定に含まれる場合、GatedCrossAttentionFusionを使用
        if 'gated' in config.fusion:
            self.layers = nn.ModuleList([
                GatedCrossAttentionFusion(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                CrossAttentionEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])

        # 各Transformer層間のLayerNormalization
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
        else:
            self.attn_pooling = None # 他のプーリング方法の場合

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
        マルチレイヤークロスアテンションTransformerエンコーダーの順伝播。

        Args:
            features (tuple): 2つのテンソルを含むタプル `(src, memory)`。
                              `src` はクエリ（例: 音声埋め込み）、`memory` はキー/バリュー（例: テキスト埋め込み）です。
                              音声特徴抽出器を使用する場合、`src` または `memory` は生の特徴量となります。
            mask (torch.Tensor, optional): アテンションマスク。
            key_padding_mask (torch.BoolTensor, optional): キーパディングマスク。
        Returns:
            torch.Tensor: 分類器の出力（ロジット）。
        """
        src, memory = features

        audio_model_type = self.model_name.split('_')[1] if hasattr(self.config, 'audio_model') and self.config.audio_model != '' else ''

        # 音声特徴抽出（必要に応じて）
        # srcまたはmemoryのどちらか（configで指定された方）にResNetAudioを適用
        if self.mel_extractor is not None:
            if 'mel' == audio_model_type or 'egemaps' == audio_model_type:
                # ここでどちらに適用するかはモデルの設計によるため、今回はsrcに適用としています。
                # 実際のモデルではconfigなどで明示的に指定されることが多いです。
                src = self.mel_extractor(src)
            # elif 'some_other_audio_type' == audio_model_type:
            #     memory = self.mel_extractor(memory) # memoryが音声特徴の場合
        
        # Transformer層を順次適用
        for i, layer in enumerate(self.layers):
            # srcがmemoryにアテンションする形式
            src = layer(src, memory, src_mask=mask, src_key_padding_mask=key_padding_mask)
            # 各層の間にLayerNormalizationとDropoutを適用（最後の層を除く）
            if i < len(self.norm_layers):
                src = self.norm_layers[i](src)
                src = self.dropout(src)

        # プーリング戦略の適用
        if self.pooling == 'mean':
            src = src.mean(dim=1)  # 時間次元で平均を取るグローバルプーリング
        elif self.pooling == 'cls':
            src = src[:, 0, :]  # 最初のトークン（CLSトークン）を使用
        elif 'attn' in self.pooling and self.attn_pooling is not None:
            # アテンションプーリングまたはゲート付きアテンションプーリング
            src = self.attn_pooling(src, mask=key_padding_mask) # プーリングにはkey_padding_maskを使用
        
        # 最終分類器でクラスロジットを出力
        return self.classifier(src)


class BidirectionalCrossAttentionTransformerEncoder(nn.Module):
    """
    双方向クロスアテンションTransformerエンコーダー

    音声→テキスト、テキスト→音声の両方向でクロスアテンションを実行し、
    それぞれの融合結果をさらに結合して最終的な表現を生成します。
    """
    def __init__(self, config):
        """
        Args:
            config (object): モデル設定を含むオブジェクト。
                             CrossAttentionTransformerEncoderのconfigに加えて、
                             - fusion (str): 最終的な融合戦略（'concat', 'sum', 'mul', 'mean' など）
        """
        super(BidirectionalCrossAttentionTransformerEncoder, self).__init__()

        self.num_layers = config.n_layers
        self.model_name = config.model_name
        self.fusion = config.fusion
        self.config = config

        # 音声特徴抽出器の初期化
        audio_model_type = self.model_name.split('_')[1] if hasattr(self.config, 'audio_model') and self.config.audio_model != '' else ''
        if 'mel' == audio_model_type or 'egemaps' == audio_model_type:
            self.mel_extractor = ResNetAudio(in_channels=1, out_channels=config.hidden_size, dropout=config.dropout)
        else:
            self.mel_extractor = None

        # 双方向のクロスアテンション層の初期化
        # 'gated' が fusion 設定に含まれる場合、GatedCrossAttentionFusionを使用
        if 'gated' in config.fusion:
            # 1つ目の方向（例: 音声がテキストに注意）
            self.layers_1 = nn.ModuleList([
                GatedCrossAttentionFusion(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])
            # 2つ目の方向（例: テキストが音声に注意）
            self.layers_2 = nn.ModuleList([
                GatedCrossAttentionFusion(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])
        else:
            # 1つ目の方向（例: 音声がテキストに注意）
            self.layers_1 = nn.ModuleList([
                CrossAttentionEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])
            # 2つ目の方向（例: テキストが音声に注意）
            self.layers_2 = nn.ModuleList([
                CrossAttentionEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])

        # 各層間のLayerNormalization（各方向に対応）
        self.norm_layers_1 = nn.ModuleList([
            nn.LayerNorm(config.hidden_size) for _ in range(config.n_layers - 1)
        ])
        self.norm_layers_2 = nn.ModuleList([
            nn.LayerNorm(config.hidden_size) for _ in range(config.n_layers - 1)
        ])

        self.dropout = nn.Dropout(config.dropout)
        self.pooling = config.pooling

        # 最終的な分類器の入力サイズは、双方向融合の方法によって異なる
        # 'concat' の場合、隠れ層サイズが2倍になります。
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
        双方向クロスアテンションTransformerエンコーダーの順伝播。

        Args:
            features (tuple): `(src, memory)`。`src` と `memory` はそれぞれのモーダル特徴量。
            mask (torch.Tensor, optional): アテンションマスク。
            key_padding_mask (torch.BoolTensor, optional): キーパディングマスク。
        Returns:
            torch.Tensor: 分類器の出力（ロジット）。
        """
        src, memory = features

        audio_model_type = self.model_name.split('_')[1] if hasattr(self.config, 'audio_model') and self.config.audio_model != '' else ''

        # 音声特徴抽出（必要に応じて）
        if self.mel_extractor is not None:
            if 'mel' == audio_model_type or 'egemaps' == audio_model_type:
                # ここではsrcが音声、memoryがテキストという前提で進めます
                src = self.mel_extractor(src)
            # elif 'some_other_audio_type' == audio_model_type:
            #     memory = self.mel_extractor(memory)

        # 1. 音声→テキスト方向の処理 (src1 が memory1 にアテンション)
        src1 = src.clone() # 元のsrcをクエリとして使用
        memory1 = memory.clone() # 元のmemoryをキー/バリューとして使用
        for i, layer in enumerate(self.layers_1):
            src1 = layer(src1, memory1, src_mask=mask, src_key_padding_mask=key_padding_mask)
            if i < len(self.norm_layers_1):
                src1 = self.norm_layers_1[i](src1)
                src1 = self.dropout(src1)
        
        # 2. テキスト→音声方向の処理 (src2 が memory2 にアテンション)
        src2 = memory.clone() # 元のmemoryをクエリとして使用 (テキスト)
        memory2 = src.clone() # 元のsrcをキー/バリューとして使用 (音声)
        for i, layer in enumerate(self.layers_2):
            src2 = layer(src2, memory2, src_mask=mask, src_key_padding_mask=key_padding_mask)
            if i < len(self.norm_layers_2):
                src2 = self.norm_layers_2[i](src2)
                src2 = self.dropout(src2)

        # src1 と src2 を融合
        if 'concat' in self.fusion:
            # 連結融合: 隠れ次元で2つの出力を結合
            src = torch.cat((src1, src2), dim=2)  # (B, T, 2*d_model)
        elif 'sum' in self.fusion:
            # 加算融合
            src = src1 + src2
        elif 'mul' in self.fusion:
            # 要素ごとの乗算融合
            src = src1 * src2
        elif 'mean' in self.fusion:
            # 平均融合
            src = (src1 + src2) / 2
        else:
            # デフォルトは加算融合
            src = src1 + src2
            
        # プーリング戦略の適用
        if self.pooling == 'mean':
            src = src.mean(dim=1)  # 時間次元で平均
        elif self.pooling == 'cls':
            src = src[:, 0, :]  # 最初のトークン（CLS）

        # 最終分類器でクラスロジットを出力
        return self.classifier(src)


class ElementWiseFusionEncoder(nn.Module):
    """
    要素ごと融合エンコーダー

    音声とテキストの埋め込みを初期段階で要素ごとに（連結、加算など）融合し、
    その後、単一のTransformerエンコーダーで処理します。
    """
    def __init__(self, config):
        """
        Args:
            config (object): モデル設定を含むオブジェクト。
                             BidirectionalCrossAttentionTransformerEncoderのconfigに加えて、
                             - fusion (str): 初期融合戦略（'concat', 'mean', 'sum', 'mul', 'selfattn'）
        """
        super(ElementWiseFusionEncoder, self).__init__()

        self.model_name = config.model_name
        self.fusion = config.fusion
        self.config = config

        # 融合方法に応じてTransformerの隠れ層サイズを調整
        # 'concat' の場合、入力次元が2倍になるため、Transformerの d_model も2倍に設定
        hidden_size = config.hidden_size * 2 if self.fusion == 'concat' else config.hidden_size

        # 音声特徴抽出器の初期化
        audio_model_type = self.model_name.split('_')[1] if hasattr(self.config, 'audio_model') and self.config.audio_model != '' else ''
        if 'mel' == audio_model_type or 'egemaps' == audio_model_type:
            self.mel_extractor = ResNetAudio(in_channels=1, out_channels=config.hidden_size, dropout=config.dropout)
        else:
            self.mel_extractor = None
        
        # 標準のTransformerEncoderを初期化
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size, # 融合後の次元数に合わせる
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
            nn.LayerNorm(hidden_size), # 融合後の次元数に合わせる
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, config.hidden_mlp_size),
            nn.ReLU(),
            nn.Linear(config.hidden_mlp_size, config.num_classes)
        )
        
    def forward(self, features, mask=None, key_padding_mask=None):
        """
        要素ごと融合エンコーダーの順伝播。

        Args:
            features (tuple): `(src, memory)`。`src` と `memory` はそれぞれのモーダル特徴量。
            mask (torch.Tensor, optional): アテンションマスク (TransformerEncoderに渡される)。
            key_padding_mask (torch.BoolTensor, optional): キーパディングマスク (TransformerEncoderに渡される)。
        Returns:
            torch.Tensor: 分類器の出力（ロジット）。
        """
        src, memory = features

        audio_model_type = self.model_name.split('_')[1] if hasattr(self.config, 'audio_model') and self.config.audio_model != '' else ''

        # 音声特徴抽出（必要に応じて）
        if self.mel_extractor is not None:
            if 'mel' == audio_model_type or 'egemaps' == audio_model_type:
                src = self.mel_extractor(src)
            # elif 'some_other_audio_type' == audio_model_type:
            #     memory = self.mel_extractor(memory)

        # 要素ごとの融合方法
        if self.fusion == 'concat':
            # 連結融合: srcとmemoryを次元2（特徴量次元）で連結
            features = torch.cat((src, memory), dim=2)
        elif self.fusion == 'selfattn':
            # self-attentionベースの融合（ここでは単純な平均プーリングとスタック）
            # 注意: この'selfattn'融合は、通常の要素ごと融合とは異なり、
            # 各モーダルの時間軸をプールしてからスタックするため、
            # 後続のTransformerEncoderの入力形状に注意が必要です。
            # 通常の要素ごと融合は時間軸が揃っていることを前提とします。
            src = src.mean(dim=1)  # (B, D)
            memory = memory.mean(dim=1) # (B, D)
            features = torch.stack((src, memory), dim=1)  # (B, 2, D)
            # この場合、TransformerEncoderの入力は (B, 2, D) となり、
            # シーケンス長が2（音声とテキストの各表現）となる点に注意。
            # このため、key_padding_maskもそれに合わせて変更する必要があるかもしれません。
            # 実際には、より複雑なセルフアテンションメカニズムがここで使用されることがあります。
        elif self.fusion == 'mean':
            # 平均融合
            features = (src + memory) / 2
        elif self.fusion == 'sum':
            # 加算融合
            features = src + memory
        elif self.fusion == 'mul':
            # 要素ごとの乗算融合
            features = src * memory
        
        # 融合された特徴量をTransformerEncoderで処理
        features = self.encoder(features, src_key_padding_mask=key_padding_mask)
                
        # プーリング戦略の適用
        if self.pooling == 'mean':
            features = features.mean(dim=1)  # 時間次元で平均
        elif self.pooling == 'cls':
            features = features[:, 0, :]  # 最初のトークン（CLS）
        
        # 最終分類器でクラスロジットを出力
        return self.classifier(features)


class MyTransformerEncoder(nn.Module):
    """
    単一モーダル用Transformerエンコーダー

    音声またはテキストの単一モーダルデータのみを処理するモデルです。
    マルチモーダル融合は行わず、単一の入力シーケンスをTransformerでエンコードします。
    """
    def __init__(self, config):
        """
        Args:
            config (object): モデル設定を含むオブジェクト。
                             ElementWiseFusionEncoderのconfigと同様。
        """
        super(MyTransformerEncoder, self).__init__()

        self.model_name = config.model_name

        # 音声特徴抽出器の初期化
        if 'mel' in self.model_name or 'egemaps' in self.model_name:
            self.mel_extractor = ResNetAudio(in_channels=1, out_channels=config.hidden_size)
        else:
            self.mel_extractor = None
        
        # 標準のTransformerEncoderを初期化
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
        単一モーダルTransformerエンコーダーの順伝播。

        Args:
            features (torch.Tensor): 入力モーダル（音声またはテキスト）のテンソル。
                                     音声特徴抽出器を使用する場合、生の特徴量。
            mask (torch.Tensor, optional): アテンションマスク。
            key_padding_mask (torch.BoolTensor, optional): キーパディングマスク。
        Returns:
            torch.Tensor: 分類器の出力（ロジット）。
        """

        # 音声特徴抽出（必要に応じて）
        if self.mel_extractor is not None:
            features = self.mel_extractor(features)
        
        # TransformerEncoderで特徴量を処理
        features = self.encoder(features, src_key_padding_mask=key_padding_mask)
                
        # プーリング戦略の適用
        if self.pooling == 'mean':
            features = features.mean(dim=1)  # 時間次元で平均
        elif self.pooling == 'cls':
            features = features[:, 0, :]  # 最初のトークン（CLSトークン）を使用
        
        # 最終分類器でクラスロジットを出力
        return self.classifier(features)
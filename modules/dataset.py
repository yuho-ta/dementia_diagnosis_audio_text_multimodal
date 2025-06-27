# =============================
# ADReSSoデータセット用のPyTorchデータセットクラス
# - 音声・テキスト埋め込みデータの読み込み
# - マルチモーダル（音声+テキスト）対応
# - K-Fold交差検証用のデータ分割
# - データローダーの生成
# =============================

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import KFold

# データパス設定
root_text_path = os.path.join('dataset', 'diagnosis', 'train', 'text')    # テキスト埋め込みファイルのパス
root_audio_path = os.path.join('dataset', 'diagnosis', 'train', 'audio')  # 音声ファイルのパス

# ラベル情報（MMSEスコア）のCSVファイルパス
csv_labels_path = os.path.join('dataset', 'diagnosis', 'train', 'adresso-train-mmse-scores.csv')

# GPUが利用可能な場合はGPU、そうでなければCPUを使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length_wav2vec = 4000  # wav2vec2の最大長

# ADReSSoデータセット用のPyTorchデータセットクラス
class AdressoDataset(Dataset):
    def __init__(self, features, labels):
        """
        データセットの初期化
        Args:
            features: 特徴量（埋め込みベクトル）のリスト
            labels: ラベル（診断結果）のリスト
        """
        self.features = features
        self.labels = labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        """データセットのサイズを返す"""
        return len(self.labels)

    def __getitem__(self, idx):
        """指定されたインデックスのデータを返す"""
        return self.features[idx], self.labels[idx]

# テキストモデル名とファイル名のマッピング
name_mapping_text = {
    'bert': '',
    'distil': 'distil',
    'roberta': 'roberta',
    'mistral': 'mistral',
    'qwen': 'qwen',
    'stella': 'stella',
    'distilbert': ''
}

# 音声モデル名とファイル名のマッピング
name_mapping_audio = {
    'wav2vec2': 'audio',
    'egemaps': 'egemaps',
    'mel': 'mel'
}


def read_CSV(config):
    """
    CSVファイルからラベルと埋め込みデータを読み込む
    Args:
        config: 設定オブジェクト（モデル設定、ポーズ設定など）
    Returns:
        uids: ユーザーIDのリスト
        features: 特徴量（埋め込みベクトル）のリスト
        labels: ラベル（診断結果）のリスト
    """
    # ラベル情報のCSVを読み込み
    labels_pd = pd.read_csv(csv_labels_path)

    uids = []
    features = []
    labels = []

    # ポーズ情報の有無によるファイル名の変更
    pauses_data = '_pauses' if config.model.pauses else ''
    # 音声モデルによるファイル名の変更
    audio_data = '_' + name_mapping_audio[config.model.audio_model] if config.model.audio_model != '' else ''

    # 各行（各音声ファイル）を処理
    for index, row in labels_pd.iterrows():
        uids.append(row['adressfname'])
        # 診断結果を数値に変換（cn=0, ad=1）
        labels.append(torch.tensor(0 if row['dx'] == "cn" else 1).to(device).float())

        # テキスト埋め込みファイルのパスを構築
        if config.model.textual_model != '':
            text_embeddings_path = os.path.join(root_text_path, row['dx'], row['adressfname'] + 
                                                    name_mapping_text[config.model.textual_model] + pauses_data + '.pt')
        
        # 音声埋め込みファイルのパスを構築
        if config.model.audio_model != '':
            textual_data = name_mapping_text[config.model.textual_model] if config.model.textual_model != '' else 'distil'
            audio_embeddings_path = os.path.join(root_text_path, row['dx'], row['adressfname'] + textual_data 
                                                 + pauses_data + audio_data + '.pt')
        
        # マルチモーダル（音声+テキスト）の場合
        if config.model.multimodality:
            # 音声とテキストの埋め込みをタプルとして保存
            features.append((torch.load(audio_embeddings_path).to(device), torch.load(text_embeddings_path).to(device)))
        else:
            # 単一モーダルの場合
            if config.model.textual_model != '':
                # テキストのみ
                features.append(torch.load(text_embeddings_path).to(device))
            elif config.model.audio_model != '':
                # 音声のみ
                features.append(torch.load(audio_embeddings_path).to(device))

    return uids, features, labels


def get_dataloaders(config, kfold_number = 0):
    """
    K-Fold交差検証用のデータローダーを生成
    Args:
        config: 設定オブジェクト
        kfold_number: K-Foldの番号（0-4）
    Returns:
        train_dataloader: 訓練用データローダー
        validation_dataloader: 検証用データローダー
    """
    # CSVからデータを読み込み
    uids, features, labels = read_CSV(config)
    
    # K-Fold分割ファイルのディレクトリ
    splits_dir = os.path.join('dataset', 'diagnosis', 'train', 'splits')
    # 指定されたfoldの検証用UIDを読み込み
    validation_split = np.load(os.path.join(splits_dir, 'val_uids' + str(kfold_number) + '.npy'))

    batch_size = config.train.batch_size
    
    # 訓練用と検証用にデータを分割
    train_uids = []
    train_features = []
    train_labels = []

    validation_uids = []
    validation_features = []
    validation_labels = []

    # 各データを訓練用か検証用かに振り分け
    for i in range(len(uids)):
        if uids[i] in validation_split:
            # 検証用データ
            validation_uids.append(uids[i])
            validation_features.append(features[i])
            validation_labels.append(labels[i])
        else:
            # 訓練用データ
            train_uids.append(uids[i])
            train_features.append(features[i])
            train_labels.append(labels[i])

    # データセットとデータローダーを作成
    train_dataset = AdressoDataset(train_features, train_labels)
    validation_dataset = AdressoDataset(validation_features, validation_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, validation_dataloader

def set_splits():
    """
    K-Fold交差検証用のデータ分割を作成・保存
    - 5分割のK-Fold交差検証
    - 各foldの訓練用・検証用UIDをnumpyファイルとして保存
    """
    # ラベル情報のCSVを読み込み
    labels_pd = pd.read_csv(csv_labels_path)
    uids = []

    # 全UIDを取得
    for index, row in labels_pd.iterrows():
        uids.append(row['adressfname'])

    # 分割ファイル保存用ディレクトリを作成
    splits_dir = os.path.join('dataset', 'diagnosis', 'train', 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    # 5分割のK-Fold交差検証を実行
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # 各foldの訓練用・検証用インデックスを取得し、ファイルに保存
    for i, (train_index, test_index) in enumerate(kfold.split(uids)):
        print("TRAIN:", train_index, "TEST:", test_index)
        np.save(os.path.join(splits_dir, 'train_uids' + str(i)), np.array(uids)[train_index])
        np.save(os.path.join(splits_dir, 'val_uids' + str(i)), np.array(uids)[test_index])

def get_splits_stats():
    """
    各K-Fold分割の統計情報を表示
    - 各foldの訓練用・検証用データのクラス分布を確認
    """
    # ラベル情報のCSVを読み込み
    labels_pd = pd.read_csv(csv_labels_path)
    uids = []

    # 全UIDを取得
    for index, row in labels_pd.iterrows():
        uids.append(row['adressfname'])

    splits_dir = os.path.join('dataset', 'diagnosis', 'train', 'splits')

    # 各foldの統計情報を計算・表示
    for i in range(5):
        # 各foldの訓練用・検証用UIDを読み込み
        training_split = np.load(os.path.join(splits_dir, 'train_uids' + str(i) + '.npy'))
        validation_split = np.load(os.path.join(splits_dir, 'val_uids' + str(i) + '.npy'))
        
        # クラス別サンプル数をカウント
        n_cn_train = 0  # 訓練用の正常（CN）サンプル数
        n_ad_train = 0  # 訓練用のアルツハイマー（AD）サンプル数
        n_cn_val = 0    # 検証用の正常（CN）サンプル数
        n_ad_val = 0    # 検証用のアルツハイマー（AD）サンプル数

        # 訓練用データのクラス分布をカウント
        for uid in training_split:
            if labels_pd[labels_pd['adressfname'] == uid]['dx'].values[0] == 'cn':
                n_cn_train += 1
            else:
                n_ad_train += 1

        # 検証用データのクラス分布をカウント
        for uid in validation_split:
            if labels_pd[labels_pd['adressfname'] == uid]['dx'].values[0] == 'cn':
                n_cn_val += 1
            else:
                n_ad_val += 1

        # 統計情報を表示
        print(f"Fold {i}:")
        print(f"Training CN: {n_cn_train}, Training AD: {n_ad_train}")
        print(f"Validation CN: {n_cn_val}, Validation AD: {n_ad_val}")

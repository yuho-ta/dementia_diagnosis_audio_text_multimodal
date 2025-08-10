# =============================
# サイレンス特徴量用のデータセット処理スクリプト
# - サイレンス特徴量データの読み込み
# - K-Fold交差検証用のデータ分割
# - データローダーの生成
# =============================

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import StratifiedKFold

# データパス設定
root_silence_path = os.path.join('dataset', 'diagnosis', 'train', 'silence_features')  # サイレンス特徴量ファイルのパス
root_audio_path = os.path.join('dataset', 'diagnosis', 'train', 'audio')  # 音声ファイルのパス

# ラベル情報（診断結果）のCSVファイルパス
csv_labels_path = os.path.join('dataset', 'diagnosis', 'train', 'text_transcriptions.csv')

# GPUが利用可能な場合はGPU、そうでなければCPUを使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# サイレンス特徴量用のPyTorchデータセットクラス
class SilenceFeaturesDataset(Dataset):
    def __init__(self, features, labels, uids):
        """
        データセットの初期化
        Args:
            features: サイレンス特徴量のリスト
            labels: ラベル（診断結果）のリスト
            uids: ユーザーIDのリスト
        """
        self.features = features
        self.labels = labels
        self.uids = uids
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        """データセットのサイズを返す"""
        return len(self.labels)

    def __getitem__(self, idx):
        """指定されたインデックスのデータを返す"""
        return self.features[idx], self.labels[idx], idx, self.uids[idx]

def read_silence_features_CSV(config):
    """
    CSVファイルからラベルとサイレンス特徴量データを読み込む
    Args:
        config: 設定オブジェクト（音声モデル設定など）
    Returns:
        uids: ユーザーIDのリスト
        features: サイレンス特徴量のリスト
        labels: ラベル（診断結果）のリスト
    """
    uids = []
    features = []
    labels = []

    # 音声モデルによるファイル名の変更
    audio_data = '_' + config.model.audio_model if config.model.audio_model != '' else ''

    # CSVファイルを読み込み
    labels_pd = pd.read_csv(csv_labels_path)

    # 各行（各音声ファイル）を処理
    for index, row in labels_pd.iterrows():
        uids.append(row['uid'])
        # 診断結果を数値に変換（cn=0, ad=1）
        labels.append(torch.tensor(0 if row['diagno'] == "cn" else 1).to(device).float())

        # サイレンス特徴量ファイルのパスを構築
        silence_features_path = os.path.join(root_silence_path, row['diagno'], 
                                            row['uid'] + '_silence_only' + audio_data + '.pt')
        
        # サイレンス特徴量ファイルが存在するかチェック
        if os.path.exists(silence_features_path):
            features.append(torch.load(silence_features_path).to(device))
        else:
            print(f"Warning: Missing silence features file for {row['uid']}: {silence_features_path}")
            # 欠損データに対応するため、uidsとlabelsからもこのエントリを削除
            uids.pop()
            labels.pop()
            continue

    print(f"Loaded {len(features)} silence features files")
    return uids, features, labels

def get_silence_dataloaders(config, kfold_number=0):
    """
    K-Fold交差検証用のサイレンス特徴量データローダーを生成
    Args:
        config: 設定オブジェクト
        kfold_number: K-Foldの番号（0-4）
    Returns:
        train_dataloader: 訓練用データローダー
        validation_dataloader: 検証用データローダー
    """
    # CSVから訓練/検証データを読み込み
    uids, features, labels = read_silence_features_CSV(config)
    
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

    print(f"Training samples: {len(train_features)}")
    print(f"Validation samples: {len(validation_features)}")

    # データセットとデータローダーを作成
    train_dataset = SilenceFeaturesDataset(train_features, train_labels, train_uids)
    validation_dataset = SilenceFeaturesDataset(validation_features, validation_labels, validation_uids)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, validation_dataloader

def set_silence_splits():
    """
    サイレンス特徴量用のK-Fold交差検証分割を作成・保存
    - 5分割のStratified K-Fold交差検証（クラスバランスを保持）
    - 各foldの訓練用・検証用UIDをnumpyファイルとして保存
    - 実際に存在するサイレンス特徴量ファイルのみを使用
    """
    # ラベル情報のCSVを読み込み
    labels_pd = pd.read_csv(csv_labels_path)
    available_uids = []
    available_labels = []

    # 音声モデルによるファイル名の変更（簡易的な設定オブジェクトを作成）
    class Config:
        class Model:
            def __init__(self):
                self.audio_model = 'wav2vec2'
        def __init__(self):
            self.model = self.Model()
    
    config = Config()
    audio_data = '_' + config.model.audio_model if config.model.audio_model != '' else ''

    # 実際に存在するサイレンス特徴量ファイルのUIDとラベルのみを取得
    for index, row in labels_pd.iterrows():
        silence_features_path = os.path.join(root_silence_path, row['diagno'], 
                                            row['uid'] + '_silence_only' + audio_data + '.pt')
        
        if os.path.exists(silence_features_path):
            available_uids.append(row['uid'])
            # 診断結果を数値に変換（cn=0, ad=1）
            available_labels.append(0 if row['diagno'] == "cn" else 1)
        else:
            print(f"Skipping {row['uid']}: silence features file not found")

    print(f"Available files for splitting: {len(available_uids)} out of {len(labels_pd)}")

    # 分割ファイル保存用ディレクトリを作成
    splits_dir = os.path.join('dataset', 'diagnosis', 'train', 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    # 5分割のStratified K-Fold交差検証を実行
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 各foldの訓練用・検証用インデックスを取得し、ファイルに保存
    for i, (train_index, test_index) in enumerate(skf.split(available_uids, available_labels)):
        print(f"Fold {i}: TRAIN:", len(train_index), "TEST:", len(test_index))
        np.save(os.path.join(splits_dir, 'train_uids' + str(i)), np.array(available_uids)[train_index])
        np.save(os.path.join(splits_dir, 'val_uids' + str(i)), np.array(available_uids)[test_index])

def get_silence_splits_stats():
    """
    サイレンス特徴量用の各K-Fold分割の統計情報を表示
    - 各foldの訓練用・検証用データのクラス分布を確認
    """
    # ラベル情報のCSVを読み込み
    labels_pd = pd.read_csv(csv_labels_path)
    uids = []

    # 全UIDを取得
    for index, row in labels_pd.iterrows():
        uids.append(row['uid'])

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
            if labels_pd[labels_pd['uid'] == uid]['diagno'].values[0] == 'cn':
                n_cn_train += 1
            else:
                n_ad_train += 1

        # 検証用データのクラス分布をカウント
        for uid in validation_split:
            if labels_pd[labels_pd['uid'] == uid]['diagno'].values[0] == 'cn':
                n_cn_val += 1
            else:
                n_ad_val += 1

        # 統計情報を表示
        print(f"Silence Features - Fold {i}:")
        print(f"  Training CN: {n_cn_train}, Training AD: {n_ad_train}")
        print(f"  Validation CN: {n_cn_val}, Validation AD: {n_ad_val}")
        print(f"  Total Training: {n_cn_train + n_ad_train}, Total Validation: {n_cn_val + n_ad_val}")

def check_silence_features_availability():
    """
    サイレンス特徴量ファイルの存在確認
    - 各診断カテゴリごとのファイル数を表示
    - 欠損ファイルの確認
    """
    print("Checking silence features availability...")
    
    total_files = 0
    missing_files = 0
    
    for diagno in ['ad', 'cn']:
        silence_dir = os.path.join(root_silence_path, diagno)
        if os.path.exists(silence_dir):
            files = [f for f in os.listdir(silence_dir) if f.endswith('.pt')]
            print(f"{diagno.upper()}: {len(files)} silence features files")
            total_files += len(files)
        else:
            print(f"{diagno.upper()}: Directory not found")
            missing_files += 1
    
    print(f"Total silence features files: {total_files}")
    
    # CSVファイルと比較して欠損を確認
    labels_pd = pd.read_csv(csv_labels_path)
    csv_count = len(labels_pd)
    print(f"CSV entries: {csv_count}")
    print(f"Missing files: {csv_count - total_files}")
    
    return total_files, missing_files 
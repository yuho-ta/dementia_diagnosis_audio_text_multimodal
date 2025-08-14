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
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import re

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

def extract_par_id_from_cha_file(cha_file_path):
    """CHAファイルからPARのIDを抽出する"""
    try:
        with open(cha_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # @ID行でPARのIDを探す
        par_id_match = re.search(r'@ID:\s*eng\|Pitt\|PAR\|.*', content)
        if par_id_match:
            return par_id_match.group(0)
        return None
    except Exception as e:
        print(f"エラー: {cha_file_path} を読み込めませんでした: {e}")
        return None

def get_subject_id_from_cha_file(uid, diagno):
    """UIDと診断カテゴリからCHAファイルを探してPAR IDを抽出する"""
    # CHAファイルのパスを構築
    cha_file_path = os.path.join('dataset', 'diagnosis', 'train', 'segmentation', diagno, f"{uid}.cha")
    
    if os.path.exists(cha_file_path):
        par_id = extract_par_id_from_cha_file(cha_file_path)
        if par_id:
            # PAR IDから被験者ID部分を抽出（例：@ID: eng|Pitt|PAR|59;|male|ProbableAD||Participant|11|| → 59）
            par_id_match = re.search(r'@ID:\s*eng\|Pitt\|PAR\|(\d+);', par_id)
            if par_id_match:
                return par_id_match.group(1)
    
    # CHAファイルが見つからない場合やPAR IDが抽出できない場合は、ファイル名から抽出
    print(f"Warning: CHA file not found or PAR ID extraction failed for {uid}, falling back to filename extraction")
    return uid.split('-')[0] if '-' in uid else uid

def set_silence_splits():
    """
    サイレンス特徴量用のK-Fold交差検証分割を作成・保存（被験者IDベース）
    - 5分割のStratifiedGroupKFold交差検証（クラスバランスを保持、被験者IDベース分割）
    - 各foldの訓練用・検証用UIDをnumpyファイルとして保存
    - 実際に存在するサイレンス特徴量ファイルのみを使用
    - 同じ被験者のデータがtrain/testに混在しないように分割
    """
    # ラベル情報のCSVを読み込み
    labels_pd = pd.read_csv(csv_labels_path)
    available_uids = []
    available_labels = []
    available_subject_ids = []  # 被験者IDを追加

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
            uid = row['uid']
            diagno = row['diagno']
            available_uids.append(uid)
            # 診断結果を数値に変換（cn=0, ad=1）
            available_labels.append(0 if diagno == "cn" else 1)
            
            # CHAファイルから被験者IDを抽出
            subject_id = get_subject_id_from_cha_file(uid, diagno)
            available_subject_ids.append(subject_id)
        else:
            print(f"Skipping {row['uid']}: silence features file not found")

    print(f"Available files for splitting: {len(available_uids)} out of {len(labels_pd)}")

    # 分割ファイル保存用ディレクトリを作成
    splits_dir = os.path.join('dataset', 'diagnosis', 'train', 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    # 5分割のStratifiedGroupKFold交差検証を実行
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 各foldの訓練用・検証用インデックスを取得し、ファイルに保存
    for i, (train_index, test_index) in enumerate(sgkf.split(available_uids, available_labels, groups=available_subject_ids)):
        print(f"Fold {i}: TRAIN subjects: {len(set([available_subject_ids[j] for j in train_index]))}, TEST subjects: {len(set([available_subject_ids[j] for j in test_index]))}")
        print(f"Fold {i}: TRAIN samples: {len(train_index)}, TEST samples: {len(test_index)}")
        
        # 被験者IDの重複チェック
        train_subjects = set([available_subject_ids[j] for j in train_index])
        test_subjects = set([available_subject_ids[j] for j in test_index])
        overlap = train_subjects.intersection(test_subjects)
        
        if overlap:
            print(f"WARNING: Fold {i} has overlapping subjects: {overlap}")
        else:
            print(f"Fold {i}: No overlapping subjects ✓")
        
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
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
from sklearn.model_selection import StratifiedKFold

# データパス設定
root_text_path = os.path.join('dataset', 'diagnosis', 'train', 'text')     # テキスト埋め込みファイルのパス
root_audio_path = os.path.join('dataset', 'diagnosis', 'train', 'audio')  # 音声ファイルのパス

# ラベル情報（MMSEスコア）のCSVファイルパス
csv_labels_path = os.path.join('dataset', 'diagnosis', 'train', 'adresso-train-mmse-scores.csv')
# [変更点] テストデータ用のCSVファイルパスを追加
csv_test_labels_path = os.path.join('dataset', 'diagnosis', 'test', 'adresso-test-mmse-scores.csv')


# GPUが利用可能な場合はGPU、そうでなければCPUを使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length_wav2vec = 4000  # wav2vec2の最大長

# ADReSSoデータセット用のPyTorchデータセットクラス
class AdressoDataset(Dataset):
    def __init__(self, features, labels, uids):
        """
        データセットの初期化
        Args:
            features: 特徴量（埋め込みベクトル）のリスト
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


# [変更点] read_CSV関数に is_test パラメータを追加し、テスト時のラベル処理を修正
def read_CSV(config, is_test=False):
    """
    CSVファイルからラベルと埋め込みデータを読み込む
    Args:
        config: 設定オブジェクト（モデル設定、ポーズ設定など）
        is_test (bool): テストデータを読み込む場合はTrue。デフォルトはFalse。
    Returns:
        uids: ユーザーIDのリスト
        features: 特徴量（埋め込みベクトル）のリスト
        labels: ラベル（診断結果）のリスト
    """
    uids = []
    features = []
    labels = []

    # [変更点] テストデータの場合、CSVファイルが存在しない可能性を考慮
    current_csv_path = csv_test_labels_path if is_test else csv_labels_path
    
    if is_test and not os.path.exists(current_csv_path):
        print(f"Warning: Test label CSV file not found at {current_csv_path}. Proceeding without labels for test set.")
        
        # テストデータのルートパス
        root_data_path = os.path.join('dataset', 'diagnosis', 'test')
        # テストセットのデータは通常、cn/adのサブディレクトリに分かれて格納されていると仮定
        # これは、診断結果（dx）情報がファイルパスに含まれることを前提としている
        for dx_type in ['cn', 'ad']: 
            # テキスト埋め込みファイルのディレクトリパス
            if config.model.textual_model != '':
                dx_text_path = os.path.join(root_data_path, 'text', dx_type)
                if os.path.exists(dx_text_path):
                    for filename in os.listdir(dx_text_path):
                        if filename.endswith('.pt'):
                            # ファイル名から uid を抽出するロジック (例: AD_001_distil_pauses.pt -> AD_001)
                            # `adressfname` がファイル名全体ではない場合、適宜調整が必要
                            # ここではファイル名の先頭部分をuidとして仮定
                            uid_raw = os.path.splitext(filename)[0]
                            # model.pyのname_mapping_text, audio_data, pauses_dataを考慮してuidを生成する必要がある
                            # シンプルにファイル名から最初の部分をUIDとして抽出する例
                            uid_parts = uid_raw.split('_')
                            if len(uid_parts) >= 2: # AD_001_... の形式を想定
                                actual_uid = f"{uid_parts[0]}_{uid_parts[1]}"
                            else: # ファイル名が短い場合や異なる形式の場合
                                actual_uid = uid_raw # 完全なファイル名をUIDとして使用

                            # 重複を避けるため、既に追加されていないかチェック
                            if actual_uid not in uids:
                                uids.append(actual_uid)
                                # テストセットにはMMSEラベルがないので、ダミー値 -1 を設定
                                labels.append(torch.tensor(-1).to(device).float()) 
                else:
                    print(f"Warning: Text data directory for {dx_type} not found at {dx_text_path}.")

            # 音声埋め込みファイルのディレクトリパス
            if config.model.audio_model != '':
                dx_audio_path = os.path.join(root_data_path, 'text', dx_type)
                if os.path.exists(dx_audio_path):
                    for filename in os.listdir(dx_audio_path):
                        if filename.endswith('.pt'):
                            uid_raw = os.path.splitext(filename)[0]
                            uid_parts = uid_raw.split('_')
                            if len(uid_parts) >= 2:
                                actual_uid = f"{uid_parts[0]}_{uid_parts[1]}"
                            else:
                                actual_uid = uid_raw
                            
                            if actual_uid not in uids:
                                uids.append(actual_uid)
                                labels.append(torch.tensor(-1).to(device).float())
                else:
                    print(f"Warning: Audio data directory for {dx_type} not found at {dx_audio_path}.")

        # 実際にロードできる特徴量ファイルを収集
        temp_features = []
        temp_labels = [] # ここでラベルを管理する必要がある
        temp_uids = []

        # uids リストが構築された後に、各UIDに対応する特徴量ファイルをロード
        for uid_idx, current_uid in enumerate(uids):
            # 診断タイプをUIDから推測（例: "AD_001" -> "ad", "CN_001" -> "cn"）
            dx_type_from_uid = 'ad' if current_uid.startswith('AD') else 'cn'
            
            pauses_data = '_pauses' if config.model.pauses else ''
            audio_data = '_' + name_mapping_audio[config.model.audio_model] if config.model.audio_model != '' else ''
            textual_data = name_mapping_text[config.model.textual_model] if config.model.textual_model != '' else 'distil' # distilはデフォルト的なもの

            text_embeddings_path = os.path.join(root_data_path, 'text', dx_type_from_uid, 
                                                current_uid + name_mapping_text[config.model.textual_model] + pauses_data + '.pt')
            audio_embeddings_path = os.path.join(root_data_path, 'text', dx_type_from_uid, 
                                                 current_uid + textual_data + pauses_data + audio_data + '.pt')

            feature_loaded = False
            if config.model.multimodality:
                if os.path.exists(audio_embeddings_path) and os.path.exists(text_embeddings_path):
                    temp_features.append((torch.load(audio_embeddings_path).to(device), torch.load(text_embeddings_path).to(device)))
                    feature_loaded = True
            else:
                if config.model.textual_model != '':
                    if os.path.exists(text_embeddings_path):
                        temp_features.append(torch.load(text_embeddings_path).to(device))
                        feature_loaded = True
                elif config.model.audio_model != '':
                    if os.path.exists(audio_embeddings_path):
                        temp_features.append(torch.load(audio_embeddings_path).to(device))
                        feature_loaded = True
            
            if feature_loaded:
                temp_labels.append(labels[uid_idx]) # ダミーラベル
                temp_uids.append(current_uid)
            else:
                print(f"Warning: Missing feature files for {current_uid} ({dx_type_from_uid}). Skipping.")
        
        features = temp_features
        labels = temp_labels
        uids = temp_uids

    else:
        # 訓練データ、またはテストデータにCSVが存在する場合
        labels_pd = pd.read_csv(current_csv_path)

        # ポーズ情報の有無によるファイル名の変更
        pauses_data = '_pauses' if config.model.pauses else ''
        # 音声モデルによるファイル名の変更
        audio_data = '_' + name_mapping_audio[config.model.audio_model] if config.model.audio_model != '' else ''

        # 各行（各音声ファイル）を処理
        for index, row in labels_pd.iterrows():
            uids.append(row['adressfname'])
            # 診断結果を数値に変換（cn=0, ad=1）
            labels.append(torch.tensor(0 if row['dx'] == "cn" else 1).to(device).float())

            # [変更点] データパスを訓練用とテスト用で分ける
            root_data_path = os.path.join('dataset', 'diagnosis', 'test' if is_test else 'train')

            # テキスト埋め込みファイルのパスを構築
            text_embeddings_path = None
            if config.model.textual_model != '':
                text_embeddings_path = os.path.join(root_data_path, 'text', row['dx'], row['adressfname'] + 
                                                    name_mapping_text[config.model.textual_model] + pauses_data + '.pt')
            
            # 音声埋め込みファイルのパスを構築
            audio_embeddings_path = None
            if config.model.audio_model != '':
                textual_data_for_audio_path = name_mapping_text[config.model.textual_model] if config.model.textual_model != '' else 'distil'
                audio_embeddings_path = os.path.join(root_data_path, 'text', row['dx'], row['adressfname']  + pauses_data + audio_data + '.pt')
            
            # マルチモーダル（音声+テキスト）の場合
            if config.model.multimodality:
                # 音声とテキストの埋め込みをタプルとして保存
                if text_embeddings_path and audio_embeddings_path and \
                   os.path.exists(audio_embeddings_path) and os.path.exists(text_embeddings_path):
                    features.append((torch.load(audio_embeddings_path).to(device), torch.load(text_embeddings_path).to(device)))
                else:
                    print(audio_embeddings_path)
                    print(text_embeddings_path)
                    print(f"Warning: Missing embedding files for {row['adressfname']} in multimodality. Skipping.")
                    # 欠損データに対応するため、uidsとlabelsからもこのエントリを削除する必要がある
                    uids.pop()
                    labels.pop()
                    continue
            else:
                # 単一モーダルの場合
                if config.model.textual_model != '':
                    # テキストのみ
                    if text_embeddings_path and os.path.exists(text_embeddings_path):
                        features.append(torch.load(text_embeddings_path).to(device))
                    else:
                        print(f"Warning: Missing text embedding file for {row['adressfname']}. Skipping.")
                        uids.pop()
                        labels.pop()
                        continue
                elif config.model.audio_model != '':
                    # 音声のみ
                    if audio_embeddings_path and os.path.exists(audio_embeddings_path):
                        features.append(torch.load(audio_embeddings_path).to(device))
                    else:
                        print(audio_embeddings_path)
                        print(f"Warning: Missing audio embedding file for {row['adressfname']}. Skipping.")
                        uids.pop()
                        labels.pop()
                        continue
    return uids, features, labels


# [変更点] get_dataloaders関数に return_test_dataloader パラメータを追加
def get_dataloaders(config, kfold_number = 0, return_test_dataloader=False):
    """
    K-Fold交差検証用のデータローダーを生成
    Args:
        config: 設定オブジェクト
        kfold_number: K-Foldの番号（0-4）
        return_test_dataloader (bool): テストデータローダーも返す場合はTrue。デフォルトはFalse。
    Returns:
        train_dataloader: 訓練用データローダー
        validation_dataloader: 検証用データローダー
        test_dataloader (optional): テストデータローダー（return_test_dataloaderがTrueの場合のみ）
    """
    # CSVから訓練/検証データを読み込み
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
    train_dataset = AdressoDataset(train_features, train_labels, train_uids)
    validation_dataset = AdressoDataset(validation_features, validation_labels, validation_uids)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    if return_test_dataloader:
        # テストデータを読み込み
        test_uids, test_features, test_labels = read_CSV(config, is_test=True)
        test_dataset = AdressoDataset(test_features, test_labels, test_uids)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader, validation_dataloader, test_dataloader
    else:
        return train_dataloader, validation_dataloader

def set_splits():
    """
    K-Fold交差検証用のデータ分割を作成・保存
    - 5分割のStratified K-Fold交差検証（クラスバランスを保持）
    - 各foldの訓練用・検証用UIDをnumpyファイルとして保存
    """
    # ラベル情報のCSVを読み込み
    labels_pd = pd.read_csv(csv_labels_path)
    uids = []
    labels = []

    # 全UIDとラベルを取得
    for index, row in labels_pd.iterrows():
        uids.append(row['adressfname'])
        # 診断結果を数値に変換（cn=0, ad=1）
        labels.append(0 if row['dx'] == "cn" else 1)

    # 分割ファイル保存用ディレクトリを作成
    splits_dir = os.path.join('dataset', 'diagnosis', 'train', 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    # 5分割のStratified K-Fold交差検証を実行
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 各foldの訓練用・検証用インデックスを取得し、ファイルに保存
    for i, (train_index, test_index) in enumerate(skf.split(uids, labels)):
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
#!/usr/bin/env python3
# =============================
# ノイズ追加分類器用データセットクラス
# - NoiseAugmentedDataset
# - CombinedNoiseDataset
# =============================

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import logging
import json
from noise_augmented_utils import extract_subject_id_from_filename, get_subject_id_from_cha_file

# 診断カテゴリ
diagnosis = ['ad', 'cn']
label_mapping = {'ad': 1, 'cn': 0}

class SubjectSplitter:
    """被験者IDの分割を管理するクラス（再現性保証）"""
    
    def __init__(self, cache_file="subject_splits_common.json"):
        self.cache_file = cache_file
        # 再現性のためのseed固定
        np.random.seed(42)
    
    def load_or_create_splits(self, subject_list, label_list):
        """分割結果をキャッシュから読み込むか、新しく生成する"""
        # キャッシュファイルが存在する場合は読み込む
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cached_splits = json.load(f)
                
                # キャッシュされた被験者IDが現在のデータと一致するかチェック
                cached_subjects = set(cached_splits['train'] + cached_splits['validation'] + cached_splits['test'])
                current_subjects = set(subject_list)
                
                if cached_subjects == current_subjects:
                    logging.info(f"Loading subject splits from cache: {self.cache_file}")
                    return cached_splits['train'], cached_splits['validation'], cached_splits['test']
                else:
                    logging.warning(f"Cache mismatch. Regenerating splits. Cached: {len(cached_subjects)}, Current: {len(current_subjects)}")
            except Exception as e:
                logging.warning(f"Failed to load cache file: {e}")
        
        # 新しい分割を生成
        logging.info("Generating new subject splits")
        
        # まず、train+validation と test に分割（80% vs 20%）
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        train_val_subjects, test_subjects = next(sgkf_outer.split(subject_list, label_list, groups=subject_list))
        
        # 次に、train+validation を train と validation に分割（75% vs 25%）
        train_val_subject_list = [subject_list[i] for i in train_val_subjects]
        train_val_label_list = [label_list[i] for i in train_val_subjects]
        
        sgkf_inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
        train_subjects, val_subjects = next(sgkf_inner.split(train_val_subject_list, train_val_label_list, groups=train_val_subject_list))
        
        train_subject_ids = [train_val_subject_list[i] for i in train_subjects]
        val_subject_ids = [train_val_subject_list[i] for i in val_subjects]
        test_subject_ids = [subject_list[i] for i in test_subjects]
        
        # 分割結果をキャッシュに保存
        splits_data = {
            'train': train_subject_ids,
            'validation': val_subject_ids,
            'test': test_subject_ids,
            'metadata': {
                'total_subjects': len(subject_list),
                'train_count': len(train_subject_ids),
                'validation_count': len(val_subject_ids),
                'test_count': len(test_subject_ids),
                'random_state': 42
            }
        }
        
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(splits_data, f, indent=2)
            logging.info(f"Saved subject splits to cache: {self.cache_file}")
        except Exception as e:
            logging.warning(f"Failed to save cache file: {e}")
        
        return train_subject_ids, val_subject_ids, test_subject_ids

class NoiseAugmentedDataset(Dataset):
    """ノイズ追加付き特徴量データセット（被験者IDベース3分割対応）"""
    
    def __init__(self, features_path, train_noise_types='original', val_noise_types='original', test_noise_type='original', n_splits=5, splitter=None):
        # 再現性のためのseed固定
        np.random.seed(42)
        
        self.features_path = features_path
        self.train_noise_types = train_noise_types
        self.val_noise_types = val_noise_types
        self.test_noise_type = test_noise_type
        self.splitter = splitter or SubjectSplitter()
        self.data = []
        self.labels = []
        self.subject_ids = []
        self.data_types = []  # 'train', 'validation', 'test' を記録
        self.fold_indices = []  # クロスバリデーション用のインデックス
        
        # まず、すべての被験者IDとそのファイルを収集
        all_subject_files = {}  # {subject_id: {diagno: [file_paths]}}
        all_subject_labels = {}  # {subject_id: label}
        
        for diagno in diagnosis:
            diagno_path = os.path.join(features_path, diagno)
            if not os.path.exists(diagno_path):
                continue
                
            label = label_mapping[diagno]
            
            # すべてのノイズタイプのファイルを取得（後で分割時に適切なノイズタイプを選択）
            for file in os.listdir(diagno_path):
            if file.endswith('.pt'):  # すべてのノイズタイプ
                file_path = os.path.join(diagno_path, file)
                # ファイル名からUIDを抽出
                uid = extract_subject_id_from_filename(file)
                
                if uid:  # UIDが抽出できた場合のみ追加
                    # CHAファイルから被験者IDを抽出
                    subject_id = get_subject_id_from_cha_file(uid, diagno)
                    
                    if subject_id not in all_subject_files:
                        all_subject_files[subject_id] = {}
                        all_subject_labels[subject_id] = label
                    
                    if diagno not in all_subject_files[subject_id]:
                        all_subject_files[subject_id][diagno] = []
                    
                    all_subject_files[subject_id][diagno].append(file_path)
        
        # 被験者IDのリストを作成（診断カテゴリごとにグループ化）
        subject_ids_by_label = {}
        for subject_id, label in all_subject_labels.items():
            if label not in subject_ids_by_label:
                subject_ids_by_label[label] = []
            subject_ids_by_label[label].append(subject_id)
        
        # 被験者IDとラベルのリストを作成
        subject_list = []
        label_list = []
        for label, subjects in subject_ids_by_label.items():
            subject_list.extend(subjects)
            label_list.extend([label] * len(subjects))
        
        # 共通の分割ロジックを使用
        train_subject_ids, val_subject_ids, test_subject_ids = self.splitter.load_or_create_splits(
            subject_list, label_list
        )
        
        logging.info(f"Subject-level 3-way split: {len(train_subject_ids)} train, {len(val_subject_ids)} validation, {len(test_subject_ids)} test subjects")
        
        # 分割結果をログに出力
        logging.info(f"Train subjects: {sorted(train_subject_ids)}")
        logging.info(f"Validation subjects: {sorted(val_subject_ids)}")
        logging.info(f"Test subjects: {sorted(test_subject_ids)}")
        
        # トレインデータを構築（指定されたノイズタイプ）
        for subject_id in train_subject_ids:
            if subject_id in all_subject_files:
                for diagno, file_paths in all_subject_files[subject_id].items():
                    label = all_subject_labels[subject_id]
                    
                    # 指定されたノイズタイプのファイルのみを追加
                    for file_path in file_paths:
                        file_name = os.path.basename(file_path)
                        # train_noise_typesがリストの場合は複数のノイズタイプを許可
                        if isinstance(self.train_noise_types, list):
                            for noise_type in self.train_noise_types:
                                if file_name.endswith(f'_{noise_type}.pt'):
                                    self.data.append(file_path)
                                    self.labels.append(label)
                                    self.subject_ids.append(subject_id)
                                    self.data_types.append('train')
                                    break  # 1つのファイルが複数のノイズタイプにマッチしないように
                        else:
                            # 単一のノイズタイプの場合
                            if file_name.endswith(f'_{self.train_noise_types}.pt'):
                                self.data.append(file_path)
                                self.labels.append(label)
                                self.subject_ids.append(subject_id)
                                self.data_types.append('train')
        
        # バリデーションデータを構築（指定されたノイズタイプ）
        for subject_id in val_subject_ids:
            if subject_id in all_subject_files:
                for diagno, file_paths in all_subject_files[subject_id].items():
                    label = all_subject_labels[subject_id]
                    
                    # 指定されたノイズタイプのファイルのみを追加
                    for file_path in file_paths:
                        file_name = os.path.basename(file_path)
                        # val_noise_typesがリストの場合は複数のノイズタイプを許可
                        if isinstance(self.val_noise_types, list):
                            for noise_type in self.val_noise_types:
                                if file_name.endswith(f'_{noise_type}.pt'):
                                    self.data.append(file_path)
                                    self.labels.append(label)
                                    self.subject_ids.append(subject_id)
                                    self.data_types.append('validation')
                                    break  # 1つのファイルが複数のノイズタイプにマッチしないように
                        else:
                            # 単一のノイズタイプの場合
                            if file_name.endswith(f'_{self.val_noise_types}.pt'):
                                self.data.append(file_path)
                                self.labels.append(label)
                                self.subject_ids.append(subject_id)
                                self.data_types.append('validation')
        
        # テストデータを構築（指定されたノイズタイプ）
        for subject_id in test_subject_ids:
            if subject_id in all_subject_files:
                for diagno, file_paths in all_subject_files[subject_id].items():
                    label = all_subject_labels[subject_id]
                    
                    # 指定されたノイズタイプのファイルのみを追加
                    for file_path in file_paths:
                        file_name = os.path.basename(file_path)
                        if file_name.endswith(f'_{self.test_noise_type}.pt'):
                            self.data.append(file_path)
                            self.labels.append(label)
                            self.subject_ids.append(subject_id)
                            self.data_types.append('test')
        
        # クロスバリデーション用のインデックスを事前に生成（trainデータのみで分割）
        self._generate_cv_indices(n_splits)
        
        # ノイズタイプの情報をログに出力
        train_noise_str = str(self.train_noise_types) if isinstance(self.train_noise_types, list) else self.train_noise_types
        val_noise_str = str(self.val_noise_types) if isinstance(self.val_noise_types, list) else self.val_noise_types
        logging.info(f"Loaded {len([d for d in self.data_types if d == 'train'])} train, {len([d for d in self.data_types if d == 'validation'])} validation, {len([d for d in self.data_types if d == 'test'])} test samples")
        logging.info(f"Train noise types: {train_noise_str}, Validation noise types: {val_noise_str}, Test noise type: {self.test_noise_type}")
        
        # クラス別サンプル数を計算
        train_class_counts = {}
        val_class_counts = {}
        test_class_counts = {}
        for i, (label, data_type) in enumerate(zip(self.labels, self.data_types)):
            if data_type == 'train':
                train_class_counts[label] = train_class_counts.get(label, 0) + 1
            elif data_type == 'validation':
                val_class_counts[label] = val_class_counts.get(label, 0) + 1
            else:  # test
                test_class_counts[label] = test_class_counts.get(label, 0) + 1
        
        logging.info(f"Train class distribution: {train_class_counts}")
        logging.info(f"Validation class distribution: {val_class_counts}")
        logging.info(f"Test class distribution: {test_class_counts}")
        
        # 被験者IDの統計情報
        train_subjects = set([self.subject_ids[i] for i, dt in enumerate(self.data_types) if dt == 'train'])
        val_subjects = set([self.subject_ids[i] for i, dt in enumerate(self.data_types) if dt == 'validation'])
        test_subjects = set([self.subject_ids[i] for i, dt in enumerate(self.data_types) if dt == 'test'])
        
        # 重複チェック
        overlap_train_val = train_subjects.intersection(val_subjects)
        overlap_train_test = train_subjects.intersection(test_subjects)
        overlap_val_test = val_subjects.intersection(test_subjects)
        
        logging.info(f"Train subjects: {len(train_subjects)}, Validation subjects: {len(val_subjects)}, Test subjects: {len(test_subjects)}")
        
        if overlap_train_val or overlap_train_test or overlap_val_test:
            logging.warning(f"Found overlapping subjects: train-val={overlap_train_val}, train-test={overlap_train_test}, val-test={overlap_val_test}")
        else:
            logging.info("No overlapping subjects between any sets ✓")
        
        # 被験者ごとのサンプル数分布
        subject_counts = {}
        for subject_id in self.subject_ids:
            subject_counts[subject_id] = subject_counts.get(subject_id, 0) + 1
        
        logging.info(f"Subject sample distribution: min={min(subject_counts.values())}, max={max(subject_counts.values())}, mean={np.mean(list(subject_counts.values())):.2f}")
    
    def _generate_cv_indices(self, n_splits):
        """クロスバリデーション用のインデックスを生成（trainデータのみで分割）"""
        # トレインデータのインデックスを取得
        train_indices = [i for i, dt in enumerate(self.data_types) if dt == 'train']
        train_subject_ids = [self.subject_ids[i] for i in train_indices]
        train_labels = [self.labels[i] for i in train_indices]
        
        # 被験者IDベースのクロスバリデーション（trainデータのみで分割）
        from sklearn.model_selection import StratifiedGroupKFold
        # 再現性のためのseed固定
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        self.fold_indices = []
        for train_fold_idx, val_fold_idx in sgkf.split(train_indices, train_labels, groups=train_subject_ids):
            # 実際のデータインデックスに変換
            actual_train_idx = [train_indices[i] for i in train_fold_idx]
            actual_val_idx = [train_indices[i] for i in val_fold_idx]
            self.fold_indices.append((actual_train_idx, actual_val_idx))
    
    def get_fold_indices(self, fold):
        """指定されたフォールドのtrain/validationインデックスを取得"""
        if fold >= len(self.fold_indices):
            raise ValueError(f"Fold {fold} does not exist. Available folds: 0-{len(self.fold_indices)-1}")
        return self.fold_indices[fold]
    
    def get_test_indices(self):
        """テストデータのインデックスを取得"""
        return [i for i, dt in enumerate(self.data_types) if dt == 'test']
    
    def get_validation_indices(self):
        """バリデーションデータのインデックスを取得"""
        return [i for i, dt in enumerate(self.data_types) if dt == 'validation']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 特徴量を読み込み（元の長さのまま）
        features = torch.load(self.data[idx])
        
        # マスクを作成（ゼロパディング部分を検出）
        mask = (features.sum(dim=-1) == 0)  # (time_steps,)
        
        return features.float(), self.labels[idx], mask

class CombinedNoiseDataset(Dataset):
    """ノイズあり・なしを組み合わせた特徴量データセット（被験者IDベース3分割対応）"""
    
    def __init__(self, features_path, train_noise_types=['original', 'gaussian_noise_light'], test_noise_type='original', n_splits=5, splitter=None):
        # 再現性のためのseed固定
        np.random.seed(42)
        
        self.features_path = features_path
        self.train_noise_types = train_noise_types
        self.test_noise_type = test_noise_type
        self.splitter = splitter or SubjectSplitter()
        self.data = []
        self.labels = []
        self.subject_ids = []
        self.data_types = []  # 'train', 'validation', 'test' を記録
        self.fold_indices = []  # クロスバリデーション用のインデックス
        
        # まず、すべての被験者IDとそのファイルを収集
        all_subject_files = {}  # {subject_id: {diagno: [file_paths]}}
        all_subject_labels = {}  # {subject_id: label}
        
        for diagno in diagnosis:
            diagno_path = os.path.join(features_path, diagno)
            if not os.path.exists(diagno_path):
                continue
                
            label = label_mapping[diagno]
            
            # すべてのノイズタイプのファイルを取得
            for file in os.listdir(diagno_path):
                if file.endswith('.pt'):  # すべてのノイズタイプ
                    file_path = os.path.join(diagno_path, file)
                    uid = extract_subject_id_from_filename(file)
                    
                    if uid:
                        subject_id = get_subject_id_from_cha_file(uid, diagno)
                        
                        if subject_id not in all_subject_files:
                            all_subject_files[subject_id] = {}
                            all_subject_labels[subject_id] = label
                        
                        if diagno not in all_subject_files[subject_id]:
                            all_subject_files[subject_id][diagno] = []
                        
                        all_subject_files[subject_id][diagno].append(file_path)
        
        # 被験者IDのリストを作成（診断カテゴリごとにグループ化）
        subject_ids_by_label = {}
        for subject_id, label in all_subject_labels.items():
            if label not in subject_ids_by_label:
                subject_ids_by_label[label] = []
            subject_ids_by_label[label].append(subject_id)
        
        # 被験者IDとラベルのリストを作成
        subject_list = []
        label_list = []
        for label, subjects in subject_ids_by_label.items():
            subject_list.extend(subjects)
            label_list.extend([label] * len(subjects))
        
        # 共通の分割ロジックを使用
        train_subject_ids, val_subject_ids, test_subject_ids = self.splitter.load_or_create_splits(
            subject_list, label_list
        )
        
        logging.info(f"Subject-level 3-way split: {len(train_subject_ids)} train, {len(val_subject_ids)} validation, {len(test_subject_ids)} test subjects")
        
        # 分割結果をログに出力
        logging.info(f"Train subjects: {sorted(train_subject_ids)}")
        logging.info(f"Validation subjects: {sorted(val_subject_ids)}")
        logging.info(f"Test subjects: {sorted(test_subject_ids)}")
        
        # トレインデータを構築（複数のノイズタイプ）
        for subject_id in train_subject_ids:
            if subject_id in all_subject_files:
                for diagno, file_paths in all_subject_files[subject_id].items():
                    label = all_subject_labels[subject_id]
                    
                    # 指定されたノイズタイプのファイルのみを追加
                    for file_path in file_paths:
                        file_name = os.path.basename(file_path)
                        for noise_type in train_noise_types:
                            if file_name.endswith(f'_{noise_type}.pt'):
                                self.data.append(file_path)
                                self.labels.append(label)
                                self.subject_ids.append(subject_id)
                                self.data_types.append('train')
                                break  # 1つのファイルが複数のノイズタイプにマッチしないように
        
        # バリデーションデータを構築（複数のノイズタイプ）
        for subject_id in val_subject_ids:
            if subject_id in all_subject_files:
                for diagno, file_paths in all_subject_files[subject_id].items():
                    label = all_subject_labels[subject_id]
                    
                    # 指定されたノイズタイプのファイルのみを追加
                    for file_path in file_paths:
                        file_name = os.path.basename(file_path)
                        for noise_type in train_noise_types:
                            if file_name.endswith(f'_{noise_type}.pt'):
                                self.data.append(file_path)
                                self.labels.append(label)
                                self.subject_ids.append(subject_id)
                                self.data_types.append('validation')
                                break  # 1つのファイルが複数のノイズタイプにマッチしないように
        
        # テストデータを構築（単一のノイズタイプ）
        for subject_id in test_subject_ids:
            if subject_id in all_subject_files:
                for diagno, file_paths in all_subject_files[subject_id].items():
                    label = all_subject_labels[subject_id]
                    
                    # 指定されたノイズタイプのファイルのみを追加
                    for file_path in file_paths:
                        file_name = os.path.basename(file_path)
                        if file_name.endswith(f'_{test_noise_type}.pt'):
                            self.data.append(file_path)
                            self.labels.append(label)
                            self.subject_ids.append(subject_id)
                            self.data_types.append('test')
                            break  # 1つのファイルが複数のノイズタイプにマッチしないように
        
        # クロスバリデーション用のインデックスを事前に生成（trainデータのみで分割）
        self._generate_cv_indices(n_splits)
        
        logging.info(f"Loaded {len([d for d in self.data_types if d == 'train'])} train, {len([d for d in self.data_types if d == 'validation'])} validation, {len([d for d in self.data_types if d == 'test'])} test samples")
        
        # クラス別サンプル数を計算
        train_class_counts = {}
        val_class_counts = {}
        test_class_counts = {}
        for i, (label, data_type) in enumerate(zip(self.labels, self.data_types)):
            if data_type == 'train':
                train_class_counts[label] = train_class_counts.get(label, 0) + 1
            elif data_type == 'validation':
                val_class_counts[label] = val_class_counts.get(label, 0) + 1
            else:  # test
                test_class_counts[label] = test_class_counts.get(label, 0) + 1
        
        logging.info(f"Train class distribution: {train_class_counts}")
        logging.info(f"Validation class distribution: {val_class_counts}")
        logging.info(f"Test class distribution: {test_class_counts}")
        
        # 被験者IDの統計情報
        train_subjects = set([self.subject_ids[i] for i, dt in enumerate(self.data_types) if dt == 'train'])
        val_subjects = set([self.subject_ids[i] for i, dt in enumerate(self.data_types) if dt == 'validation'])
        test_subjects = set([self.subject_ids[i] for i, dt in enumerate(self.data_types) if dt == 'test'])
        
        # 重複チェック
        overlap_train_val = train_subjects.intersection(val_subjects)
        overlap_train_test = train_subjects.intersection(test_subjects)
        overlap_val_test = val_subjects.intersection(test_subjects)
        
        logging.info(f"Train subjects: {len(train_subjects)}, Validation subjects: {len(val_subjects)}, Test subjects: {len(test_subjects)}")
        
        if overlap_train_val or overlap_train_test or overlap_val_test:
            logging.warning(f"Found overlapping subjects: train-val={overlap_train_val}, train-test={overlap_train_test}, val-test={overlap_val_test}")
        else:
            logging.info("No overlapping subjects between any sets ✓")
    
    def _generate_cv_indices(self, n_splits):
        """クロスバリデーション用のインデックスを生成（trainデータのみで分割）"""
        # トレインデータのインデックスを取得
        train_indices = [i for i, dt in enumerate(self.data_types) if dt == 'train']
        train_subject_ids = [self.subject_ids[i] for i in train_indices]
        train_labels = [self.labels[i] for i in train_indices]
        
        # 被験者IDベースのクロスバリデーション（trainデータのみで分割）
        from sklearn.model_selection import StratifiedGroupKFold
        # 再現性のためのseed固定
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        self.fold_indices = []
        for train_fold_idx, val_fold_idx in sgkf.split(train_indices, train_labels, groups=train_subject_ids):
            # 実際のデータインデックスに変換
            actual_train_idx = [train_indices[i] for i in train_fold_idx]
            actual_val_idx = [train_indices[i] for i in val_fold_idx]
            self.fold_indices.append((actual_train_idx, actual_val_idx))
    
    def get_fold_indices(self, fold):
        """指定されたフォールドのtrain/validationインデックスを取得"""
        if fold >= len(self.fold_indices):
            raise ValueError(f"Fold {fold} does not exist. Available folds: 0-{len(self.fold_indices)-1}")
        return self.fold_indices[fold]
    
    def get_test_indices(self):
        """テストデータのインデックスを取得"""
        return [i for i, dt in enumerate(self.data_types) if dt == 'test']
    
    def get_validation_indices(self):
        """バリデーションデータのインデックスを取得"""
        return [i for i, dt in enumerate(self.data_types) if dt == 'validation']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 特徴量を読み込み（元の長さのまま）
        features = torch.load(self.data[idx])
        
        # マスクを作成（ゼロパディング部分を検出）
        mask = (features.sum(dim=-1) == 0)  # (time_steps,)
        
        return features.float(), self.labels[idx], mask

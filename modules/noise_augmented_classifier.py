# =============================
# ノイズ追加付きwav2vec特徴量分類スクリプト
# - ノイズ追加付きwav2vec特徴量を使用して分類
# - 各ノイズタイプごとの性能比較
# - クロスバリデーションによる評価
# - 結果の可視化と保存
# - 被験者IDベースの分割（同じ被験者のデータがtrain/testに混在しない）
# =============================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import json
from tqdm import tqdm
import wandb
import argparse
import sys
import re

# ログ設定
log_filename = f"noise_augmented_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import yaml

# 再現性のためのシード設定
def set_seed(seed=42):
    """再現性のためのシード設定"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # マルチGPUの場合
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# シードを設定
set_seed(42)

# 設定ファイルを読み込み
config_path = os.path.join('configs', 'noise_augmented_wav2vec.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# パス設定
features_path = os.path.join('dataset', 'diagnosis', 'train', 'noise_augmented_features')
output_path = os.path.join('results', 'noise_augmented_classification')
os.makedirs(output_path, exist_ok=True)

# 診断カテゴリ
diagnosis = ['ad', 'cn']
label_mapping = {'ad': 1, 'cn': 0}

# 分類設定
classification_config = config['classification']

# wandb初期化は各フォールドで行うため、ここでは行わない

def extract_subject_id_from_filename(filename):
    """ファイル名から番号を抽出する（例：714-0_original.pt → 714-0）"""
    # ファイル名から番号部分を抽出
    match = re.match(r'(\d+-\d+)', filename)
    if match:
        return match.group(1)
    return None

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
        logger.warning(f"エラー: {cha_file_path} を読み込めませんでした: {e}")
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
    logger.warning(f"CHA file not found or PAR ID extraction failed for {uid}, falling back to filename extraction")
    return uid.split('-')[0] if '-' in uid else uid

class NoiseAugmentedDataset(Dataset):
    """ノイズ追加付き特徴量データセット（被験者IDベース分割対応）"""
    
    def __init__(self, features_path, noise_type='original'):
        self.features_path = features_path
        self.noise_type = noise_type
        self.data = []
        self.labels = []
        self.subject_ids = []  # 被験者IDを追加
        
        # データを読み込み
        for diagno in diagnosis:
            diagno_path = os.path.join(features_path, diagno)
            if not os.path.exists(diagno_path):
                continue
                
            label = label_mapping[diagno]
            
            # 指定されたノイズタイプのファイルを取得
            for file in os.listdir(diagno_path):
                if file.endswith(f'_{noise_type}.pt'):
                    file_path = os.path.join(diagno_path, file)
                    # ファイル名からUIDを抽出
                    uid = extract_subject_id_from_filename(file)
                    
                    if uid:  # UIDが抽出できた場合のみ追加
                        # CHAファイルから被験者IDを抽出
                        subject_id = get_subject_id_from_cha_file(uid, diagno)
                        
                        self.data.append(file_path)
                        self.labels.append(label)
                        self.subject_ids.append(subject_id)
        
        logger.info(f"Loaded {len(self.data)} samples for noise type: {noise_type}")
        
        # クラス別サンプル数を計算
        class_counts = {}
        for label in self.labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        logger.info(f"Class distribution: {class_counts}")
        
        # 被験者IDの統計情報
        unique_subjects = len(set(self.subject_ids))
        logger.info(f"Unique subjects: {unique_subjects}")
        
        # 被験者ごとのサンプル数分布
        subject_counts = {}
        for subject_id in self.subject_ids:
            subject_counts[subject_id] = subject_counts.get(subject_id, 0) + 1
        
        logger.info(f"Subject sample distribution: min={min(subject_counts.values())}, max={max(subject_counts.values())}, mean={np.mean(list(subject_counts.values())):.2f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 特徴量を読み込み（元の長さのまま）
        features = torch.load(self.data[idx])
        
        # マスクを作成（ゼロパディング部分を検出）
        mask = (features.sum(dim=-1) == 0)  # (time_steps,)
        
        return features.float(), self.labels[idx], mask

class Wav2VecClassifier(nn.Module):
    """wav2vec特徴量用Transformer Encoder分類器（2値分類版）"""
    
    def __init__(self):
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

def train_model(model, train_loader, val_loader, num_epochs=None, lr=None):
    """モデルを訓練する関数（設定ファイルベース）"""
    
    # 設定ファイルからパラメータを取得
    if num_epochs is None:
        num_epochs = classification_config['train']['num_epochs']
    if lr is None:
        lr = classification_config['train']['learning_rate']
    
    # wandbログ頻度設定を取得
    wandb_config = config.get('wandb', {})
    logging_config = wandb_config.get('logging', {})
    train_log_every = logging_config.get('train_log_every', 1)
    val_log_every = logging_config.get('val_log_every', 1)
    console_log_every = logging_config.get('console_log_every', 10)
    
    # 2値分類用の損失関数
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=classification_config['train']['weight_decay'])
    # より安定な学習率スケジューラー
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # 訓練
        model.train()
        train_loss = 0.0
        train_preds = []
        train_true = []
        
        for features, labels, masks in train_loader:
            features, labels = features.to(device), labels.to(device).float()  # ラベルをfloatに変換
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(features, masks)
            loss = criterion(outputs.squeeze(1), labels)  # 出力をsqueezeしてラベルと形状を合わせる
            loss.backward()
            
            # 勾配クリッピング（勾配爆発を防ぐ）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # 訓練時の予測を記録
            with torch.no_grad():
                probs = torch.sigmoid(outputs).squeeze(1)
                predictions = (probs >= 0.5).long()
                train_preds.extend(predictions.cpu().numpy())
                train_true.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 訓練時の精度とF1スコアを計算
        train_acc = accuracy_score(train_true, train_preds)
        _, _, train_f1, _ = precision_recall_fscore_support(train_true, train_preds, average='weighted', zero_division=0)
        
        # 検証
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for features, labels, masks in val_loader:
                features, labels = features.to(device), labels.to(device).float()  # ラベルをfloatに変換
                masks = masks.to(device)
                outputs = model(features, masks)
                loss = criterion(outputs.squeeze(1), labels)  # 出力をsqueezeしてラベルと形状を合わせる
                val_loss += loss.item()
                
                # 2値分類の推論処理
                probs = torch.sigmoid(outputs).squeeze(1)
                predictions = (probs >= 0.5).long()
                val_preds.extend(predictions.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)
        _, _, val_f1, _ = precision_recall_fscore_support(val_true, val_preds, average='weighted', zero_division=0)
        
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # エポックごとにスケジューラーを更新
        scheduler.step()
        
        # ベストモデルを保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # wandbにログを記録（エポックごと）
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'learning_rate': optimizer.param_groups[0]['lr']
        }, step=epoch + 1)
        
        # コンソール出力（設定された頻度に基づいて）
        if (epoch + 1) % console_log_every == 0:
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return {
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

def evaluate_model(model, test_loader):
    """モデルを評価する関数"""
    
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for features, labels, masks in test_loader:
            features, labels = features.to(device), labels.to(device)
            masks = masks.to(device)
            outputs = model(features, masks)
            
            # 2値分類の推論処理
            probs = torch.sigmoid(outputs).squeeze(1)
            predictions = (probs >= 0.5).long()
            
            all_preds.extend(predictions.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
    
    # 評価指標を計算
    accuracy = accuracy_score(all_true, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_true, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'predictions': all_preds,
        'true_labels': all_true
    }

def cross_validate_noise_type(noise_type, n_splits=None):
    """特定のノイズタイプでクロスバリデーションを実行（被験者IDベース分割）"""
    
    # 設定ファイルからパラメータを取得
    if n_splits is None:
        n_splits = classification_config['train']['cross_validation_folds']
    
    logger.info(f"Starting cross-validation for noise type: {noise_type}")
    
    # データセットを作成
    dataset = NoiseAugmentedDataset(features_path, noise_type)
    
    if len(dataset) == 0:
        logger.warning(f"No data found for noise type: {noise_type}")
        return None
    
    # 被験者IDベースのクロスバリデーション
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    
    # 被験者IDベースの分割を実行
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(dataset.data, dataset.labels, groups=dataset.subject_ids)):
        logger.info(f"Fold {fold + 1}/{n_splits}")
        
        # 被験者IDの重複チェック
        train_subjects = set([dataset.subject_ids[i] for i in train_idx])
        val_subjects = set([dataset.subject_ids[i] for i in val_idx])
        overlap = train_subjects.intersection(val_subjects)
        
        if overlap:
            logger.warning(f"Fold {fold + 1}: Found overlapping subjects between train and validation: {overlap}")
        else:
            logger.info(f"Fold {fold + 1}: No overlapping subjects between train and validation")
        
        logger.info(f"Fold {fold + 1}: Train subjects: {len(train_subjects)}, Validation subjects: {len(val_subjects)}")
        logger.info(f"Fold {fold + 1}: Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
        
        # 各フォールドごとに新しいwandb runを作成
        wandb.init(
            project="noise-augmented-wav2vec-classification",
            name=f"{noise_type}_fold_{fold + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "noise_type": noise_type,
                "fold": fold + 1,
                "model_name": classification_config['model_name'],
                "batch_size": classification_config['train']['batch_size'],
                "learning_rate": classification_config['train']['learning_rate'],
                "num_epochs": classification_config['train']['num_epochs'],
                "hidden_size": classification_config['model']['hidden_size'],
                "n_layers": classification_config['model']['n_layers'],
                "n_heads": classification_config['model']['n_heads']
            }
        )
        
        # データローダーを作成（再現性のためworker_init_fnを設定）
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        def worker_init_fn(worker_id):
            """DataLoaderのワーカー初期化関数（再現性のため）"""
            np.random.seed(42 + worker_id)
        
        train_loader = DataLoader(
            dataset, 
            batch_size=classification_config['train']['batch_size'], 
            sampler=train_sampler,
            worker_init_fn=worker_init_fn,
            num_workers=0  # 再現性のため0に設定
        )
        val_loader = DataLoader(
            dataset, 
            batch_size=classification_config['train']['batch_size'], 
            sampler=val_sampler,
            worker_init_fn=worker_init_fn,
            num_workers=0  # 再現性のため0に設定
        )
        
        # モデルを作成（設定ファイルからパラメータを取得）
        model = Wav2VecClassifier().to(device)
        
        # 訓練
        train_history = train_model(model, train_loader, val_loader)
        
        # 評価
        eval_results = evaluate_model(model, val_loader)
        
        fold_results.append({
            'fold': fold + 1,
            'best_val_acc': train_history['best_val_acc'],
            'final_val_acc': eval_results['accuracy'],
            'precision': eval_results['precision'],
            'recall': eval_results['recall'],
            'f1_score': eval_results['f1_score'],
            'train_history': train_history
        })
        
        # 各フォールドの最終結果をwandbに記録
        wandb.log({
            'final_accuracy': eval_results['accuracy'],
            'final_precision': eval_results['precision'],
            'final_recall': eval_results['recall'],
            'final_f1_score': eval_results['f1_score']
        })
        
        # メモリをクリア
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 各フォールドのwandb runを終了
        wandb.finish()
    
    # 平均結果を計算
    mean_accuracy = np.mean([r['final_val_acc'] for r in fold_results])
    std_accuracy = np.std([r['final_val_acc'] for r in fold_results])
    mean_f1 = np.mean([r['f1_score'] for r in fold_results])
    std_f1 = np.std([r['f1_score'] for r in fold_results])
    
    avg_results = {
        'noise_type': noise_type,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'fold_results': fold_results
    }
    
    # クロスバリデーション全体の結果を記録するための新しいwandb run
    wandb.init(
        project="noise-augmented-wav2vec-classification",
        name=f"{noise_type}_cv_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "noise_type": noise_type,
            "cv_folds": n_splits,
            "model_name": classification_config['model_name']
        }
    )
    
    # クロスバリデーション全体の結果を記録
    wandb.log({
        'cv_mean_accuracy': mean_accuracy,
        'cv_std_accuracy': std_accuracy,
        'cv_mean_f1': mean_f1,
        'cv_std_f1': std_f1
    })
    
    # 各フォールドの詳細結果も記録
    for i, result in enumerate(fold_results):
        wandb.log({
            f'fold_{i+1}_accuracy': result['final_val_acc'],
            f'fold_{i+1}_f1': result['f1_score']
        })
    
    wandb.finish()
    
    return avg_results


def process_single_noise_type(noise_type):
    """単一のノイズタイプを処理"""
    logger.info(f"Processing single noise type: {noise_type}")
    
    # クロスバリデーションを実行
    results = cross_validate_noise_type(noise_type)
    
    if results is None:
        logger.error(f"Failed to process noise type: {noise_type}")
        return None
    
    # 結果を保存
    noise_output_path = os.path.join(output_path, noise_type)
    os.makedirs(noise_output_path, exist_ok=True)
    
    # CSVで保存
    results_df = pd.DataFrame([{
        'noise_type': results['noise_type'],
        'mean_accuracy': results['mean_accuracy'],
        'std_accuracy': results['std_accuracy'],
        'mean_f1': results['mean_f1'],
        'std_f1': results['std_f1']
    }])
    
    results_df.to_csv(os.path.join(noise_output_path, f'{noise_type}_results.csv'), index=False)
    
    # 詳細結果をJSONで保存
    with open(os.path.join(noise_output_path, f'{noise_type}_detailed_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 結果を表示
    logger.info(f"Results for {noise_type}:")
    logger.info(f"  Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    logger.info(f"  Mean F1: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    
    return results


if __name__ == "__main__":
    logger.info("Starting noise augmented classification comparison with subject-based splitting...")
    
    # すべてのノイズタイプを処理
    logger.info("Processing all available noise types with subject-based cross-validation")
    logger.info("Using StratifiedGroupKFold to ensure no subject data appears in both train and test sets")
    available_noise_types = ["original", "gaussian_noise_light", "gaussian_noise_medium", "gaussian_noise_heavy", "uniform_noise"]
    
    logger.info(f"Found {len(available_noise_types)} noise types: {available_noise_types}")
    
    all_results = {}
    for noise_type in available_noise_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing noise type: {noise_type}")
        logger.info(f"{'='*50}")
        
        try:
            result = process_single_noise_type(noise_type)
            if result:
                all_results[noise_type] = result
                logger.info(f"✓ Completed processing for {noise_type}")
                logger.info(f"  Mean Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
                logger.info(f"  Mean F1: {result['mean_f1']:.4f} ± {result['std_f1']:.4f}")
            else:
                logger.error(f"✗ Failed to process noise type: {noise_type}")
        except Exception as e:
            logger.error(f"✗ Error processing noise type {noise_type}: {str(e)}")
            continue
    
    # 全結果のサマリーを表示
    if all_results:
        logger.info(f"\n{'='*50}")
        logger.info("SUMMARY OF ALL NOISE TYPES")
        logger.info(f"{'='*50}")
        
        # 結果を精度順にソート
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['mean_accuracy'], reverse=True)
        
        for i, (noise_type, result) in enumerate(sorted_results, 1):
            logger.info(f"{i}. {noise_type}:")
            logger.info(f"   Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
            logger.info(f"   F1 Score: {result['mean_f1']:.4f} ± {result['std_f1']:.4f}")
        
        # 最良の結果を強調
        best_noise_type, best_result = sorted_results[0]
        logger.info(f"\n🏆 BEST PERFORMANCE: {best_noise_type}")
        logger.info(f"   Accuracy: {best_result['mean_accuracy']:.4f} ± {best_result['std_accuracy']:.4f}")
        logger.info(f"   F1 Score: {best_result['mean_f1']:.4f} ± {best_result['std_f1']:.4f}")
        
        # 結果をCSVファイルに保存
        summary_df = pd.DataFrame([
            {
                'noise_type': noise_type,
                'mean_accuracy': result['mean_accuracy'],
                'std_accuracy': result['std_accuracy'],
                'mean_f1': result['mean_f1'],
                'std_f1': result['std_f1']
            }
            for noise_type, result in sorted_results
        ])
        
        summary_path = os.path.join(output_path, 'all_noise_types_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\nSummary saved to: {summary_path}")
        
    else:
        logger.warning("No results to summarize")
    
    logger.info("Noise augmented classification comparison with subject-based splitting completed!")
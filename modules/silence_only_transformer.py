import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2FeatureExtractor,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
import time
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import re

# モデル設定（英語・多言語対応wav2vec2-large-xlsr-53）
MODEL_NAME = "facebook/wav2vec2-base-960h"

# ハイパーパラメータ
BATCH_SIZE = 16  # ファインチューニング用に調整
GRAD_ACCUM = 4
EPOCHS = 5
LR = 5e-5  
WARMUP_RATIO = 0.1

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データディレクトリ
original_silence_features_dir = "dataset/diagnosis/train/silence_features"
noise_augmented_silence_features_dir = "dataset/diagnosis/train/noise_augmented_silence_features"

# wandb設定
WANDB_PROJECT = "silence-transformer-classification"
WANDB_ENTITY = None  # あなたのwandbユーザー名を設定

# ID抽出関数
def extract_id_from_filename(filename):
    """ファイル名からIDを抽出（例：684-0_silence_only_wav2vec2.pt → 684-0）"""
    match = re.match(r'(\d+-\d+)', filename)
    return match.group(1) if match else None

# クラス重み計算関数
def calculate_class_weights(labels):
    """
    クラス重みを計算する関数
    weight_for_class_i = total_samples / (num_samples_in_class_i * num_classes)
    """
    total_samples = len(labels)
    num_classes = 2
    
    # 各クラスのサンプル数を計算
    class_counts = np.bincount(labels)
    print(f"Class distribution: Class 0 (CN): {class_counts[0]}, Class 1 (AD): {class_counts[1]}")
    
    # 重みを計算
    weights = []
    for i in range(num_classes):
        weight = total_samples / (class_counts[i] * num_classes)
        weights.append(weight)
    
    print(f"Calculated weights: Class 0: {weights[0]:.4f}, Class 1: {weights[1]:.4f}")
    
    # pos_weightを計算（positive classの重み）
    pos_weight = weights[1] / weights[0]
    print(f"pos_weight for BCEWithLogitsLoss: {pos_weight:.4f}")
    
    return torch.tensor(pos_weight, dtype=torch.float32)

# サイレンス特徴量データ読み込み関数（IDベース分割対応）
def load_silence_features_with_split():
    """
    既存のサイレンス特徴量ファイル（.pt）を読み込み、IDベースでtrain/testに分割
    フォルダ名（cn/ad）からラベルを自動判定
    オリジナルとノイズ追加の特徴量を別々に管理
    """
    # 利用可能なIDを収集
    available_ids = set()
    
    # Original silence featuresからIDを収集
    for diagno in ['cn', 'ad']:
        original_dir = os.path.join(original_silence_features_dir, diagno)
        if os.path.exists(original_dir):
            for filename in os.listdir(original_dir):
                if filename.endswith('.pt'):
                    file_id = extract_id_from_filename(filename)
                    if file_id:
                        available_ids.add(file_id)
    
    print(f"Total unique IDs found: {len(available_ids)}")
    
    # 利用可能なIDをリストに変換してシャッフル
    available_ids_list = list(available_ids)
    np.random.shuffle(available_ids_list)
    
    # 80%をtrain、20%をtestに分割
    split_idx = int(len(available_ids_list) * 0.8)
    train_ids = set(available_ids_list[:split_idx])
    test_ids = set(available_ids_list[split_idx:])
    
    print(f"Train IDs: {len(train_ids)}, Test IDs: {len(test_ids)}")
    
    # データを読み込み
    original_train_data = []
    noise_augmented_train_data = []
    test_data = []
    
    # Original silence featuresからデータを読み込み
    for diagno in ['cn', 'ad']:
        label = 1 if diagno == "ad" else 0
        original_dir = os.path.join(original_silence_features_dir, diagno)
        
        if os.path.exists(original_dir):
            for filename in os.listdir(original_dir):
                if filename.endswith('.pt'):
                    file_id = extract_id_from_filename(filename)
                    if file_id:
                        # ファイル名からuidを抽出（例：684-0_silence_only_wav2vec2.pt → 684-0）
                        uid = file_id
                        original_path = os.path.join(original_dir, filename)
                        
                        # IDがtrain/testのどちらに属するかを判定
                        if file_id in train_ids:
                            # Trainデータ：original features
                            try:
                                original_features = torch.load(original_path)
                                original_train_data.append({
                                    "uid": uid + "_original",
                                    "features": original_features,
                                    "label": label,
                                    "diagno": diagno,
                                    "source": "original"
                                })
                            except Exception as e:
                                print(f"Error loading {original_path}: {e}")
                            
                            # Noise augmented featuresも追加
                            noise_augmented_path = os.path.join(noise_augmented_silence_features_dir, diagno, 
                                                              uid + '_silence_noise_augmented_wav2vec2.pt')
                            if os.path.exists(noise_augmented_path):
                                try:
                                    noise_features = torch.load(noise_augmented_path)
                                    noise_augmented_train_data.append({
                                        "uid": uid + "_noise_augmented",
                                        "features": noise_features,
                                        "label": label,
                                        "diagno": diagno,
                                        "source": "noise_augmented"
                                    })
                                except Exception as e:
                                    print(f"Error loading {noise_augmented_path}: {e}")
                            
                        elif file_id in test_ids:
                            # Testデータ：originalのみを使用
                            try:
                                original_features = torch.load(original_path)
                                test_data.append({
                                    "uid": uid + "_original",
                                    "features": original_features,
                                    "label": label,
                                    "diagno": diagno,
                                    "source": "original"
                                })
                            except Exception as e:
                                print(f"Error loading {original_path}: {e}")
    
    print(f"Original train samples loaded: {len(original_train_data)}")
    print(f"Noise augmented train samples loaded: {len(noise_augmented_train_data)}")
    print(f"Test samples loaded: {len(test_data)}")
    
    assert len(original_train_data) > 0, "No original train data found. Check data directories and file paths."
    assert len(noise_augmented_train_data) > 0, "No noise augmented train data found. Check data directories and file paths."
    assert len(test_data) > 0, "No test data found. Check data directories and file paths."
    
    return original_train_data, noise_augmented_train_data, test_data

# Transformer分類モデル
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, num_heads=8, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 入力投影層
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分類ヘッド
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # バイナリ分類用
        )
        
        # 位置エンコーディング
        self.pos_encoding = nn.Parameter(torch.randn(1, 1750, hidden_dim))  # 35秒分の最大長
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # 入力投影
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # 位置エンコーディングを追加
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer処理
        x = self.transformer(x)  # [batch_size, seq_len, hidden_dim]
        
        # グローバル平均プーリング
        x = x.mean(dim=1)  # [batch_size, hidden_dim]
        
        # 分類
        logits = self.classifier(x)  # [batch_size, 1]
        
        return logits

# カスタムデータセットクラス
class SilenceFeaturesDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # サイレンス特徴量を取得
        features = item["features"]  # [seq_len, feature_dim]
        
        # 特徴量を正規化
        features_normalized = (features - features.mean(dim=0, keepdim=True)) / (features.std(dim=0, keepdim=True) + 1e-8)
        
        # 最大長に合わせて調整
        max_length = 1750  # 35秒分
        if features_normalized.shape[0] > max_length:
            features_normalized = features_normalized[:max_length]
        elif features_normalized.shape[0] < max_length:
            # パディング
            padding = torch.zeros(max_length - features_normalized.shape[0], features_normalized.shape[1])
            features_normalized = torch.cat([features_normalized, padding], dim=0)
        
        return {
            "features": features_normalized,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "uid": item["uid"]
        }

# カスタムコラテーション関数
def collate_fn(batch):
    features = torch.stack([item["features"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    
    return {
        "features": features,
        "labels": labels
    }

# 評価関数
def evaluate_model(model, dataloader, device, criterion=None):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            
            # Use custom loss function if provided, otherwise use model's built-in loss
            if criterion is not None:
                # Get logits from model without computing loss
                outputs = model(features)
                logits = outputs
                
                # Convert labels to float for BCEWithLogitsLoss
                labels_float = labels.float().unsqueeze(1)  # [batch_size, 1]
                
                # Compute loss with custom criterion
                loss = criterion(logits, labels_float)
            else:
                # Use model's built-in loss
                outputs = model(features, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
            
            total_loss += loss.item()
            
            # 予測を計算 (apply sigmoid for binary classification)
            if criterion is not None:
                # For BCEWithLogitsLoss, apply sigmoid to get probabilities
                probs = torch.sigmoid(logits).squeeze(1)  # [batch_size]
                preds = (probs > 0.5).long()
                all_probabilities.extend(probs.cpu().numpy())
            else:
                # For regular classification, use argmax
                preds = torch.argmax(logits, dim=-1)
            
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)
            
            # 予測とラベルを保存
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct_predictions / total_predictions
    avg_loss = total_loss / len(dataloader)
    
    # Calculate additional metrics for binary classification
    if criterion is not None and len(all_predictions) > 0:
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate precision, recall, F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
        
        # Calculate AUC-ROC
        try:
            auc_roc = roc_auc_score(all_labels, all_probabilities)
        except ValueError:
            auc_roc = 0.0  # In case of single class in validation set
        
        return avg_loss, accuracy, all_predictions, all_labels, {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc
        }
    
    return avg_loss, accuracy, all_predictions, all_labels, None

# プロット作成関数
def create_training_plots(train_losses, val_losses, val_accuracies, fold, stage):
    """訓練プロットを作成してwandbにログ"""
    
    # 損失プロット
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title(f'Training and Validation Loss - Fold {fold+1} - {stage}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.title(f'Validation Accuracy - Fold {fold+1} - {stage}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # wandbにプロットをログ
    wandb.log({f"training_plots_fold_{fold+1}_{stage}": wandb.Image(plt)})
    plt.close()

# カスタム訓練関数（ファインチューニング用）
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs, grad_accum_steps, fold, stage, criterion=None):
    best_accuracy = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()  # Set to train mode for training
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, batch in enumerate(progress_bar):
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            
            # Use custom loss function if provided, otherwise use model's built-in loss
            if criterion is not None:
                # Get logits from model without computing loss
                outputs = model(features)
                logits = outputs
                
                # Convert labels to float for BCEWithLogitsLoss
                labels_float = labels.float().unsqueeze(1)  # [batch_size, 1]
                
                # Compute loss with custom criterion
                loss = criterion(logits, labels_float) / grad_accum_steps
            else:
                # Use model's built-in loss
                outputs = model(features, labels=labels)
                loss = outputs.loss / grad_accum_steps
            
            loss.backward()
            
            total_loss += loss.item() * grad_accum_steps
            
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": f"{loss.item()*grad_accum_steps:.4f}"})
        
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # 検証
        val_result = evaluate_model(model, val_dataloader, device, criterion)
        if len(val_result) == 5:  # With additional metrics
            val_loss, val_accuracy, val_predictions, val_labels, val_metrics = val_result
        else:  # Without additional metrics
            val_loss, val_accuracy, val_predictions, val_labels = val_result
            val_metrics = None
            
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Log additional metrics if available
        log_dict = {
            f"fold_{fold+1}_{stage}/train_loss": avg_train_loss,
            f"fold_{fold+1}_{stage}/val_loss": val_loss,
            f"fold_{fold+1}_{stage}/val_accuracy": val_accuracy,
            f"fold_{fold+1}_{stage}/epoch": epoch + 1,
            f"fold_{fold+1}_{stage}/learning_rate": scheduler.get_last_lr()[0]
        }
        
        if val_metrics:
            log_dict.update({
                f"fold_{fold+1}_{stage}/val_precision": val_metrics['precision'],
                f"fold_{fold+1}_{stage}/val_recall": val_metrics['recall'],
                f"fold_{fold+1}_{stage}/val_f1": val_metrics['f1'],
                f"fold_{fold+1}_{stage}/val_auc_roc": val_metrics['auc_roc']
            })
            print(f"  Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc_roc']:.4f}")
        
        # wandbにログ
        wandb.log(log_dict)
        
        # ベストモデル保存（F1-scoreを使用）
        if val_metrics and val_metrics['f1'] > best_accuracy:
            best_accuracy = val_metrics['f1']
            best_model_state = model.state_dict().copy()  # ベストモデルの状態を保存
        elif not val_metrics and val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()  # ベストモデルの状態を保存
    
    # 訓練プロットを作成
    create_training_plots(train_losses, val_losses, val_accuracies, fold, stage)
    
    return train_losses, val_losses, val_accuracies, best_accuracy, best_model_state

# メイン実行部分
if __name__ == "__main__":
    import argparse
    
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Train silence-only transformer classifier')
    parser.add_argument('--data_type', type=str, choices=['original', 'noise_augmented'], 
                       required=True, help='Type of data to use for training')
    args = parser.parse_args()
    
    data_type = args.data_type
    
    # wandb初期化
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config={
            "model_name": MODEL_NAME,
            "data_type": data_type,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "warmup_ratio": WARMUP_RATIO,
            "num_labels": 1, # Binary classification: single output
            "device": str(device)
        },
        tags=["silence", "transformer", "classification", "alzheimer", "wav2vec2-features", f"{data_type}-training"]
    )
    
    print("Loading silence features with ID-based split...")
    original_train_data, noise_augmented_train_data, test_data = load_silence_features_with_split()
    
    # データセット統計をwandbにログ
    total_original_train_samples = len(original_train_data)
    total_noise_augmented_train_samples = len(noise_augmented_train_data)
    total_test_samples = len(test_data)
    total_samples = total_original_train_samples + total_noise_augmented_train_samples + total_test_samples
    
    original_train_cn_samples = sum(1 for item in original_train_data if item["label"] == 0)
    original_train_ad_samples = sum(1 for item in original_train_data if item["label"] == 1)
    noise_augmented_train_cn_samples = sum(1 for item in noise_augmented_train_data if item["label"] == 0)
    noise_augmented_train_ad_samples = sum(1 for item in noise_augmented_train_data if item["label"] == 1)
    test_cn_samples = sum(1 for item in test_data if item["label"] == 0)
    test_ad_samples = sum(1 for item in test_data if item["label"] == 1)
    
    wandb.log({
        "dataset/total_samples": total_samples,
        "dataset/original_train_samples": total_original_train_samples,
        "dataset/noise_augmented_train_samples": total_noise_augmented_train_samples,
        "dataset/test_samples": total_test_samples,
        "dataset/original_train_cn_samples": original_train_cn_samples,
        "dataset/original_train_ad_samples": original_train_ad_samples,
        "dataset/noise_augmented_train_cn_samples": noise_augmented_train_cn_samples,
        "dataset/noise_augmented_train_ad_samples": noise_augmented_train_ad_samples,
        "dataset/test_cn_samples": test_cn_samples,
        "dataset/test_ad_samples": test_ad_samples
    })
    
    print(f"Original train data: {total_original_train_samples} samples (CN: {original_train_cn_samples}, AD: {original_train_ad_samples})")
    print(f"Noise augmented train data: {total_noise_augmented_train_samples} samples (CN: {noise_augmented_train_cn_samples}, AD: {noise_augmented_train_ad_samples})")
    print(f"Test data: {total_test_samples} samples (CN: {test_cn_samples}, AD: {test_ad_samples})")
    
    # wav2vec2モデルを特徴量抽出器として準備（実際には使用しないが、設定のため）
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    
    print(f"\n{'='*50}")
    print(f"Training model with {data_type} features")
    print(f"{'='*50}")
    
    # データタイプに応じてデータを選択
    if data_type == 'original':
        train_data = original_train_data
    else:
        train_data = noise_augmented_train_data
    
    # Stratified k-fold cross validation設定
    train_labels = [item["label"] for item in train_data]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_fold_results = []
   
    # テストデータローダーを準備（全foldで共通）
    test_dataset = SilenceFeaturesDataset(test_data)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # テスト評価の集計用
    test_f1_scores = []
    test_accuracies = []
    test_auc_rocs = []
    test_precisions = []
    
    # 最良のモデル追跡用（precision基準）
    best_test_precision = 0.0
    best_test_model_state = None
    best_fold = 0
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, train_labels)):
        print(f"\n### Fold {fold + 1} - {data_type.upper()} Features")
        
        # 出力ディレクトリ作成
        os.makedirs(f"./results_{data_type}_fold{fold+1}", exist_ok=True)
    
        # train/val データセット選択
        train_data_fold = [train_data[i] for i in train_idx]
        val_data_fold = [train_data[i] for i in val_idx]
        
        # クラス重み計算
        train_labels_fold = [item["label"] for item in train_data_fold]
        val_labels_fold = [item["label"] for item in val_data_fold]
        
        # 各foldのクラス分布を詳細に表示
        print(f"Fold {fold + 1} train label distribution:")
        train_dist = pd.Series(train_labels_fold).value_counts().sort_index()
        print(f"  CN (0): {train_dist.get(0, 0)} samples")
        print(f"  AD (1): {train_dist.get(1, 0)} samples")
        print(f"  Total: {len(train_labels_fold)} samples")
        
        print(f"Fold {fold + 1} validation label distribution:")
        val_dist = pd.Series(val_labels_fold).value_counts().sort_index()
        print(f"  CN (0): {val_dist.get(0, 0)} samples")
        print(f"  AD (1): {val_dist.get(1, 0)} samples")
        print(f"  Total: {len(val_labels_fold)} samples")
        
        # 全体の分布との比較
        total_dist = pd.Series(train_labels).value_counts().sort_index()
        print(f"Overall train dataset distribution:")
        print(f"  CN (0): {total_dist.get(0, 0)} samples")
        print(f"  AD (1): {total_dist.get(1, 0)} samples")
        print(f"  Total: {len(train_labels)} samples")
        
        # Calculate class weights for imbalanced dataset
        pos_weight = calculate_class_weights(train_labels_fold)
        print(f"Fold {fold + 1} pos_weight: {pos_weight.item():.4f}")
        
        # Create BCEWithLogitsLoss with class weighting
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        
        # wandbにfold情報をログ
        fold_cn_train = train_dist.get(0, 0)
        fold_ad_train = train_dist.get(1, 0)
        fold_cn_val = val_dist.get(0, 0)
        fold_ad_val = val_dist.get(1, 0)
        
        wandb.log({
            f"{data_type}_fold_{fold+1}/train_cn": fold_cn_train,
            f"{data_type}_fold_{fold+1}/train_ad": fold_ad_train,
            f"{data_type}_fold_{fold+1}/val_cn": fold_cn_val,
            f"{data_type}_fold_{fold+1}/val_ad": fold_ad_val,
            f"{data_type}_fold_{fold+1}/train_total": len(train_data_fold),
            f"{data_type}_fold_{fold+1}/val_total": len(val_data_fold),
            f"{data_type}_fold_{fold+1}/pos_weight": pos_weight.item(),
            f"{data_type}_fold_{fold+1}/train_cn_ratio": fold_cn_train / len(train_data_fold),
            f"{data_type}_fold_{fold+1}/train_ad_ratio": fold_ad_train / len(train_data_fold),
            f"{data_type}_fold_{fold+1}/val_cn_ratio": fold_cn_val / len(val_data_fold),
            f"{data_type}_fold_{fold+1}/val_ad_ratio": fold_ad_val / len(val_data_fold)
        })
        
        # カスタムデータセット作成
        train_dataset = SilenceFeaturesDataset(train_data_fold)
        val_dataset = SilenceFeaturesDataset(val_data_fold)
        
        # データローダー作成
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        # ラベル分布表示（既に上で詳細表示済み）
    
        # モデル準備：Transformerモデルで分類
        # wav2vec2-base-960hの出力次元は768
        model = TransformerClassifier(input_dim=768).to(device)
        
        # ファインチューニング用の設定
        # 全パラメータを学習可能に
        for param in model.parameters():
            param.requires_grad = True
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        total_steps = len(train_dataloader) * EPOCHS // GRAD_ACCUM
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        # モデルのファインチューニング
        train_losses, val_losses, val_accuracies, best_accuracy, best_model_state = train_model(
            model, train_dataloader, val_dataloader, optimizer, scheduler, device, EPOCHS, GRAD_ACCUM, fold, f"{data_type}_transformer", criterion
        )
        
        # 各epochでの最高値を計算
        best_val_accuracy = max(val_accuracies)
        best_val_loss = min(val_losses)
        
        # ベストモデルの状態を復元（テスト用）
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Best model state restored for fold {fold + 1} testing")
        else:
            print(f"Warning: No best model state available for fold {fold + 1}")
        
        # テストデータで評価（各foldで共通テストセット）
        test_eval = evaluate_model(model, test_dataloader, device, criterion)
        if len(test_eval) == 5:
            test_loss, test_acc, _, _, test_metrics = test_eval
            test_f1 = float(test_metrics.get('f1', 0.0))
            test_auc = float(test_metrics.get('auc_roc', 0.0))
            test_precision = float(test_metrics.get('precision', 0.0))
        else:
            test_loss, test_acc, _, _ = test_eval
            test_f1 = 0.0
            test_auc = 0.0
            test_precision = 0.0
       
        test_f1_scores.append(test_f1)
        test_accuracies.append(float(test_acc))
        test_auc_rocs.append(test_auc)
        test_precisions.append(test_precision)
        
        # 最良のテストprecisionのモデルを追跡
        if test_precision > best_test_precision:
            best_test_precision = test_precision
            best_test_model_state = model.state_dict().copy()
            best_fold = fold + 1
            print(f"New best test precision: {test_precision:.4f} (Fold {fold + 1})")
       
        wandb.log({
            f"{data_type}_fold_{fold+1}/test_loss": float(test_loss),
            f"{data_type}_fold_{fold+1}/test_accuracy": float(test_acc),
            f"{data_type}_fold_{fold+1}/test_f1": test_f1,
            f"{data_type}_fold_{fold+1}/test_auc": test_auc,
            f"{data_type}_fold_{fold+1}/test_precision": test_precision
        })
        
        # 最終的なfold結果をwandbにログ
        log_dict = {
            f"{data_type}_fold_{fold+1}/best_f1": best_accuracy,
            f"{data_type}_fold_{fold+1}/best_val_accuracy": best_val_accuracy,
            f"{data_type}_fold_{fold+1}/best_val_loss": best_val_loss,
            f"{data_type}_fold_{fold+1}/final_val_accuracy": val_accuracies[-1],
            f"{data_type}_fold_{fold+1}/final_val_loss": val_losses[-1]
        }
        
        wandb.log(log_dict)
        
        # 結果保存
        fold_result = {
            "fold": fold + 1,
            "data_type": data_type,
            "best_f1": best_accuracy,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "final_val_accuracy": val_accuracies[-1],
            "final_val_loss": val_losses[-1],
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies
        }
        all_fold_results.append(fold_result)
        
        print(f"Fold {fold + 1} ({data_type}) completed. Best F1-score: {best_accuracy:.4f}, Best Accuracy: {best_val_accuracy:.4f}")
    
    # 全foldでのADサンプルの使用状況を確認
    print(f"\n=== AD Sample Usage Across All Folds ({data_type.upper()}) ===")
    all_ad_samples = set()
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, train_labels)):
        fold_ad_train = [train_data[i]["uid"] for i in train_idx if train_data[i]["label"] == 1]
        fold_ad_val = [train_data[i]["uid"] for i in val_idx if train_data[i]["label"] == 1]
        fold_ad_all = fold_ad_train + fold_ad_val
        all_ad_samples.update(fold_ad_all)
        print(f"Fold {fold + 1}: {len(fold_ad_all)} AD samples")
    
    total_ad_samples = sum(1 for item in train_data if item["label"] == 1)
    print(f"Total unique AD samples used across all folds: {len(all_ad_samples)}")
    print(f"Total AD samples in train dataset: {total_ad_samples}")
    print(f"All AD samples used: {'Yes' if len(all_ad_samples) == total_ad_samples else 'No'}")
    
    # 全foldの結果表示
    df_results = pd.DataFrame([{
        "fold": result["fold"],
        "data_type": result["data_type"],
        "best_f1": result["best_f1"],
        "best_val_accuracy": result["best_val_accuracy"],
        "best_val_loss": result["best_val_loss"],
        "final_val_accuracy": result["final_val_accuracy"],
        "final_val_loss": result["final_val_loss"]
    } for result in all_fold_results])
    
    print(f"\n=== Final Results ({data_type.upper()}) ===")
    print(df_results)
    
    # テスト結果の集計と表示（fold平均）
    if len(test_f1_scores) > 0:
        mean_test_f1 = float(np.mean(test_f1_scores))
        std_test_f1 = float(np.std(test_f1_scores))
        mean_test_acc = float(np.mean(test_accuracies))
        std_test_acc = float(np.std(test_accuracies))
        mean_test_auc = float(np.mean(test_auc_rocs))
        std_test_auc = float(np.std(test_auc_rocs))
        mean_test_precision = float(np.mean(test_precisions))
        std_test_precision = float(np.std(test_precisions))
        print(f"\n=== Test Performance Across Folds ({data_type.upper()}) (Average ± Std) ===")
        print(f"Test F1-score: {mean_test_f1:.4f} ± {std_test_f1:.4f}")
        print(f"Test Accuracy: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
        print(f"Test AUC: {mean_test_auc:.4f} ± {std_test_auc:.4f}")
        print(f"Test Precision: {mean_test_precision:.4f} ± {std_test_precision:.4f}")
        
        wandb.log({
            f"{data_type}_test/mean_f1": mean_test_f1,
            f"{data_type}_test/std_f1": std_test_f1,
            f"{data_type}_test/mean_accuracy": mean_test_acc,
            f"{data_type}_test/std_accuracy": std_test_acc,
            f"{data_type}_test/mean_auc": mean_test_auc,
            f"{data_type}_test/std_auc": std_test_auc,
            f"{data_type}_test/mean_precision": mean_test_precision,
            f"{data_type}_test/std_precision": std_test_precision,
        })
    
    # 各epochでの最高値の平均を計算
    mean_best_f1 = df_results['best_f1'].mean()
    std_best_f1 = df_results['best_f1'].std()
    mean_best_accuracy = df_results['best_val_accuracy'].mean()
    std_best_accuracy = df_results['best_val_accuracy'].std()
    mean_best_loss = df_results['best_val_loss'].mean()
    std_best_loss = df_results['best_val_loss'].std()
    
    print(f"\n=== Best Values Across Epochs ({data_type.upper()}) (Average ± Std) ===")
    print(f"Best F1-score: {mean_best_f1:.4f} ± {std_best_f1:.4f}")
    print(f"Best Accuracy: {mean_best_accuracy:.4f} ± {std_best_accuracy:.4f}")
    print(f"Best Loss: {mean_best_loss:.4f} ± {std_best_loss:.4f}")
    
    # 最終結果をwandbにログ
    wandb.log({
        f"{data_type}_final/mean_best_f1": mean_best_f1,
        f"{data_type}_final/std_best_f1": std_best_f1,
        f"{data_type}_final/mean_best_accuracy": mean_best_accuracy,
        f"{data_type}_final/std_best_accuracy": std_best_accuracy,
        f"{data_type}_final/mean_best_loss": mean_best_loss,
        f"{data_type}_final/std_best_loss": std_best_loss,
        f"{data_type}_final/results_table": wandb.Table(dataframe=df_results),
        f"{data_type}_final/stratified_split_info": {
            "total_train_samples": len(train_data),
            "total_test_samples": total_test_samples,
            "total_ad_samples": total_ad_samples,
            "unique_ad_samples_used": len(all_ad_samples),
            "all_ad_samples_used": len(all_ad_samples) == total_ad_samples,
            "cv_method": "StratifiedKFold",
            "split_method": "ID-based split (80% train, 20% test)"
        }
    })
    
    # 最終的な結果プロット
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(df_results['fold'], df_results['best_f1'])
    plt.title(f'Best F1-Score by Fold ({data_type.upper()})')
    plt.xlabel('Fold')
    plt.ylabel('Best F1-Score')
    plt.ylim(0, 1)
    for i, v in enumerate(df_results['best_f1']):
        plt.text(i+1, v + 0.01, f'{v:.3f}', ha='center')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.bar(df_results['fold'], df_results['best_val_accuracy'])
    plt.title(f'Best Validation Accuracy by Fold ({data_type.upper()})')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(df_results['best_val_accuracy']):
        plt.text(i+1, v + 0.01, f'{v:.3f}', ha='center')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.bar(df_results['fold'], df_results['best_val_loss'])
    plt.title(f'Best Validation Loss by Fold ({data_type.upper()})')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    for i, v in enumerate(df_results['best_val_loss']):
        plt.text(i+1, v + 0.01, f'{v:.3f}', ha='center')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    wandb.log({f"{data_type}_final/best_metrics_by_fold": wandb.Image(plt)})
    plt.close()
    
    # テストデータでの最終評価（オプション）
    print(f"\n=== Test Data Evaluation ({data_type.upper()}) ===")
    print(f"Test data contains {len(test_data)} samples")
    print(f"Test CN samples: {test_cn_samples}, Test AD samples: {test_ad_samples}")
    
    print(f"\n{data_type.upper()} model training completed!")
    
    # 最終的なF1、Accuracy、AUC、Precisionスコアの表示（テスト指標）
    print(f"\n{'='*60}")
    print(f"FINAL PERFORMANCE SUMMARY - {data_type.upper().replace('_', ' ')} MODEL")
    print(f"{'='*60}")
    if len(test_f1_scores) > 0:
        print(f"Test F1-Score: {mean_test_f1:.4f} ± {std_test_f1:.4f}")
        print(f"Test Accuracy: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
        print(f"Test AUC: {mean_test_auc:.4f} ± {std_test_auc:.4f}")
        print(f"Test Precision: {mean_test_precision:.4f} ± {std_test_precision:.4f}")
    else:
        print("Test metrics not available.")
    print(f"Cross-validation folds: 5")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # wandbに最終サマリーをログ
    wandb.log({
        "final_summary/test_f1_score": mean_test_f1 if len(test_f1_scores) > 0 else None,
        "final_summary/test_f1_score_std": std_test_f1 if len(test_f1_scores) > 0 else None,
        "final_summary/test_accuracy": mean_test_acc if len(test_accuracies) > 0 else None,
        "final_summary/test_accuracy_std": std_test_acc if len(test_accuracies) > 0 else None,
        "final_summary/test_auc": mean_test_auc if len(test_auc_rocs) > 0 else None,
        "final_summary/test_auc_std": std_test_auc if len(test_auc_rocs) > 0 else None,
        "final_summary/test_precision": mean_test_precision if len(test_precisions) > 0 else None,
        "final_summary/test_precision_std": std_test_precision if len(test_precisions) > 0 else None,
        "final_summary/data_type": data_type,
        "final_summary/cv_folds": 5,
        "final_summary/training_samples": len(train_data),
        "final_summary/test_samples": len(test_data)
    })
    
    # 最良のモデルを保存
    if best_test_model_state is not None:
        # 最良のモデルを作成して保存
        best_model = TransformerClassifier(input_dim=768).to(device)
        best_model.load_state_dict(best_test_model_state)
        
        # モデル保存ディレクトリを作成
        model_save_dir = f"models/{data_type}_transformer"
        os.makedirs(model_save_dir, exist_ok=True)
        
        # モデルファイル名
        model_filename = f"best_model_fold{best_fold}_test_precision_{best_test_precision:.4f}.pt"
        model_path = os.path.join(model_save_dir, model_filename)
        
        # モデルを保存
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'model_config': {
                'input_dim': 768,
                'hidden_dim': 256,
                'num_layers': 2,
                'num_heads': 8,
                'dropout': 0.1
            },
            'training_info': {
                'data_type': data_type,
                'best_fold': best_fold,
                'best_test_precision': best_test_precision,
                'best_test_f1': test_f1_scores[best_fold - 1],
                'best_test_accuracy': test_accuracies[best_fold - 1],
                'best_test_auc': test_auc_rocs[best_fold - 1],
                'total_folds': 5,
                'training_samples': len(train_data),
                'test_samples': len(test_data)
            }
        }, model_path)
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL SAVED (PRECISION-BASED)")
        print(f"{'='*60}")
        print(f"Model saved to: {model_path}")
        print(f"Best fold: {best_fold}")
        print(f"Best test precision: {best_test_precision:.4f}")
        print(f"Best test F1-score: {test_f1_scores[best_fold - 1]:.4f}")
        print(f"Best test accuracy: {test_accuracies[best_fold - 1]:.4f}")
        print(f"Best test AUC: {test_auc_rocs[best_fold - 1]:.4f}")
        
        # wandbに最良モデル情報をログ
        wandb.log({
            "best_model/fold": best_fold,
            "best_model/test_precision": best_test_precision,
            "best_model/test_f1": test_f1_scores[best_fold - 1],
            "best_model/test_accuracy": test_accuracies[best_fold - 1],
            "best_model/test_auc": test_auc_rocs[best_fold - 1],
            "best_model/model_path": model_path
        })
    else:
        print("Warning: No best model found to save.")
    
    # wandb終了
    wandb.finish()

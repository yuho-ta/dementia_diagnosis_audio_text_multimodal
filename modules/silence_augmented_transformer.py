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

# モデル設定（英語・多言語対応wav2vec2-large-xlsr-53）
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"

# ハイパーパラメータ
BATCH_SIZE = 8  # ファインチューニング用に調整
GRAD_ACCUM = 4
EPOCHS = 10
LR = 3e-5  # ファインチューニング用に学習率を下げる
WARMUP_RATIO = 0.1

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データディレクトリ
silence_features_dir = "dataset/diagnosis/train/noise_augmented_silence_features"
csv_labels_path = "dataset/diagnosis/train/text_transcriptions.csv"

# wandb設定
WANDB_PROJECT = "silence-augmented-transformer-classification"
WANDB_ENTITY = None  # あなたのwandbユーザー名を設定

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

# サイレンス特徴量データ読み込み関数
def load_silence_features():
    """
    既存のサイレンス特徴量ファイル（.pt）を読み込む
    """
    data = []
    
    # CSVファイルからラベル情報を読み込み
    labels_df = pd.read_csv(csv_labels_path)
    
    # 音声モデルによるファイル名の変更
    audio_data = '_wav2vec2'  # 既存のファイル名に合わせる
    
    for index, row in labels_df.iterrows():
        uid = row['uid']
        diagno = row['diagno']
        label = 1 if diagno == "ad" else 0
        
        # サイレンス特徴量ファイルのパスを構築
        silence_features_path = os.path.join(silence_features_dir, diagno, 
                                            uid + '_silence_noise_augmented' + audio_data + '.pt')
        
        # ファイルが存在するかチェック
        if os.path.exists(silence_features_path):
            try:
                # サイレンス特徴量を読み込み
                features = torch.load(silence_features_path)
                data.append({
                    "uid": uid,
                    "features": features,
                    "label": label,
                    "diagno": diagno
                })
            except Exception as e:
                print(f"Error loading {silence_features_path}: {e}")
        else:
            print(f"Missing file: {silence_features_path}")
    
    print(f"Total samples loaded: {len(data)}")
    assert len(data) > 0, "No silence features found. Check data_dir and file paths."
    
    return data

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
        
        # ベストモデル保存（F1-scoreを使用、モデルファイルは保存しない）
        if val_metrics and val_metrics['f1'] > best_accuracy:
            best_accuracy = val_metrics['f1']
        elif not val_metrics and val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
    
    # 訓練プロットを作成
    create_training_plots(train_losses, val_losses, val_accuracies, fold, stage)
    
    return train_losses, val_losses, val_accuracies, best_accuracy

# メイン実行部分
if __name__ == "__main__":
    # wandb初期化
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config={
            "model_name": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "warmup_ratio": WARMUP_RATIO,
            "num_labels": 1, # Binary classification: single output
            "device": str(device)
        },
        tags=["silence", "transformer", "classification", "alzheimer", "wav2vec2-features"]
    )
    
    print("Loading silence features...")
    data = load_silence_features()
    
    # データセット統計をwandbにログ
    total_samples = len(data)
    cn_samples = sum(1 for item in data if item["label"] == 0)
    ad_samples = sum(1 for item in data if item["label"] == 1)
    
    wandb.log({
        "dataset/total_samples": total_samples,
        "dataset/cn_samples": cn_samples,
        "dataset/ad_samples": ad_samples,
        "dataset/cn_ratio": cn_samples / total_samples,
        "dataset/ad_ratio": ad_samples / total_samples
    })
    
    # wav2vec2モデルを特徴量抽出器として準備（実際には使用しないが、設定のため）
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    
    # Stratified k-fold cross validation設定（ADとCNの比率を保持）
    # ラベルを抽出してstratified splitに使用
    labels = [item["label"] for item in data]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
        print(f"\n### Fold {fold + 1}")
        
        # 出力ディレクトリ作成
        os.makedirs(f"./results_fold{fold+1}", exist_ok=True)
    
        # train/val データセット選択
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        
        # クラス重み計算
        train_labels = [item["label"] for item in train_data]
        val_labels = [item["label"] for item in val_data]
        
        # 各foldのクラス分布を詳細に表示
        print(f"Fold {fold + 1} train label distribution:")
        train_dist = pd.Series(train_labels).value_counts().sort_index()
        print(f"  CN (0): {train_dist.get(0, 0)} samples")
        print(f"  AD (1): {train_dist.get(1, 0)} samples")
        print(f"  Total: {len(train_labels)} samples")
        
        print(f"Fold {fold + 1} validation label distribution:")
        val_dist = pd.Series(val_labels).value_counts().sort_index()
        print(f"  CN (0): {val_dist.get(0, 0)} samples")
        print(f"  AD (1): {val_dist.get(1, 0)} samples")
        print(f"  Total: {len(val_labels)} samples")
        
        # 全体の分布との比較
        total_dist = pd.Series(labels).value_counts().sort_index()
        print(f"Overall dataset distribution:")
        print(f"  CN (0): {total_dist.get(0, 0)} samples")
        print(f"  AD (1): {total_dist.get(1, 0)} samples")
        print(f"  Total: {len(labels)} samples")
        
        # Calculate class weights for imbalanced dataset
        pos_weight = calculate_class_weights(train_labels)
        print(f"Fold {fold + 1} pos_weight: {pos_weight.item():.4f}")
        
        # Create BCEWithLogitsLoss with class weighting
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        
        # wandbにfold情報をログ
        fold_cn_train = train_dist.get(0, 0)
        fold_ad_train = train_dist.get(1, 0)
        fold_cn_val = val_dist.get(0, 0)
        fold_ad_val = val_dist.get(1, 0)
        
        wandb.log({
            f"fold_{fold+1}/train_cn": fold_cn_train,
            f"fold_{fold+1}/train_ad": fold_ad_train,
            f"fold_{fold+1}/val_cn": fold_cn_val,
            f"fold_{fold+1}/val_ad": fold_ad_val,
            f"fold_{fold+1}/train_total": len(train_data),
            f"fold_{fold+1}/val_total": len(val_data),
            f"fold_{fold+1}/pos_weight": pos_weight.item(),
            f"fold_{fold+1}/train_cn_ratio": fold_cn_train / len(train_data),
            f"fold_{fold+1}/train_ad_ratio": fold_ad_train / len(train_data),
            f"fold_{fold+1}/val_cn_ratio": fold_cn_val / len(val_data),
            f"fold_{fold+1}/val_ad_ratio": fold_ad_val / len(val_data)
        })
        
        # カスタムデータセット作成
        train_dataset = SilenceFeaturesDataset(train_data)
        val_dataset = SilenceFeaturesDataset(val_data)
        
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
        train_losses, val_losses, val_accuracies, best_accuracy = train_model(
            model, train_dataloader, val_dataloader, optimizer, scheduler, device, EPOCHS, GRAD_ACCUM, fold, "transformer", criterion
        )
        
        # 結果を保存（既にtrain_modelで計算済み）
        # best_accuracyはF1-scoreを使用
        
        # 最終的なfold結果をwandbにログ
        log_dict = {
            f"fold_{fold+1}/best_f1": best_accuracy,
            f"fold_{fold+1}/final_val_accuracy": val_accuracies[-1],
            f"fold_{fold+1}/final_val_loss": val_losses[-1]
        }
        
        wandb.log(log_dict)
        
        # 結果保存
        fold_result = {
            "fold": fold + 1,
            "best_f1": best_accuracy,
            "final_val_accuracy": val_accuracies[-1],
            "final_val_loss": val_losses[-1],
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies
        }
        all_fold_results.append(fold_result)
        
        print(f"Fold {fold + 1} completed. Best F1-score: {best_accuracy:.4f}")
    
    # 全foldでのADサンプルの使用状況を確認
    print("\n=== AD Sample Usage Across All Folds ===")
    all_ad_samples = set()
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
        fold_ad_train = [data[i]["uid"] for i in train_idx if data[i]["label"] == 1]
        fold_ad_val = [data[i]["uid"] for i in val_idx if data[i]["label"] == 1]
        fold_ad_all = fold_ad_train + fold_ad_val
        all_ad_samples.update(fold_ad_all)
        print(f"Fold {fold + 1}: {len(fold_ad_all)} AD samples")
    
    total_ad_samples = sum(1 for item in data if item["label"] == 1)
    print(f"Total unique AD samples used across all folds: {len(all_ad_samples)}")
    print(f"Total AD samples in dataset: {total_ad_samples}")
    print(f"All AD samples used: {'Yes' if len(all_ad_samples) == total_ad_samples else 'No'}")
    
    # 全foldの結果表示
    df_results = pd.DataFrame([{
        "fold": result["fold"],
        "best_f1": result["best_f1"],
        "final_val_accuracy": result["final_val_accuracy"],
        "final_val_f1": result.get("final_val_f1", 0),
        "final_val_precision": result.get("final_val_precision", 0),
        "final_val_recall": result.get("final_val_recall", 0),
        "final_val_auc_roc": result.get("final_val_auc_roc", 0),
        "final_val_loss": result["final_val_loss"]
    } for result in all_fold_results])
    
    print("\n=== Final Results ===")
    print(df_results)
    
    mean_f1 = df_results['best_f1'].mean()
    std_f1 = df_results['best_f1'].std()
    mean_accuracy = df_results['final_val_accuracy'].mean()
    std_accuracy = df_results['final_val_accuracy'].std()
    
    print(f"\nMean F1-score across folds: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Mean accuracy across folds: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    
    # 最終結果をwandbにログ
    wandb.log({
        "final/mean_f1": mean_f1,
        "final/std_f1": std_f1,
        "final/mean_accuracy": mean_accuracy,
        "final/std_accuracy": std_accuracy,
        "final/results_table": wandb.Table(dataframe=df_results),
        "final/stratified_split_info": {
            "total_ad_samples": total_ad_samples,
            "unique_ad_samples_used": len(all_ad_samples),
            "all_ad_samples_used": len(all_ad_samples) == total_ad_samples,
            "cv_method": "StratifiedKFold"
        }
    })
    
    # 最終的な結果プロット
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(df_results['fold'], df_results['best_f1'])
    plt.title('Best F1-Score by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Best F1-Score')
    plt.ylim(0, 1)
    for i, v in enumerate(df_results['best_f1']):
        plt.text(i+1, v + 0.01, f'{v:.3f}', ha='center')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.bar(df_results['fold'], df_results['final_val_accuracy'])
    plt.title('Final Validation Accuracy by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(df_results['final_val_accuracy']):
        plt.text(i+1, v + 0.01, f'{v:.3f}', ha='center')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.bar(df_results['fold'], df_results['final_val_auc_roc'])
    plt.title('Final Validation AUC-ROC by Fold')
    plt.xlabel('Fold')
    plt.ylabel('AUC-ROC')
    plt.ylim(0, 1)
    for i, v in enumerate(df_results['final_val_auc_roc']):
        plt.text(i+1, v + 0.01, f'{v:.3f}', ha='center')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    wandb.log({"final/metrics_by_fold": wandb.Image(plt)})
    plt.close()
    
    # wandb終了
    wandb.finish()

# =============================
# ノイズ追加付きwav2vec特徴量分類スクリプト
# - ノイズ追加付きwav2vec特徴量を使用して分類
# - 各ノイズタイプごとの性能比較
# - クロスバリデーションによる評価
# - 結果の可視化と保存
# =============================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import json
from tqdm import tqdm
import wandb

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

# 設定ファイルを読み込み
config_path = os.path.join('configs', 'noise_augmented_wav2vec.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# パス設定
features_path = os.path.join('dataset', 'diagnosis', 'train', 'noise_augmented_features')
output_path = os.path.join('results', 'noise_augmented_classification')
os.makedirs(output_path, exist_ok=True)

# 診断カテゴリ
diagnosis = ['ad', 'cn']
label_mapping = {'ad': 1, 'cn': 0}

# 分類設定
classification_config = config['classification']

# wandb初期化
wandb.init(
    project="noise-augmented-wav2vec-classification",
    name=f"noise_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config={
        "model_name": classification_config['model_name'],
        "batch_size": classification_config['train']['batch_size'],
        "learning_rate": classification_config['train']['learning_rate'],
        "num_epochs": classification_config['train']['num_epochs'],
        "cross_validation_folds": classification_config['train']['cross_validation_folds']
    }
)

class NoiseAugmentedDataset(Dataset):
    """ノイズ追加付き特徴量データセット"""
    
    def __init__(self, features_path, noise_type='original', max_length=200):
        self.features_path = features_path
        self.noise_type = noise_type
        self.max_length = max_length
        self.data = []
        self.labels = []
        
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
                    self.data.append(file_path)
                    self.labels.append(label)
        
        logger.info(f"Loaded {len(self.data)} samples for noise type: {noise_type}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 特徴量を読み込み
        features = torch.load(self.data[idx])
        original_length = features.shape[0]
        
        # 長さを調整
        if features.shape[0] > self.max_length:
            features = features[:self.max_length]
            mask = torch.zeros(self.max_length, dtype=torch.bool)
        else:
            # パディング
            padding = torch.zeros(self.max_length - features.shape[0], features.shape[1])
            features = torch.cat([features, padding], dim=0)
            # パディングマスクを作成（Trueがパディング部分）
            mask = torch.zeros(self.max_length, dtype=torch.bool)
            mask[original_length:] = True
        
        return features.float(), self.labels[idx], mask

class Wav2VecClassifier(nn.Module):
    """wav2vec特徴量用Transformer Encoder分類器（設定ファイルベース）"""
    
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=2, dropout=0.5, n_layers=1, n_heads=8):
        super().__init__()
        
        self.feature_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 特徴量投影層（wav2vec特徴量をTransformerの入力次元に投影）
        self.feature_projection = nn.Linear(input_dim, hidden_dim)
        
        # 位置エンコーディング
        self.pos_encoding = nn.Parameter(torch.randn(1, 200, hidden_dim))  # max_length=200
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,  # intermediate_size
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # プーリング戦略（設定ファイルから取得）
        self.pooling = classification_config['model']['pooling']
        
        # 分類器
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, classification_config['model']['hidden_mlp_size']),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classification_config['model']['hidden_mlp_size'], num_classes)
        )
    
    def forward(self, x, mask=None):
        # x shape: (batch_size, time_steps, feature_dim)
        batch_size, time_steps, _ = x.shape
        
        # 特徴量投影
        x = self.feature_projection(x)  # (batch_size, time_steps, hidden_dim)
        
        # 位置エンコーディングを追加
        x = x + self.pos_encoding[:, :time_steps, :]
        
        # パディングマスクの作成（ゼロパディング部分をマスク）
        if mask is None:
            # ゼロパディング部分を検出してマスクを作成
            mask = (x.sum(dim=-1) == 0)  # (batch_size, time_steps)
        
        # Transformer Encoderで処理
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # プーリング戦略の適用
        if self.pooling == 'mean':
            # マスクされた部分を除外して平均を計算
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1), 0)
                lengths = (~mask).sum(dim=1, keepdim=True).float()
                pooled = x.sum(dim=1) / lengths.clamp(min=1)
            else:
                pooled = x.mean(dim=1)
        elif self.pooling == 'cls':
            # 最初のトークンを使用
            pooled = x[:, 0, :]
        else:
            # デフォルトは平均プーリング
            pooled = x.mean(dim=1)
        
        # 分類
        output = self.classifier(pooled)
        
        return output

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
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=classification_config['train']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # 訓練
        model.train()
        train_loss = 0.0
        for features, labels, masks in train_loader:
            features, labels = features.to(device), labels.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(features, masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 検証
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for features, labels, masks in val_loader:
                features, labels = features.to(device), labels.to(device)
                masks = masks.to(device)
                outputs = model(features, masks)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)
        
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step(val_loss)
        
        # ベストモデルを保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # wandbにログを記録（設定された頻度に基づいて）
        if (epoch + 1) % train_log_every == 0:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch + 1)
        
        if (epoch + 1) % val_log_every == 0:
            wandb.log({
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'val_accuracy': val_acc
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
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
    
    # 評価指標を計算
    accuracy = accuracy_score(all_true, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_preds, average='weighted')
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
    """特定のノイズタイプでクロスバリデーションを実行"""
    
    # 設定ファイルからパラメータを取得
    if n_splits is None:
        n_splits = classification_config['train']['cross_validation_folds']
    
    logger.info(f"Starting cross-validation for noise type: {noise_type}")
    
    # データセットを作成
    dataset = NoiseAugmentedDataset(features_path, noise_type)
    
    if len(dataset) == 0:
        logger.warning(f"No data found for noise type: {noise_type}")
        return None
    
    # クロスバリデーション
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.data, dataset.labels)):
        logger.info(f"Fold {fold + 1}/{n_splits}")
        
        # データローダーを作成
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=classification_config['train']['batch_size'], sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=classification_config['train']['batch_size'], sampler=val_sampler)
        
        # モデルを作成（設定ファイルからパラメータを取得）
        model = Wav2VecClassifier(
            input_dim=768,  # wav2vec特徴量の次元
            hidden_dim=classification_config['model']['hidden_size'],
            num_classes=classification_config['model']['num_classes'],
            dropout=classification_config['model']['dropout'],
            n_layers=classification_config['model']['n_layers'],
            n_heads=classification_config['model']['n_heads']
        ).to(device)
        
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
            'f1_score': eval_results['f1_score']
        })
    
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
    
    # wandbに結果を記録（クロスバリデーション完了時）
    wandb.log({
        f'{noise_type}/mean_accuracy': mean_accuracy,
        f'{noise_type}/std_accuracy': std_accuracy,
        f'{noise_type}/mean_f1': mean_f1,
        f'{noise_type}/std_f1': std_f1
    }, step=wandb.run.step if hasattr(wandb.run, 'step') else None)
    
    return avg_results

def compare_all_noise_types():
    """全てのノイズタイプを比較"""
    
    # 利用可能なノイズタイプを取得
    noise_types = []
    for diagno in diagnosis:
        diagno_path = os.path.join(features_path, diagno)
        if os.path.exists(diagno_path):
            for file in os.listdir(diagno_path):
                if file.endswith('.pt'):
                    noise_type = file.split('_')[-1].replace('.pt', '')
                    if noise_type not in noise_types:
                        noise_types.append(noise_type)
    
    logger.info(f"Found noise types: {noise_types}")
    
    # 各ノイズタイプでクロスバリデーションを実行
    all_results = []
    
    for noise_type in tqdm(noise_types, desc="Processing noise types"):
        results = cross_validate_noise_type(noise_type)
        if results is not None:
            all_results.append(results)
    
    # 結果をDataFrameに変換
    results_df = pd.DataFrame([
        {
            'noise_type': r['noise_type'],
            'mean_accuracy': r['mean_accuracy'],
            'std_accuracy': r['std_accuracy'],
            'mean_f1': r['mean_f1'],
            'std_f1': r['std_f1']
        }
        for r in all_results
    ])
    
    # 結果を保存
    results_df.to_csv(os.path.join(output_path, 'noise_comparison_results.csv'), index=False)
    
    # 詳細結果をJSONで保存
    with open(os.path.join(output_path, 'detailed_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # wandbにテーブルとして結果を記録（全ノイズタイプ比較完了時）
    wandb.log({
        "noise_comparison_table": wandb.Table(
            dataframe=results_df
        )
    }, step=wandb.run.step if hasattr(wandb.run, 'step') else None)
    
    return results_df, all_results

def plot_comparison_results(results_df, all_results):
    """比較結果を可視化"""
    
    # 精度比較
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(len(results_df)), results_df['mean_accuracy'])
    plt.errorbar(range(len(results_df)), results_df['mean_accuracy'], 
                yerr=results_df['std_accuracy'], fmt='none', color='black', capsize=5)
    plt.xlabel('Noise Type')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison Across Noise Types')
    plt.xticks(range(len(results_df)), results_df['noise_type'], rotation=45)
    
    # バーの色を設定
    colors = ['green' if 'original' in nt else 'blue' if 'light' in nt else 'orange' if 'medium' in nt else 'red' for nt in results_df['noise_type']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # F1スコア比較
    plt.subplot(1, 2, 2)
    bars = plt.bar(range(len(results_df)), results_df['mean_f1'])
    plt.errorbar(range(len(results_df)), results_df['mean_f1'], 
                yerr=results_df['std_f1'], fmt='none', color='black', capsize=5)
    plt.xlabel('Noise Type')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison Across Noise Types')
    plt.xticks(range(len(results_df)), results_df['noise_type'], rotation=45)
    
    # バーの色を設定
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'noise_comparison_plot.png'), dpi=300, bbox_inches='tight')
    
    # wandbにプロットを記録
    wandb.log({
        "noise_comparison_plot": wandb.Image(plt)
    }, step=wandb.run.step if hasattr(wandb.run, 'step') else None)
    
    # インタラクティブなバーチャートも記録
    wandb.log({
        "accuracy_comparison": wandb.plot.bar(
            wandb.Table(
                data=[[nt, acc] for nt, acc in zip(results_df['noise_type'], results_df['mean_accuracy'])],
                columns=["noise_type", "accuracy"]
            ),
            "noise_type",
            "accuracy",
            title="Accuracy Comparison Across Noise Types"
        ),
        "f1_comparison": wandb.plot.bar(
            wandb.Table(
                data=[[nt, f1] for nt, f1 in zip(results_df['noise_type'], results_df['mean_f1'])],
                columns=["noise_type", "f1_score"]
            ),
            "noise_type",
            "f1_score",
            title="F1 Score Comparison Across Noise Types"
        )
    }, step=wandb.run.step if hasattr(wandb.run, 'step') else None)
    
    plt.show()
    
    # 結果テーブルを表示
    print("\n=== Noise Type Comparison Results ===")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    logger.info("Starting noise augmented classification comparison...")
    
    # 全てのノイズタイプを比較
    results_df, detailed_results = compare_all_noise_types()
    
    # 結果を可視化
    plot_comparison_results(results_df, detailed_results)
    
    logger.info("Classification comparison completed!")
    logger.info(f"Results saved to: {output_path}")
    
    # wandbを終了
    wandb.finish()
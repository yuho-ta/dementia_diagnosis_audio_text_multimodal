#!/usr/bin/env python3
# =============================
# ノイズ追加分類器用訓練・評価関数
# - train_model
# - evaluate_model
# =============================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import wandb
import logging

def train_model(model, train_loader, val_loader, classification_config, config, device, num_epochs=None, lr=None):
    """モデルを訓練する関数（設定ファイルベース）"""
    
    # 設定ファイルからパラメータを取得
    if num_epochs is None:
        num_epochs = classification_config['train']['num_epochs']
    if lr is None:
        lr = classification_config['train']['learning_rate']
    
    # 早期停止の設定を取得
    early_stopping = classification_config['train'].get('early_stopping', False)
    early_stopping_patience = classification_config['train'].get('early_stopping_patience', 5)
    
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
    best_model_state = None  # ベストモデルの状態を保存
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # 早期停止用のカウンター
    patience_counter = 0
    
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
            best_model_state = model.state_dict().copy()  # ベストモデルの状態を保存
            patience_counter = 0  # パフォーマンスが改善したらカウンターをリセット
        else:
            patience_counter += 1  # パフォーマンスが改善しなかったらカウンターを増加
        
        # wandbにログを記録（エポックごと）
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'patience_counter': patience_counter
        }, step=epoch + 1)
        
        # コンソール出力（設定された頻度に基づいて）
        if (epoch + 1) % console_log_every == 0:
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Patience: {patience_counter}/{early_stopping_patience}')
        
        # 早期停止のチェック
        if early_stopping and patience_counter >= early_stopping_patience:
            logging.info(f'Early stopping triggered at epoch {epoch+1}. Best validation accuracy: {best_val_acc:.4f}')
            break
    
    return {
        'best_val_acc': best_val_acc,
        'best_model_state': best_model_state,  # ベストモデルの状態を返す
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'epochs_trained': epoch + 1,  # 実際に訓練したエポック数
        'early_stopping_triggered': early_stopping and patience_counter >= early_stopping_patience
    }

def evaluate_model(model, test_loader, device):
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

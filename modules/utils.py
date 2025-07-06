# =============================
# ユーティリティ関数群
# - 設定ファイルの読み込み・保存
# - 再現性のためのシード設定
# - モデル訓練・評価関数
# - 分類指標の計算
# - 実験結果の統計分析
# =============================

import torch
import numpy as np
import random
import yaml
import os
from dotmap import DotMap
import wandb
from tqdm import tqdm
import time
import copy
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pandas as pd


def set_seed(seed):
    """
    再現性のためのシード設定
    PyTorch、NumPy、Pythonの乱数生成器を固定
    Args:
        seed: シード値
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)


def get_config(config_file):
    """
    YAMLファイルから設定を読み込み、ログディレクトリの存在を確認
    Args:
        config_file: 設定ファイルのパス
    Returns:
        config: DotMap形式の設定オブジェクト
    """
    
    with open(config_file, 'r') as f:
        config_yaml = yaml.safe_load(f)
    
    config = DotMap(config_yaml)
    config.path_name = f"{config.model_name}_{config.model.pooling}"
    log_path = os.path.join('logs', config.path_name)
    
    # ログディレクトリの作成（現在はコメントアウト）
    # os.makedirs(log_path, exist_ok=True)
    
    # 設定ファイルの保存（現在はコメントアウト）
    # config_file_path = os.path.join(log_path, 'config.yaml')
    
    # with open(config_file_path, 'w') as f:
        # yaml.dump(config_yaml, f, default_flow_style=False)
    
    return config

def save_config(config):
    """
    設定をYAMLファイルに保存し、ログディレクトリの存在を確認
    Args:
        config: 設定オブジェクト
    """
    
    log_path = os.path.join('logs', config.path_name)
    os.makedirs(log_path, exist_ok=True)
    
    config_file_path = os.path.join(log_path, 'config.yaml')

    # マルチモーダル設定の自動判定
    config.model.multimodality = config.model.textual_model != '' and config.model.audio_model != ''

    # モデル名の自動生成
    textual_data = config.model.textual_model + '_' if config.model.textual_model != '' else ''
    audio_data = config.model.audio_model + '_' if config.model.audio_model != '' else ''
    pauses_data = 'P_' if config.model.pauses else ''

    config.model_name = f"{textual_data}{audio_data}{pauses_data}{config.model.fusion}"
    config.model.model_name = config.model_name

    config.path_name = f"{config.model_name}_{config.model.pooling}"

    
    # DotMapを標準辞書に変換
    config_dict = config.toDict()
    
    with open(config_file_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def get_metrics_classification(true_labels, pred_labels):
    """
    分類指標を安全に計算
    Args:
        true_labels: 真のラベル
        pred_labels: 予測ラベル
    Returns:
        accuracy, f1, recall, precision: 分類指標
    """
    # ゼロ除算を避けるためのフラグ（クラスが1つしかない場合）
    zero_div = 1 if len(set(true_labels)) == 1 else 0
    
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=zero_div)
    recall = recall_score(true_labels, pred_labels, average='macro', zero_division=zero_div)
    precision = precision_score(true_labels, pred_labels, average='macro', zero_division=zero_div)
    
    return accuracy, f1, recall, precision

def train(model, train_dataloader, valid_dataloader, lossfn, optimizer, lr_scheduler, num_epochs, model_name, early_stopping, early_stopping_patience, cross_val=False, num_cross_val=0):
    """
    早期停止付きモデル訓練
    Args:
        model: 訓練するモデル
        train_dataloader: 訓練用データローダー
        valid_dataloader: 検証用データローダー
        lossfn: 損失関数
        optimizer: オプティマイザー
        lr_scheduler: 学習率スケジューラー
        num_epochs: エポック数
        model_name: モデル名
        early_stopping: 早期停止の有効/無効
        early_stopping_patience: 早期停止の忍耐回数
        cross_val: 交差検証フラグ
        num_cross_val: 交差検証の番号
    Returns:
        model: 訓練済みモデル
        best_value: 最良の検証精度
        rest_best_values: 最良のその他の指標
    """
    # Weights & Biasesの初期化はmain関数で行うため、ここではコメントアウト
    # wandb.init(project="WordLevelFusion", config={"epochs": num_epochs})
    # wandb.watch(model)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # ログファイルパスの設定
    log_path = f'logs/{model_name}/train_stats_{num_cross_val}.txt' if cross_val else f'logs/{model_name}/train_stats.txt'
    
    best_value, patience = 0, 0
    best_epoch, best_weights, rest_best_values = 0, None, []
    best_mismatched_uids_with_probs = []  # 最良のaccuracyの時のmismatch IDと確率
    best_mismatched_uids_with_confidence = [] # 最良のaccuracyの時のmismatch IDと信頼度
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    
    with open(log_path, "w") as log:
        for epoch in range(num_epochs):
            model.train()
            total_true, total_pred, total_loss = [], [], 0
            
            progress_bar.set_description(f"Epoch {epoch + 1}")
            log.write(f'Epoch {epoch + 1}:\n')
            
            # 訓練ループ
            for features, labels, _, _ in train_dataloader:    
                # 順伝播
                outputs = model(features).squeeze(-1)
                loss = lossfn(outputs, labels)
                
                # 逆伝播
                loss.backward()
                # 勾配クリッピング（勾配爆発を防ぐ）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # 予測の計算
                probs = torch.sigmoid(outputs)
                predictions = torch.round(probs)
                
                # NaN値の検出と処理
                if torch.isnan(predictions).any():
                    print("⚠️ Warning: NaN detected in predictions! Skipping batch.")
                    continue
                
                predictions = predictions.detach().cpu().numpy().astype(int)
                labels = labels.detach().cpu().numpy().astype(int)
                
                total_true.extend(labels)
                total_pred.extend(predictions)
                progress_bar.update(1)
            
            # 訓練指標の計算
            accuracy, f1, recall, precision = get_metrics_classification(total_true, total_pred)
            avg_loss = total_loss / len(train_dataloader)
            
            log.write(f'Training completed in: {time.time()} seconds\n')
            log.write(f'Loss: {avg_loss}\nAccuracy: {accuracy}\nF1 Score: {f1}\nRecall: {recall}\nPrecision: {precision}\n')
            wandb.log({"train_loss": avg_loss, "train_ACC": accuracy, "train_F1": f1})
            
            # 検証
            validation_value, rest_values, mismatched_uids_with_probs, predictions_output = evaluation(model, valid_dataloader, lossfn, log, test=False)
            
            # 最良モデルの保存
            if validation_value > best_value:
                best_epoch, best_weights = epoch + 1, copy.deepcopy(model.state_dict())
                best_value, rest_best_values = validation_value, rest_values
                best_mismatched_uids_with_probs = mismatched_uids_with_probs.copy()  # 最良のaccuracyの時のmismatch IDと確率を保存
                best_predictions_output = predictions_output.copy() # 最良のaccuracyの時の予測結果を保存
                patience = 0
            else:
                patience += 1
            
            # 早期停止のチェック
            if patience == early_stopping_patience and early_stopping:
                print(f'Early stopping at epoch {epoch + 1}')
                break
        
        # 最良指標の初期化（エラー対策）
        if not rest_best_values:
            rest_best_values = [0, 0, 0]
        
        # 最終結果の記録
        log.write(f'Best validation accuracy: {best_value}\n')
        log.write(f'Best validation F1: {rest_best_values[0]}\nBest validation Recall: {rest_best_values[1]}\nBest validation Precision: {rest_best_values[2]}\n')
        log.write(f'Best epoch: {best_epoch}\n')
    
    # 最良の重みをモデルに読み込み
    model.load_state_dict(best_weights)
    
    # 最良のaccuracyの時のmismatch IDと確率を保存
    mmse_csv_path = './dataset/diagnosis/train/adresso-train-mmse-scores.csv'
    if os.path.exists(mmse_csv_path):
        mmse_df = pd.read_csv(mmse_csv_path)
        mmse_dict = {str(row['adressfname']): (row['mmse'], row['dx']) for _, row in mmse_df.iterrows()}
    else:
        mmse_dict = {}
        print("MMSE scores file not found. Skipping MMSE scores.")
    best_mismatched_uids_path = f'logs/{model_name}/{num_cross_val}_best_mismatched_uids.txt' if cross_val else f'logs/{model_name}/best_mismatched_uids.txt'
    with open(best_mismatched_uids_path, 'w') as f:
        f.write(f"Best accuracy: {best_value:.4f}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write("UID, Prob, MMSE, DX\n")
        for uid, prob in best_mismatched_uids_with_probs:
            mmse, dx = mmse_dict.get(str(uid), ("N/A", "N/A"))
            f.write(f"{uid}, {prob:.4f}, {mmse}, {dx}\n")
    best_predictions_output_path = f'logs/{model_name}/{num_cross_val}_best_predictions_output.txt' if cross_val else f'logs/{model_name}/best_predictions_output.txt'
    with open(best_predictions_output_path, 'w') as f:
        f.write(f"Best accuracy: {best_value:.4f}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write("UID, Prob, MMSE, DX\n")
        for uid, prob in best_predictions_output:
            mmse, dx = mmse_dict.get(str(uid), ("N/A", "N/A"))
            f.write(f"{uid}, {prob:.4f}, {mmse}, {dx}\n")
    return model, best_value, rest_best_values

def evaluation(model, dataloader, lossfn, log, test=False):
    """
    指定されたデータセットでモデルを評価
    Args:
        model: 評価するモデル
        dataloader: 評価用データローダー
        lossfn: 損失関数
        log: ログファイル
        test: テストモードフラグ
        save_logits_path: ログitsを保存するパス
    Returns:
        accuracy: 精度 (MMSEラベルがない場合は -1.0)
        [f1, recall, precision]: その他の指標 (MMSEラベルがない場合は [-1.0, -1.0, -1.0])
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 評価モードに設定
    
    total_true, total_pred, total_loss = [], [], 0
    predictions_output = [] # 予測結果を保存するためのリスト
    mismatched_uids_with_probs = [] # 予測と正解が異なるUIDと確率のペアを保存

    # MMSEラベルがあるかどうかをチェック
    has_mmse_labels = False
    # データローダーが空でないか、かつラベルリストが空でないかを確認
    if dataloader is not None and len(dataloader.dataset.labels) > 0:
        # dataloader.dataset.labels の要素が全て -1 でないかを確認
        # 最初の要素が -1 でなければMMSEラベルがあると判断
        if dataloader.dataset.labels[0] != -1:
            has_mmse_labels = True
        else:
            print("Detected dummy labels (-1) in the dataset. Skipping accuracy calculation for this evaluation.")

    with torch.no_grad():  # 勾配計算を無効化
        for features, labels, indices, uids in dataloader:
            outputs = model(features).squeeze(-1)
            
            # MMSEラベルがある場合のみ損失を計算
            if has_mmse_labels:
                loss = lossfn(outputs, labels)
                total_loss += loss.item()
            
            # 予測の計算
            probs = torch.sigmoid(outputs)
            predictions = torch.round(probs)
            
            # NaN値の検出と処理
            if torch.isnan(predictions).any():
                print("⚠️ Warning: NaN detected in predictions! Skipping batch.")
                continue
            
            predictions_np = predictions.detach().cpu().numpy().astype(int)
            probs_np = probs.detach().cpu().numpy()
        

            if has_mmse_labels:
                labels_np = labels.detach().cpu().numpy().astype(int)
                total_true.extend(labels_np)
                total_pred.extend(predictions_np)
                for i, prob in enumerate(probs_np):
                    if predictions_np[i] != labels_np[i]:
                        mismatched_uids_with_probs.append((uids[i], prob))
                    predictions_output.append((uids[i], prob))
    
    # 評価指標の計算
    if has_mmse_labels:
        accuracy, f1, recall, precision = get_metrics_classification(total_true, total_pred)
        avg_loss = total_loss / len(dataloader)
        if log is not None:
            log.write(f'Loss: {avg_loss:.4f}\nAccuracy: {accuracy:.4f}\nF1 Score: {f1:.4f}\nRecall: {recall:.4f}\nPrecision: {precision:.4f}\n')
        # wandbに結果を記録（テストか検証かで異なるキーを使用）
        wandb.log({"test_loss": avg_loss, "test_UAR": accuracy, "test_F1": f1} if test else {"validation_loss": avg_loss, "validation_ACC": accuracy, "validation_F1": f1})
        return accuracy, [f1, recall, precision], mismatched_uids_with_probs, predictions_output
    else:
        # MMSEラベルがない場合、精度は計算しない
        accuracy = -1.0
        f1 = -1.0
        recall = -1.0
        precision = -1.0
        avg_loss = -1.0 # 損失も計算しない、またはダミー値
        
        if log is not None:
            log.write(f'Loss: N/A (No MMSE labels)\nAccuracy: N/A\nF1 Score: N/A\nRecall: N/A\nPrecision: N/A\n')
        # wandbにはダミー値を記録する
        wandb.log({"test_loss": avg_loss, "test_UAR": accuracy, "test_F1": f1} if test else {"validation_loss": avg_loss, "validation_ACC": accuracy, "validation_F1": f1})
        
        print("Predictions for test set (without MMSE labels):")
        # 予測結果をログファイルに追記
        if log is not None:
            log.write(f"All predictions: {predictions_output}\n") # すべての予測を出力する
        
        return accuracy, [f1, recall, precision], mismatched_uids_with_probs, predictions_output


def get_model_statistics(model='all'):
    """
    実験結果の統計分析とLaTeX表の生成
    Args:
        model: 分析対象のモデル名（'all'の場合は全モデル）
    """
    directory = 'logs/'
    folder_names = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

    # 結果をグループ化するための辞書
    grouped_results = {}
    models_used = set()

    for folder_name in folder_names:
        try:
            # フォルダ名からモデル名とプーリング方法を抽出（例: "distilbert_base_cls"）
            # マルチモーダルの場合、フォルダ名が 'text_audio_P_fusion_pooling' のようになる可能性を考慮
            parts = folder_name.split('_')
            pooling = parts[-1] # 最後の要素がpooling
            model_name = '_'.join(parts[:-1]) # それ以外がモデル名

        except ValueError:
            print(f"Warning: Unexpected folder name format {folder_name}, skipping.")
            continue
        
        # 指定されたモデルのみを処理
        if model != 'all' and model not in model_name:
            continue
        
        file_path = os.path.join(directory, folder_name, 'cross_fold_summary.txt')
        test_log_file_path = os.path.join(directory, folder_name, 'test_results_summary.txt') # テスト結果ログファイルを追加

        if not os.path.exists(file_path):
            print(f"Warning: Missing validation summary file {file_path}")
            continue
        
        # テスト結果の読み込み
        test_accuracy_list = []
        test_f1_list = []
        if os.path.exists(test_log_file_path):
            try:
                with open(test_log_file_path, "r") as test_result_file:
                    test_lines = test_result_file.readlines()
                    # 各foldのテスト結果をパース
                    current_test_acc = None
                    current_test_f1 = None
                    for line in test_lines:
                        if 'Test Accuracy:' in line:
                            test_acc_str = line.split(':')[-1].strip()
                            if test_acc_str != 'N/A': # MMSEラベルがない場合はN/Aなのでスキップ
                                current_test_acc = float(test_acc_str) * 100
                            else:
                                current_test_acc = np.nan # N/Aの場合はNaNとして扱う
                        elif 'Test F1:' in line:
                            test_f1_str = line.split(':')[-1].strip()
                            if test_f1_str != 'N/A':
                                current_test_f1 = float(test_f1_str) * 100
                            else:
                                current_test_f1 = np.nan
                        
                        # Foldの終わり、または次のFoldの始まりでリストに追加
                        if '-----------------------------------' in line and current_test_acc is not None and current_test_f1 is not None:
                            test_accuracy_list.append(current_test_acc)
                            test_f1_list.append(current_test_f1)
                            current_test_acc = None # リセット
                            current_test_f1 = None # リセット
            except Exception as e:
                print(f"Error reading test results from {test_log_file_path}: {e}")
        else:
            print(f"Warning: Missing test summary file {test_log_file_path}")


        try:
            with open(file_path, "r") as result_file:
                lines = result_file.readlines()
            
            if not lines:
                print(f"Warning: Empty validation summary file {file_path}")
                continue

            # 各foldの指標を格納
            metrics = {'acc': [], 'f1': [], 'recall': [], 'precision': []}
            
            # 4行ごとに指標を読み込み
            for i in range(0, len(lines), 4):
                try:
                    metrics['acc'].append(float(lines[i].split()[-1]) * 100)
                    metrics['f1'].append(float(lines[i+1].split()[-1]) * 100)
                    metrics['recall'].append(float(lines[i+2].split()[-1]) * 100)
                    metrics['precision'].append(float(lines[i+3].split()[-1]) * 100)
                except (IndexError, ValueError) as e:
                    print(f"Warning: Malformed line in {file_path} - {e}")
                    continue

            if not all(metrics[key] for key in metrics):
                print(f"Warning: Incomplete validation statistics in {file_path}")
                continue

            # 平均と標準偏差を計算
            means = np.array([np.mean(metrics[key]) for key in metrics])
            stds = np.array([np.std(metrics[key]) for key in metrics])

            if model_name not in grouped_results:
                grouped_results[model_name] = {}
            
            # 結果を保存（平均±標準偏差の形式）
            test_acc_mean, test_acc_std = (np.nanmean(test_accuracy_list), np.nanstd(test_accuracy_list)) if test_accuracy_list else (np.nan, np.nan)
            test_f1_mean, test_f1_std = (np.nanmean(test_f1_list), np.nanstd(test_f1_list)) if test_f1_list else (np.nan, np.nan)
            
            grouped_results[model_name][pooling] = (
                round(means[0], 2), round(stds[0], 1),   # Acc (Validation)
                round(means[1], 2), round(stds[1], 1),   # F1 (Validation)
                round(means[2], 2), round(stds[2], 1),   # Recall (Validation)
                round(means[3], 2), round(stds[3], 1),   # Precision (Validation)
                round(test_acc_mean, 2) if not np.isnan(test_acc_mean) else 'N/A', round(test_acc_std, 1) if not np.isnan(test_acc_std) else 'N/A', # Acc (Test)
                round(test_f1_mean, 2) if not np.isnan(test_f1_mean) else 'N/A', round(test_f1_std, 1) if not np.isnan(test_f1_std) else 'N/A' # F1 (Test)
            )
            models_used.add(model_name)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # LaTeX形式の表を出力
    for model_name, poolings in grouped_results.items():
        print("\n\n\\begin{table}[H]")
        print("\\centering")
        # 新しい列（Test AccとTest F1）のために列数を増やす
        print("\\begin{tabular}{l|cccc|cc}") 
        print("\\hline")
        print("Pooling & Val Acc & Val F1 & Val Recall & Val Precision & Test Acc & Test F1 \\\\")
        print("\\Xhline{1pt}")
        
        for pooling in sorted(poolings.keys()):  # 一貫した順序を保証
            values = poolings[pooling]
            # N/A を適切に処理して出力
            val_acc_str = f"{values[0]} $\\pm$ {values[1]}"
            val_f1_str = f"{values[2]} $\\pm$ {values[3]}"
            val_recall_str = f"{values[4]} $\\pm$ {values[5]}"
            val_precision_str = f"{values[6]} $\\pm$ {values[7]}"
            
            test_acc_val = values[8]
            test_acc_std = values[9]
            test_acc_str = f"{test_acc_val} $\\pm$ {test_acc_std}" if test_acc_val != 'N/A' else 'N/A'

            test_f1_val = values[10]
            test_f1_std = values[11]
            test_f1_str = f"{test_f1_val} $\\pm$ {test_f1_std}" if test_f1_val != 'N/A' else 'N/A'


            print(f"{pooling}  &  {val_acc_str}  &  {val_f1_str}  &  {val_recall_str}  &  {val_precision_str} & {test_acc_str} & {test_f1_str} \\\\")
        print("\\hline")
        
        print("\\end{tabular}")
        print(f"\\caption{{{model_name}}}")
        print("\\end{table}")
# =============================
# メイン訓練スクリプト
# - マルチモーダル（音声+テキスト）モデルの訓練
# - K-Fold交差検証対応
# - Weights & Biases（wandb）による実験管理
# - 早期停止、学習率スケジューリング対応
# - テストセットでの評価追加
# =============================

from dataset import get_dataloaders
from utils import set_seed, get_config, train, save_config, evaluation # [変更点] evaluationをインポート
from model import CrossAttentionTransformerEncoder, MyTransformerEncoder, BidirectionalCrossAttentionTransformerEncoder, ElementWiseFusionEncoder
import torch
import wandb
import sys
import torch.nn as nn
from transformers import get_scheduler
from torch.optim import AdamW
wandb.login()
import os


def set_up(config, train_dataloader, device, fold=0):
    """
    モデル、オプティマイザー、損失関数、スケジューラーの設定
    Args:
        config: 設定オブジェクト
        train_dataloader: 訓練用データローダー
        device: 使用デバイス（GPU/CPU）
        fold: K-Foldの番号（デフォルト: 0）
    Returns:
        model: 初期化されたモデル
        optimizer: オプティマイザー
        lossfn: 損失関数
        lr_scheduler: 学習率スケジューラー
    """
    # 再現性のためのシード設定
    set_seed(42)
    
    # マルチモーダル（音声+テキスト）の場合
    if config.model.multimodality:
        # 融合方法に応じてモデルを選択
        if 'bicross' in config.model.fusion:
            # 双方向クロスアテンション融合
            model = BidirectionalCrossAttentionTransformerEncoder(config.model).to(device)
        elif 'cross' in config.model.fusion:
            # クロスアテンション融合
            model = CrossAttentionTransformerEncoder(config.model).to(device)
        else:
            # 要素ごとの融合
            model = ElementWiseFusionEncoder(config.model).to(device)
    else:
        # 単一モーダルの場合
        model = MyTransformerEncoder(config.model).to(device)

    # オプティマイザーの設定（AdamW）
    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    # 損失関数の設定（バイナリクロスエントロピー）
    lossfn = nn.BCEWithLogitsLoss()
    
    # 学習率スケジューラーの設定
    num_training_steps = config.train.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=20, num_training_steps=num_training_steps
    )

    # Weights & Biases（wandb）の初期化
    # 実験管理とログ記録のため
    wandb.init(
        project="WordLevelFusion",
        name=f"{config.model_name}_{fold}" if config.train.cross_validation else config.model_name,
        config={
            "learning_rate": config.train.learning_rate,
            "architecture": config.model_name,
            "dataset": "ADReSSo",
            "epochs": config.train.num_epochs,
            "batch_size": config.train.batch_size,
        }
    )
    
    # モデルの監視開始
    wandb.watch(model)
    return model, optimizer, lossfn, lr_scheduler

def main(config):
    """
    メイン関数：モデルの訓練と保存、テストセットでの評価
    K-Fold交差検証に対応
    Args:
        config: 設定オブジェクト
    """
    # GPUが利用可能な場合はGPU、そうでなければCPUを使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ログ保存用ディレクトリの作成
    log_path = os.path.join('logs', config.path_name)
    os.makedirs(log_path, exist_ok=True)
    
    # K-Fold交差検証の場合
    if config.train.cross_validation:
        # 交差検証結果のサマリーファイル
        log_file = os.path.join(log_path, 'cross_fold_summary.txt')
        # [変更点] テスト結果を保存するファイルを追加
        test_log_file = os.path.join(log_path, 'test_results_summary.txt') 

        with open(log_file, "w") as log_cv, open(test_log_file, "w") as log_test: # [変更点] log_testファイルも開く
            # 各foldで訓練・評価を実行
            for fold in range(config.train.cross_validation_folds):
                # [変更点] テストデータローダーも取得するように変更
                train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(config, kfold_number=fold, return_test_dataloader=True)
                
                # モデル、オプティマイザー等の設定
                model, optimizer, lossfn, lr_scheduler = set_up(config, train_dataloader, device, fold)
                # モデルの訓練
                model, best_value, rest_best_values = train(
                    model, train_dataloader, validation_dataloader, lossfn, optimizer, lr_scheduler,
                    config.train.num_epochs, config.path_name, config.train.early_stopping, 
                    config.train.early_stopping_patience, config.train.cross_validation, fold
                )
                
                # 結果をログファイルに記録
                log_cv.write(f'Fold {fold}: Best Value = {best_value}\n')
                log_cv.write(f'Best F1: {rest_best_values[0]}\nBest Recall: {rest_best_values[1]}\nBest Precision: {rest_best_values[2]}\n')
                
                # モデルの保存
                torch.save(model.state_dict(), os.path.join(log_path, f'model_fold_{fold}.pth'))
                print(f'Model for fold {fold} saved')

                # [変更点] テストセットでの評価 (MMSE CSVがない場合は精度計算をスキップ)
                print(f"Evaluating on test set for Fold {fold}...")
                log_test.write(f'--- Test Results for Fold {fold} ---\n')
                
                # test_dataloaderのラベルがダミー値（-1）かどうかで判断
                if test_dataloader is not None and len(test_dataloader.dataset.labels) > 0 and test_dataloader.dataset.labels[0] != -1:
                    # ラベルが存在し、ダミー値でない場合のみ精度計算を行う
                    test_accuracy, test_rest_values = evaluation(model, test_dataloader, lossfn, log_test, test=True)
                    print(f'Test Accuracy: {test_accuracy:.4f}, Test F1: {test_rest_values[0]:.4f}')
                    log_test.write(f'Test Accuracy: {test_accuracy:.4f}\nTest F1: {test_rest_values[0]:.4f}\nTest Recall: {test_rest_values[1]:.4f}\nTest Precision: {test_rest_values[2]:.4f}\n')
                    wandb.log({
                        "best_value_validation": best_value,
                        "best_f1_validation": rest_best_values[0],
                        "best_recall_validation": rest_best_values[1],
                        "best_precision_validation": rest_best_values[2],
                        "test_accuracy": test_accuracy,
                        "test_f1": test_rest_values[0]
                    })
                else:
                    # ラベルがダミー値の場合、予測のみ行う
                    print("Test set has no MMSE labels. Performing prediction only.")
                    # evaluation関数が予測のみを返すように修正されているため、その結果をログに記録
                    test_accuracy, test_rest_values = evaluation(model, test_dataloader, lossfn, log_test, test=True)
                    # evaluation関数がN/Aを書き込むので、ここでは追加で書き込まない
                    # wandbにもダミー値を記録
                    wandb.log({
                        "best_value_validation": best_value,
                        "best_f1_validation": rest_best_values[0],
                        "best_recall_validation": rest_best_values[1],
                        "best_precision_validation": rest_best_values[2],
                        "test_accuracy": test_accuracy, # -1.0 が記録される
                        "test_f1": test_rest_values[0] # -1.0 が記録される
                    })


                log_test.write(f'-----------------------------------\n')
                wandb.finish()
    else:
        # 通常の訓練（交差検証なし）
        # [変更点] テストデータローダーも取得するように変更
        train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(config, return_test_dataloader=True)
        
        # モデル、オプティマイザー等の設定
        model, optimizer, lossfn, lr_scheduler = set_up(config, train_dataloader, device)
        # モデルの訓練
        model, best_value, rest_best_values = train(
            model, train_dataloader, validation_dataloader, lossfn, optimizer, lr_scheduler, 
            config.train.num_epochs, config.path_name, config.train.early_stopping, 
            config.train.early_stopping_patience
        )
        
        # モデルの保存
        model_save_path = os.path.join(log_path, 'model.pt')
        torch.save(model.state_dict(), model_save_path)
        print('Model saved')

        # [変更点] テストセットでの評価 (MMSE CSVがない場合は精度計算をスキップ)
        print(f"Evaluating on test set...")
        test_log_file = os.path.join(log_path, 'test_results.txt')
        with open(test_log_file, "w") as log_test:
            log_test.write('--- Test Results ---\n')
            if test_dataloader is not None and len(test_dataloader.dataset.labels) > 0 and test_dataloader.dataset.labels[0] != -1:
                test_accuracy, test_rest_values = evaluation(model, test_dataloader, lossfn, log_test, test=True)
                print(f'Test Accuracy: {test_accuracy:.4f}, Test F1: {test_rest_values[0]:.4f}')
                log_test.write(f'Test Accuracy: {test_accuracy:.4f}\nTest F1: {test_rest_values[0]:.4f}\nTest Recall: {test_rest_values[1]:.4f}\nTest Precision: {test_rest_values[2]:.4f}\n')
                wandb.log({
                    "best_value_validation": best_value,
                    "best_f1_validation": rest_best_values[0],
                    "best_recall_validation": rest_best_values[1],
                    "best_precision_validation": rest_best_values[2],
                    "test_accuracy": test_accuracy,
                    "test_f1": test_rest_values[0]
                })
            else:
                print("Test set has no MMSE labels. Performing prediction only.")
                test_accuracy, test_rest_values = evaluation(model, test_dataloader, lossfn, log_test, test=True)
                wandb.log({
                    "best_value_validation": best_value,
                    "best_f1_validation": rest_best_values[0],
                    "best_recall_validation": rest_best_values[1],
                    "best_precision_validation": rest_best_values[2],
                    "test_accuracy": test_accuracy,
                    "test_f1": test_rest_values[0]
                })

            log_test.write(f'--------------------\n')
        wandb.finish()


# スクリプトのメイン実行部分
if __name__ == '__main__':
    # コマンドライン引数から設定ファイルのパスを取得
    # スクリプトを実行する際には、`python your_script_name.py --config path/to/your/config.yaml` のように実行します。
    try:
        config_path_index = sys.argv.index('--config') + 1
        config_path = sys.argv[config_path_index]
    except (ValueError, IndexError):
        print("Usage: python main_script.py --config /path/to/your/config.yaml")
        sys.exit(1)

    # 設定ファイルを読み込み
    config = get_config(config_path)
    
    # ハイパーパラメータのグリッドサーチ用コード（現在はコメントアウト）
    """
    for model_name_iter in ['qwen']:
        config.model_name = model_name_iter
        config.model.model_name = config.model_name

        for fusion_iter in ['crossgated']:
            config.model.fusion = fusion_iter

            for pooling_iter in ['mean', 'cls']:
                config.model.pooling = pooling_iter
                # 設定を保存
                save_config(config)    
                # メイン関数を実行
                main(config)
    """
    
    # 設定を保存 (グリッドサーチを行わない場合)
    save_config(config)    
    # メイン関数を実行
    main(config)
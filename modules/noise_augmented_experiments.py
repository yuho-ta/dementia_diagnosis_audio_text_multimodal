#!/usr/bin/env python3
# =============================
# ノイズ追加分類器用実験関数
# - cross_validate_noise_type
# - cross_validate_noise_combination
# - process_single_noise_type
# - process_noise_combination_experiment
# =============================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
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

from noise_augmented_utils import set_seed
from noise_augmented_datasets import NoiseAugmentedDataset, CombinedNoiseDataset
from noise_augmented_model import Wav2VecClassifier
from noise_augmented_training import train_model, evaluate_model

def cross_validate_noise_type(noise_type, features_path, classification_config, config, device, output_path, n_splits=None):
    """特定のノイズタイプでクロスバリデーションを実行（被験者IDベース3分割）"""
    
    # 設定ファイルからパラメータを取得
    if n_splits is None:
        n_splits = classification_config['train']['cross_validation_folds']
    
    logging.info(f"Starting cross-validation for noise type: {noise_type} (3-way split)")
    
    # データセットを作成（被験者レベルで3分割済み、クロスバリデーションインデックスも事前生成済み）
    dataset = NoiseAugmentedDataset(features_path, noise_type, n_splits)
    
    if len(dataset) == 0:
        logging.warning(f"No data found for noise type: {noise_type}")
        return None
    
    # 各セットのインデックスを取得
    test_indices = dataset.get_test_indices()
    validation_indices = dataset.get_validation_indices()
    
    logging.info(f"Train samples: {len([d for d in dataset.data_types if d == 'train'])}, Validation samples: {len(validation_indices)}, Test samples: {len(test_indices)}")
    
    fold_results = []
    
    # 事前に生成されたクロスバリデーションインデックスを使用
    for fold in range(n_splits):
        logging.info(f"Fold {fold + 1}/{n_splits}")
        
        # 事前に生成されたインデックスを取得（trainデータ内での分割）
        train_fold_idx, train_val_fold_idx = dataset.get_fold_indices(fold)
        
        # 被験者IDの重複チェック
        train_subjects = set([dataset.subject_ids[i] for i in train_fold_idx])
        train_val_subjects = set([dataset.subject_ids[i] for i in train_val_fold_idx])
        val_subjects = set([dataset.subject_ids[i] for i in validation_indices])
        test_subjects = set([dataset.subject_ids[i] for i in test_indices])
        
        overlap_train_trainval = train_subjects.intersection(train_val_subjects)
        overlap_train_val = train_subjects.intersection(val_subjects)
        overlap_train_test = train_subjects.intersection(test_subjects)
        overlap_trainval_val = train_val_subjects.intersection(val_subjects)
        overlap_trainval_test = train_val_subjects.intersection(test_subjects)
        overlap_val_test = val_subjects.intersection(test_subjects)
        
        if overlap_train_trainval:
            logging.warning(f"Fold {fold + 1}: Found overlapping subjects between train and train-validation: {overlap_train_trainval}")
        else:
            logging.info(f"Fold {fold + 1}: No overlapping subjects between train and train-validation")
        
        if overlap_train_val or overlap_train_test or overlap_trainval_val or overlap_trainval_test or overlap_val_test:
            logging.warning(f"Fold {fold + 1}: Found overlapping subjects between sets")
        else:
            logging.info(f"Fold {fold + 1}: No overlapping subjects between any sets ✓")
        
        logging.info(f"Fold {fold + 1}: Train subjects: {len(train_subjects)}, Train-val subjects: {len(train_val_subjects)}")
        logging.info(f"Fold {fold + 1}: Train samples: {len(train_fold_idx)}, Train-val samples: {len(train_val_fold_idx)}")
        
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
        
        # データローダーを作成
        train_sampler = torch.utils.data.SubsetRandomSampler(train_fold_idx)
        train_val_sampler = torch.utils.data.SubsetRandomSampler(train_val_fold_idx)
        validation_sampler = torch.utils.data.SubsetRandomSampler(validation_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
        
        def worker_init_fn(worker_id):
            """DataLoaderのワーカー初期化関数（再現性のため）"""
            np.random.seed(42 + worker_id)
        
        train_loader = DataLoader(
            dataset, 
            batch_size=classification_config['train']['batch_size'], 
            sampler=train_sampler,
            worker_init_fn=worker_init_fn,
            num_workers=0
        )
        train_val_loader = DataLoader(
            dataset, 
            batch_size=classification_config['train']['batch_size'], 
            sampler=train_val_sampler,
            worker_init_fn=worker_init_fn,
            num_workers=0
        )
        validation_loader = DataLoader(
            dataset, 
            batch_size=classification_config['train']['batch_size'], 
            sampler=validation_sampler,
            worker_init_fn=worker_init_fn,
            num_workers=0
        )
        test_loader = DataLoader(
            dataset, 
            batch_size=classification_config['train']['batch_size'], 
            sampler=test_sampler,
            worker_init_fn=worker_init_fn,
            num_workers=0
        )
        
        # モデルを作成
        model = Wav2VecClassifier(classification_config).to(device)
        
        # 訓練（trainデータで訓練、train-valデータで早期停止）
        train_history = train_model(model, train_loader, train_val_loader, classification_config, config, device)
        
        # ベストモデルの状態を復元
        if train_history['best_model_state'] is not None:
            model.load_state_dict(train_history['best_model_state'])
            logging.info(f"Best model state restored (validation accuracy: {train_history['best_val_acc']:.4f})")
        else:
            logging.warning("No best model state found, using final model state")
        
        # 独立したバリデーションデータで評価
        val_results = evaluate_model(model, validation_loader, device)
        
        # テストデータで評価
        test_results = evaluate_model(model, test_loader, device)
        
        fold_results.append({
            'fold': fold + 1,
            'best_train_val_acc': train_history['best_val_acc'],
            'validation_accuracy': val_results['accuracy'],
            'validation_f1': val_results['f1_score'],
            'test_accuracy': test_results['accuracy'],
            'test_f1': test_results['f1_score'],
            'test_precision': test_results['precision'],
            'test_recall': test_results['recall'],
            'train_history': train_history
        })
        
        # 各フォールドの結果をwandbに記録
        wandb.log({
            'validation_accuracy': val_results['accuracy'],
            'validation_f1': val_results['f1_score'],
            'test_accuracy': test_results['accuracy'],
            'test_f1': test_results['f1_score'],
            'test_precision': test_results['precision'],
            'test_recall': test_results['recall']
        })
        
        # メモリをクリア
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 各フォールドのwandb runを終了
        wandb.finish()
    
    # 平均結果を計算
    mean_validation_acc = np.mean([r['validation_accuracy'] for r in fold_results])
    std_validation_acc = np.std([r['validation_accuracy'] for r in fold_results])
    mean_test_acc = np.mean([r['test_accuracy'] for r in fold_results])
    std_test_acc = np.std([r['test_accuracy'] for r in fold_results])
    mean_test_f1 = np.mean([r['test_f1'] for r in fold_results])
    std_test_f1 = np.std([r['test_f1'] for r in fold_results])
    
    avg_results = {
        'noise_type': noise_type,
        'mean_validation_accuracy': mean_validation_acc,
        'std_validation_accuracy': std_validation_acc,
        'mean_test_accuracy': mean_test_acc,
        'std_test_accuracy': std_test_acc,
        'mean_test_f1': mean_test_f1,
        'std_test_f1': std_test_f1,
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
        'cv_mean_validation_accuracy': mean_validation_acc,
        'cv_std_validation_accuracy': std_validation_acc,
        'cv_mean_test_accuracy': mean_test_acc,
        'cv_std_test_accuracy': std_test_acc,
        'cv_mean_test_f1': mean_test_f1,
        'cv_std_test_f1': std_test_f1
    })
    
    # 各フォールドの詳細結果も記録
    for i, result in enumerate(fold_results):
        wandb.log({
            f'fold_{i+1}_validation_accuracy': result['validation_accuracy'],
            f'fold_{i+1}_test_accuracy': result['test_accuracy'],
            f'fold_{i+1}_test_f1': result['test_f1']
        })
    
    wandb.finish()
    
    return avg_results

def cross_validate_noise_combination(train_noise_types, test_noise_type, features_path, classification_config, config, device, output_path, n_splits=None):
    """ノイズの組み合わせでクロスバリデーションを実行（3分割アプローチ）"""
    
    # 設定ファイルからパラメータを取得
    if n_splits is None:
        n_splits = classification_config['train']['cross_validation_folds']
    
    logging.info(f"Starting cross-validation for noise combination (3-way split):")
    logging.info(f"  Train noise types: {train_noise_types}")
    logging.info(f"  Test noise type: {test_noise_type}")
    
    # データセットを作成（被験者レベルで3分割済み、クロスバリデーションインデックスも事前生成済み）
    dataset = CombinedNoiseDataset(features_path, train_noise_types, test_noise_type, n_splits)
    
    if len(dataset) == 0:
        logging.warning("No data found for the specified noise combination")
        return None
    
    # 各セットのインデックスを取得
    test_indices = dataset.get_test_indices()
    validation_indices = dataset.get_validation_indices()
    
    logging.info(f"Train samples: {len([d for d in dataset.data_types if d == 'train'])}, Validation samples: {len(validation_indices)}, Test samples: {len(test_indices)}")
    
    fold_results = []
    
    # 事前に生成されたクロスバリデーションインデックスを使用
    for fold in range(n_splits):
        logging.info(f"Fold {fold + 1}/{n_splits}")
        
        # 事前に生成されたインデックスを取得（trainデータ内での分割）
        train_fold_idx, train_val_fold_idx = dataset.get_fold_indices(fold)
        
        # 被験者IDの重複チェック
        train_subjects = set([dataset.subject_ids[i] for i in train_fold_idx])
        train_val_subjects = set([dataset.subject_ids[i] for i in train_val_fold_idx])
        val_subjects = set([dataset.subject_ids[i] for i in validation_indices])
        test_subjects = set([dataset.subject_ids[i] for i in test_indices])
        
        overlap_train_trainval = train_subjects.intersection(train_val_subjects)
        overlap_train_val = train_subjects.intersection(val_subjects)
        overlap_train_test = train_subjects.intersection(test_subjects)
        overlap_trainval_val = train_val_subjects.intersection(val_subjects)
        overlap_trainval_test = train_val_subjects.intersection(test_subjects)
        overlap_val_test = val_subjects.intersection(test_subjects)
        
        if overlap_train_trainval:
            logging.warning(f"Fold {fold + 1}: Found overlapping subjects between train and train-validation: {overlap_train_trainval}")
        else:
            logging.info(f"Fold {fold + 1}: No overlapping subjects between train and train-validation")
        
        if overlap_train_val or overlap_train_test or overlap_trainval_val or overlap_trainval_test or overlap_val_test:
            logging.warning(f"Fold {fold + 1}: Found overlapping subjects between sets")
        else:
            logging.info(f"Fold {fold + 1}: No overlapping subjects between any sets ✓")
        
        logging.info(f"Fold {fold + 1}: Train subjects: {len(train_subjects)}, Train-val subjects: {len(train_val_subjects)}")
        logging.info(f"Fold {fold + 1}: Train samples: {len(train_fold_idx)}, Train-val samples: {len(train_val_fold_idx)}")
        
        # 各フォールドごとに新しいwandb runを作成
        wandb.init(
            project="noise-combination-classification",
            name=f"train_{'_'.join(train_noise_types)}_test_{test_noise_type}_fold_{fold + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "train_noise_types": train_noise_types,
                "test_noise_type": test_noise_type,
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
        
        # データローダーを作成
        train_sampler = torch.utils.data.SubsetRandomSampler(train_fold_idx)
        train_val_sampler = torch.utils.data.SubsetRandomSampler(train_val_fold_idx)
        validation_sampler = torch.utils.data.SubsetRandomSampler(validation_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
        
        def worker_init_fn(worker_id):
            """DataLoaderのワーカー初期化関数（再現性のため）"""
            np.random.seed(42 + worker_id)
        
        train_loader = DataLoader(
            dataset, 
            batch_size=classification_config['train']['batch_size'], 
            sampler=train_sampler,
            worker_init_fn=worker_init_fn,
            num_workers=0
        )
        train_val_loader = DataLoader(
            dataset, 
            batch_size=classification_config['train']['batch_size'], 
            sampler=train_val_sampler,
            worker_init_fn=worker_init_fn,
            num_workers=0
        )
        validation_loader = DataLoader(
            dataset, 
            batch_size=classification_config['train']['batch_size'], 
            sampler=validation_sampler,
            worker_init_fn=worker_init_fn,
            num_workers=0
        )
        test_loader = DataLoader(
            dataset, 
            batch_size=classification_config['train']['batch_size'], 
            sampler=test_sampler,
            worker_init_fn=worker_init_fn,
            num_workers=0
        )
        
        # モデルを作成
        model = Wav2VecClassifier(classification_config).to(device)
        
        # 訓練（trainデータで訓練、train-valデータで早期停止）
        train_history = train_model(model, train_loader, train_val_loader, classification_config, config, device)
        
        # ベストモデルの状態を復元
        if train_history['best_model_state'] is not None:
            model.load_state_dict(train_history['best_model_state'])
            logging.info(f"Best model state restored (validation accuracy: {train_history['best_val_acc']:.4f})")
        else:
            logging.warning("No best model state found, using final model state")
        
        # 独立したバリデーションデータで評価
        val_results = evaluate_model(model, validation_loader, device)
        
        # テストデータで評価
        test_results = evaluate_model(model, test_loader, device)
        
        fold_results.append({
            'fold': fold + 1,
            'best_train_val_acc': train_history['best_val_acc'],
            'validation_accuracy': val_results['accuracy'],
            'validation_f1': val_results['f1_score'],
            'test_accuracy': test_results['accuracy'],
            'test_f1': test_results['f1_score'],
            'test_precision': test_results['precision'],
            'test_recall': test_results['recall'],
            'train_history': train_history
        })
        
        # 各フォールドの結果をwandbに記録
        wandb.log({
            'validation_accuracy': val_results['accuracy'],
            'validation_f1': val_results['f1_score'],
            'test_accuracy': test_results['accuracy'],
            'test_f1': test_results['f1_score'],
            'test_precision': test_results['precision'],
            'test_recall': test_results['recall']
        })

        logging.info(f"Fold {fold + 1} results:")
        logging.info(f"  Validation Accuracy: {val_results['accuracy']:.4f}")
        logging.info(f"  Validation F1: {val_results['f1_score']:.4f}")
        logging.info(f"  Test Accuracy: {test_results['accuracy']:.4f}")
        logging.info(f"  Test F1: {test_results['f1_score']:.4f}")
        logging.info(f"  Test Precision: {test_results['precision']:.4f}")
        logging.info(f"  Test Recall: {test_results['recall']:.4f}")
        
        # メモリをクリア
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 各フォールドのwandb runを終了
        wandb.finish()
    
    # 平均結果を計算
    mean_validation_acc = np.mean([r['validation_accuracy'] for r in fold_results])
    std_validation_acc = np.std([r['validation_accuracy'] for r in fold_results])
    mean_test_acc = np.mean([r['test_accuracy'] for r in fold_results])
    std_test_acc = np.std([r['test_accuracy'] for r in fold_results])
    mean_test_f1 = np.mean([r['test_f1'] for r in fold_results])
    std_test_f1 = np.std([r['test_f1'] for r in fold_results])
    
    avg_results = {
        'train_noise_types': train_noise_types,
        'test_noise_type': test_noise_type,
        'mean_validation_accuracy': mean_validation_acc,
        'std_validation_accuracy': std_validation_acc,
        'mean_test_accuracy': mean_test_acc,
        'std_test_accuracy': std_test_acc,
        'mean_test_f1': mean_test_f1,
        'std_test_f1': std_test_f1,
        'fold_results': fold_results
    }
    
    # クロスバリデーション全体の結果を記録
    wandb.init(
        project="noise-combination-classification",
        name=f"train_{'_'.join(train_noise_types)}_test_{test_noise_type}_cv_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "train_noise_types": train_noise_types,
            "test_noise_type": test_noise_type,
            "cv_folds": n_splits,
            "model_name": classification_config['model_name']
        }
    )
    
    wandb.log({
        'cv_mean_validation_accuracy': mean_validation_acc,
        'cv_std_validation_accuracy': std_validation_acc,
        'cv_mean_test_accuracy': mean_test_acc,
        'cv_std_test_accuracy': std_test_acc,
        'cv_mean_test_f1': mean_test_f1,
        'cv_std_test_f1': std_test_f1
    })
    
    # 各フォールドの詳細結果も記録
    for i, result in enumerate(fold_results):
        wandb.log({
            f'fold_{i+1}_validation_accuracy': result['validation_accuracy'],
            f'fold_{i+1}_test_accuracy': result['test_accuracy'],
            f'fold_{i+1}_test_f1': result['test_f1']
        })
    
    wandb.finish()
    
    return avg_results

def process_single_noise_type(noise_type, features_path, classification_config, config, device, output_path):
    """単一のノイズタイプを処理"""
    logging.info(f"Processing single noise type: {noise_type}")
    
    # クロスバリデーションを実行
    results = cross_validate_noise_type(noise_type, features_path, classification_config, config, device, output_path)
    
    if results is None:
        logging.error(f"Failed to process noise type: {noise_type}")
        return None
    
    # 結果を保存
    noise_output_path = os.path.join(output_path, noise_type)
    os.makedirs(noise_output_path, exist_ok=True)
    
    # CSVで保存
    results_df = pd.DataFrame([{
        'noise_type': results['noise_type'],
        'mean_validation_accuracy': results['mean_validation_accuracy'],
        'std_validation_accuracy': results['std_validation_accuracy'],
        'mean_test_accuracy': results['mean_test_accuracy'],
        'std_test_accuracy': results['std_test_accuracy'],
        'mean_test_f1': results['mean_test_f1'],
        'std_test_f1': results['std_test_f1']
    }])
    
    results_df.to_csv(os.path.join(noise_output_path, f'{noise_type}_results.csv'), index=False)
    
    # 詳細結果をJSONで保存
    with open(os.path.join(noise_output_path, f'{noise_type}_detailed_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 結果を表示
    logging.info(f"Results for {noise_type}:")
    logging.info(f"  Validation Accuracy: {results['mean_validation_accuracy']:.4f} ± {results['std_validation_accuracy']:.4f}")
    logging.info(f"  Test Accuracy: {results['mean_test_accuracy']:.4f} ± {results['std_test_accuracy']:.4f}")
    logging.info(f"  Test F1: {results['mean_test_f1']:.4f} ± {results['std_test_f1']:.4f}")
    
    return results

def process_noise_combination_experiment(features_path, classification_config, config, device, output_path):
    """ノイズの組み合わせ実験を実行"""
    logging.info("Starting noise combination experiment...")
    
    # 実験設定
    experiments = [
        {
            'name': 'test_original_train_combined',
            'train_noise_types': ['original', 'gaussian_noise_light'],
            'test_noise_type': 'original'
        }
    ]
    #  {
    #     'name': 'test_noisy_train_combined',
    #     'train_noise_types': ['original', 'gaussian_noise_light', 'gaussian_noise_medium'],
    #     'test_noise_type': 'gaussian_noise_light'
    # }
    
    all_results = {}
    
    for exp in experiments:
        logging.info(f"\n{'='*60}")
        logging.info(f"Experiment: {exp['name']}")
        logging.info(f"Train noise types: {exp['train_noise_types']}")
        logging.info(f"Test noise type: {exp['test_noise_type']}")
        logging.info(f"{'='*60}")
        
        try:
            result = cross_validate_noise_combination(
                exp['train_noise_types'], 
                exp['test_noise_type'],
                features_path,
                classification_config,
                config,
                device,
                output_path
            )
            
            if result:
                all_results[exp['name']] = result
                logging.info(f"✓ Completed experiment: {exp['name']}")
                logging.info(f"  Mean Test Accuracy: {result['mean_test_accuracy']:.4f} ± {result['std_test_accuracy']:.4f}")
                logging.info(f"  Mean Test F1: {result['mean_test_f1']:.4f} ± {result['std_test_f1']:.4f}")
            else:
                logging.error(f"✗ Failed to complete experiment: {exp['name']}")
                
        except Exception as e:
            logging.error(f"✗ Error in experiment {exp['name']}: {str(e)}")
            continue
    
    # 結果を保存
    if all_results:
        # 結果をCSVファイルに保存
        results_data = []
        for exp_name, result in all_results.items():
            results_data.append({
                'experiment': exp_name,
                'train_noise_types': '_'.join(result['train_noise_types']),
                'test_noise_type': result['test_noise_type'],
                'mean_validation_accuracy': result['mean_validation_accuracy'],
                'std_validation_accuracy': result['std_validation_accuracy'],
                'mean_test_accuracy': result['mean_test_accuracy'],
                'std_test_accuracy': result['std_test_accuracy'],
                'mean_test_f1': result['mean_test_f1'],
                'std_test_f1': result['std_test_f1']
            })
        
        results_df = pd.DataFrame(results_data)
        results_path = os.path.join(output_path, 'noise_combination_experiment_results.csv')
        results_df.to_csv(results_path, index=False)
        logging.info(f"\nResults saved to: {results_path}")
        
        # 詳細結果をJSONで保存
        with open(os.path.join(output_path, 'noise_combination_experiment_detailed_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # 結果の保存のみ行い、表示はメイン関数で行う
        logging.info(f"Results saved to: {results_path}")
        logging.info(f"Detailed results saved to: {os.path.join(output_path, 'noise_combination_experiment_detailed_results.json')}")
    
    return all_results

#!/usr/bin/env python3
# =============================
# ノイズ追加付きwav2vec特徴量分類スクリプト（メインファイル）
# - ノイズ追加付きwav2vec特徴量を使用して分類
# - 各ノイズタイプごとの性能比較
# - クロスバリデーションによる評価
# - 結果の可視化と保存
# - 被験者IDベースの分割（同じ被験者のデータがtrain/testに混在しない）
# =============================

import os
import yaml
import argparse
import logging
import pandas as pd
import json

from noise_augmented_utils import set_seed, setup_logging
from noise_augmented_experiments import process_single_noise_type, process_noise_combination_experiment

def main():
    """メイン関数"""
    # シードを設定
    set_seed(42)
    
    # ログ設定
    logger = setup_logging()
    
    # 設定ファイルを読み込み
    config_path = os.path.join('configs', 'noise_augmented_wav2vec.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # デバイス設定
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # パス設定
    features_path = os.path.join('dataset', 'diagnosis', 'train', 'noise_augmented_features')
    output_path = os.path.join('results', 'noise_augmented_classification')
    os.makedirs(output_path, exist_ok=True)
    
    # 分類設定
    classification_config = config['classification']
    
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Noise augmented classification with subject-based splitting')
    parser.add_argument('--experiment', type=str, default='single_noise', 
                       choices=['single_noise', 'noise_combination'],
                       help='Experiment type: single_noise or noise_combination')
    
    args = parser.parse_args()
    
    if args.experiment == 'single_noise':
        # 単一ノイズタイプの実験
        logger.info("Starting noise augmented classification comparison with subject-based splitting...")
        
        # すべてのノイズタイプを処理
        logger.info("Processing all available noise types with subject-based cross-validation")
        logger.info("Using StratifiedGroupKFold to ensure no subject data appears in both train and test sets")
        available_noise_types = [
            ('gaussian_noise_light', 'gaussian_noise_light', 'original')
        ]
        logger.info(f"Found {len(available_noise_types)} noise types: {available_noise_types}")
        
        all_results = {}
        for train_noise_types, val_noise_types, test_noise_type in available_noise_types:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing noise type: train_noise_types={train_noise_types}, val_noise_types={val_noise_types}, test_noise_type={test_noise_type}")
            logger.info(f"{'='*50}")
            
            try:
                result = process_single_noise_type(train_noise_types, val_noise_types, test_noise_type, features_path, classification_config, config, device, output_path)
                if result:
                    all_results[train_noise_types + '_' + val_noise_types + '_' + test_noise_type] = result
                    logger.info(f"✓ Completed processing for train_noise_types={train_noise_types}, val_noise_types={val_noise_types}, test_noise_type={test_noise_type}")
                    logger.info(f"  Validation Accuracy: {result['mean_validation_accuracy']:.4f} ± {result['std_validation_accuracy']:.4f}")
                    logger.info(f"  Test Accuracy: {result['mean_test_accuracy']:.4f} ± {result['std_test_accuracy']:.4f}")
                    logger.info(f"  Test F1: {result['mean_test_f1']:.4f} ± {result['std_test_f1']:.4f}")
                else:
                    logger.error(f"✗ Failed to process noise type: train_noise_types={train_noise_types}, val_noise_types={val_noise_types}, test_noise_type={test_noise_type}")
            except Exception as e:
                logger.error(f"✗ Error processing noise type train_noise_types={train_noise_types}, val_noise_types={val_noise_types}, test_noise_type={test_noise_type}: {str(e)}")
                continue
        
        # 全結果のサマリーを表示
        if all_results:
            logger.info(f"\n{'='*50}")
            logger.info("SUMMARY OF ALL NOISE COMBINATIONS")
            logger.info(f"{'='*50}")
            
            # 結果をテスト精度順にソート
            sorted_results = sorted(all_results.items(), key=lambda x: x[1]['mean_test_accuracy'], reverse=True)
            
            for i, (noise_combination, result) in enumerate(sorted_results, 1):
                logger.info(f"{i}. {noise_combination}:")
                logger.info(f"   Validation Accuracy: {result['mean_validation_accuracy']:.4f} ± {result['std_validation_accuracy']:.4f}")
                logger.info(f"   Test Accuracy: {result['mean_test_accuracy']:.4f} ± {result['std_test_accuracy']:.4f}")
                logger.info(f"   Test F1 Score: {result['mean_test_f1']:.4f} ± {result['std_test_f1']:.4f}")
            
            # 最良の結果を強調（テスト精度で評価）
            best_noise_combination, best_result = sorted_results[0]
            logger.info(f"\n🏆 BEST PERFORMANCE: {best_noise_combination}")
            logger.info(f"   Validation Accuracy: {best_result['mean_validation_accuracy']:.4f} ± {best_result['std_validation_accuracy']:.4f}")
            logger.info(f"   Test Accuracy: {best_result['mean_test_accuracy']:.4f} ± {best_result['std_test_accuracy']:.4f}")
            logger.info(f"   Test F1 Score: {best_result['mean_test_f1']:.4f} ± {best_result['std_test_f1']:.4f}")
            
            # 結果をCSVファイルに保存
            summary_df = pd.DataFrame([
                {
                    'noise_combination': noise_combination,
                    'mean_validation_accuracy': result['mean_validation_accuracy'],
                    'std_validation_accuracy': result['std_validation_accuracy'],
                    'mean_test_accuracy': result['mean_test_accuracy'],
                    'std_test_accuracy': result['std_test_accuracy'],
                    'mean_test_f1': result['mean_test_f1'],
                    'std_test_f1': result['std_test_f1']
                }
                for noise_combination, result in sorted_results
            ])
            
            summary_path = os.path.join(output_path, 'all_noise_types_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"\nSummary saved to: {summary_path}")
            
        else:
            logger.warning("No results to summarize")
        
        logger.info("Noise augmented classification comparison with subject-based splitting completed!")
        
    elif args.experiment == 'noise_combination':
        # ノイズの組み合わせ実験
        logger.info("Starting noise combination experiment with subject-based splitting...")
        logger.info("Testing combinations of train and test noise types")
        logger.info("Ensuring no subject overlap between train and test sets")
        
        all_results = process_noise_combination_experiment(features_path, classification_config, config, device, output_path)
        
        if all_results:
            logger.info("Noise combination experiment completed successfully!")
            
            # 結果のサマリーを表示
            logger.info(f"\n{'='*60}")
            logger.info("NOISE COMBINATION EXPERIMENT SUMMARY")
            logger.info(f"{'='*60}")
            
            # 結果をテスト精度順にソート
            sorted_results = sorted(all_results.items(), key=lambda x: x[1]['mean_test_accuracy'], reverse=True)
            
            for i, (exp_name, result) in enumerate(sorted_results, 1):
                logger.info(f"\n{i}. {exp_name}:")
                logger.info(f"   Train noise types: {result['train_noise_types']}")
                logger.info(f"   Test noise type: {result['test_noise_type']}")
                logger.info(f"   Validation Accuracy: {result['mean_validation_accuracy']:.4f} ± {result['std_validation_accuracy']:.4f}")
                logger.info(f"   Test Accuracy: {result['mean_test_accuracy']:.4f} ± {result['std_test_accuracy']:.4f}")
                logger.info(f"   Test F1 Score: {result['mean_test_f1']:.4f} ± {result['std_test_f1']:.4f}")
            
            # 最良の結果を強調（テスト精度で評価）
            best_exp_name, best_result = sorted_results[0]
            logger.info(f"\n🏆 BEST PERFORMANCE: {best_exp_name}")
            logger.info(f"   Train noise types: {best_result['train_noise_types']}")
            logger.info(f"   Test noise type: {best_result['test_noise_type']}")
            logger.info(f"   Validation Accuracy: {best_result['mean_validation_accuracy']:.4f} ± {best_result['std_validation_accuracy']:.4f}")
            logger.info(f"   Test Accuracy: {best_result['mean_test_accuracy']:.4f} ± {best_result['std_test_accuracy']:.4f}")
            logger.info(f"   Test F1 Score: {best_result['mean_test_f1']:.4f} ± {best_result['std_test_f1']:.4f}")
            
            # 結果をCSVファイルに保存
            summary_df = pd.DataFrame([
                {
                    'experiment': exp_name,
                    'train_noise_types': '_'.join(result['train_noise_types']),
                    'test_noise_type': result['test_noise_type'],
                    'mean_validation_accuracy': result['mean_validation_accuracy'],
                    'std_validation_accuracy': result['std_validation_accuracy'],
                    'mean_test_accuracy': result['mean_test_accuracy'],
                    'std_test_accuracy': result['std_test_accuracy'],
                    'mean_test_f1': result['mean_test_f1'],
                    'std_test_f1': result['std_test_f1']
                }
                for exp_name, result in sorted_results
            ])
            
            summary_path = os.path.join(output_path, 'noise_combination_experiment_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"\nSummary saved to: {summary_path}")
            
            # 詳細結果をJSONで保存
            detailed_path = os.path.join(output_path, 'noise_combination_experiment_detailed_results.json')
            with open(detailed_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"Detailed results saved to: {detailed_path}")
            
        else:
            logger.warning("No results from noise combination experiment")
        
        logger.info("Noise combination experiment completed!")

if __name__ == "__main__":
    main()

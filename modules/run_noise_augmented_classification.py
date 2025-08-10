#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ノイズ追加付きwav2vec特徴量分類システム実行スクリプト
Transformer Encoderを使用した分類器で、各ノイズタイプの性能を比較します。
"""

import os
import sys
import logging
from datetime import datetime

# ログ設定
log_filename = f"run_noise_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """メイン実行関数"""
    
    logger.info("=" * 60)
    logger.info("ノイズ追加付きwav2vec特徴量分類システム開始")
    logger.info("Transformer Encoderを使用した分類器")
    logger.info("=" * 60)
    
    try:
        # 必要なファイルの存在確認
        required_files = [
            'noise_augmented_classifier.py',
            'configs/noise_augmented_wav2vec.yaml'
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                logger.error(f"必要なファイルが見つかりません: {file_path}")
                return False
        
        # 特徴量ディレクトリの確認
        features_path = os.path.join('dataset', 'diagnosis', 'train', 'noise_augmented_features')
        if not os.path.exists(features_path):
            logger.error(f"特徴量ディレクトリが見つかりません: {features_path}")
            logger.info("先に特徴量抽出を実行してください: python preprocess/preprocess_noise_augmented_wav2vec.py")
            return False
        
        # 分類スクリプトをインポートして実行
        logger.info("分類スクリプトをインポート中...")
        from noise_augmented_classifier import compare_all_noise_types, plot_comparison_results
        
        logger.info("全ノイズタイプの比較を開始...")
        results_df, detailed_results = compare_all_noise_types()
        
        logger.info("結果の可視化を開始...")
        plot_comparison_results(results_df, detailed_results)
        
        logger.info("=" * 60)
        logger.info("分類システムが正常に完了しました！")
        logger.info("結果は以下の場所に保存されています:")
        logger.info(f"- CSV結果: results/noise_augmented_classification/noise_comparison_results.csv")
        logger.info(f"- 詳細結果: results/noise_augmented_classification/detailed_results.json")
        logger.info(f"- プロット: results/noise_augmented_classification/noise_comparison_plot.png")
        logger.info(f"- wandbダッシュボードで詳細な結果を確認できます")
        logger.info("=" * 60)
        
        return True
        
    except ImportError as e:
        logger.error(f"モジュールのインポートエラー: {e}")
        logger.info("必要な依存関係がインストールされているか確認してください")
        return False
        
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
# =============================
# サイレンス特徴量用のK-Fold交差検証分割生成スクリプト
# - 5分割のK-Fold交差検証用データ分割を作成
# - 各foldの訓練用・検証用データをnumpyファイルとして保存
# - 分割結果の統計情報を表示
# - サイレンス特徴量ファイルの存在確認
# =============================

"""
Script to generate cross-validation splits for silence features.
サイレンス特徴量用の交差検証分割を生成するスクリプト
"""

import os
import sys
from silence_features_dataset import set_silence_splits, get_silence_splits_stats, check_silence_features_availability

def main():
    """
    メイン関数
    - サイレンス特徴量ファイルの存在確認
    - K-Fold分割の生成
    - 分割結果の統計情報表示
    """
    print("=" * 60)
    print("Generating cross-validation splits for silence features...")
    print("サイレンス特徴量用の交差検証分割を生成中...")
    print("=" * 60)
    
    # サイレンス特徴量ファイルの存在確認
    print("\n1. Checking silence features availability...")
    total_files, missing_files = check_silence_features_availability()
    
    if total_files == 0:
        print("Error: No silence features files found!")
        print("Please run preprocess_silence_embeddings.py first to generate silence features.")
        return
    
    if missing_files > 0:
        print(f"Warning: {missing_files} files are missing compared to CSV entries.")
        print("Proceeding with available files...")
    
    # K-Fold分割を生成
    print("\n2. Generating K-Fold splits...")
    set_silence_splits()
    
    print("Splits generated successfully!")
    print("分割が正常に生成されました！")
    
    # 各foldの統計情報を表示
    print("\n3. Split statistics:")
    print("分割統計情報:")
    print("-" * 40)
    get_silence_splits_stats()
    
    print("\n" + "=" * 60)
    print("Silence features cross-validation splits generation completed!")
    print("サイレンス特徴量用の交差検証分割生成が完了しました！")
    print("=" * 60)

def test_silence_features_loading():
    """
    サイレンス特徴量の読み込みテスト
    """
    print("\nTesting silence features loading...")
    
    # 簡易的な設定オブジェクトを作成
    class Config:
        class Model:
            def __init__(self):
                self.audio_model = 'wav2vec2'
        class Train:
            def __init__(self):
                self.batch_size = 32
        
        def __init__(self):
            self.model = self.Model()
            self.train = self.Train()
    
    config = Config()
    
    try:
        from silence_features_dataset import read_silence_features_CSV
        uids, features, labels = read_silence_features_CSV(config)
        print(f"Successfully loaded {len(features)} silence features")
        print(f"Sample feature shape: {features[0].shape if features else 'No features'}")
        return True
    except Exception as e:
        print(f"Error loading silence features: {e}")
        return False

# スクリプトが直接実行された場合の処理
if __name__ == "__main__":
    main()
    
    # オプション: 読み込みテストを実行
    print("\n" + "-" * 40)
    print("Running loading test...")
    test_silence_features_loading() 
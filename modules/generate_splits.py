#!/usr/bin/env python3
# =============================
# ADReSSoデータセット用のK-Fold交差検証分割生成スクリプト
# - 5分割のK-Fold交差検証用データ分割を作成
# - 各foldの訓練用・検証用データをnumpyファイルとして保存
# - 分割結果の統計情報を表示
# =============================

"""
Script to generate cross-validation splits for the ADReSSo dataset.
ADReSSoデータセット用の交差検証分割を生成するスクリプト
"""

import os
import sys
from dataset import set_splits, get_splits_stats

def main():
    """
    メイン関数
    - CSVファイルの存在確認
    - K-Fold分割の生成
    - 分割結果の統計情報表示
    """
    print("Generating cross-validation splits...")
    print("交差検証分割を生成中...")
    
    # K-Fold分割を生成
    # 5分割の交差検証用データ分割を作成し、numpyファイルとして保存
    set_splits()
    
    print("Splits generated successfully!")
    print("分割が正常に生成されました！")
    print("\nSplit statistics:")
    print("分割統計情報:")
    
    # 各foldの統計情報を表示
    # 訓練用・検証用データのクラス分布（正常/アルツハイマー）を確認
    get_splits_stats()

# スクリプトが直接実行された場合の処理
if __name__ == "__main__":
    main() 
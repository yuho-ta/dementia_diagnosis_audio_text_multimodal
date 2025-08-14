#!/usr/bin/env python3
# =============================
# サイレンス特徴量用のK-Fold交差検証分割生成スクリプト（被験者IDベース）
# - 5分割のStratifiedGroupKFold交差検証用データ分割を作成
# - 各foldの訓練用・検証用データをnumpyファイルとして保存
# - 分割結果の統計情報を表示
# - 同じ被験者のデータがtrain/testに混在しないように分割
# =============================

"""
Script to generate cross-validation splits for the silence features dataset.
サイレンス特徴量データセット用の被験者IDベース交差検証分割を生成するスクリプト
"""

import os
import sys
from silence_features_dataset import set_silence_splits, get_silence_splits_stats

def main():
    """
    メイン関数
    - CSVファイルの存在確認
    - 被験者IDベースのサイレンス特徴量K-Fold分割の生成
    - 分割結果の統計情報表示
    """
    print("Generating subject-based cross-validation splits for silence features...")
    print("サイレンス特徴量用の被験者IDベース交差検証分割を生成中...")
    
    # 被験者IDベースのサイレンス特徴量K-Fold分割を生成
    # 5分割のStratifiedGroupKFold交差検証用データ分割を作成し、numpyファイルとして保存
    # 同じ被験者のデータがtrain/testに混在しないように分割
    set_silence_splits()
    
    print("Subject-based silence splits generated successfully!")
    print("サイレンス特徴量用の被験者IDベース分割が正常に生成されました！")
    print("\nSplit statistics:")
    print("分割統計情報:")
    
    # 各foldの統計情報を表示
    # 訓練用・検証用データのクラス分布（正常/アルツハイマー）を確認
    get_silence_splits_stats()

# スクリプトが直接実行された場合の処理
if __name__ == "__main__":
    main() 
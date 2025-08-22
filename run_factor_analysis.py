#!/usr/bin/env python3
# =============================
# wav2vecベクトルの因子分析実行スクリプト
# =============================

import os
import sys
import argparse

# modulesディレクトリをパスに追加
sys.path.append('modules')

from factor_analysis_wav2vec import Wav2VecFactorAnalyzer

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Factor analysis of wav2vec features')
    parser.add_argument('--features_path', type=str, 
                       default='dataset/diagnosis/train/noise_augmented_features',
                       help='Path to wav2vec features directory')
    parser.add_argument('--output_dir', type=str, 
                       default='plots/factor_analysis',
                       help='Output directory for plots and results')
    parser.add_argument('--install_deps', action='store_true',
                       help='Install required dependencies')
    
    args = parser.parse_args()
    
    # 依存関係のインストール
    if args.install_deps:
        print("Installing dependencies...")
        os.system("python modules/install_factor_analysis_deps.py")
        print("Dependencies installed!")
    
    # 特徴量パスの確認
    if not os.path.exists(args.features_path):
        print(f"Error: Features path '{args.features_path}' does not exist!")
        print("Please make sure the wav2vec features are generated and the path is correct.")
        return
    
    print(f"Starting factor analysis...")
    print(f"Features path: {args.features_path}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # 因子分析の実行
        analyzer = Wav2VecFactorAnalyzer(args.features_path, args.output_dir)
        results = analyzer.run_complete_analysis()
        
        print("\n" + "="*50)
        print("FACTOR ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Results saved to: {args.output_dir}")
        print("\nGenerated files:")
        print("- pca_visualization.png: PCAによる2次元可視化")
        print("- tsne_visualization.png: t-SNEによる2次元可視化")
        print("- umap_visualization.png: UMAPによる2次元可視化")
        print("- pca_variance_analysis.png: PCAの寄与率分析")
        print("- data_distribution.png: データ分布の可視化")
        print("- factor_analysis_results.json: 詳細な分析結果")
        
        print("\nKey metrics:")
        print(f"PCA Silhouette Score: {results['pca_metrics']['silhouette_score']:.4f}")
        print(f"t-SNE Silhouette Score: {results['tsne_metrics']['silhouette_score']:.4f}")
        print(f"UMAP Silhouette Score: {results['umap_metrics']['silhouette_score']:.4f}")
        
    except Exception as e:
        print(f"Error during factor analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
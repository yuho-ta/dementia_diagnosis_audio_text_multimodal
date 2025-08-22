#!/usr/bin/env python3
# =============================
# AD/CN特徴量比較分析実行スクリプト
# =============================

import os
import sys
import argparse

# modulesディレクトリをパスに追加
sys.path.append('modules')

from ad_cn_feature_comparison import ADCNFeatureComparator

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='AD/CN feature comparison analysis')
    parser.add_argument('--features_path', type=str, 
                       default='dataset/diagnosis/train/noise_augmented_features',
                       help='Path to wav2vec features directory')
    parser.add_argument('--output_dir', type=str, 
                       default='plots/ad_cn_comparison',
                       help='Output directory for plots and results')
    parser.add_argument('--noise_type', type=str, 
                       default='gaussian_noise_light',
                       help='Type of noise to compare with original')
    
    args = parser.parse_args()
    
    # 特徴量パスの確認
    if not os.path.exists(args.features_path):
        print(f"Error: Features path '{args.features_path}' does not exist!")
        print("Please make sure the wav2vec features are generated and the path is correct.")
        return
    
    print(f"Starting AD/CN feature comparison analysis...")
    print(f"Features path: {args.features_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Noise type: {args.noise_type}")
    
    try:
        # AD/CN特徴量比較分析の実行
        comparator = ADCNFeatureComparator(args.features_path, args.output_dir)
        summary = comparator.run_complete_analysis()
        
        print("\n" + "="*60)
        print("AD/CN FEATURE COMPARISON COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved to: {args.output_dir}")
        
        print("\nGenerated files:")
        print("- feature_distributions.png: AD/CN特徴量分布の比較")
        print("- statistical_significance.png: 統計的有意性の比較")
        print("- effect_sizes.png: 効果量の比較")
        print("- noise_impact.png: ノイズの影響の可視化")
        print("- pca_2d_visualization.png: PCAによる2次元可視化")
        print("- pca_variance_comparison.png: PCA寄与率の比較")
        print("- pca_axis_similarity.png: PCA軸の類似性分析")
        print("- pca_separation_analysis.png: PCA空間での分離分析")
        print("- tsne_2d_visualization.png: t-SNEによる2次元可視化")
        print("- tsne_separation_analysis.png: t-SNE空間での分離分析")
        print("- tsne_perplexity_comparison.png: 異なるperplexityでのt-SNE比較")
        print("- pca_vs_tsne_comparison.png: PCA vs t-SNEの比較")
        print("- ad_cn_comparison_results.json: 詳細な分析結果")
        
        print("\nKey findings:")
        print(f"NOISE-FREE DATA:")
        print(f"  Significant features: {summary['noise_free']['significant_features']} / {summary['noise_free']['total_features']} ({summary['noise_free']['significant_ratio']:.2%})")
        print(f"  Mean effect size: {summary['noise_free']['mean_effect_size']:.4f}")
        
        print(f"\nNOISY DATA:")
        print(f"  Significant features: {summary['noisy']['significant_features']} / {summary['noisy']['total_features']} ({summary['noisy']['significant_ratio']:.2%})")
        print(f"  Mean effect size: {summary['noisy']['mean_effect_size']:.4f}")
        
        print(f"\nNOISE IMPACT:")
        print(f"  Features improved: {summary['comparison']['features_improved']}")
        print(f"  Features degraded: {summary['comparison']['features_degraded']}")
        print(f"  Mean separation change: {summary['comparison']['mean_separation_change']:.4f}")
        
        # 解釈のガイダンス
        print(f"\n" + "="*60)
        print("INTERPRETATION GUIDE:")
        print("="*60)
        
        if summary['comparison']['mean_separation_change'] > 0:
            print("✓ Overall, noise appears to IMPROVE feature separation")
        else:
            print("✗ Overall, noise appears to DEGRADE feature separation")
        
        if summary['noise_free']['significant_ratio'] > summary['noisy']['significant_ratio']:
            print("✓ Noise-free data has more statistically significant features")
        else:
            print("✗ Noisy data has more statistically significant features")
        
        if summary['noise_free']['mean_effect_size'] > summary['noisy']['mean_effect_size']:
            print("✓ Noise-free data has larger effect sizes on average")
        else:
            print("✗ Noisy data has larger effect sizes on average")
        
    except Exception as e:
        print(f"Error during AD/CN feature comparison: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
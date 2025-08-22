#!/usr/bin/env python3
# =============================
# wav2vecベクトルの因子分析と可視化
# - 主成分分析（PCA）による次元削減
# - t-SNEによる可視化
# - UMAPによる可視化
# - AD/CNの分離状況の分析
# =============================

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import logging
from noise_augmented_datasets import NoiseAugmentedDataset
from noise_augmented_utils import setup_logging

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class Wav2VecFactorAnalyzer:
    """wav2vecベクトルの因子分析と可視化クラス"""
    
    def __init__(self, features_path, output_dir="plots/factor_analysis"):
        self.features_path = features_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # ログ設定
        self.logger = setup_logging()
        
        # データセットの読み込み
        self.dataset = NoiseAugmentedDataset(
            features_path=features_path,
            train_noise_types='original',
            val_noise_types='original', 
            test_noise_type='original'
        )
        
        # 特徴量とラベルの抽出
        self.features = []
        self.labels = []
        self.subject_ids = []
        self.data_types = []
        
        self._extract_features()
        
    def _extract_features(self):
        """特徴量とラベルを抽出"""
        self.logger.info("Extracting features and labels...")
        
        for i in range(len(self.dataset)):
            features, label, mask = self.dataset[i]
            
            # マスクを使用して有効な特徴量のみを抽出
            valid_features = features[~mask]  # ゼロパディング部分を除外
            
            if len(valid_features) > 0:
                # 時間次元で平均を取って1つのベクトルに
                feature_vector = valid_features.mean(dim=0).numpy()
                
                self.features.append(feature_vector)
                self.labels.append(label)
                self.subject_ids.append(self.dataset.subject_ids[i])
                self.data_types.append(self.dataset.data_types[i])
        
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        self.subject_ids = np.array(self.subject_ids)
        self.data_types = np.array(self.data_types)
        
        self.logger.info(f"Extracted {len(self.features)} feature vectors with shape {self.features.shape}")
        self.logger.info(f"Label distribution: {np.bincount(self.labels)}")
        
    def perform_pca_analysis(self, n_components=2):
        """主成分分析（PCA）を実行"""
        self.logger.info(f"Performing PCA with {n_components} components...")
        
        # データの標準化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        # PCA実行
        pca = PCA(n_components=n_components, random_state=42)
        features_pca = pca.fit_transform(features_scaled)
        
        # 寄与率の計算
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        self.logger.info(f"Explained variance ratio: {explained_variance_ratio}")
        self.logger.info(f"Cumulative explained variance ratio: {cumulative_variance_ratio}")
        
        return features_pca, pca, scaler, explained_variance_ratio
    
    def perform_tsne_analysis(self, n_components=2, perplexity=30):
        """t-SNE分析を実行"""
        self.logger.info(f"Performing t-SNE with {n_components} components and perplexity {perplexity}...")
        
        # データの標準化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        # t-SNE実行
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        features_tsne = tsne.fit_transform(features_scaled)
        
        return features_tsne, tsne, scaler
    
    def perform_umap_analysis(self, n_components=2, n_neighbors=15, min_dist=0.1):
        """UMAP分析を実行"""
        self.logger.info(f"Performing UMAP with {n_components} components...")
        
        # データの標準化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        # UMAP実行
        umap_reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                                min_dist=min_dist, random_state=42)
        features_umap = umap_reducer.fit_transform(features_scaled)
        
        return features_umap, umap_reducer, scaler
    
    def calculate_separation_metrics(self, features_2d):
        """分離指標を計算"""
        # シルエットスコアの計算
        silhouette = silhouette_score(features_2d, self.labels)
        
        # クラス間距離の計算
        ad_indices = np.where(self.labels == 1)[0]
        cn_indices = np.where(self.labels == 0)[0]
        
        ad_center = np.mean(features_2d[ad_indices], axis=0)
        cn_center = np.mean(features_2d[cn_indices], axis=0)
        
        class_distance = np.linalg.norm(ad_center - cn_center)
        
        # クラス内分散の計算
        ad_variance = np.var(features_2d[ad_indices], axis=0).mean()
        cn_variance = np.var(features_2d[cn_indices], axis=0).mean()
        
        self.logger.info(f"Silhouette score: {silhouette:.4f}")
        self.logger.info(f"Class distance: {class_distance:.4f}")
        self.logger.info(f"AD variance: {ad_variance:.4f}")
        self.logger.info(f"CN variance: {cn_variance:.4f}")
        
        return {
            'silhouette_score': silhouette,
            'class_distance': class_distance,
            'ad_variance': ad_variance,
            'cn_variance': cn_variance
        }
    
    def plot_2d_scatter(self, features_2d, method_name, metrics=None):
        """2次元散布図を作成"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # ラベルに基づいて色分け
        colors = ['blue' if label == 0 else 'red' for label in self.labels]
        labels_text = ['CN' if label == 0 else 'AD' for label in self.labels]
        
        # 散布図の作成
        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                           c=colors, alpha=0.6, s=50)
        
        # 凡例の作成
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', alpha=0.6, label='CN'),
                          Patch(facecolor='red', alpha=0.6, label='AD')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # タイトルとラベル
        ax.set_title(f'{method_name} Visualization of wav2vec Features', fontsize=16)
        ax.set_xlabel(f'{method_name} Component 1', fontsize=12)
        ax.set_ylabel(f'{method_name} Component 2', fontsize=12)
        
        # グリッド
        ax.grid(True, alpha=0.3)
        
        # メトリクスの表示
        if metrics:
            metrics_text = f"Silhouette: {metrics['silhouette_score']:.3f}\n"
            metrics_text += f"Class Distance: {metrics['class_distance']:.3f}\n"
            metrics_text += f"AD Variance: {metrics['ad_variance']:.3f}\n"
            metrics_text += f"CN Variance: {metrics['cn_variance']:.3f}"
            
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, f'{method_name.lower()}_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved {method_name} visualization to {output_path}")
    
    def plot_pca_variance(self, explained_variance_ratio):
        """PCAの寄与率をプロット"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 寄与率の棒グラフ
        n_components = len(explained_variance_ratio)
        ax1.bar(range(1, n_components + 1), explained_variance_ratio, alpha=0.7)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('PCA Explained Variance Ratio')
        ax1.grid(True, alpha=0.3)
        
        # 累積寄与率の折れ線グラフ
        cumulative_variance = np.cumsum(explained_variance_ratio)
        ax2.plot(range(1, n_components + 1), cumulative_variance, 'bo-', linewidth=2, markersize=6)
        ax2.set_xlabel('Number of Principal Components')
        ax2.set_ylabel('Cumulative Explained Variance Ratio')
        ax2.set_title('Cumulative Explained Variance Ratio')
        ax2.grid(True, alpha=0.3)
        
        # 95%の線を追加
        ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
        ax2.legend()
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, 'pca_variance_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved PCA variance analysis to {output_path}")
    
    def plot_data_distribution(self):
        """データ分布の可視化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # ラベル分布
        label_counts = np.bincount(self.labels)
        ax1.bar(['CN', 'AD'], label_counts, color=['blue', 'red'], alpha=0.7)
        ax1.set_title('Label Distribution')
        ax1.set_ylabel('Count')
        for i, v in enumerate(label_counts):
            ax1.text(i, v + 0.1, str(v), ha='center', va='bottom')
        
        # データタイプ分布
        data_type_counts = pd.Series(self.data_types).value_counts()
        ax2.pie(data_type_counts.values, labels=data_type_counts.index, autopct='%1.1f%%')
        ax2.set_title('Data Type Distribution')
        
        # 特徴量の統計
        feature_mean = np.mean(self.features, axis=0)
        feature_std = np.std(self.features, axis=0)
        
        ax3.hist(feature_mean, bins=50, alpha=0.7, color='green')
        ax3.set_title('Distribution of Feature Means')
        ax3.set_xlabel('Feature Mean')
        ax3.set_ylabel('Count')
        
        ax4.hist(feature_std, bins=50, alpha=0.7, color='orange')
        ax4.set_title('Distribution of Feature Standard Deviations')
        ax4.set_xlabel('Feature Standard Deviation')
        ax4.set_ylabel('Count')
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, 'data_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved data distribution to {output_path}")
    
    def run_complete_analysis(self):
        """完全な分析を実行"""
        self.logger.info("Starting complete factor analysis...")
        
        # データ分布の可視化
        self.plot_data_distribution()
        
        # PCA分析
        features_pca, pca, scaler, explained_variance_ratio = self.perform_pca_analysis(n_components=10)
        pca_metrics = self.calculate_separation_metrics(features_pca[:, :2])
        self.plot_pca_variance(explained_variance_ratio)
        self.plot_2d_scatter(features_pca[:, :2], 'PCA', pca_metrics)
        
        # t-SNE分析
        features_tsne, tsne, scaler = self.perform_tsne_analysis()
        tsne_metrics = self.calculate_separation_metrics(features_tsne)
        self.plot_2d_scatter(features_tsne, 't-SNE', tsne_metrics)
        
        # UMAP分析
        features_umap, umap_reducer, scaler = self.perform_umap_analysis()
        umap_metrics = self.calculate_separation_metrics(features_umap)
        self.plot_2d_scatter(features_umap, 'UMAP', umap_metrics)
        
        # 結果の保存
        results = {
            'pca_metrics': pca_metrics,
            'tsne_metrics': tsne_metrics,
            'umap_metrics': umap_metrics,
            'data_info': {
                'total_samples': len(self.features),
                'feature_dimension': self.features.shape[1],
                'label_distribution': np.bincount(self.labels).tolist()
            }
        }
        
        import json
        results_path = os.path.join(self.output_dir, 'factor_analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Complete analysis results saved to {results_path}")
        
        return results

def main():
    """メイン関数"""
    # パス設定
    features_path = os.path.join('dataset', 'diagnosis', 'train', 'noise_augmented_features')
    output_dir = os.path.join('plots', 'factor_analysis')
    
    # 分析の実行
    analyzer = Wav2VecFactorAnalyzer(features_path, output_dir)
    results = analyzer.run_complete_analysis()
    
    print("Factor analysis completed!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 
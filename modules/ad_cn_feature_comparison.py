#!/usr/bin/env python3
# =============================
# AD/CN特徴量比較分析
# - ノイズ付き vs ノイズなしの特徴量の違い
# - AD/CN間の特徴量の違い
# - 統計的検定による有意差の確認
# - PCAによる次元削減と可視化
# =============================

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind
import logging
from noise_augmented_datasets import NoiseAugmentedDataset
from noise_augmented_utils import setup_logging

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ADCNFeatureComparator:
    """AD/CN特徴量比較分析クラス"""
    
    def __init__(self, features_path, output_dir="plots/ad_cn_comparison"):
        self.features_path = features_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # ログ設定
        self.logger = setup_logging()
        
        # データの読み込み
        self.noise_free_data = self._load_noise_free_data()
        self.noisy_data = self._load_noisy_data()
        
    def _load_noise_free_data(self):
        """ノイズなしデータの読み込み"""
        self.logger.info("Loading noise-free (original) data...")
        
        dataset = NoiseAugmentedDataset(
            features_path=self.features_path,
            train_noise_type='original',
            val_noise_type='original',
            test_noise_type='original'
        )
        
        return self._extract_features_from_dataset(dataset, 'original')
    
    def _load_noisy_data(self):
        """ノイズ付きデータの読み込み"""
        self.logger.info("Loading noisy data...")
        
        dataset = NoiseAugmentedDataset(
            features_path=self.features_path,
            train_noise_type='gaussian_noise_light',
            val_noise_type='gaussian_noise_light',
            test_noise_type='gaussian_noise_light'
        )
        
        return self._extract_features_from_dataset(dataset, 'gaussian_noise_light')
    
    def _extract_features_from_dataset(self, dataset, noise_type):
        """データセットから特徴量を抽出"""
        features = []
        labels = []
        subject_ids = []
        
        for i in range(len(dataset)):
            feature_tensor, label, mask = dataset[i]
            
            # マスクを使用して有効な特徴量のみを抽出
            valid_features = feature_tensor[~mask]
            
            if len(valid_features) > 0:
                # 時間次元で平均を取って1つのベクトルに
                feature_vector = valid_features.mean(dim=0).numpy()
                
                features.append(feature_vector)
                labels.append(label)
                subject_ids.append(dataset.subject_ids[i])
        
        return {
            'features': np.array(features),
            'labels': np.array(labels),
            'subject_ids': np.array(subject_ids),
            'noise_type': noise_type
        }
    
    def perform_pca_analysis(self, data, n_components=2):
        """PCA分析を実行"""
        self.logger.info(f"Performing PCA analysis with {n_components} components...")
        
        # データの標準化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(data['features'])
        
        # PCA実行
        pca = PCA(n_components=n_components, random_state=42)
        features_pca = pca.fit_transform(features_scaled)
        
        # 寄与率の計算
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        return {
            'features_pca': features_pca,
            'pca_model': pca,
            'scaler': scaler,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance_ratio': cumulative_variance_ratio
        }
    
    def analyze_feature_differences(self):
        """特徴量の違いを分析"""
        self.logger.info("Analyzing feature differences between AD and CN...")
        
        # ノイズなしデータの分析
        noise_free_stats = self._analyze_single_dataset(self.noise_free_data, "Noise-Free")
        
        # ノイズ付きデータの分析
        noisy_stats = self._analyze_single_dataset(self.noisy_data, "Noisy")
        
        # 比較分析
        comparison_stats = self._compare_noise_free_vs_noisy()
        
        return {
            'noise_free': noise_free_stats,
            'noisy': noisy_stats,
            'comparison': comparison_stats
        }
    
    def _analyze_single_dataset(self, data, dataset_name):
        """単一データセットの分析"""
        features = data['features']
        labels = data['labels']
        
        # AD/CNの分離
        ad_indices = np.where(labels == 1)[0]
        cn_indices = np.where(labels == 0)[0]
        
        ad_features = features[ad_indices]
        cn_features = features[cn_indices]
        
        self.logger.info(f"{dataset_name} - AD: {len(ad_features)}, CN: {len(cn_features)}")
        
        # 統計的検定
        p_values = []
        effect_sizes = []
        
        for i in range(features.shape[1]):
            # Mann-Whitney U検定（ノンパラメトリック）
            stat, p_value = mannwhitneyu(ad_features[:, i], cn_features[:, i], alternative='two-sided')
            p_values.append(p_value)
            
            # 効果量（Cohen's d）
            pooled_std = np.sqrt(((len(ad_features) - 1) * np.var(ad_features[:, i]) + 
                                 (len(cn_features) - 1) * np.var(cn_features[:, i])) / 
                                (len(ad_features) + len(cn_features) - 2))
            effect_size = (np.mean(ad_features[:, i]) - np.mean(cn_features[:, i])) / pooled_std
            effect_sizes.append(effect_size)
        
        # 多重比較補正（FDR）
        from statsmodels.stats.multitest import multipletests
        _, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        
        # 有意な特徴量の特定
        significant_features = np.where(p_values_corrected < 0.05)[0]
        
        return {
            'ad_features': ad_features,
            'cn_features': cn_features,
            'p_values': np.array(p_values),
            'p_values_corrected': p_values_corrected,
            'effect_sizes': np.array(effect_sizes),
            'significant_features': significant_features,
            'dataset_name': dataset_name
        }
    
    def _compare_noise_free_vs_noisy(self):
        """ノイズなし vs ノイズ付きの比較"""
        self.logger.info("Comparing noise-free vs noisy data...")
        
        # 共通の特徴量次元で比較
        min_features = min(self.noise_free_data['features'].shape[1], 
                          self.noisy_data['features'].shape[1])
        
        # 各特徴量でのAD/CN分離の違いを比較
        separation_differences = []
        
        for i in range(min_features):
            # ノイズなしでの効果量
            noise_free_effect = self._analyze_single_dataset(self.noise_free_data, "temp")['effect_sizes'][i]
            
            # ノイズ付きでの効果量
            noisy_effect = self._analyze_single_dataset(self.noisy_data, "temp")['effect_sizes'][i]
            
            # 効果量の差
            separation_differences.append(abs(noisy_effect) - abs(noise_free_effect))
        
        return {
            'separation_differences': np.array(separation_differences),
            'min_features': min_features
        }
    
    def perform_tsne_analysis(self, data, n_components=2, perplexity=30):
        """t-SNE分析を実行"""
        self.logger.info(f"Performing t-SNE analysis with {n_components} components and perplexity {perplexity}...")
        
        from sklearn.manifold import TSNE
        
        # データの標準化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(data['features'])
        
        # t-SNE実行
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        features_tsne = tsne.fit_transform(features_scaled)
        
        return {
            'features_tsne': features_tsne,
            'tsne_model': tsne,
            'scaler': scaler
        }
    
    def plot_tsne_comparison(self):
        """t-SNE比較の可視化"""
        self.logger.info("Creating t-SNE comparison visualizations...")
        
        # ノイズなしデータのt-SNE
        noise_free_tsne = self.perform_tsne_analysis(self.noise_free_data, n_components=2, perplexity=30)
        
        # ノイズ付きデータのt-SNE
        noisy_tsne = self.perform_tsne_analysis(self.noisy_data, n_components=2, perplexity=30)
        
        # 1. t-SNE可視化（2次元）
        self._plot_tsne_2d_visualization(noise_free_tsne, noisy_tsne)
        
        # 2. t-SNE空間でのAD/CN分離
        self._plot_tsne_separation_analysis(noise_free_tsne, noisy_tsne)
        
        # 3. 異なるperplexityでの比較
        self._plot_tsne_perplexity_comparison()
    
    def _plot_tsne_2d_visualization(self, noise_free_tsne, noisy_tsne):
        """t-SNE 2次元可視化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ノイズなしデータのt-SNE可視化
        features_tsne = noise_free_tsne['features_tsne']
        labels = self.noise_free_data['labels']
        
        # AD/CNで色分け
        ad_indices = np.where(labels == 1)[0]
        cn_indices = np.where(labels == 0)[0]
        
        ax1.scatter(features_tsne[cn_indices, 0], features_tsne[cn_indices, 1], 
                   c='blue', alpha=0.6, s=50, label='CN')
        ax1.scatter(features_tsne[ad_indices, 0], features_tsne[ad_indices, 1], 
                   c='red', alpha=0.6, s=50, label='AD')
        ax1.set_title('t-SNE: Noise-Free Data')
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ノイズ付きデータのt-SNE可視化
        features_tsne_noisy = noisy_tsne['features_tsne']
        labels_noisy = self.noisy_data['labels']
        
        ad_indices_noisy = np.where(labels_noisy == 1)[0]
        cn_indices_noisy = np.where(labels_noisy == 0)[0]
        
        ax2.scatter(features_tsne_noisy[cn_indices_noisy, 0], features_tsne_noisy[cn_indices_noisy, 1], 
                   c='blue', alpha=0.6, s=50, label='CN')
        ax2.scatter(features_tsne_noisy[ad_indices_noisy, 0], features_tsne_noisy[ad_indices_noisy, 1], 
                   c='red', alpha=0.6, s=50, label='AD')
        ax2.set_title('t-SNE: Noisy Data')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, 'tsne_2d_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved t-SNE 2D visualization to {output_path}")
    
    def _plot_tsne_separation_analysis(self, noise_free_tsne, noisy_tsne):
        """t-SNE空間でのAD/CN分離分析"""
        from sklearn.metrics import silhouette_score
        
        # ノイズなしデータの分離指標
        features_tsne = noise_free_tsne['features_tsne']
        labels = self.noise_free_data['labels']
        noise_free_silhouette = silhouette_score(features_tsne, labels)
        
        # ノイズ付きデータの分離指標
        features_tsne_noisy = noisy_tsne['features_tsne']
        labels_noisy = self.noisy_data['labels']
        noisy_silhouette = silhouette_score(features_tsne_noisy, labels_noisy)
        
        # クラス間距離の計算
        ad_indices = np.where(labels == 1)[0]
        cn_indices = np.where(labels == 0)[0]
        ad_center = np.mean(features_tsne[ad_indices], axis=0)
        cn_center = np.mean(features_tsne[cn_indices], axis=0)
        noise_free_distance = np.linalg.norm(ad_center - cn_center)
        
        ad_indices_noisy = np.where(labels_noisy == 1)[0]
        cn_indices_noisy = np.where(labels_noisy == 0)[0]
        ad_center_noisy = np.mean(features_tsne_noisy[ad_indices_noisy], axis=0)
        cn_center_noisy = np.mean(features_tsne_noisy[cn_indices_noisy], axis=0)
        noisy_distance = np.linalg.norm(ad_center_noisy - cn_center_noisy)
        
        # 結果の可視化
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['Noise-Free', 'Noisy']
        silhouette_scores = [noise_free_silhouette, noisy_silhouette]
        class_distances = [noise_free_distance, noisy_distance]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, silhouette_scores, width, label='Silhouette Score', alpha=0.7)
        bars2 = ax.bar(x + width/2, class_distances, width, label='Class Distance', alpha=0.7)
        
        ax.set_xlabel('Data Type')
        ax.set_ylabel('Score')
        ax.set_title('t-SNE Space Separation Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # バーの上に値を表示
        for bar, score in zip(bars1, silhouette_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{score:.3f}', ha='center', va='bottom')
        
        for bar, distance in zip(bars2, class_distances):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{distance:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, 'tsne_separation_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved t-SNE separation analysis to {output_path}")
        self.logger.info(f"Noise-Free - Silhouette: {noise_free_silhouette:.4f}, Distance: {noise_free_distance:.4f}")
        self.logger.info(f"Noisy - Silhouette: {noisy_silhouette:.4f}, Distance: {noisy_distance:.4f}")
    
    def _plot_tsne_perplexity_comparison(self):
        """異なるperplexityでのt-SNE比較"""
        self.logger.info("Comparing t-SNE with different perplexity values...")
        
        perplexities = [5, 15, 30, 50]
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, perplexity in enumerate(perplexities):
            # ノイズなしデータでt-SNE実行
            tsne_result = self.perform_tsne_analysis(self.noise_free_data, n_components=2, perplexity=perplexity)
            features_tsne = tsne_result['features_tsne']
            labels = self.noise_free_data['labels']
            
            # AD/CNで色分け
            ad_indices = np.where(labels == 1)[0]
            cn_indices = np.where(labels == 0)[0]
            
            ax = axes[i]
            ax.scatter(features_tsne[cn_indices, 0], features_tsne[cn_indices, 1], 
                      c='blue', alpha=0.6, s=50, label='CN')
            ax.scatter(features_tsne[ad_indices, 0], features_tsne[ad_indices, 1], 
                      c='red', alpha=0.6, s=50, label='AD')
            ax.set_title(f't-SNE: Perplexity = {perplexity}')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, 'tsne_perplexity_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved t-SNE perplexity comparison to {output_path}")
    
    def plot_pca_comparison(self):
        """PCA比較の可視化"""
        self.logger.info("Creating PCA comparison visualizations...")
        
        # ノイズなしデータのPCA
        noise_free_pca = self.perform_pca_analysis(self.noise_free_data, n_components=10)
        
        # ノイズ付きデータのPCA
        noisy_pca = self.perform_pca_analysis(self.noisy_data, n_components=10)
        
        # 1. PCA可視化（2次元）
        self._plot_pca_2d_visualization(noise_free_pca, noisy_pca)
        
        # 2. 寄与率の比較
        self._plot_pca_variance_comparison(noise_free_pca, noisy_pca)
        
        # 3. PCA軸の類似性分析
        self._plot_pca_axis_similarity(noise_free_pca, noisy_pca)
        
        # 4. PCA空間でのAD/CN分離
        self._plot_pca_separation_analysis(noise_free_pca, noisy_pca)
    
    def _plot_pca_2d_visualization(self, noise_free_pca, noisy_pca):
        """PCA 2次元可視化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # ノイズなしデータのPCA可視化
        features_pca = noise_free_pca['features_pca']
        labels = self.noise_free_data['labels']
        
        # AD/CNで色分け
        ad_indices = np.where(labels == 1)[0]
        cn_indices = np.where(labels == 0)[0]
        
        ax1.scatter(features_pca[cn_indices, 0], features_pca[cn_indices, 1], 
                   c='blue', alpha=0.6, s=50, label='CN')
        ax1.scatter(features_pca[ad_indices, 0], features_pca[ad_indices, 1], 
                   c='red', alpha=0.6, s=50, label='AD')
        ax1.set_title('PCA: Noise-Free Data (PC1 vs PC2)')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ノイズ付きデータのPCA可視化
        features_pca_noisy = noisy_pca['features_pca']
        labels_noisy = self.noisy_data['labels']
        
        ad_indices_noisy = np.where(labels_noisy == 1)[0]
        cn_indices_noisy = np.where(labels_noisy == 0)[0]
        
        ax2.scatter(features_pca_noisy[cn_indices_noisy, 0], features_pca_noisy[cn_indices_noisy, 1], 
                   c='blue', alpha=0.6, s=50, label='CN')
        ax2.scatter(features_pca_noisy[ad_indices_noisy, 0], features_pca_noisy[ad_indices_noisy, 1], 
                   c='red', alpha=0.6, s=50, label='AD')
        ax2.set_title('PCA: Noisy Data (PC1 vs PC2)')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # PC1 vs PC3
        ax3.scatter(features_pca[cn_indices, 0], features_pca[cn_indices, 2], 
                   c='blue', alpha=0.6, s=50, label='CN')
        ax3.scatter(features_pca[ad_indices, 0], features_pca[ad_indices, 2], 
                   c='red', alpha=0.6, s=50, label='AD')
        ax3.set_title('PCA: Noise-Free Data (PC1 vs PC3)')
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC3')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # PC1 vs PC3 (ノイズ付き)
        ax4.scatter(features_pca_noisy[cn_indices_noisy, 0], features_pca_noisy[cn_indices_noisy, 2], 
                   c='blue', alpha=0.6, s=50, label='CN')
        ax4.scatter(features_pca_noisy[ad_indices_noisy, 0], features_pca_noisy[ad_indices_noisy, 2], 
                   c='red', alpha=0.6, s=50, label='AD')
        ax4.set_title('PCA: Noisy Data (PC1 vs PC3)')
        ax4.set_xlabel('PC1')
        ax4.set_ylabel('PC3')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, 'pca_2d_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved PCA 2D visualization to {output_path}")
    
    def _plot_pca_variance_comparison(self, noise_free_pca, noisy_pca):
        """PCA寄与率の比較"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 寄与率の比較
        n_components = len(noise_free_pca['explained_variance_ratio'])
        x = range(1, n_components + 1)
        
        ax1.plot(x, noise_free_pca['explained_variance_ratio'], 'bo-', 
                label='Noise-Free', linewidth=2, markersize=6)
        ax1.plot(x, noisy_pca['explained_variance_ratio'], 'ro-', 
                label='Noisy', linewidth=2, markersize=6)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('PCA Explained Variance Ratio Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 累積寄与率の比較
        ax2.plot(x, noise_free_pca['cumulative_variance_ratio'], 'bo-', 
                label='Noise-Free', linewidth=2, markersize=6)
        ax2.plot(x, noisy_pca['cumulative_variance_ratio'], 'ro-', 
                label='Noisy', linewidth=2, markersize=6)
        ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95% threshold')
        ax2.set_xlabel('Number of Principal Components')
        ax2.set_ylabel('Cumulative Explained Variance Ratio')
        ax2.set_title('Cumulative Explained Variance Ratio Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, 'pca_variance_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved PCA variance comparison to {output_path}")
    
    def _plot_pca_axis_similarity(self, noise_free_pca, noisy_pca):
        """PCA軸の類似性分析"""
        # 主成分軸の類似性を計算
        pca1 = noise_free_pca['pca_model']
        pca2 = noisy_pca['pca_model']
        
        n_components = min(pca1.components_.shape[0], pca2.components_.shape[0])
        similarity_matrix = np.zeros((n_components, n_components))
        
        for i in range(n_components):
            for j in range(n_components):
                axis1 = pca1.components_[i]
                axis2 = pca2.components_[j]
                
                # コサイン類似度
                cos_sim = np.dot(axis1, axis2) / (np.linalg.norm(axis1) * np.linalg.norm(axis2))
                similarity_matrix[i, j] = abs(cos_sim)
        
        # 類似性マトリックスの可視化
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(n_components))
        ax.set_yticks(range(n_components))
        ax.set_xticklabels([f'PC{i+1}' for i in range(n_components)])
        ax.set_yticklabels([f'PC{i+1}' for i in range(n_components)])
        ax.set_title('PCA Axes Similarity Matrix\n(Noise-Free vs Noisy)')
        
        # 数値を表示
        for i in range(n_components):
            for j in range(n_components):
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                             ha="center", va="center", 
                             color="white" if similarity_matrix[i, j] < 0.5 else "black")
        
        plt.colorbar(im, ax=ax, label='Similarity Score')
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, 'pca_axis_similarity.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved PCA axis similarity to {output_path}")
        
        # 類似性の統計
        self.logger.info("PCA Axes Similarity Summary:")
        for i in range(n_components):
            max_sim = np.max(similarity_matrix[i, :])
            max_idx = np.argmax(similarity_matrix[i, :])
            self.logger.info(f"PC{i+1} (Noise-Free) vs PC{max_idx+1} (Noisy): {max_sim:.4f}")
    
    def _plot_pca_separation_analysis(self, noise_free_pca, noisy_pca):
        """PCA空間でのAD/CN分離分析"""
        from sklearn.metrics import silhouette_score
        
        # ノイズなしデータの分離指標
        features_pca = noise_free_pca['features_pca'][:, :2]  # 最初の2成分
        labels = self.noise_free_data['labels']
        noise_free_silhouette = silhouette_score(features_pca, labels)
        
        # ノイズ付きデータの分離指標
        features_pca_noisy = noisy_pca['features_pca'][:, :2]  # 最初の2成分
        labels_noisy = self.noisy_data['labels']
        noisy_silhouette = silhouette_score(features_pca_noisy, labels_noisy)
        
        # クラス間距離の計算
        ad_indices = np.where(labels == 1)[0]
        cn_indices = np.where(labels == 0)[0]
        ad_center = np.mean(features_pca[ad_indices], axis=0)
        cn_center = np.mean(features_pca[cn_indices], axis=0)
        noise_free_distance = np.linalg.norm(ad_center - cn_center)
        
        ad_indices_noisy = np.where(labels_noisy == 1)[0]
        cn_indices_noisy = np.where(labels_noisy == 0)[0]
        ad_center_noisy = np.mean(features_pca_noisy[ad_indices_noisy], axis=0)
        cn_center_noisy = np.mean(features_pca_noisy[cn_indices_noisy], axis=0)
        noisy_distance = np.linalg.norm(ad_center_noisy - cn_center_noisy)
        
        # 結果の可視化
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['Noise-Free', 'Noisy']
        silhouette_scores = [noise_free_silhouette, noisy_silhouette]
        class_distances = [noise_free_distance, noisy_distance]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, silhouette_scores, width, label='Silhouette Score', alpha=0.7)
        bars2 = ax.bar(x + width/2, class_distances, width, label='Class Distance', alpha=0.7)
        
        ax.set_xlabel('Data Type')
        ax.set_ylabel('Score')
        ax.set_title('PCA Space Separation Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # バーの上に値を表示
        for bar, score in zip(bars1, silhouette_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{score:.3f}', ha='center', va='bottom')
        
        for bar, distance in zip(bars2, class_distances):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{distance:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, 'pca_separation_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved PCA separation analysis to {output_path}")
        self.logger.info(f"Noise-Free - Silhouette: {noise_free_silhouette:.4f}, Distance: {noise_free_distance:.4f}")
        self.logger.info(f"Noisy - Silhouette: {noisy_silhouette:.4f}, Distance: {noisy_distance:.4f}")
    
    def plot_feature_comparison(self, analysis_results):
        """特徴量比較の可視化"""
        self.logger.info("Creating feature comparison visualizations...")
        
        # 1. AD/CN特徴量分布の比較（ノイズなし vs ノイズ付き）
        self._plot_feature_distributions(analysis_results)
        
        # 2. 統計的有意性の比較
        self._plot_statistical_significance(analysis_results)
        
        # 3. 効果量の比較
        self._plot_effect_sizes(analysis_results)
        
        # 4. ノイズの影響
        self._plot_noise_impact(analysis_results)
    
    def _plot_feature_distributions(self, analysis_results):
        """特徴量分布の比較プロット"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ノイズなしデータの分布
        noise_free = analysis_results['noise_free']
        ad_features = noise_free['ad_features']
        cn_features = noise_free['cn_features']
        
        # 最初の特徴量の分布
        ax1 = axes[0, 0]
        ax1.hist(ad_features[:, 0], bins=20, alpha=0.7, label='AD', color='red', density=True)
        ax1.hist(cn_features[:, 0], bins=20, alpha=0.7, label='CN', color='blue', density=True)
        ax1.set_title('Noise-Free: Feature 0 Distribution')
        ax1.set_xlabel('Feature Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2番目の特徴量の分布
        ax2 = axes[0, 1]
        ax2.hist(ad_features[:, 1], bins=20, alpha=0.7, label='AD', color='red', density=True)
        ax2.hist(cn_features[:, 1], bins=20, alpha=0.7, label='CN', color='blue', density=True)
        ax2.set_title('Noise-Free: Feature 1 Distribution')
        ax2.set_xlabel('Feature Value')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # ノイズ付きデータの分布
        noisy = analysis_results['noisy']
        ad_features_noisy = noisy['ad_features']
        cn_features_noisy = noisy['cn_features']
        
        # 最初の特徴量の分布（ノイズ付き）
        ax3 = axes[1, 0]
        ax3.hist(ad_features_noisy[:, 0], bins=20, alpha=0.7, label='AD', color='red', density=True)
        ax3.hist(cn_features_noisy[:, 0], bins=20, alpha=0.7, label='CN', color='blue', density=True)
        ax3.set_title('Noisy: Feature 0 Distribution')
        ax3.set_xlabel('Feature Value')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 2番目の特徴量の分布（ノイズ付き）
        ax4 = axes[1, 1]
        ax4.hist(ad_features_noisy[:, 1], bins=20, alpha=0.7, label='AD', color='red', density=True)
        ax4.hist(cn_features_noisy[:, 1], bins=20, alpha=0.7, label='CN', color='blue', density=True)
        ax4.set_title('Noisy: Feature 1 Distribution')
        ax4.set_xlabel('Feature Value')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, 'feature_distributions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved feature distributions to {output_path}")
    
    def _plot_statistical_significance(self, analysis_results):
        """統計的有意性の比較プロット"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ノイズなしデータのp値
        noise_free = analysis_results['noise_free']
        p_values_noise_free = -np.log10(noise_free['p_values_corrected'])
        
        ax1.hist(p_values_noise_free, bins=50, alpha=0.7, color='blue', label='Noise-Free')
        ax1.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p=0.05 threshold')
        ax1.set_title('Statistical Significance: Noise-Free Data')
        ax1.set_xlabel('-log10(p-value)')
        ax1.set_ylabel('Number of Features')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ノイズ付きデータのp値
        noisy = analysis_results['noisy']
        p_values_noisy = -np.log10(noisy['p_values_corrected'])
        
        ax2.hist(p_values_noisy, bins=50, alpha=0.7, color='orange', label='Noisy')
        ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p=0.05 threshold')
        ax2.set_title('Statistical Significance: Noisy Data')
        ax2.set_xlabel('-log10(p-value)')
        ax2.set_ylabel('Number of Features')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, 'statistical_significance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved statistical significance plot to {output_path}")
    
    def _plot_effect_sizes(self, analysis_results):
        """効果量の比較プロット"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ノイズなしデータの効果量
        noise_free = analysis_results['noise_free']
        effect_sizes_noise_free = noise_free['effect_sizes']
        
        ax1.hist(effect_sizes_noise_free, bins=50, alpha=0.7, color='blue', label='Noise-Free')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_title('Effect Sizes: Noise-Free Data')
        ax1.set_xlabel("Cohen's d")
        ax1.set_ylabel('Number of Features')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ノイズ付きデータの効果量
        noisy = analysis_results['noisy']
        effect_sizes_noisy = noisy['effect_sizes']
        
        ax2.hist(effect_sizes_noisy, bins=50, alpha=0.7, color='orange', label='Noisy')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Effect Sizes: Noisy Data')
        ax2.set_xlabel("Cohen's d")
        ax2.set_ylabel('Number of Features')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, 'effect_sizes.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved effect sizes plot to {output_path}")
    
    def _plot_noise_impact(self, analysis_results):
        """ノイズの影響の可視化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 効果量の変化
        comparison = analysis_results['comparison']
        separation_differences = comparison['separation_differences']
        
        ax1.hist(separation_differences, bins=50, alpha=0.7, color='green')
        ax1.axvline(x=0, color='red', linestyle='--', label='No change')
        ax1.set_title('Impact of Noise on Feature Separation')
        ax1.set_xlabel('Change in Effect Size (Noisy - Noise-Free)')
        ax1.set_ylabel('Number of Features')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 上位特徴量の比較
        noise_free = analysis_results['noise_free']
        noisy = analysis_results['noisy']
        
        # 効果量の絶対値でソート
        top_features_noise_free = np.argsort(np.abs(noise_free['effect_sizes']))[-20:]
        top_features_noisy = np.argsort(np.abs(noisy['effect_sizes']))[-20:]
        
        # 上位20特徴量の効果量を比較
        x = range(20)
        y1 = np.abs(noise_free['effect_sizes'][top_features_noise_free])
        y2 = np.abs(noisy['effect_sizes'][top_features_noisy])
        
        ax2.plot(x, y1, 'bo-', label='Noise-Free', markersize=6)
        ax2.plot(x, y2, 'ro-', label='Noisy', markersize=6)
        ax2.set_title('Top 20 Features: Effect Size Comparison')
        ax2.set_xlabel('Feature Rank')
        ax2.set_ylabel('Absolute Effect Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, 'noise_impact.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved noise impact plot to {output_path}")
    
    def plot_pca_vs_tsne_comparison(self):
        """PCA vs t-SNEの比較"""
        self.logger.info("Creating PCA vs t-SNE comparison...")
        
        # PCA分析
        noise_free_pca = self.perform_pca_analysis(self.noise_free_data, n_components=2)
        noisy_pca = self.perform_pca_analysis(self.noisy_data, n_components=2)
        
        # t-SNE分析
        noise_free_tsne = self.perform_tsne_analysis(self.noise_free_data, n_components=2, perplexity=30)
        noisy_tsne = self.perform_tsne_analysis(self.noisy_data, n_components=2, perplexity=30)
        
        # 比較プロット
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # ノイズなしデータのPCA
        features_pca = noise_free_pca['features_pca']
        labels = self.noise_free_data['labels']
        ad_indices = np.where(labels == 1)[0]
        cn_indices = np.where(labels == 0)[0]
        
        ax1.scatter(features_pca[cn_indices, 0], features_pca[cn_indices, 1], 
                   c='blue', alpha=0.6, s=50, label='CN')
        ax1.scatter(features_pca[ad_indices, 0], features_pca[ad_indices, 1], 
                   c='red', alpha=0.6, s=50, label='AD')
        ax1.set_title('PCA: Noise-Free Data')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ノイズなしデータのt-SNE
        features_tsne = noise_free_tsne['features_tsne']
        ax2.scatter(features_tsne[cn_indices, 0], features_tsne[cn_indices, 1], 
                   c='blue', alpha=0.6, s=50, label='CN')
        ax2.scatter(features_tsne[ad_indices, 0], features_tsne[ad_indices, 1], 
                   c='red', alpha=0.6, s=50, label='AD')
        ax2.set_title('t-SNE: Noise-Free Data')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # ノイズ付きデータのPCA
        features_pca_noisy = noisy_pca['features_pca']
        labels_noisy = self.noisy_data['labels']
        ad_indices_noisy = np.where(labels_noisy == 1)[0]
        cn_indices_noisy = np.where(labels_noisy == 0)[0]
        
        ax3.scatter(features_pca_noisy[cn_indices_noisy, 0], features_pca_noisy[cn_indices_noisy, 1], 
                   c='blue', alpha=0.6, s=50, label='CN')
        ax3.scatter(features_pca_noisy[ad_indices_noisy, 0], features_pca_noisy[ad_indices_noisy, 1], 
                   c='red', alpha=0.6, s=50, label='AD')
        ax3.set_title('PCA: Noisy Data')
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC2')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # ノイズ付きデータのt-SNE
        features_tsne_noisy = noisy_tsne['features_tsne']
        ax4.scatter(features_tsne_noisy[cn_indices_noisy, 0], features_tsne_noisy[cn_indices_noisy, 1], 
                   c='blue', alpha=0.6, s=50, label='CN')
        ax4.scatter(features_tsne_noisy[ad_indices_noisy, 0], features_tsne_noisy[ad_indices_noisy, 1], 
                   c='red', alpha=0.6, s=50, label='AD')
        ax4.set_title('t-SNE: Noisy Data')
        ax4.set_xlabel('t-SNE 1')
        ax4.set_ylabel('t-SNE 2')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, 'pca_vs_tsne_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved PCA vs t-SNE comparison to {output_path}")
        
        # 分離指標の比較
        from sklearn.metrics import silhouette_score
        
        # PCA分離指標
        pca_silhouette_noise_free = silhouette_score(features_pca, labels)
        pca_silhouette_noisy = silhouette_score(features_pca_noisy, labels_noisy)
        
        # t-SNE分離指標
        tsne_silhouette_noise_free = silhouette_score(features_tsne, labels)
        tsne_silhouette_noisy = silhouette_score(features_tsne_noisy, labels_noisy)
        
        self.logger.info("PCA vs t-SNE Separation Comparison:")
        self.logger.info(f"PCA - Noise-Free: {pca_silhouette_noise_free:.4f}, Noisy: {pca_silhouette_noisy:.4f}")
        self.logger.info(f"t-SNE - Noise-Free: {tsne_silhouette_noise_free:.4f}, Noisy: {tsne_silhouette_noisy:.4f}")
    
    def generate_summary_report(self, analysis_results):
        """分析結果のサマリーレポートを生成"""
        self.logger.info("Generating summary report...")
        
        noise_free = analysis_results['noise_free']
        noisy = analysis_results['noisy']
        comparison = analysis_results['comparison']
        
        # 統計情報の計算
        summary = {
            'noise_free': {
                'total_features': len(noise_free['effect_sizes']),
                'significant_features': len(noise_free['significant_features']),
                'significant_ratio': len(noise_free['significant_features']) / len(noise_free['effect_sizes']),
                'mean_effect_size': np.mean(np.abs(noise_free['effect_sizes'])),
                'max_effect_size': np.max(np.abs(noise_free['effect_sizes'])),
                'ad_samples': len(noise_free['ad_features']),
                'cn_samples': len(noise_free['cn_features'])
            },
            'noisy': {
                'total_features': len(noisy['effect_sizes']),
                'significant_features': len(noisy['significant_features']),
                'significant_ratio': len(noisy['significant_features']) / len(noisy['effect_sizes']),
                'mean_effect_size': np.mean(np.abs(noisy['effect_sizes'])),
                'max_effect_size': np.max(np.abs(noisy['effect_sizes'])),
                'ad_samples': len(noisy['ad_features']),
                'cn_samples': len(noisy['cn_features'])
            },
            'comparison': {
                'mean_separation_change': np.mean(comparison['separation_differences']),
                'features_improved': np.sum(comparison['separation_differences'] > 0),
                'features_degraded': np.sum(comparison['separation_differences'] < 0)
            }
        }
        
        # レポートの出力
        self.logger.info("="*60)
        self.logger.info("AD/CN FEATURE COMPARISON SUMMARY")
        self.logger.info("="*60)
        
        self.logger.info(f"NOISE-FREE DATA:")
        self.logger.info(f"  Total features: {summary['noise_free']['total_features']}")
        self.logger.info(f"  Significant features: {summary['noise_free']['significant_features']} ({summary['noise_free']['significant_ratio']:.2%})")
        self.logger.info(f"  Mean effect size: {summary['noise_free']['mean_effect_size']:.4f}")
        self.logger.info(f"  Max effect size: {summary['noise_free']['max_effect_size']:.4f}")
        self.logger.info(f"  AD samples: {summary['noise_free']['ad_samples']}, CN samples: {summary['noise_free']['cn_samples']}")
        
        self.logger.info(f"\nNOISY DATA:")
        self.logger.info(f"  Total features: {summary['noisy']['total_features']}")
        self.logger.info(f"  Significant features: {summary['noisy']['significant_features']} ({summary['noisy']['significant_ratio']:.2%})")
        self.logger.info(f"  Mean effect size: {summary['noisy']['mean_effect_size']:.4f}")
        self.logger.info(f"  Max effect size: {summary['noisy']['max_effect_size']:.4f}")
        self.logger.info(f"  AD samples: {summary['noisy']['ad_samples']}, CN samples: {summary['noisy']['cn_samples']}")
        
        self.logger.info(f"\nNOISE IMPACT:")
        self.logger.info(f"  Mean separation change: {summary['comparison']['mean_separation_change']:.4f}")
        self.logger.info(f"  Features improved: {summary['comparison']['features_improved']}")
        self.logger.info(f"  Features degraded: {summary['comparison']['features_degraded']}")
        
        # JSONファイルに保存
        import json
        results_path = os.path.join(self.output_dir, 'ad_cn_comparison_results.json')
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"\nDetailed results saved to: {results_path}")
        
        return summary
    
    def run_complete_analysis(self):
        """完全な分析を実行"""
        self.logger.info("Starting AD/CN feature comparison analysis...")
        
        # 特徴量の違いを分析
        analysis_results = self.analyze_feature_differences()
        
        # PCA分析
        self.plot_pca_comparison()
        
        # t-SNE分析
        self.plot_tsne_comparison()

        # PCA vs t-SNE比較
        self.plot_pca_vs_tsne_comparison()
        
        # 可視化
        self.plot_feature_comparison(analysis_results)
        
        # サマリーレポート
        summary = self.generate_summary_report(analysis_results)
        
        return summary

def main():
    """メイン関数"""
    # パス設定
    features_path = os.path.join('dataset', 'diagnosis', 'train', 'noise_augmented_features')
    output_dir = os.path.join('plots', 'ad_cn_comparison')
    
    # 分析の実行
    comparator = ADCNFeatureComparator(features_path, output_dir)
    summary = comparator.run_complete_analysis()
    
    print("AD/CN feature comparison completed!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 
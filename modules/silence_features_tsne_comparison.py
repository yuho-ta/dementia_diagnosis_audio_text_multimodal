#!/usr/bin/env python3
# =============================
# Silence Features t-SNE比較分析
# - ノイズ付きサイレンス特徴量 vs ノイズなしサイレンス特徴量
# - t-SNEによる可視化と分離分析
# - 統計的比較と分離指標の計算
# =============================

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.stats import mannwhitneyu
import logging
import json

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('silence_features_tsne_comparison.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class SilenceFeaturesTSNEComparator:
    """サイレンス特徴量t-SNE比較分析クラス"""
    
    def __init__(self, silence_features_path, noise_augmented_path, output_dir="plots/silence_features_tsne"):
        self.silence_features_path = silence_features_path
        self.noise_augmented_path = noise_augmented_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # ログ設定
        self.logger = setup_logging()
        
        # データの読み込み
        self.silence_data = self._load_silence_features()
        self.noise_augmented_data = self._load_noise_augmented_features()
        
    def _load_silence_features(self):
        """ノイズなしサイレンス特徴量の読み込み"""
        self.logger.info("Loading silence features (noise-free)...")
        
        features = []
        labels = []
        subject_ids = []
        
        # ADとCNフォルダからデータを読み込み
        for class_folder in ['ad', 'cn']:
            class_path = os.path.join(self.silence_features_path, class_folder)
            
            if not os.path.exists(class_path):
                self.logger.warning(f"Class folder {class_path} does not exist")
                continue
            
            # ラベルを設定（AD=1, CN=0）
            label = 1 if class_folder == 'ad' else 0
            
            # フォルダ内のファイルを読み込み
            for filename in os.listdir(class_path):
                if filename.endswith('.pt'):
                    file_path = os.path.join(class_path, filename)
                    
                    try:
                        # PyTorchテンソルとして読み込み
                        data = torch.load(file_path)
                        
                        if isinstance(data, dict):
                            # 辞書形式の場合
                            feature_tensor = data.get('features', data.get('feature', None))
                            subject_id = data.get('subject_id', filename.replace('.pt', ''))
                        else:
                            # テンソル形式の場合
                            feature_tensor = data
                            subject_id = filename.replace('.pt', '')
                        
                        if feature_tensor is not None:
                            # テンソルをnumpy配列に変換
                            if isinstance(feature_tensor, torch.Tensor):
                                feature_array = feature_tensor.numpy()
                            else:
                                feature_array = feature_tensor
                            
                            # 1次元ベクトルに変換
                            if len(feature_array.shape) > 1:
                                feature_vector = feature_array.flatten()
                            else:
                                feature_vector = feature_array
                            
                            features.append(feature_vector)
                            labels.append(label)
                            subject_ids.append(subject_id)
                            
                    except Exception as e:
                        self.logger.warning(f"Error loading {file_path}: {e}")
                        continue
        
        if not features:
            raise ValueError("No valid features found in silence features directory")
        
        # 特徴量の次元を統一
        max_dim = max(len(f) for f in features)
        padded_features = []
        
        for feature in features:
            if len(feature) < max_dim:
                # ゼロパディング
                padded = np.pad(feature, (0, max_dim - len(feature)), 'constant')
            else:
                padded = feature[:max_dim]
            padded_features.append(padded)
        
        self.logger.info(f"Loaded {len(padded_features)} silence features: {sum(labels)} AD, {len(labels) - sum(labels)} CN")
        
        return {
            'features': np.array(padded_features),
            'labels': np.array(labels),
            'subject_ids': np.array(subject_ids),
            'feature_type': 'silence_features'
        }
    
    def _load_noise_augmented_features(self):
        """ノイズ付きサイレンス特徴量の読み込み"""
        self.logger.info("Loading noise-augmented silence features...")
        
        features = []
        labels = []
        subject_ids = []
        
        # ADとCNフォルダからデータを読み込み
        for class_folder in ['ad', 'cn']:
            class_path = os.path.join(self.noise_augmented_path, class_folder)
            
            if not os.path.exists(class_path):
                self.logger.warning(f"Class folder {class_path} does not exist")
                continue
            
            # ラベルを設定（AD=1, CN=0）
            label = 1 if class_folder == 'ad' else 0
            
            # フォルダ内のファイルを読み込み
            for filename in os.listdir(class_path):
                if filename.endswith('.pt'):
                    file_path = os.path.join(class_path, filename)
                    
                    try:
                        # PyTorchテンソルとして読み込み
                        data = torch.load(file_path)
                        
                        if isinstance(data, dict):
                            # 辞書形式の場合
                            feature_tensor = data.get('features', data.get('feature', None))
                            subject_id = data.get('subject_id', filename.replace('.pt', ''))
                        else:
                            # テンソル形式の場合
                            feature_tensor = data
                            subject_id = filename.replace('.pt', '')
                        
                        if feature_tensor is not None:
                            # テンソルをnumpy配列に変換
                            if isinstance(feature_tensor, torch.Tensor):
                                feature_array = feature_tensor.numpy()
                            else:
                                feature_array = feature_tensor
                            
                            # 1次元ベクトルに変換
                            if len(feature_array.shape) > 1:
                                feature_vector = feature_array.flatten()
                            else:
                                feature_vector = feature_array
                            
                            features.append(feature_vector)
                            labels.append(label)
                            subject_ids.append(subject_id)
                            
                    except Exception as e:
                        self.logger.warning(f"Error loading {file_path}: {e}")
                        continue
        
        if not features:
            raise ValueError("No valid features found in noise-augmented features directory")
        
        # 特徴量の次元を統一
        max_dim = max(len(f) for f in features)
        padded_features = []
        
        for feature in features:
            if len(feature) < max_dim:
                # ゼロパディング
                padded = np.pad(feature, (0, max_dim - len(feature)), 'constant')
            else:
                padded = feature[:max_dim]
            padded_features.append(padded)
        
        self.logger.info(f"Loaded {len(padded_features)} noise-augmented features: {sum(labels)} AD, {len(labels) - sum(labels)} CN")
        
        return {
            'features': np.array(padded_features),
            'labels': np.array(labels),
            'subject_ids': np.array(subject_ids),
            'feature_type': 'noise_augmented_silence_features'
        }
    
    def perform_tsne_analysis(self, data, n_components=2, perplexity=30, max_samples=1000, use_pca=True):
        """t-SNE分析を実行（メモリ効率化版）"""
        self.logger.info(f"Performing t-SNE analysis with {n_components} components and perplexity {perplexity}...")
        
        features = data['features']
        labels = data['labels']
        
        # データサイズの確認
        n_samples, n_features = features.shape
        self.logger.info(f"Original data: {n_samples} samples, {n_features} features")
        
        # サンプル数が多すぎる場合はサンプリング
        if n_samples > max_samples:
            self.logger.info(f"Sampling {max_samples} samples from {n_samples} total samples")
            indices = np.random.choice(n_samples, max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]
            n_samples = max_samples
        
        # データの標準化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 次元削減（PCA）を適用してメモリ使用量を削減
        if use_pca and n_features > 50:
            from sklearn.decomposition import PCA
            # 50次元に削減（t-SNEの計算を軽くするため）
            pca = PCA(n_components=min(50, n_features))
            features_scaled = pca.fit_transform(features_scaled)
            self.logger.info(f"Applied PCA: reduced to {features_scaled.shape[1]} dimensions")
        
        # perplexityをサンプル数に応じて調整
        adjusted_perplexity = min(perplexity, n_samples // 3)
        if adjusted_perplexity < perplexity:
            self.logger.info(f"Adjusted perplexity from {perplexity} to {adjusted_perplexity} due to sample size")
        
        # メモリ効率的なt-SNE実行
        try:
            # より軽量な設定でt-SNEを実行
            tsne = TSNE(
                n_components=n_components, 
                perplexity=adjusted_perplexity, 
                random_state=42,
                max_iter=1000,  # 反復回数を減らす（n_iterからmax_iterに変更）
                learning_rate='auto',  # 自動学習率
                method='barnes_hut' if n_samples > 5000 else 'exact'  # 大規模データの場合はBarnes-Hut
            )
            features_tsne = tsne.fit_transform(features_scaled)
            
        except MemoryError:
            self.logger.warning("Memory error with current settings, trying with reduced parameters...")
            # さらに軽量な設定で再試行
            tsne = TSNE(
                n_components=n_components, 
                perplexity=max(5, adjusted_perplexity // 2), 
                random_state=42,
                max_iter=500,  # さらに反復回数を減らす（n_iterからmax_iterに変更）
                learning_rate='auto',
                method='barnes_hut'
            )
            features_tsne = tsne.fit_transform(features_scaled)
        
        return {
            'features_tsne': features_tsne,
            'tsne_model': tsne,
            'scaler': scaler,
            'labels': labels,
            'n_samples_used': n_samples
        }
    
    def calculate_separation_metrics(self, tsne_result):
        """分離指標を計算"""
        features_tsne = tsne_result['features_tsne']
        labels = tsne_result['labels']
        
        # Silhouette score
        silhouette = silhouette_score(features_tsne, labels)
        
        # クラス間距離
        ad_indices = np.where(labels == 1)[0]
        cn_indices = np.where(labels == 0)[0]
        
        if len(ad_indices) > 0 and len(cn_indices) > 0:
            ad_center = np.mean(features_tsne[ad_indices], axis=0)
            cn_center = np.mean(features_tsne[cn_indices], axis=0)
            class_distance = np.linalg.norm(ad_center - cn_center)
        else:
            class_distance = 0
        
        # クラス内分散
        ad_variance = np.var(features_tsne[ad_indices], axis=0).mean() if len(ad_indices) > 0 else 0
        cn_variance = np.var(features_tsne[cn_indices], axis=0).mean() if len(cn_indices) > 0 else 0
        
        return {
            'silhouette_score': silhouette,
            'class_distance': class_distance,
            'ad_variance': ad_variance,
            'cn_variance': cn_variance,
            'ad_samples': len(ad_indices),
            'cn_samples': len(cn_indices)
        }
    
    def plot_tsne_comparison(self):
        """t-SNE比較の可視化"""
        self.logger.info("Creating t-SNE comparison visualizations...")
        
        # サイレンス特徴量のt-SNE
        silence_tsne = self.perform_tsne_analysis(self.silence_data, n_components=2, perplexity=30)
        
        # ノイズ付き特徴量のt-SNE
        noise_augmented_tsne = self.perform_tsne_analysis(self.noise_augmented_data, n_components=2, perplexity=30)
        
        # 1. t-SNE可視化（2次元）
        self._plot_tsne_2d_visualization(silence_tsne, noise_augmented_tsne)
        
        # 2. t-SNE空間でのAD/CN分離分析
        self._plot_tsne_separation_analysis(silence_tsne, noise_augmented_tsne)
        
        # 3. 異なるperplexityでの比較
        self._plot_tsne_perplexity_comparison()
        
        # 4. 統計的比較
        self._perform_statistical_comparison(silence_tsne, noise_augmented_tsne)
    
    def _plot_tsne_2d_visualization(self, silence_tsne, noise_augmented_tsne):
        """t-SNE 2次元可視化"""
        # フォントサイズを大きく設定
        plt.rcParams.update({'font.size': 14})
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # サイレンス特徴量のt-SNE可視化
        features_tsne = silence_tsne['features_tsne']
        labels = silence_tsne['labels']  # サンプリングされたラベルを使用
        
        # AD/CNで色分け
        ad_indices = np.where(labels == 1)[0]
        cn_indices = np.where(labels == 0)[0]
        
        ax1.scatter(features_tsne[cn_indices, 0], features_tsne[cn_indices, 1], 
                   c='blue', alpha=0.6, s=50, label='CN')
        ax1.scatter(features_tsne[ad_indices, 0], features_tsne[ad_indices, 1], 
                   c='red', alpha=0.6, s=50, label='AD')
        ax1.set_title('t-SNE: Silence Features (Noise-Free)', fontsize=16, fontweight='bold')
        ax1.set_xlabel('t-SNE 1', fontsize=14)
        ax1.set_ylabel('t-SNE 2', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.grid(True, alpha=0.3)
        
        # ノイズ付き特徴量のt-SNE可視化
        features_tsne_noisy = noise_augmented_tsne['features_tsne']
        labels_noisy = noise_augmented_tsne['labels']  # サンプリングされたラベルを使用
        
        ad_indices_noisy = np.where(labels_noisy == 1)[0]
        cn_indices_noisy = np.where(labels_noisy == 0)[0]
        
        ax2.scatter(features_tsne_noisy[cn_indices_noisy, 0], features_tsne_noisy[cn_indices_noisy, 1], 
                   c='blue', alpha=0.6, s=50, label='CN')
        ax2.scatter(features_tsne_noisy[ad_indices_noisy, 0], features_tsne_noisy[ad_indices_noisy, 1], 
                   c='red', alpha=0.6, s=50, label='AD')
        ax2.set_title('t-SNE: Noise-Augmented Silence Features', fontsize=16, fontweight='bold')
        ax2.set_xlabel('t-SNE 1', fontsize=14)
        ax2.set_ylabel('t-SNE 2', fontsize=14)
        ax2.legend(fontsize=12)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, 'tsne_2d_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved t-SNE 2D visualization to {output_path}")
    
    def _plot_tsne_separation_analysis(self, silence_tsne, noise_augmented_tsne):
        """t-SNE空間でのAD/CN分離分析"""
        # サイレンス特徴量の分離指標
        silence_metrics = self.calculate_separation_metrics(silence_tsne)
        
        # ノイズ付き特徴量の分離指標
        noise_augmented_metrics = self.calculate_separation_metrics(noise_augmented_tsne)
        
        # 結果の可視化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Silhouette Score比較
        methods = ['Silence Features', 'Noise-Augmented']
        silhouette_scores = [silence_metrics['silhouette_score'], noise_augmented_metrics['silhouette_score']]
        
        bars1 = ax1.bar(methods, silhouette_scores, color=['blue', 'orange'], alpha=0.7)
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Silhouette Score Comparison')
        ax1.grid(True, alpha=0.3)
        
        # バーの上に値を表示
        for bar, score in zip(bars1, silhouette_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 2. クラス間距離比較
        class_distances = [silence_metrics['class_distance'], noise_augmented_metrics['class_distance']]
        
        bars2 = ax2.bar(methods, class_distances, color=['blue', 'orange'], alpha=0.7)
        ax2.set_ylabel('Class Distance')
        ax2.set_title('Class Distance Comparison')
        ax2.grid(True, alpha=0.3)
        
        for bar, distance in zip(bars2, class_distances):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{distance:.3f}', ha='center', va='bottom')
        
        # 3. クラス内分散比較
        variances = [silence_metrics['ad_variance'], silence_metrics['cn_variance'],
                    noise_augmented_metrics['ad_variance'], noise_augmented_metrics['cn_variance']]
        variance_labels = ['Silence AD', 'Silence CN', 'Noisy AD', 'Noisy CN']
        
        bars3 = ax3.bar(variance_labels, variances, color=['red', 'blue', 'red', 'blue'], alpha=0.7)
        ax3.set_ylabel('Variance')
        ax3.set_title('Class Variance Comparison')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        for bar, var in zip(bars3, variances):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{var:.3f}', ha='center', va='bottom')
        
        # 4. サンプル数比較
        sample_counts = [silence_metrics['ad_samples'], silence_metrics['cn_samples'],
                        noise_augmented_metrics['ad_samples'], noise_augmented_metrics['cn_samples']]
        
        bars4 = ax4.bar(variance_labels, sample_counts, color=['red', 'blue', 'red', 'blue'], alpha=0.7)
        ax4.set_ylabel('Number of Samples')
        ax4.set_title('Sample Count Comparison')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        for bar, count in zip(bars4, sample_counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, 'tsne_separation_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved t-SNE separation analysis to {output_path}")
        self.logger.info(f"Silence Features - Silhouette: {silence_metrics['silhouette_score']:.4f}, Distance: {silence_metrics['class_distance']:.4f}")
        self.logger.info(f"Noise-Augmented - Silhouette: {noise_augmented_metrics['silhouette_score']:.4f}, Distance: {noise_augmented_metrics['class_distance']:.4f}")
    
    def _plot_tsne_perplexity_comparison(self):
        """異なるperplexityでのt-SNE比較"""
        self.logger.info("Comparing t-SNE with different perplexity values...")
        
        perplexities = [5, 15, 30, 50]
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        for i, perplexity in enumerate(perplexities):
            # サイレンス特徴量でt-SNE実行
            silence_tsne = self.perform_tsne_analysis(self.silence_data, n_components=2, perplexity=perplexity)
            features_tsne = silence_tsne['features_tsne']
            labels = silence_tsne['labels']
            
            # AD/CNで色分け
            ad_indices = np.where(labels == 1)[0]
            cn_indices = np.where(labels == 0)[0]
            
            ax1 = axes[0, i]
            ax1.scatter(features_tsne[cn_indices, 0], features_tsne[cn_indices, 1], 
                       c='blue', alpha=0.6, s=50, label='CN')
            ax1.scatter(features_tsne[ad_indices, 0], features_tsne[ad_indices, 1], 
                       c='red', alpha=0.6, s=50, label='AD')
            ax1.set_title(f'Silence Features\nPerplexity = {perplexity}')
            ax1.set_xlabel('t-SNE 1')
            ax1.set_ylabel('t-SNE 2')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # ノイズ付き特徴量でt-SNE実行
            noise_tsne = self.perform_tsne_analysis(self.noise_augmented_data, n_components=2, perplexity=perplexity)
            features_tsne_noisy = noise_tsne['features_tsne']
            labels_noisy = noise_tsne['labels']
            
            ad_indices_noisy = np.where(labels_noisy == 1)[0]
            cn_indices_noisy = np.where(labels_noisy == 0)[0]
            
            ax2 = axes[1, i]
            ax2.scatter(features_tsne_noisy[cn_indices_noisy, 0], features_tsne_noisy[cn_indices_noisy, 1], 
                       c='blue', alpha=0.6, s=50, label='CN')
            ax2.scatter(features_tsne_noisy[ad_indices_noisy, 0], features_tsne_noisy[ad_indices_noisy, 1], 
                       c='red', alpha=0.6, s=50, label='AD')
            ax2.set_title(f'Noise-Augmented\nPerplexity = {perplexity}')
            ax2.set_xlabel('t-SNE 1')
            ax2.set_ylabel('t-SNE 2')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, 'tsne_perplexity_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved t-SNE perplexity comparison to {output_path}")
    
    def _perform_statistical_comparison(self, silence_tsne, noise_augmented_tsne):
        """統計的比較を実行"""
        self.logger.info("Performing statistical comparison...")
        
        # 特徴量の統計的比較
        silence_features = self.silence_data['features']
        noise_augmented_features = self.noise_augmented_data['features']
        
        # 共通の次元数で比較
        min_dim = min(silence_features.shape[1], noise_augmented_features.shape[1])
        
        # 各特徴量での統計的検定
        p_values = []
        effect_sizes = []
        
        for i in range(min_dim):
            # Mann-Whitney U検定
            stat, p_value = mannwhitneyu(silence_features[:, i], noise_augmented_features[:, i], alternative='two-sided')
            p_values.append(p_value)
            
            # 効果量（Cohen's d）
            pooled_std = np.sqrt(((len(silence_features) - 1) * np.var(silence_features[:, i]) + 
                                 (len(noise_augmented_features) - 1) * np.var(noise_augmented_features[:, i])) / 
                                (len(silence_features) + len(noise_augmented_features) - 2))
            effect_size = (np.mean(silence_features[:, i]) - np.mean(noise_augmented_features[:, i])) / pooled_std
            effect_sizes.append(effect_size)
        
        # 多重比較補正
        from statsmodels.stats.multitest import multipletests
        _, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        
        # 有意な特徴量の特定
        significant_features = np.where(p_values_corrected < 0.05)[0]
        
        # 結果の可視化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # p値の分布
        ax1.hist(-np.log10(p_values_corrected), bins=50, alpha=0.7, color='green')
        ax1.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p=0.05 threshold')
        ax1.set_title('Statistical Significance Distribution')
        ax1.set_xlabel('-log10(p-value)')
        ax1.set_ylabel('Number of Features')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 効果量の分布
        ax2.hist(effect_sizes, bins=50, alpha=0.7, color='purple')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Effect Size Distribution')
        ax2.set_xlabel("Cohen's d")
        ax2.set_ylabel('Number of Features')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, 'statistical_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Saved statistical comparison to {output_path}")
        self.logger.info(f"Significant features: {len(significant_features)} out of {min_dim}")
        
        return {
            'p_values': np.array(p_values),
            'p_values_corrected': p_values_corrected,
            'effect_sizes': np.array(effect_sizes),
            'significant_features': significant_features
        }
    
    def generate_summary_report(self):
        """分析結果のサマリーレポートを生成"""
        self.logger.info("Generating summary report...")
        
        # 基本的な統計情報
        silence_tsne = self.perform_tsne_analysis(self.silence_data, n_components=2, perplexity=30)
        noise_augmented_tsne = self.perform_tsne_analysis(self.noise_augmented_data, n_components=2, perplexity=30)
        
        silence_metrics = self.calculate_separation_metrics(silence_tsne)
        noise_augmented_metrics = self.calculate_separation_metrics(noise_augmented_tsne)
        
        # 統計的比較
        statistical_results = self._perform_statistical_comparison(silence_tsne, noise_augmented_tsne)
        
        summary = {
            'silence_features': {
                'total_samples': len(self.silence_data['features']),
                'feature_dimensions': self.silence_data['features'].shape[1],
                'ad_samples': silence_metrics['ad_samples'],
                'cn_samples': silence_metrics['cn_samples'],
                'silhouette_score': silence_metrics['silhouette_score'],
                'class_distance': silence_metrics['class_distance'],
                'ad_variance': silence_metrics['ad_variance'],
                'cn_variance': silence_metrics['cn_variance']
            },
            'noise_augmented_features': {
                'total_samples': len(self.noise_augmented_data['features']),
                'feature_dimensions': self.noise_augmented_data['features'].shape[1],
                'ad_samples': noise_augmented_metrics['ad_samples'],
                'cn_samples': noise_augmented_metrics['cn_samples'],
                'silhouette_score': noise_augmented_metrics['silhouette_score'],
                'class_distance': noise_augmented_metrics['class_distance'],
                'ad_variance': noise_augmented_metrics['ad_variance'],
                'cn_variance': noise_augmented_metrics['cn_variance']
            },
            'comparison': {
                'significant_features': len(statistical_results['significant_features']),
                'mean_effect_size': np.mean(np.abs(statistical_results['effect_sizes'])),
                'max_effect_size': np.max(np.abs(statistical_results['effect_sizes']))
            }
        }
        
        # レポートの出力
        self.logger.info("="*60)
        self.logger.info("SILENCE FEATURES T-SNE COMPARISON SUMMARY")
        self.logger.info("="*60)
        
        self.logger.info(f"SILENCE FEATURES:")
        self.logger.info(f"  Total samples: {summary['silence_features']['total_samples']}")
        self.logger.info(f"  Feature dimensions: {summary['silence_features']['feature_dimensions']}")
        self.logger.info(f"  AD samples: {summary['silence_features']['ad_samples']}, CN samples: {summary['silence_features']['cn_samples']}")
        self.logger.info(f"  Silhouette score: {summary['silence_features']['silhouette_score']:.4f}")
        self.logger.info(f"  Class distance: {summary['silence_features']['class_distance']:.4f}")
        
        self.logger.info(f"\nNOISE-AUGMENTED FEATURES:")
        self.logger.info(f"  Total samples: {summary['noise_augmented_features']['total_samples']}")
        self.logger.info(f"  Feature dimensions: {summary['noise_augmented_features']['feature_dimensions']}")
        self.logger.info(f"  AD samples: {summary['noise_augmented_features']['ad_samples']}, CN samples: {summary['noise_augmented_features']['cn_samples']}")
        self.logger.info(f"  Silhouette score: {summary['noise_augmented_features']['silhouette_score']:.4f}")
        self.logger.info(f"  Class distance: {summary['noise_augmented_features']['class_distance']:.4f}")
        
        self.logger.info(f"\nCOMPARISON:")
        self.logger.info(f"  Significant features: {summary['comparison']['significant_features']}")
        self.logger.info(f"  Mean effect size: {summary['comparison']['mean_effect_size']:.4f}")
        self.logger.info(f"  Max effect size: {summary['comparison']['max_effect_size']:.4f}")
        
        # JSONファイルに保存
        results_path = os.path.join(self.output_dir, 'silence_features_tsne_results.json')
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"\nDetailed results saved to: {results_path}")
        
        return summary
    
    def run_complete_analysis(self):
        """完全な分析を実行"""
        self.logger.info("Starting silence features t-SNE comparison analysis...")
        
        # t-SNE比較
        self.plot_tsne_comparison()
        
        # サマリーレポート
        summary = self.generate_summary_report()
        
        return summary

def main():
    """メイン関数"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Silence Features t-SNE Comparison Analysis')
    parser.add_argument('--max-samples', type=int, default=1000, 
                       help='Maximum number of samples to use for t-SNE (default: 1000)')
    parser.add_argument('--perplexity', type=int, default=30, 
                       help='Perplexity for t-SNE (default: 30)')
    parser.add_argument('--use-pca', action='store_true', default=True,
                       help='Use PCA for dimensionality reduction (default: True)')
    parser.add_argument('--silence-path', type=str, 
                       default=os.path.join('dataset', 'diagnosis', 'train', 'silence_features'),
                       help='Path to silence features directory')
    parser.add_argument('--noise-path', type=str, 
                       default=os.path.join('dataset', 'diagnosis', 'train', 'noise_augmented_silence_features'),
                       help='Path to noise-augmented features directory')
    parser.add_argument('--output-dir', type=str, 
                       default=os.path.join('plots', 'silence_features_tsne'),
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # パス設定
    silence_features_path = args.silence_path
    noise_augmented_path = args.noise_path
    output_dir = args.output_dir
    
    print(f"Using max_samples: {args.max_samples}")
    print(f"Using perplexity: {args.perplexity}")
    print(f"Using PCA: {args.use_pca}")
    
    # 分析の実行
    comparator = SilenceFeaturesTSNEComparator(silence_features_path, noise_augmented_path, output_dir)
    
    try:
        # メモリ効率的な設定で実行
        silence_tsne = comparator.perform_tsne_analysis(
            comparator.silence_data, 
            n_components=2, 
            perplexity=args.perplexity, 
            max_samples=args.max_samples,
            use_pca=args.use_pca
        )
        noise_tsne = comparator.perform_tsne_analysis(
            comparator.noise_augmented_data, 
            n_components=2, 
            perplexity=args.perplexity, 
            max_samples=args.max_samples,
            use_pca=args.use_pca
        )
        
        # 基本的な可視化を実行
        comparator._plot_tsne_2d_visualization(silence_tsne, noise_tsne)
        
        # 分離分析を実行
        comparator._plot_tsne_separation_analysis(silence_tsne, noise_tsne)
        
        print("Silence features t-SNE comparison completed!")
        print(f"Results saved to: {output_dir}")
        
    except MemoryError as e:
        print(f"Memory error occurred: {e}")
        print("Trying with even more reduced memory settings...")
        
        # さらに軽量な設定で再試行
        try:
            silence_tsne = comparator.perform_tsne_analysis(
                comparator.silence_data, 
                n_components=2, 
                perplexity=10, 
                max_samples=200,  # さらにサンプル数を削減
                use_pca=True
            )
            noise_tsne = comparator.perform_tsne_analysis(
                comparator.noise_augmented_data, 
                n_components=2, 
                perplexity=10, 
                max_samples=200,  # さらにサンプル数を削減
                use_pca=True
            )
            
            # 基本的な可視化のみ実行
            comparator._plot_tsne_2d_visualization(silence_tsne, noise_tsne)
            
            print("Basic t-SNE analysis completed with minimal memory usage!")
            print(f"Results saved to: {output_dir}")
            
        except Exception as e2:
            print(f"Still encountering issues: {e2}")
            print("Please try reducing the dataset size or using a machine with more memory.")
            print("You can also try running with: --max-samples 100 --perplexity 5")

if __name__ == "__main__":
    main()

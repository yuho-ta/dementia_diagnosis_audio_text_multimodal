import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import librosa
import librosa.display
from scipy import signal
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 既存のモデルとデータローダーをインポート
from silence_only_transformer import (
    TransformerClassifier, 
    SilenceFeaturesDataset, 
    collate_fn,
    load_silence_features_with_split,
    extract_id_from_filename
)
from torch.utils.data import DataLoader

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradCAM:
    """Grad-CAM実装クラス"""
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.handlers = []
        
        # フックを登録
        self._register_hooks()
    
    def get_available_layers(self):
        """利用可能な層のリストを取得"""
        layers = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # リーフノードのみ
                layers.append(name)
        return layers
    
    def _register_hooks(self):
        """勾配とアクティベーションのフックを登録"""
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # ターゲット層を探してフックを登録
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                self.handlers.append(module.register_forward_hook(forward_hook))
                self.handlers.append(module.register_backward_hook(backward_hook))
                break
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Grad-CAMを生成"""
        self.model.eval()
        
        # フォワードパス
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # バックワードパス
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # 勾配とアクティベーションを取得
        gradients = self.gradients[0]  # [seq_len, hidden_dim]
        activations = self.activations[0]  # [seq_len, hidden_dim]
        
        # Grad-CAMの正しい計算: 勾配のグローバル平均プーリング
        # 各特徴次元での勾配の平均を計算
        weights = torch.mean(gradients, dim=0)  # [hidden_dim] - 特徴次元方向の平均
        
        # CAMを計算: 重み × アクティベーションの時間方向の合計
        cam = torch.zeros(activations.shape[0])  # [seq_len]
        for t in range(activations.shape[0]):  # 時間フレームごと
            cam[t] = torch.sum(weights * activations[t])  # 重み付き特徴量の合計
        
        # 正規化（ReLU + 最大値正規化）
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def generate_detailed_cam(self, input_tensor, class_idx=None):
        """詳細なGrad-CAM分析（勾配とアクティベーションの詳細情報付き）"""
        self.model.eval()
        
        # フォワードパス
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # バックワードパス
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # 勾配とアクティベーションを取得
        gradients = self.gradients[0]  # [seq_len, hidden_dim]
        activations = self.activations[0]  # [seq_len, hidden_dim]
        
        # 詳細な分析情報
        analysis_info = {
            'gradients_shape': gradients.shape,
            'activations_shape': activations.shape,
            'gradients_mean': gradients.mean().item(),
            'gradients_std': gradients.std().item(),
            'activations_mean': activations.mean().item(),
            'activations_std': activations.std().item(),
            'class_idx': class_idx.item() if hasattr(class_idx, 'item') else class_idx,
            'output_confidence': torch.sigmoid(output[0, class_idx]).item()
        }
        
        # Grad-CAMの正しい計算
        weights = torch.mean(gradients, dim=0)  # [hidden_dim]
        cam = torch.zeros(activations.shape[0])  # [seq_len]
        
        for t in range(activations.shape[0]):
            cam[t] = torch.sum(weights * activations[t])
        
        # 正規化
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # 時間フレームごとの詳細情報
        frame_details = []
        for t in range(activations.shape[0]):
            frame_details.append({
                'time_frame': t,
                'cam_score': cam[t].item(),
                'activation_norm': torch.norm(activations[t]).item(),
                'gradient_norm': torch.norm(gradients[t]).item(),
                'weighted_sum': torch.sum(weights * activations[t]).item()
            })
        
        return {
            'cam': cam.detach().cpu().numpy(),
            'analysis_info': analysis_info,
            'frame_details': frame_details,
            'weights': weights.detach().cpu().numpy(),
            'gradients': gradients.detach().cpu().numpy(),
            'activations': activations.detach().cpu().numpy()
        }
    
    def remove_hooks(self):
        """フックを削除"""
        for handler in self.handlers:
            handler.remove()

class AttentionVisualizer:
    """Attention重みの可視化クラス"""
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = None
        self.handlers = []
        
        # Attention層のフックを登録
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """Attention層のフックを登録"""
        def attention_hook(module, input, output):
            # TransformerEncoderLayerのattention_weightsを取得
            if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'attention_weights'):
                self.attention_weights = module.self_attn.attention_weights
        
        # TransformerEncoderLayerにフックを登録
        for name, module in self.model.named_modules():
            if 'transformer.layers' in name and isinstance(module, nn.TransformerEncoderLayer):
                self.handlers.append(module.register_forward_hook(attention_hook))
    
    def get_attention_weights(self, input_tensor):
        """Attention重みを取得"""
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        return self.attention_weights
    
    def remove_hooks(self):
        """フックを削除"""
        for handler in self.handlers:
            handler.remove()

class AudioSpectrogramGenerator:
    """音声ファイルからスペクトログラムを生成するクラス"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def load_audio_file(self, audio_path):
        """音声ファイルを読み込み"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return None, None
    
    def generate_spectrogram(self, audio, sr=None):
        """スペクトログラムを生成"""
        if sr is None:
            sr = self.sample_rate
        
        # 短時間フーリエ変換
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        
        # デシベルスケールに変換
        db_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        return db_spectrogram, sr
    
    def generate_mel_spectrogram(self, audio, sr=None, n_mels=128):
        """メルスペクトログラムを生成"""
        if sr is None:
            sr = self.sample_rate
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512
        )
        
        # デシベルスケールに変換
        db_mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)
        
        return db_mel_spec, sr

class XAIAnalyzer:
    """XAI分析のメインクラス"""
    
    def __init__(self, model_path, data_type='original'):
        self.model_path = model_path
        self.data_type = data_type
        self.model = None
        self.gradcam = None
        self.attention_viz = None
        self.spectrogram_gen = AudioSpectrogramGenerator()
        
        # 結果保存用
        self.results = []
        
    def load_model(self):
        """訓練済みモデルを読み込み"""
        print(f"Loading model from {self.model_path}")
        
        # モデル構造を再構築
        self.model = TransformerClassifier(input_dim=768).to(device)
        
        # モデルファイルが存在するかチェック
        if os.path.exists(self.model_path):
            # 重みを読み込み
            checkpoint = torch.load(self.model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("Trained model loaded successfully")
        else:
            # モデルファイルが存在しない場合は、ランダム初期化されたモデルを使用
            print("Model file not found. Using randomly initialized model for demonstration.")
            print("Note: This is for demonstration purposes only. For real analysis, train a model first.")
        
        self.model.eval()
        
        # XAIツールを初期化
        self.gradcam = GradCAM(self.model, 'transformer.layers.0')
        self.attention_viz = AttentionVisualizer(self.model)
        
        print("Model ready for XAI analysis")
    
    def analyze_sample(self, features, label, uid, audio_path=None, detailed=False):
        """
        単一サンプルのXAI分析
        
        データフロー:
        1. 音声波形 (raw waveform) → スペクトログラム（可視化用）
        2. 音声波形 → Wav2Vec2 → 特徴量ベクトル系列 (T, D) → Transformer
        3. Grad-CAM計算: 勾配 × 特徴マップ → 時間フレームごとの寄与度
        4. 時間軸補間: Wav2Vec2フレーム(20ms) → スペクトログラムフレーム(10ms)
        5. 可視化: スペクトログラム + Grad-CAM重ね合わせ
        """
        # 特徴量をバッチ形式に変換 (Wav2Vec2出力: [seq_len, feature_dim])
        features_batch = features.unsqueeze(0).to(device)  # [1, seq_len, feature_dim]
        
        # 予測を取得
        with torch.no_grad():
            logits = self.model(features_batch)
            probs = torch.sigmoid(logits)
            prediction = (probs > 0.5).long().item()
            confidence = probs.item()
        
        # Grad-CAMを生成（詳細版または簡易版）
        # Transformerの特定層（例: transformer.layers.0）での勾配 × 特徴マップ
        # 出力: 時間フレームごとの寄与度スコア [seq_len]
        if detailed:
            cam_result = self.gradcam.generate_detailed_cam(features_batch, class_idx=prediction)
            cam = cam_result['cam']
            cam_details = cam_result
        else:
            cam = self.gradcam.generate_cam(features_batch, class_idx=prediction)
            cam_details = None
        
        # Attention重みを取得
        attention_weights = self.attention_viz.get_attention_weights(features_batch)
        
        # スペクトログラムを生成（音声ファイルが利用可能な場合）
        # 可視化用: 元音声の時間周波数表現
        spectrogram = None
        mel_spectrogram = None
        if audio_path and os.path.exists(audio_path):
            audio, sr = self.spectrogram_gen.load_audio_file(audio_path)
            if audio is not None:
                # スペクトログラム: 10msフレーム (hop_length=512, sr=16000)
                spectrogram, _ = self.spectrogram_gen.generate_spectrogram(audio, sr)
                # メルスペクトログラム: より音声に適した周波数表現
                mel_spectrogram, _ = self.spectrogram_gen.generate_mel_spectrogram(audio, sr)
        
        result = {
            'uid': uid,
            'true_label': label,
            'predicted_label': prediction,
            'confidence': confidence,
            'correct': (label == prediction),
            'gradcam': cam,
            'gradcam_details': cam_details,
            'attention_weights': attention_weights,
            'spectrogram': spectrogram,
            'mel_spectrogram': mel_spectrogram,
            'audio_path': audio_path
        }
        
        return result
    
    def visualize_analysis(self, result, save_path=None, detailed=False):
        """分析結果を可視化"""
        if detailed and result.get('gradcam_details'):
            # 詳細版の可視化（6つのサブプロット）
            fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        else:
            # 標準版の可視化（4つのサブプロット）
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        fig.suptitle(f'XAI Analysis: {result["uid"]} (True: {result["true_label"]}, Pred: {result["predicted_label"]}, Conf: {result["confidence"]:.3f})', 
                     fontsize=14)
        
        # 1. Grad-CAM可視化
        ax1 = axes[0, 0] if not detailed else axes[0, 0]
        cam = result['gradcam']
        time_axis = np.arange(len(cam)) / 50.0  # 50Hzでサンプリングされていると仮定
        
        ax1.plot(time_axis, cam, 'b-', linewidth=2)
        ax1.fill_between(time_axis, 0, cam, alpha=0.3, color='blue')
        ax1.set_title('Grad-CAM Attention Map')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Attention Weight')
        ax1.grid(True, alpha=0.3)
        
        # Grad-CAMの統計情報を表示
        if detailed and result.get('gradcam_details'):
            details = result['gradcam_details']['analysis_info']
            ax1.text(0.02, 0.98, f'Max: {cam.max():.3f}\nMean: {cam.mean():.3f}\nStd: {cam.std():.3f}', 
                    transform=ax1.transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. スペクトログラム（利用可能な場合）
        ax2 = axes[0, 1]
        if result['spectrogram'] is not None:
            spec = result['spectrogram']
            librosa.display.specshow(spec, sr=16000, hop_length=512, 
                                   x_axis='time', y_axis='hz', ax=ax2)
            ax2.set_title('Audio Spectrogram')
        else:
            ax2.text(0.5, 0.5, 'Audio file not available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Audio Spectrogram (N/A)')
        
        # 3. メルスペクトログラム（利用可能な場合）
        ax3 = axes[1, 0]
        if result['mel_spectrogram'] is not None:
            mel_spec = result['mel_spectrogram']
            librosa.display.specshow(mel_spec, sr=16000, hop_length=512, 
                                   x_axis='time', y_axis='mel', ax=ax3)
            ax3.set_title('Mel Spectrogram')
        else:
            ax3.text(0.5, 0.5, 'Audio file not available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Mel Spectrogram (N/A)')
        
        # 4. Grad-CAMとスペクトログラムの重ね合わせ（利用可能な場合）
        ax4 = axes[1, 1]
        if result['spectrogram'] is not None:
            spec = result['spectrogram']
            librosa.display.specshow(spec, sr=16000, hop_length=512, 
                                   x_axis='time', y_axis='hz', ax=ax4, alpha=0.7)
            
            # 時間軸の補間: Wav2Vec2フレーム（20ms）→ スペクトログラムフレーム（10ms）
            # Wav2Vec2: 50Hz (20ms per frame)
            # スペクトログラム: 100Hz (10ms per frame, hop_length=512, sr=16000)
            spec_time_axis = np.linspace(0, spec.shape[1] * 512 / 16000, spec.shape[1])
            
            # 高精度補間でGrad-CAMをスペクトログラムの時間軸に合わせる
            cam_interp = np.interp(spec_time_axis, time_axis, cam)
            
            # カラーマップでGrad-CAMを重ね合わせ
            ax4_twin = ax4.twinx()
            ax4_twin.plot(spec_time_axis, cam_interp, 'r-', linewidth=3, alpha=0.8)
            ax4_twin.fill_between(spec_time_axis, 0, cam_interp, alpha=0.3, color='red')
            ax4_twin.set_ylabel('Grad-CAM Weight', color='red')
            ax4_twin.set_ylim(0, 1)
            
            # 時間軸の情報を表示
            ax4.set_title(f'Spectrogram + Grad-CAM Overlay\n(Wav2Vec2: 50Hz → Spec: 100Hz)')
        else:
            ax4.text(0.5, 0.5, 'Audio file not available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Spectrogram + Grad-CAM (N/A)')
        
        # 詳細版の追加プロット
        if detailed and result.get('gradcam_details'):
            details = result['gradcam_details']
            
            # 5. 勾配の分布
            ax5 = axes[2, 0]
            gradients = details['gradients']
            gradient_norms = np.linalg.norm(gradients, axis=1)
            ax5.plot(time_axis, gradient_norms, 'g-', linewidth=2)
            ax5.fill_between(time_axis, 0, gradient_norms, alpha=0.3, color='green')
            ax5.set_title('Gradient Norms Over Time')
            ax5.set_xlabel('Time (seconds)')
            ax5.set_ylabel('Gradient Norm')
            ax5.grid(True, alpha=0.3)
            
            # 6. アクティベーションの分布
            ax6 = axes[2, 1]
            activations = details['activations']
            activation_norms = np.linalg.norm(activations, axis=1)
            ax6.plot(time_axis, activation_norms, 'm-', linewidth=2)
            ax6.fill_between(time_axis, 0, activation_norms, alpha=0.3, color='magenta')
            ax6.set_title('Activation Norms Over Time')
            ax6.set_xlabel('Time (seconds)')
            ax6.set_ylabel('Activation Norm')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        return fig
    
    def analyze_dataset(self, dataset, output_dir='xai_results', detailed=False):
        """データセット全体のXAI分析"""
        os.makedirs(output_dir, exist_ok=True)
        
        # サンプルを選択（正解・不正解をバランス良く）
        correct_samples = []
        incorrect_samples = []
        
        print("Analyzing samples...")
        for i, sample in enumerate(tqdm(dataset.data[:100])):  # 最初の100サンプルから選択
            features = sample["features"]
            label = sample["label"]
            uid = sample["uid"]
            
            # 音声ファイルパスを構築（ラベル情報を使って効率的に検索）
            audio_path = self._construct_audio_path(uid, label)
            result = self.analyze_sample(features, label, uid, audio_path, detailed=detailed)
            
            if result['correct']:
                correct_samples.append(result)
            else:
                incorrect_samples.append(result)
    
    def analyze_test_dataset(self, test_data, output_dir='xai_results', detailed=False):
        """テストデータセット全体のXAI分析"""
        os.makedirs(output_dir, exist_ok=True)
        
        # クラス別にサンプルを分離
        ad_samples_list = []
        cn_samples_list = []
        
        for sample in test_data:
            if sample["label"] == 1:  # AD
                ad_samples_list.append(sample)
            else:  # CN
                cn_samples_list.append(sample)
        
        print(f"Found {len(ad_samples_list)} AD samples and {len(cn_samples_list)} CN samples in test data")
        print(f"Analyzing all {len(test_data)} test samples...")
        
        all_results = []
        
        # 全テストサンプルを分析
        for i, sample in enumerate(tqdm(test_data, desc="Analyzing all test samples")):
            features = sample["features"]
            label = sample["label"]
            uid = sample["uid"]
            
            # 音声ファイルパスを構築
            audio_path = self._construct_audio_path(uid, label)
            
            result = self.analyze_sample(features, label, uid, audio_path, detailed=detailed)
            result['sample_type'] = 'AD' if label == 1 else 'CN'
            all_results.append(result)
        
        # 可視化を生成（correct/incorrectをファイル名に含める）
        for i, result in enumerate(all_results):
            correctness = "correct" if result['correct'] else "incorrect"
            save_path = os.path.join(output_dir, f'{result["sample_type"].lower()}_{correctness}_{result["uid"]}.png')
            self.visualize_analysis(result, save_path, detailed=detailed)
            
            # 結果を保存
            self.results.append(result)
        
        # 統計情報を生成
        self._generate_test_summary_report(output_dir, all_results)
        
        return all_results
    
    def _generate_test_summary_report(self, output_dir, all_results):
        """テストデータ分析結果のサマリーレポートを生成"""
        if not all_results:
            return
        
        # クラス別に結果を分離
        ad_results = [r for r in all_results if r['sample_type'] == 'AD']
        cn_results = [r for r in all_results if r['sample_type'] == 'CN']
        
        # 統計情報を計算
        total_samples = len(all_results)
        correct_predictions = sum(1 for r in all_results if r['correct'])
        accuracy = correct_predictions / total_samples
        
        # クラス別統計
        ad_correct = sum(1 for r in ad_results if r['correct'])
        ad_accuracy = ad_correct / len(ad_results) if ad_results else 0
        
        cn_correct = sum(1 for r in cn_results if r['correct'])
        cn_accuracy = cn_correct / len(cn_results) if cn_results else 0
        
        # 平均信頼度
        avg_confidence = np.mean([r['confidence'] for r in all_results])
        
        # レポートを生成
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'data_type': self.data_type,
            'analysis_type': 'test_dataset',
            'total_samples_analyzed': total_samples,
            'ad_samples': len(ad_results),
            'cn_samples': len(cn_results),
            'overall_accuracy': accuracy,
            'ad_accuracy': ad_accuracy,
            'cn_accuracy': cn_accuracy,
            'average_confidence': avg_confidence,
            'samples_with_audio': sum(1 for r in all_results if r['audio_path'] is not None)
        }
        
        # JSONファイルとして保存
        report_path = os.path.join(output_dir, 'xai_test_analysis_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # テキストレポートも生成
        txt_report_path = os.path.join(output_dir, 'xai_test_analysis_summary.txt')
        with open(txt_report_path, 'w') as f:
            f.write("XAI Test Dataset Analysis Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {report['analysis_timestamp']}\n")
            f.write(f"Model Path: {report['model_path']}\n")
            f.write(f"Data Type: {report['data_type']}\n")
            f.write(f"Analysis Type: {report['analysis_type']}\n\n")
            f.write(f"Total Samples Analyzed: {report['total_samples_analyzed']}\n")
            f.write(f"AD Samples: {report['ad_samples']} (Accuracy: {report['ad_accuracy']:.3f})\n")
            f.write(f"CN Samples: {report['cn_samples']} (Accuracy: {report['cn_accuracy']:.3f})\n\n")
            f.write(f"Overall Accuracy: {report['overall_accuracy']:.3f}\n")
            f.write(f"Average Confidence: {report['average_confidence']:.3f}\n")
            f.write(f"Samples with Audio: {report['samples_with_audio']}\n")
        
        print(f"Test analysis summary report saved to {report_path} and {txt_report_path}")
    
    def _construct_audio_path(self, uid, label=None):
        """UIDから音声ファイルパスを構築"""
        # 実際の音声ファイルのパス構造に合わせて調整
        # 例: uid = "684-0_original" -> "684-0"
        base_uid = uid.replace('_original', '').replace('_noise_augmented', '')
        
        # ラベル情報を使って直接正しいディレクトリを指定
        if label is not None:
            # label: 0=CN, 1=AD
            subdir = "cn" if label == 0 else "ad"
            direct_path = f"dataset/diagnosis/train/silence_audio/{subdir}/{base_uid}_silence_combined.mp3"
            if os.path.exists(direct_path):
                return direct_path
        print(f"Audio path not found: {direct_path}")
        return None
    
    def _generate_summary_report(self, output_dir):
        """分析結果のサマリーレポートを生成"""
        if not self.results:
            return
        
        # 統計情報を計算
        total_samples = len(self.results)
        correct_predictions = sum(1 for r in self.results if r['correct'])
        accuracy = correct_predictions / total_samples
        
        # クラス別統計
        cn_samples = [r for r in self.results if r['true_label'] == 0]
        ad_samples = [r for r in self.results if r['true_label'] == 1]
        
        cn_accuracy = sum(1 for r in cn_samples if r['correct']) / len(cn_samples) if cn_samples else 0
        ad_accuracy = sum(1 for r in ad_samples if r['correct']) / len(ad_samples) if ad_samples else 0
        
        # 平均信頼度
        avg_confidence = np.mean([r['confidence'] for r in self.results])
        
        # レポートを生成
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'data_type': self.data_type,
            'total_samples_analyzed': total_samples,
            'overall_accuracy': accuracy,
            'cn_samples': len(cn_samples),
            'cn_accuracy': cn_accuracy,
            'ad_samples': len(ad_samples),
            'ad_accuracy': ad_accuracy,
            'average_confidence': avg_confidence,
            'samples_with_audio': sum(1 for r in self.results if r['audio_path'] is not None)
        }
        
        # JSONファイルとして保存
        report_path = os.path.join(output_dir, 'xai_analysis_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # テキストレポートも生成
        txt_report_path = os.path.join(output_dir, 'xai_analysis_summary.txt')
        with open(txt_report_path, 'w') as f:
            f.write("XAI Analysis Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {report['analysis_timestamp']}\n")
            f.write(f"Model Path: {report['model_path']}\n")
            f.write(f"Data Type: {report['data_type']}\n\n")
            f.write(f"Total Samples Analyzed: {report['total_samples_analyzed']}\n")
            f.write(f"Overall Accuracy: {report['overall_accuracy']:.3f}\n\n")
            f.write(f"CN Samples: {report['cn_samples']} (Accuracy: {report['cn_accuracy']:.3f})\n")
            f.write(f"AD Samples: {report['ad_samples']} (Accuracy: {report['ad_accuracy']:.3f})\n\n")
            f.write(f"Average Confidence: {report['average_confidence']:.3f}\n")
            f.write(f"Samples with Audio: {report['samples_with_audio']}\n")
        
        print(f"Summary report saved to {report_path} and {txt_report_path}")

def main():
    parser = argparse.ArgumentParser(description='XAI Analysis for Silence Transformer Model')
    parser.add_argument('--data_type', type=str, choices=['original', 'noise_augmented'], 
                       required=True, help='Type of data to analyze')
    
    args = parser.parse_args()
    
    # デフォルト値を設定
    model_path = None  # 自動検索
    output_dir = 'xai_results'
    detailed = False
    target_layer = 'transformer.layers.0'
    
    print("Starting XAI Analysis...")
    print(f"Data Type: {args.data_type}")
    print(f"Output Directory: {output_dir}")
    
    # モデルファイルを自動検索
    if model_path is None:
        # データタイプに応じてモデルを検索
        model_search_dir = f"models/{args.data_type}_transformer"
        if os.path.exists(model_search_dir):
            model_files = [f for f in os.listdir(model_search_dir) if f.endswith('.pt')]
            if model_files:
                # 最良のテストF1スコアのモデルを選択（ファイル名にF1スコアが含まれている）
                best_model_file = None
                best_f1_score = 0.0
                
                for model_file in model_files:
                    # ファイル名からF1スコアを抽出（例：best_model_fold3_test_f1_0.8234.pt）
                    import re
                    match = re.search(r'test_f1_(\d+\.\d+)', model_file)
                    if match:
                        f1_score = float(match.group(1))
                        if f1_score > best_f1_score:
                            best_f1_score = f1_score
                            best_model_file = model_file
                
                if best_model_file:
                    model_path = os.path.join(model_search_dir, best_model_file)
                    print(f"Auto-selected best model: {model_path}")
                    print(f"Best test F1-score: {best_f1_score:.4f}")
                else:
                    # F1スコアが見つからない場合は最新のファイルを使用
                    model_files.sort(reverse=True)
                    model_path = os.path.join(model_search_dir, model_files[0])
                    print(f"Auto-selected latest model: {model_path}")
            else:
                print(f"No model files found in {model_search_dir}")
                print("Creating a new model for demonstration...")
                model_path = "demo_model.pt"
        else:
            print(f"Model directory {model_search_dir} not found")
            print("Creating a new model for demonstration...")
            model_path = "demo_model.pt"
    
    # XAI分析器を初期化
    analyzer = XAIAnalyzer(model_path, args.data_type)
    
    # モデルを読み込み
    analyzer.load_model()
    
    # ターゲット層を設定
    if target_layer != 'transformer.layers.0':
        analyzer.gradcam.remove_hooks()
        analyzer.gradcam = GradCAM(analyzer.model, target_layer)
        print(f"Using target layer: {target_layer}")
    
    # 利用可能な層を表示
    available_layers = analyzer.gradcam.get_available_layers()
    print(f"Available layers: {available_layers[:10]}...")  # 最初の10個を表示
    
    # データを読み込み
    print("Loading data...")
    original_train_data, noise_augmented_train_data, test_data = load_silence_features_with_split()
    
    # テストデータを使用してXAI分析を実行
    print(f"Using test data for XAI analysis...")
    print(f"Test data contains {len(test_data)} samples")
    
    # テストデータのクラス分布を確認
    test_ad_count = sum(1 for item in test_data if item["label"] == 1)
    test_cn_count = sum(1 for item in test_data if item["label"] == 0)
    print(f"Test data distribution: AD={test_ad_count}, CN={test_cn_count}")
    
    # XAI分析を実行（全テストデータ）
    results = analyzer.analyze_test_dataset(test_data, output_dir=output_dir, detailed=detailed)
    
    print(f"XAI analysis completed! Results saved to {output_dir}")
    print(f"Analyzed {len(results)} samples")

if __name__ == "__main__":
    main()

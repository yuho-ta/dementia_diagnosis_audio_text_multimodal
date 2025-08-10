# =============================
# ノイズ追加付きwav2vec音声特徴量抽出スクリプト
# - 音声ファイルにノイズを追加してロバスト性を向上
# - wav2vec2で音声特徴量を抽出
# - ノイズの影響を減らすためのデータ拡張
# - 結果をPyTorchテンソルとして保存
# =============================

import os
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import torchaudio
import numpy as np
import time
import logging
from datetime import datetime
import random

# ログ設定
log_filename = f"preprocess_noise_augmented_wav2vec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hugging Faceのダウンロードタイムアウトを延長
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'

# GPUが利用可能な場合はGPU、そうでなければCPUを使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# パス設定
root_path = os.path.join('dataset', 'diagnosis', 'train', 'audio')
output_path = os.path.join('dataset', 'diagnosis', 'train', 'noise_augmented_features')
textual_data = os.path.join('dataset', 'diagnosis', 'train', 'text_transcriptions.csv')
max_length = 200

# 診断カテゴリ（ad: アルツハイマー, cn: 正常）
diagnosis = ['ad', 'cn']

# wav2vec2モデルの初期化
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
wav2vec_model.eval()

def add_noise(audio, noise_type='gaussian', snr_db=20.0, noise_level=0.1):
    """
    音声にノイズを追加する関数
    
    Args:
        audio (torch.Tensor): 入力音声テンソル
        noise_type (str): ノイズの種類 ('gaussian', 'uniform', 'pink')
        snr_db (float): 信号対雑音比（dB）
        noise_level (float): ノイズレベル（0.0-1.0）
    
    Returns:
        torch.Tensor: ノイズが追加された音声テンソル
    """
    # 音声のRMSを計算
    signal_rms = torch.sqrt(torch.mean(audio ** 2))
    
    if noise_type == 'gaussian':
        # ガウシアンノイズ
        noise = torch.randn_like(audio) * noise_level
    elif noise_type == 'uniform':
        # 一様分布ノイズ
        noise = (torch.rand_like(audio) - 0.5) * 2 * noise_level
    elif noise_type == 'pink':
        # ピンクノイズ（1/fノイズ）
        fft = torch.fft.fft(audio)
        freqs = torch.fft.fftfreq(len(audio))
        # 1/f特性を持つフィルタ
        pink_filter = 1.0 / torch.sqrt(torch.abs(freqs) + 1e-8)
        pink_filter[0] = 0  # DC成分を除去
        pink_noise_freq = torch.randn_like(fft) * pink_filter * noise_level
        noise = torch.real(torch.fft.ifft(pink_noise_freq))
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # SNRに基づいてノイズレベルを調整
    noise_rms = torch.sqrt(torch.mean(noise ** 2))
    if noise_rms > 0:
        target_snr_linear = 10 ** (snr_db / 20.0)
        noise = noise * (signal_rms / (noise_rms * target_snr_linear))
    
    # ノイズを追加
    noisy_audio = audio + noise
    
    return noisy_audio

def add_reverberation(audio, room_size=0.1, damping=0.5):
    """
    残響を追加する関数（簡易版）
    
    Args:
        audio (torch.Tensor): 入力音声テンソル
        room_size (float): 部屋のサイズ（0.0-1.0）
        damping (float): 減衰率（0.0-1.0）
    
    Returns:
        torch.Tensor: 残響が追加された音声テンソル
    """
    # 簡易的な残響効果（エコー）
    delay_samples = int(room_size * 16000)  # 16kHzサンプリングレート
    decay = damping
    
    # 遅延信号を作成
    delayed = torch.zeros_like(audio)
    if delay_samples < len(audio):
        delayed[delay_samples:] = audio[:-delay_samples] * decay
    
    # 元の信号と遅延信号を合成
    reverberated = audio + delayed
    
    return reverberated

def augment_audio(audio, augmentation_config):
    """
    音声データ拡張を適用する関数
    
    Args:
        audio (torch.Tensor): 入力音声テンソル
        augmentation_config (dict): データ拡張設定
    
    Returns:
        torch.Tensor: 拡張された音声テンソル
    """
    augmented_audio = audio.clone()
    
    # ノイズ追加
    if augmentation_config.get('add_noise', False):
        noise_type = augmentation_config.get('noise_type', 'gaussian')
        snr_db = augmentation_config.get('snr_db', 20.0)
        noise_level = augmentation_config.get('noise_level', 0.1)
        augmented_audio = add_noise(augmented_audio, noise_type, snr_db, noise_level)
    
    # 残響追加
    if augmentation_config.get('add_reverberation', False):
        room_size = augmentation_config.get('room_size', 0.1)
        damping = augmentation_config.get('damping', 0.5)
        augmented_audio = add_reverberation(augmented_audio, room_size, damping)
    
    # 音量調整
    if augmentation_config.get('adjust_volume', False):
        volume_factor = augmentation_config.get('volume_factor', 1.0)
        augmented_audio = augmented_audio * volume_factor
    
    return augmented_audio

def extract_wav2vec_features(audio_path, augmentation_config=None):
    """
    wav2vec2で音声特徴量を抽出する関数
    
    Args:
        audio_path (str): 音声ファイルのパス
        augmentation_config (dict): データ拡張設定
    
    Returns:
        torch.Tensor: 音声特徴量テンソル
    """
    try:
        # 音声ファイルを読み込み
        wave_form, sample_rate = torchaudio.load(audio_path)
        
        # ステレオの場合はモノラルに変換
        if wave_form.shape[0] > 1:
            wave_form = wave_form.mean(dim=0, keepdim=True)
        
        # サンプリングレートを16kHzに変換
        wave_form = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(wave_form)
        sample_rate = 16000
        wave_form = wave_form.squeeze(0)
        
        # データ拡張を適用
        if augmentation_config:
            wave_form = augment_audio(wave_form, augmentation_config)
        
        # wav2vec2で音声特徴量を抽出
        inputs_audio = processor(wave_form, sampling_rate=sample_rate, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs_audio = wav2vec_model(**inputs_audio)
        
        # 最後の隠れ状態を取得
        last_hidden_states_audio = outputs_audio.last_hidden_state.squeeze(0).cpu()
        
        # NaN値の処理
        if torch.isnan(last_hidden_states_audio).any():
            last_hidden_states_audio = torch.nan_to_num(last_hidden_states_audio, nan=0.0)
        
        return last_hidden_states_audio
        
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        return None

def preprocess_noise_augmented_wav2vec():
    """
    ノイズ追加付きwav2vec特徴量抽出のメイン関数
    """
    # 出力ディレクトリを作成
    os.makedirs(output_path, exist_ok=True)
    
    # データ拡張設定
    augmentation_configs = [
        {
            'name': 'original',
            'config': None  # 元の音声
        },
        {
            'name': 'gaussian_noise',
            'config': {
                'add_noise': True,
                'noise_type': 'gaussian',
                'snr_db': 20.0,
                'noise_level': 0.1
            }
        },
        {
            'name': 'uniform_noise',
            'config': {
                'add_noise': True,
                'noise_type': 'uniform',
                'snr_db': 15.0,
                'noise_level': 0.15
            }
        },
        {
            'name': 'pink_noise',
            'config': {
                'add_noise': True,
                'noise_type': 'pink',
                'snr_db': 18.0,
                'noise_level': 0.12
            }
        },
        {
            'name': 'reverberation',
            'config': {
                'add_reverberation': True,
                'room_size': 0.1,
                'damping': 0.5
            }
        },
        {
            'name': 'combined',
            'config': {
                'add_noise': True,
                'noise_type': 'gaussian',
                'snr_db': 20.0,
                'noise_level': 0.1,
                'add_reverberation': True,
                'room_size': 0.05,
                'damping': 0.7
            }
        }
    ]
    
    # 各診断カテゴリごとに処理
    for diagno in diagnosis:
        diagno_path = os.path.join(root_path, diagno)
        diagno_output_path = os.path.join(output_path, diagno)
        os.makedirs(diagno_output_path, exist_ok=True)
        
        logger.info(f"Processing {diagno} category...")
        
        # 音声ファイルを取得
        audio_files = [f for f in os.listdir(diagno_path) if f.endswith(('.mp3', '.wav'))]
        
        for file in audio_files:
            audio_path = os.path.join(diagno_path, file)
            uid = file.replace('.mp3', '').replace('.wav', '')
            
            logger.info(f"Processing {uid}...")
            
            # 各データ拡張設定で特徴量を抽出
            for aug_config in augmentation_configs:
                aug_name = aug_config['name']
                config = aug_config['config']
                
                # 出力ファイル名
                output_file = os.path.join(diagno_output_path, f"{uid}_{aug_name}.pt")
                
                # 既に処理済みの場合はスキップ
                if os.path.exists(output_file):
                    logger.info(f"Skipping {uid}_{aug_name} (already exists)")
                    continue
                
                # 特徴量を抽出
                features = extract_wav2vec_features(audio_path, config)
                
                if features is not None:
                    # 特徴量を保存
                    torch.save(features, output_file)
                    logger.info(f"Saved {uid}_{aug_name} features: {features.shape}")
                else:
                    logger.error(f"Failed to extract features for {uid}_{aug_name}")
    
    logger.info("Noise augmented wav2vec feature extraction completed!")

if __name__ == "__main__":
    preprocess_noise_augmented_wav2vec()

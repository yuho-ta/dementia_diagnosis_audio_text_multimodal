# =============================
# サイレンス音声ファイルにノイズ追加付き音声特徴量抽出スクリプト
# - 既存のサイレンス音声ファイル（{uid}_silence_combined.mp3）を使用
# - ガウシアンノイズ N(0,1) * 0.002 を追加
# - 複数の音声モデル（wav2vec2, eGeMAPS, mel）に対応
# - ノイズ追加によるロバスト性向上
# - 結果をPyTorchテンソルとして保存
# =============================

import os
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import torchaudio
import opensmile
import librosa
import math
import numpy as np
import time
import logging
import re
from datetime import datetime
from pydub import AudioSegment

# ログ設定
log_filename = f"preprocess_noise_augmented_silence_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

# モデル設定
audio_model = 'wav2vec2'      # 音声埋め込み用モデル

# 音声モデル名とファイル名のマッピング
name_mapping_audio = {
    'wav2vec2': 'audio',
    'egemaps': 'egemaps',
    'mel': 'mel'
}
audio_model_data = '_' + audio_model

# ノイズ設定
NOISE_STD = 0.002  # N(0,1) * 0.002 の標準偏差

# 音声モデルの初期化
if audio_model == 'wav2vec2':
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    segment_length = 50
    # 35秒分の音声に対応するmax_lengthを計算
    # wav2vec2は20msフレームで処理されるため、35秒 = 35000ms / 20ms = 1750フレーム
    max_length = 1750
elif audio_model == 'egemaps':
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    segment_length = 10
    # 35秒分の音声に対応するmax_lengthを計算
    # eGeMAPSは10msセグメントで処理されるため、35秒 = 35000ms / 10ms = 3500フレーム
    max_length = 3500
else:
    segment_length = 50
    # 35秒分の音声に対応するmax_lengthを計算
    # デフォルトは20msフレームで処理されるため、35秒 = 35000ms / 20ms = 1750フレーム
    max_length = 1750

# パス設定
silence_audio_path = os.path.join('dataset', 'diagnosis', 'train', 'silence_audio')  # サイレンス音声ファイル
root_text_path = os.path.join('dataset', 'diagnosis', 'train', 'noise_augmented_silence_features')
textual_data = os.path.join('dataset', 'diagnosis', 'train', 'text_transcriptions.csv')

def add_gaussian_noise(audio_features, noise_std=NOISE_STD):
    """
    音声特徴量にガウシアンノイズを追加する関数
    
    Args:
        audio_features (torch.Tensor): 音声特徴量テンソル
        noise_std (float): ノイズの標準偏差
    
    Returns:
        torch.Tensor: ノイズが追加された音声特徴量テンソル
    """
    # N(0,1) * 0.002 のガウシアンノイズを生成
    noise = torch.randn_like(audio_features) * noise_std
    
    # ノイズを追加
    noisy_features = audio_features + noise
    
    return noisy_features

def extract_noise_augmented_silence_embeddings():
    """
    サイレンス音声ファイルにノイズ追加付き音声特徴量を生成するメイン関数
    """
    # CSVファイルの存在確認
    if not os.path.exists(textual_data):
        logger.error(f"Error: {textual_data} not found!")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Looking for file at: {os.path.abspath(textual_data)}")
        return

    # 出力ディレクトリの作成
    os.makedirs(root_text_path, exist_ok=True)
    for diagno in ['ad', 'cn']:
        os.makedirs(os.path.join(root_text_path, diagno), exist_ok=True)

    # CSVから書き起こしデータを読み込み
    df = pd.read_csv(textual_data, encoding='utf-8')

    completed_audios = 0

    # 各行（音声ファイル）を処理
    for index, row in df.iterrows():
        logger.info(f"------------------------------------------")
        logger.info(f"Processing {row['uid']}, {row['diagno']}")

        # 出力ファイルのパス設定
        audio_embedding_path = os.path.join(root_text_path, row['diagno'], row['uid'] + '_silence_noise_augmented' + audio_model_data + '.pt')

        # サイレンス音声ファイルのパス
        silence_audio_file_path = os.path.join(silence_audio_path, row['diagno'], row['uid'] + '_silence_combined.mp3')

        if not os.path.exists(silence_audio_file_path):
            logger.warning(f"Silence audio file not found: {silence_audio_file_path}")
            continue

        # 音声埋め込みの生成（ノイズ追加付き）
        if audio_model != '':
            if audio_model == 'wav2vec2':
                wave_form, sample_rate = torchaudio.load(silence_audio_file_path)
                
                if wave_form.shape[0] > 1:
                    wave_form = wave_form.mean(dim=0, keepdim=True)

                wave_form = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(wave_form)
                sample_rate = 16000
                wave_form = wave_form.squeeze(0)

                inputs_audio = processor(wave_form, sampling_rate=sample_rate, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs_audio = wav2vec_model(**inputs_audio)

                last_hidden_states_audio = outputs_audio.last_hidden_state.squeeze(0).cpu()
                features_audio = last_hidden_states_audio
                
                # デバッグ情報: 実際のフレームレートを確認
                audio_duration = len(wave_form) / sample_rate  # 音声の実際の長さ（秒）
                num_frames = last_hidden_states_audio.shape[0]  # wav2vec2の出力フレーム数
                actual_frame_rate = audio_duration / num_frames  # 実際のフレームレート（秒/フレーム）
                
                logger.info(f"Silence audio duration: {audio_duration:.2f}s")
                logger.info(f"Wav2Vec2 output frames: {num_frames}")
                logger.info(f"Actual frame rate: {actual_frame_rate:.3f}s per frame")
                logger.info(f"Expected frame rate (50ms): {0.05}s per frame")
                logger.info(f"Expected frame rate (20ms): {0.02}s per frame")
                
                processed_audio_tensor = torch.zeros((max_length, last_hidden_states_audio.shape[1]))

                if torch.isnan(last_hidden_states_audio).any():
                    last_hidden_states_audio = torch.nan_to_num(last_hidden_states_audio, nan=0.0)
                    features_audio = last_hidden_states_audio

            elif audio_model == 'egemaps':
                y, sr = librosa.load(silence_audio_file_path)
                frame_size = 0.1
                frame_samples = int(frame_size * sr)
                frames = librosa.util.frame(y, frame_length=frame_samples, hop_length=frame_samples).T

                features = []
                for frame in frames:
                    features.append(smile.process_signal(frame, sr))
                
                features = np.vstack(features)
                features_audio = torch.tensor(features).float().to(device)
                processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1]))

                if torch.isnan(features_audio).any():
                    features_audio = torch.nan_to_num(features_audio, nan=0.0)

            elif audio_model == 'mel':
                y, sr = librosa.load(silence_audio_file_path)
                win_length = int(0.02 * sr)
                hop_length = int(0.02 * sr)
                n_mels = 80

                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=win_length, hop_length=hop_length, n_mels=n_mels)
                features_audio = torch.tensor(mel).float().permute(1,0)
                processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1]))

                if torch.isnan(features_audio).any():
                    features_audio = torch.nan_to_num(features_audio, nan=0.0)

            # 音声特徴量にガウシアンノイズを追加
            logger.info(f"Adding Gaussian noise N(0,1) * {NOISE_STD} to silence audio features")
            noisy_features_audio = add_gaussian_noise(features_audio, NOISE_STD)

            # ノイズ追加後の音声特徴量をテンソルに格納
            if noisy_features_audio.shape[0] > max_length:
                # 長すぎる場合は最初の部分を抽出
                noisy_features_audio = noisy_features_audio[:max_length]
                logger.debug(f"Truncated noisy silence audio from {features_audio.shape[0]} to {max_length} frames")
            elif noisy_features_audio.shape[0] < max_length:
                # 短すぎる場合はパディング（ゼロパディング）
                padding_length = max_length - noisy_features_audio.shape[0]
                padding = torch.zeros(padding_length, noisy_features_audio.shape[1], dtype=torch.float32)
                noisy_features_audio = torch.cat([noisy_features_audio, padding], dim=0)
                logger.debug(f"Padded noisy silence audio from {features_audio.shape[0]} to {max_length} frames")

            # ノイズ追加後の特徴量を保存用テンソルに格納
            processed_audio_tensor[:noisy_features_audio.shape[0]] = noisy_features_audio

            # 音声埋め込みを保存
            torch.save(processed_audio_tensor, audio_embedding_path)
            logger.info(f"Saved noise-augmented silence features for {row['uid']}: {processed_audio_tensor.shape}")

        completed_audios += 1
        logger.info(f"Completed processing noise-augmented silence features for {row['uid']}")

    logger.info(f"------------------------------------------")
    logger.info(f"COMPLETED PROCESSING NOISE-AUGMENTED SILENCE FEATURES")
    logger.info(f"Completed audios: {completed_audios}")

# メイン実行部分
if __name__ == "__main__":
    try:
        logger.info("Starting noise-augmented silence features preprocessing...")
        logger.info(f"Using Gaussian noise: N(0,1) * {NOISE_STD}")
        logger.info(f"Using silence audio files from: {silence_audio_path}")
        extract_noise_augmented_silence_embeddings()
        logger.info("Noise-augmented silence features preprocessing completed successfully!")
    except Exception as e:
        logger.error(f"Error during noise-augmented silence features preprocessing: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# =============================
# 音声データのサイレンス部分のみを用いた音声特徴量抽出スクリプト
# - .chaファイルから単語間のサイレンス情報を抽出（50ミリ秒以上）
# - サイレンス部分に対応する音声特徴量を抽出
# - サイレンス部分のみの音声特徴量を保存
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
log_filename = f"preprocess_silence_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
root_path = os.path.join('dataset', 'diagnosis', 'train', 'audio')
root_text_path = os.path.join('dataset', 'diagnosis', 'train', 'silence_features')
root_silence_audio_path = os.path.join('dataset', 'diagnosis', 'train', 'silence_audio')  # サイレンス音声ファイル保存用
textual_data = os.path.join('dataset', 'diagnosis', 'train', 'text_transcriptions.csv')

def extract_silence_from_cha(cha_file_path):
    """
    .chaファイルから単語間のサイレンス情報を抽出（50ミリ秒以上）
    
    Args:
        cha_file_path: .chaファイルのパス
    
    Returns:
        silence_timestamps: サイレンス部分の時間情報
    """
    silence_timestamps = []
    
    if not os.path.exists(cha_file_path):
        logger.warning(f"CHA file not found: {cha_file_path}")
        return silence_timestamps
    
    try:
        with open(cha_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # %wor:行から単語と時間情報を抽出
        word_timestamps = []
        for line in lines:
            if line.startswith('%wor:'):
                # %wor:行をパース
                parts = line.strip().split()
                #logger.info(f"{parts}")
                # %wor:をスキップして、単語と時間情報のペアを処理
                for i in range(1, len(parts)):
                    time_info = parts[i]
                    
                    # 時間情報をパース（例: 1360_1820）
                    # 特殊文字（U+0015）で囲まれた数字_数字のパターンを検索
                    time_match = re.search(r'\u0015(\d+)_(\d+)\u0015', time_info)
                    if time_match:
                        start_time = int(time_match.group(1)) / 1000.0  # ミリ秒を秒に変換
                        end_time = int(time_match.group(2)) / 1000.0
                        word_timestamps.append({
                            'start': start_time,
                            'end': end_time
                        })
        
        # 単語間のサイレンスを計算（50ミリ秒以上）
        for i in range(1, len(word_timestamps)):
            prev_word = word_timestamps[i-1]
            curr_word = word_timestamps[i]
            
            # 前の単語の終了時間と現在の単語の開始時間の間隔
            silence_duration = curr_word['start'] - prev_word['end']
            
            # 50ミリ秒（0.05秒）以上のサイレンスを抽出
           
            silence_timestamps.append({
                'start': prev_word['end'],
                'end': curr_word['start'],
                'duration': silence_duration,
                'type': 'silence'
            })
        
        # サイレンスの全体の長さを計算
        total_silence_duration = sum(silence['duration'] for silence in silence_timestamps)
        
        # 35秒以上のサイレンスのみを保持
        if total_silence_duration >=35.0:
            logger.info(f"Extracted {len(silence_timestamps)} silence segments (total duration: {total_silence_duration:.2f}s) from {cha_file_path}")
        else:
            silence_timestamps = []  # 35秒未満の場合は空にする
            logger.info(f"Total silence duration ({total_silence_duration:.2f}s) is less than 35s, skipping {cha_file_path}")
        
    except Exception as e:
        logger.error(f"Error parsing CHA file {cha_file_path}: {e}")
    
    return silence_timestamps

def extract_silence_audio_segments(audio_path, silence_timestamps, output_dir, uid):
    """
    元の音声ファイルからサイレンス部分を切り抜いて結合したMP3ファイルとして保存

    Args:
        audio_path: 元の音声ファイルのパス
        silence_timestamps: サイレンス部分の時間情報
        output_dir: 出力ディレクトリ
        uid: ユーザーID
    """
    try:
        audio = AudioSegment.from_mp3(audio_path)
        os.makedirs(output_dir, exist_ok=True)
        combined_silence = AudioSegment.empty()
        total_duration_ms = 0
        max_duration_ms = 35 * 1000  # 35秒をミリ秒に変換

        for i, silence_info in enumerate(silence_timestamps):
            adjusted_start = silence_info['start']
            adjusted_end = silence_info['end']

            if adjusted_start < adjusted_end:
                start_time = int(adjusted_start * 1000)
                end_time = int(adjusted_end * 1000)
                silence_segment = audio[start_time:end_time]

                segment_duration = len(silence_segment)
                
                # すでに35秒以上追加していたら終了
                if total_duration_ms >= max_duration_ms:
                    break

                # 追加すると35秒を超える場合は、切り詰める
                if total_duration_ms + segment_duration > max_duration_ms:
                    remaining = max_duration_ms - total_duration_ms
                    silence_segment = silence_segment[:remaining]
                    segment_duration = len(silence_segment)

                combined_silence += silence_segment
                total_duration_ms += segment_duration
            else:
                logger.warning(f"Skipping silence segment {i} for {uid}: adjusted time range invalid (start: {adjusted_start:.3f}s, end: {adjusted_end:.3f}s)")
                continue

        output_filename = f"{uid}_silence_combined.mp3"
        output_path = os.path.join(output_dir, output_filename)
        combined_silence.export(output_path, format="mp3")

        logger.info(f"Extracted and combined {len(silence_timestamps)} silence segments for {uid} (capped at 35s, actual: {total_duration_ms/1000:.2f}s)")

    except Exception as e:
        logger.error(f"Error extracting silence audio for {uid}: {e}")

def preprocess_silence_embeddings():
    """
    サイレンス部分のみの音声特徴量を生成するメイン関数
    """
    # CSVファイルの存在確認
    if not os.path.exists(textual_data):
        logger.error(f"Error: {textual_data} not found!")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Looking for file at: {os.path.abspath(textual_data)}")
        return

    # 出力ディレクトリの作成
    os.makedirs(root_text_path, exist_ok=True)
    os.makedirs(root_silence_audio_path, exist_ok=True)
    for diagno in ['ad', 'cn']:
        os.makedirs(os.path.join(root_text_path, diagno), exist_ok=True)
        os.makedirs(os.path.join(root_silence_audio_path, diagno), exist_ok=True)

    # CSVから書き起こしデータを読み込み
    df = pd.read_csv(textual_data, encoding='utf-8')

    completed_audios = 0

    # 各行（音声ファイル）を処理
    for index, row in df.iterrows():
        logger.info(f"------------------------------------------")
        logger.info(f"Processing {row['uid']}, {row['diagno']}")

        # 出力ファイルのパス設定
        audio_embedding_path = os.path.join(root_text_path, row['diagno'], row['uid'] + '_silence_only' + audio_model_data + '.pt')

        # .chaファイルのパス
        cha_file_path = os.path.join('dataset', 'diagnosis', 'train', 'segmentation', row['diagno'], row['uid'] + '.cha')

        if not os.path.exists(cha_file_path):
            logger.warning(f"CHA file not found: {cha_file_path}")
            continue

        # .chaファイルからサイレンス情報を抽出
        silence_timestamps = extract_silence_from_cha(cha_file_path)
        
        logger.info(f"Original UID: {row['uid']}")
        logger.info(f"Number of silence segments: {len(silence_timestamps)}")

        if len(silence_timestamps) == 0:
            logger.info(f"No silence information found for {row['uid']}")
            continue

        # サイレンス音声セグメントの抽出と保存
        audio_path = os.path.join(root_path, row['diagno'], row['uid'] + '.mp3')
        silence_audio_output_dir = os.path.join(root_silence_audio_path, row['diagno'])
        extract_silence_audio_segments(audio_path, silence_timestamps, silence_audio_output_dir, row['uid'])
        silence_audio_path = os.path.join(root_silence_audio_path, row['diagno'], f"{row['uid']}_silence_combined.mp3")

        # 音声埋め込みの生成（サイレンス部分のみ）
        if audio_model != '':
            # audio_pathは既に上で定義済み

            if audio_model == 'wav2vec2':
                wave_form, sample_rate = torchaudio.load(silence_audio_path)
                
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
                
                logger.info(f"Audio duration: {audio_duration:.2f}s")
                logger.info(f"Wav2Vec2 output frames: {num_frames}")
                logger.info(f"Actual frame rate: {actual_frame_rate:.3f}s per frame")
                logger.info(f"Expected frame rate (50ms): {0.05}s per frame")
                logger.info(f"Expected frame rate (20ms): {0.02}s per frame")
                
                processed_audio_tensor = torch.zeros((max_length, last_hidden_states_audio.shape[1]))

                if torch.isnan(last_hidden_states_audio).any():
                    last_hidden_states_audio = torch.nan_to_num(last_hidden_states_audio, nan=0.0)
                    features_audio = last_hidden_states_audio

            elif audio_model == 'egemaps':
                y, sr = librosa.load(silence_audio_path)
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
                y, sr = librosa.load(silence_audio_path)
                win_length = int(0.02 * sr)
                hop_length = int(0.02 * sr)
                n_mels = 80

                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=win_length, hop_length=hop_length, n_mels=n_mels)
                features_audio = torch.tensor(mel).float().permute(1,0)
                processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1]))

                if torch.isnan(features_audio).any():
                    features_audio = torch.nan_to_num(features_audio, nan=0.0)

            # サイレンス部分の音声特徴量を抽出
            silence_audio_features = []
            
            for silence_info in silence_timestamps:
                start_segment = math.floor(silence_info['start'] * segment_length)
                end_segment = math.ceil(silence_info['end'] * segment_length)
                
                if end_segment > features_audio.shape[0]:
                    end_segment = features_audio.shape[0]
                
                if start_segment < end_segment:
                    silence_segment = features_audio[start_segment:end_segment]
                    silence_audio_features.append(silence_segment.mean(dim=0))
                else:
                    # 短いサイレンスの場合はゼロベクトルを使用
                    silence_audio_features.append(torch.zeros(features_audio.shape[1]))

            # サイレンス音声特徴量をテンソルに格納
            if silence_audio_features:
                # サイレンス特徴量を順次格納
                for i, feature in enumerate(silence_audio_features[:max_length]):
                    processed_audio_tensor[i] = feature

            # 音声埋め込みを保存
            torch.save(processed_audio_tensor, audio_embedding_path)

        completed_audios += 1
        logger.info(f"Completed processing silence features for {row['uid']}")

    logger.info(f"------------------------------------------")
    logger.info(f"COMPLETED PROCESSING SILENCE FEATURES")
    logger.info(f"Completed audios: {completed_audios}")

# メイン実行部分
if __name__ == "__main__":
    try:
        logger.info("Starting silence features preprocessing...")
        preprocess_silence_embeddings()
        logger.info("Silence features preprocessing completed successfully!")
    except Exception as e:
        logger.error(f"Error during silence features preprocessing: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise 
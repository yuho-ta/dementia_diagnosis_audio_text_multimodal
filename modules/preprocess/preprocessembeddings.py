# =============================
# 音声・テキスト埋め込み前処理スクリプト
# - テキスト書き起こしから言語モデル埋め込みを抽出
# - 音声ファイルから音声特徴量埋め込みを抽出
# - 単語レベルでの音声-テキストアライメントを実行
# - 結果をPyTorchテンソルとして保存
# =============================

import os
import pandas as pd
from transformers import AutoTokenizer, RobertaModel, Wav2Vec2Processor, Wav2Vec2Model, BertTokenizer, BertModel, DistilBertModel, AutoModel
import torch
import torchaudio
import opensmile
import unicodedata
import librosa
import math
import numpy as np
import time
import logging
from datetime import datetime

# ログ設定
# タイムスタンプ付きのログファイル名を作成
log_filename = f"preprocess_embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # コンソールにも出力
    ]
)
logger = logging.getLogger(__name__)

# Hugging Faceのダウンロードタイムアウトを延長（5分）
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'

# GPUが利用可能な場合はGPU、そうでなければCPUを使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデル設定
# 利用可能: bert, roberta, distilbert, stella, mistral, qwen
textual_model = 'distilbert'  # テキスト埋め込み用モデル
audio_model = 'wav2vec2'      # 音声埋め込み用モデル
pauses = True                 # ポーズ情報を使用するかどうか

# ポーズ情報の有無によるファイル名の変更
pauses_data = '_pauses' if pauses else ''

# テキストモデル名とファイル名のマッピング
name_mapping_text = {
    'bert': '',
    'distil': 'distil',
    'roberta': 'roberta',
    'mistral': 'mistral',
    'qwen': 'qwen',
    'stella': 'stella'
}
textual_model_data = textual_model

# 音声モデル名とファイル名のマッピング
name_mapping_audio = {
    'wav2vec2': 'audio',
    'egemaps': 'egemaps',
    'mel': 'mel'
}
audio_model_data = '_' + audio_model

# ネットワーク問題に対応したモデルロード関数（リトライ機能付き）
def load_model_with_retry(model_name, tokenizer_class, model_class, max_retries=3):
    """ネットワーク問題に対応したモデルロード関数（リトライ機能付き）"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to load {model_name} (attempt {attempt + 1}/{max_retries})")
            tokenizer = tokenizer_class.from_pretrained(model_name)
            model = model_class.from_pretrained(model_name).to(device)
            logger.info(f"Successfully loaded {model_name}")
            return tokenizer, model
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                logger.error(f"Failed to load {model_name} after {max_retries} attempts")
                raise

# テキストモデルの初期化
if textual_model == 'bert':
    tokenizer, model = load_model_with_retry("bert-base-uncased", BertTokenizer, BertModel)
elif textual_model == 'roberta':
    tokenizer, model = load_model_with_retry("roberta-base", AutoTokenizer, RobertaModel)
elif textual_model == 'distilbert':
    tokenizer, model = load_model_with_retry('distilbert-base-uncased', AutoTokenizer, DistilBertModel)
elif textual_model == 'stella':
    tokenizer = AutoTokenizer.from_pretrained("NovaSearch/stella_en_1.5B_v5", trust_remote_code=True)
    model = AutoModel.from_pretrained("NovaSearch/stella_en_1.5B_v5", trust_remote_code=True)
elif textual_model == 'mistral':
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    # アクセストークンが必要
    model = AutoModel.from_pretrained("mistralai/Mistral-7B-v0.1", use_auth_token=True)
elif textual_model == 'qwen':
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    # アクセストークンが必要
    model = AutoModel.from_pretrained("Qwen/Qwen2.5-7B")
else:
    # モデルが指定されていない場合はdistilbertをデフォルトとして使用
    logger.info(f"No textual model specified, using distilbert as default")
    tokenizer, model = load_model_with_retry('distilbert-base-uncased', AutoTokenizer, DistilBertModel)

model.eval()  # 評価モードに設定

# 音声モデルの初期化
if audio_model == 'wav2vec2':
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    segment_length = 50  # 音声セグメントの長さ（フレーム数）
elif audio_model == 'egemaps':
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    segment_length = 10
else:
    segment_length = 50

# パス設定
root_path = os.path.join('dataset', 'diagnosis', 'train', 'audio')      # 音声ファイルのルートパス
root_text_path = os.path.join('dataset', 'diagnosis', 'train', 'text')  # テキスト埋め込みの出力パス

textual_data = os.path.join('dataset', 'diagnosis', 'train', 'text_transcriptions.csv')  # 書き起こしデータのCSV
max_length = 200  # 最大トークン数


# メイン前処理関数
def preprocess_text():

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

    # ポーズ情報の有無に応じて使用するカラムを選択
    row_data = 'transcription_pause' if pauses else 'transcription'

    # Unicode正規化（文字の正規化）
    df[row_data] = df[row_data].apply(lambda x: unicodedata.normalize("NFC", str(x)))

    completed_audios = 0

    # 各行（音声ファイル）を処理
    for index, row in df.iterrows():

        logger.info(f"------------------------------------------")
        logger.info(f"------------------------------------------")
        logger.info(f"Processing {row['uid']}, {row['diagno']}")

        # 出力ファイルのパス設定
        text_embedding_path = os.path.join(root_text_path, row['diagno'], row['uid'] + textual_model_data + pauses_data + '.pt')
        audio_embedding_path = os.path.join(root_text_path, row['diagno'], row['uid'] + textual_model_data + pauses_data + audio_model_data + '.pt')
        
        # 既に処理済みのファイルはスキップ
        if os.path.exists(text_embedding_path) and (audio_model == '' or os.path.exists(audio_embedding_path)):
            logger.info(f"Skipping {row['uid']} - embedding files already exist")
            completed_audios += 1
            continue

        # 書き起こしテキストを取得
        transcription = row[row_data]

        # テキストをトークン化
        inputs_text = tokenizer(
            transcription,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        ).to(device)

        # テキスト埋め込みを取得
        with torch.no_grad():
            outputs_text = model(**inputs_text)

        # テキスト埋め込みを保存
        last_hidden_states_text = outputs_text.last_hidden_state.squeeze(0).cpu()
        torch.save(last_hidden_states_text, text_embedding_path)

        # 音声モデルが指定されている場合、音声埋め込みも処理
        if audio_model != '':
            audio_path = os.path.join(root_path, row['diagno'], row['uid'] + '.wav')

            # wav2vec2モデルの場合
            if audio_model == 'wav2vec2':
                wave_form, sample_rate = torchaudio.load(audio_path)
                        
                # ステレオの場合はモノラルに変換
                if wave_form.shape[0] > 1:
                    wave_form = wave_form.mean(dim=0, keepdim=True)

                # サンプリングレートを16kHzに変換
                wave_form = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(wave_form)
                sample_rate = 16000
                wave_form = wave_form.squeeze(0)

                # wav2vec2で音声特徴量を抽出
                inputs_audio = processor(wave_form, sampling_rate=sample_rate, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs_audio = wav2vec_model(**inputs_audio)

                last_hidden_states_audio = outputs_audio.last_hidden_state.squeeze(0).cpu()
                features_audio = last_hidden_states_audio  # wav2vec2用にfeatures_audioを定義
                processed_audio_tensor = torch.zeros((max_length, last_hidden_states_audio.shape[1]))

                # NaN値の処理
                if torch.isnan(last_hidden_states_audio).any():
                    last_hidden_states_audio = torch.nan_to_num(last_hidden_states_audio, nan=0.0)
                    features_audio = last_hidden_states_audio  # features_audioも更新

            # eGeMAPS特徴量の場合
            elif audio_model == 'egemaps':
                y, sr = librosa.load(audio_path)
                frame_size = 0.1  # フレームサイズ（秒）

                frame_samples = int(frame_size * sr)  # フレームあたりのサンプル数
                frames = librosa.util.frame(y, frame_length=frame_samples, hop_length=frame_samples).T

                features = []
                for frame in frames:
                    features.append(smile.process_signal(frame, sr))
                
                features = np.vstack(features)

                features_audio = torch.tensor(features).float().to(device)
                logger.info(f"Features shape: {features_audio.shape}")

                processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1]))

                # NaN値の処理
                if torch.isnan(features_audio).any():
                    logger.error(f"ERROR BEFORE in {row['diagno']}, {row['uid']}: NaN values in features_audio")
                    features_audio = torch.nan_to_num(features_audio, nan=0.0)

            # メルスペクトログラム特徴量の場合
            elif audio_model == 'mel':
                y, sr = librosa.load(audio_path)

                win_length = int(0.02 * sr)  # 20msをサンプル数に変換
                hop_length = int(0.02 * sr)  # 20ms（50セグメント/秒）
                n_mels = 80  # メルフィルタバンクの数

                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=win_length, hop_length=hop_length, n_mels=n_mels)

                features_audio = torch.tensor(mel).float().permute(1,0)

                processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1]))

                # NaN値の処理
                if torch.isnan(features_audio).any():
                    features_audio = torch.nan_to_num(features_audio, nan=0.0)
        
            # 音声特徴量の平均を最初の位置に設定
            processed_audio_tensor[0] = features_audio.mean(dim=0)

            # トークン化とオフセットマッピングの取得
            inputs_offset = tokenizer(
                transcription,
                return_tensors="pt",
                return_offsets_mapping=True,  # トークン-オフセットマッピングを取得
                padding="max_length",
                truncation=True,
                max_length=max_length
            ).to(device)

            # 単語-トークンマッピングの抽出
            input_ids = inputs_offset["input_ids"][-1]
            offset_mapping = inputs_offset["offset_mapping"][-1]

            tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
            word_mapping = []

            current_word = ""
            current_tokens = []
            current_token_ids = []

            # トークンを単語にグループ化
            for token, offset, token_id in zip(tokens, offset_mapping.tolist(), input_ids.tolist()):
                start, end = offset

                # 特殊トークン（[CLS], [SEP], [PAD]）をスキップ
                if start == 0 and end == 0:
                    continue

                # サブワード（##）をチェックしてトークンを単語にグループ化
                if token.startswith("##"):
                    current_word += token[2:]
                    current_tokens.append(token)
                    current_token_ids.append(token_id)
                else:
                    # 前の単語を保存
                    if current_word:
                        word_mapping.append((current_word, current_tokens, current_token_ids))
                    # 新しい単語を開始
                    current_word = token
                    current_tokens = [token]
                    current_token_ids = [token_id]

            # 最後の単語を保存
            if current_word:
                word_mapping.append((current_word, current_tokens, current_token_ids))

            # 単語レベルのタイムスタンプファイルのパス
            word_level_timestamp_path = os.path.join(root_text_path, row['diagno'], row['uid'] + '.csv')

            # 単語レベルのタイムスタンプを読み込み
            df_word_level = pd.read_csv(word_level_timestamp_path)
            # カラム: ['word', 'start', 'end', 'probability']
            words = []
            for index, data in df_word_level.iterrows():
                words.append((data['word'], data['start'], data['end']))

            idx_probs = 0
            act_word = ''

            idx_att = 0
            idx_start_att = 0

            idx_start_map = 0
            idx_map = 0

            n_audio_segments = 0

            # 各テキストトークンに対応する音声特徴量を抽出して processed_audio_tensor に格納する
            for word, tokens, token_ids in word_mapping:
                cleaned_word = word.replace('Ġ', '')
                act_word += cleaned_word.replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower()

                logger.debug(f"Word: {word}, Tokens: {tokens}, Token IDs: {token_ids}")
                logger.debug(f"Act Word: {act_word}")
                if idx_probs < len(words):
                    # words[idx_probs][0]が文字列かどうかをチェックしてから出力
                    if isinstance(words[idx_probs][0], str):
                        logger.debug(f"Expected Word: {words[idx_probs][0].replace('Ġ', '').replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower()}")

                # 句読点の処理(もし現在のwordが句読点（.、, など）であると判断された場合、その句読点に対応する音声セグメントを推定します。)
                if word.strip() in ['.', ',', '?', '!', ';', 'Ġ','Ġ.', 'Ġ,', 'Ġ?', 'Ġ!', 'Ġ;', 'Ġ...', '...']:
                    if idx_probs > 0:  # インデックスエラーを避ける
                        start = words[idx_probs-1][2]  # 前の単語の終了時間を取得
                    else:
                        start = 0  # 最初の単語の場合は0をデフォルト
                    end = words[idx_probs][1] if idx_probs < len(words) else None  # 安全なチェック

                    start_segment = math.floor(start * segment_length)
                    end_segment = math.ceil(end * segment_length if end is not None else last_hidden_states_audio.shape[0])
                    logger.debug(f"FOUND PUNCTUATION: {word}, Start: {start}, End: {end}, Start Segment: {start_segment}, End Segment: {end_segment}")
                    logger.debug("Token IDs:")
                    for idx in range(idx_start_map, idx_map + 1):
                        logger.debug(f"{idx}: {word_mapping[idx]}")
                        logger.debug('------------------------------------------')

                    for idx in range(idx_start_att, idx_att + len(token_ids)):
                        n_audio_segments += 1

                        if end_segment - start_segment < 3:
                            start_segment = max(0, start_segment - 2)
                            end_segment = min(last_hidden_states_audio.shape[0], end_segment + 2)

                        audio_features_segment = last_hidden_states_audio[start_segment:end_segment]
                        processed_audio_tensor[idx + 1] = torch.clamp(audio_features_segment.mean(dim=0), min=-1e3, max=1e3)

                    idx_start_att = idx_att + len(token_ids)
                    idx_start_map = idx_map + 1

                # 単語の処理
                if idx_probs < len(words) and isinstance(words[idx_probs][0], str) and act_word == words[idx_probs][0].replace('Ġ', '').replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower():
                    
                        start = words[idx_probs][1]
                        end = words[idx_probs][2]

                        start_segment = math.floor(start * segment_length)
                        end_segment = math.ceil(end * segment_length if end is not None else last_hidden_states_audio.shape[0])

                        logger.debug(f"FOUND WORD: {act_word}, Start: {start}, End: {end}, Start Segment: {start_segment}, End Segment: {end_segment}")
                        logger.debug("Token IDs:")
                        for idx in range(idx_start_map, idx_map + 1):
                            logger.debug(f"{idx}: {word_mapping[idx]}")
                            logger.debug('------------------------------------------')

                        for idx in range(idx_start_att, idx_att + len(token_ids)):
                            n_audio_segments += 1
                            
                            if end_segment - start_segment < 3:
                                start_segment = max(0, start_segment - 2)
                                end_segment = min(last_hidden_states_audio.shape[0], end_segment + 2)

                            audio_features_segment = last_hidden_states_audio[start_segment:end_segment]
                            processed_audio_tensor[idx + 1] = torch.clamp(audio_features_segment.mean(dim=0), min=-1e3, max=1e3)

                        idx_probs += 1
                        act_word = ''
                        idx_start_att = idx_att + len(token_ids)
                        idx_start_map = idx_map + 1

                idx_att += len(token_ids)
                idx_map += 1

            # 最後の単語の処理
            if idx_probs < len(words) and isinstance(words[idx_probs][0], str) and act_word in words[idx_probs][0].replace('Ġ', '').replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower():
                start = words[idx_probs][1]
                end = words[idx_probs][2]

                start_segment = math.floor(start * segment_length)
                end_segment = math.ceil(end * segment_length if end is not None else last_hidden_states_audio.shape[0])

                logger.debug(f"FOUND WORD: {act_word}, Start: {start}, End: {end}, Start Segment: {start_segment}, End Segment: {end_segment}")
                logger.debug("Token IDs:")
                for idx in range(idx_start_map, idx_map):
                    logger.debug(f"{idx}: {word_mapping[idx]}")
                    logger.debug('------------------------------------------')

                for idx in range(idx_start_att, idx_att):
                    n_audio_segments += 1

                    if end_segment - start_segment < 3:
                        start_segment = max(0, start_segment - 2)
                        end_segment = min(last_hidden_states_audio.shape[0], end_segment + 2)

                    audio_features_segment = last_hidden_states_audio[start_segment:end_segment]
                    processed_audio_tensor[idx + 1] = torch.clamp(audio_features_segment.mean(dim=0), min=-1e3, max=1e3)

                idx_probs += 1
                act_word = ''
                idx_start_att = idx_att + len(token_ids)
                idx_start_map = idx_map + 1

            logger.debug(f"Number of audio segments: {n_audio_segments}")        
            # 入力トークン数と音声セグメント数を比較（PADトークンを除く）
            total_tokens = torch.sum(inputs_text['attention_mask'][0]).item()
            logger.debug(f"Total tokens: {total_tokens}")
            if n_audio_segments + 2 != total_tokens:
                logger.debug(f"ERROR in {row['diagno']}, {row['uid']}: Number of audio segments ({n_audio_segments}) does not match the number of tokens ({total_tokens})")
                logger.debug(f"Completed audios: {completed_audios}")
                return -1

            # NaN値のチェック
            if torch.isnan(processed_audio_tensor).any():
                logger.debug(f"ERROR in {row['diagno']}, {row['uid']}: NaN values in processed_audio_tensor")
                logger.debug(f"Completed audios: {completed_audios}")
                return -1
            
            # 音声埋め込みを保存
            torch.save(processed_audio_tensor, audio_embedding_path)

            completed_audios += 1

        logger.info(f"------------------------------------------")
        logger.info(f"CORRECTLY PROCESSED ALL AUDIOS")
        logger.info(f"Completed audios: {completed_audios}")

# メイン実行部分
if __name__ == "__main__":
    try:
        logger.info("Starting preprocessing embeddings...")
        preprocess_text()
        logger.info("Preprocessing completed successfully!")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
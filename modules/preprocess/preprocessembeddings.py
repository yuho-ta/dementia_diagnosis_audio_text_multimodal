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
textual_model = ''              # テキスト埋め込み用モデル (空文字列にするとテキスト処理をスキップ)
audio_model = 'wav2vec2'        # 音声埋め込み用モデル (空文字列にすると音声処理をスキップ)
pauses = True                   # ポーズ情報を使用するかどうか

# ポーズ情報の有無によるファイル名の変更
pauses_data_suffix = '_pauses' if pauses else ''

# テキストモデル名とファイル名のマッピング
name_mapping_text = {
    'bert': 'bert', # 'bert'の場合、ファイル名に'bert'と含める
    'distilbert': 'distil',
    'roberta': 'roberta',
    'mistral': 'mistral',
    'qwen': 'qwen',
    'stella': 'stella'
}
# textual_model が設定されている場合のみ、そのモデルのサフィックスを取得
textual_model_suffix = textual_model

# 音声モデル名とファイル名のマッピング
name_mapping_audio = {
    'wav2vec2': 'audio',
    'egemaps': 'egemaps',
    'mel': 'mel'
}
# audio_model が設定されている場合のみ、そのモデルのサフィックスを取得
audio_model_suffix = audio_model

# ネットワーク問題に対応したモデルロード関数（リトライ機能付き）
def load_model_with_retry(model_name_or_path, tokenizer_class, model_class, max_retries=3, **kwargs):
    """ネットワーク問題に対応したモデルロード関数（リトライ機能付き）"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to load {model_name_or_path} (attempt {attempt + 1}/{max_retries})")
            tokenizer = tokenizer_class.from_pretrained(model_name_or_path, **kwargs)
            model = model_class.from_pretrained(model_name_or_path, **kwargs).to(device)
            logger.info(f"Successfully loaded {model_name_or_path}")
            return tokenizer, model
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                logger.error(f"Failed to load {model_name_or_path} after {max_retries} attempts")
                raise

# テキストモデルの初期化
tokenizer, model = None, None # 初期値をNoneに設定
if textual_model == 'bert':
    tokenizer, model = load_model_with_retry("bert-base-uncased", BertTokenizer, BertModel)
elif textual_model == 'roberta':
    tokenizer, model = load_model_with_retry("roberta-base", AutoTokenizer, RobertaModel)
elif textual_model == 'distilbert':
    tokenizer, model = load_model_with_retry('distilbert-base-uncased', AutoTokenizer, DistilBertModel)
elif textual_model == 'stella':
    tokenizer, model = load_model_with_retry("NovaSearch/stella_en_1.5B_v5", AutoTokenizer, AutoModel, trust_remote_code=True)
elif textual_model == 'mistral':
    tokenizer, model = load_model_with_retry("mistralai/Mistral-7B-v0.1", AutoTokenizer, AutoModel, use_auth_token=True)
    if tokenizer: # Mistral special case
        tokenizer.pad_token = tokenizer.eos_token
elif textual_model == 'qwen':
    tokenizer, model = load_model_with_retry("Qwen/Qwen2.5-7B", AutoTokenizer, AutoModel)
else: # textual_model が空文字列の場合、このブロックが実行される
    logger.info(f"No specific textual model set. Textual embeddings will not be generated and no text model will be loaded.")
    # ここでは何もしない (tokenizer と model は None のまま)

if model: # modelがNoneでない場合のみeval()を呼び出す
    model.eval() # 評価モードに設定

# 音声モデルの初期化
processor, wav2vec_model, smile = None, None, None
segment_length = 50 # Default segment length (for wav2vec2, mel)

if audio_model == 'wav2vec2':
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    wav2vec_model.eval() # Set to eval mode
    segment_length = 50  # wav2vec2のフレームレートに依存 (16kHz / 20ms = 50フレーム/秒)
elif audio_model == 'egemaps':
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        # FeatureLevel.Functionalsは通常、発話レベルの要約特徴量です。
        # 単語レベルのアライメントには、より細かい時間解像度の特徴量（LLDなど）が必要です。
        # 現在のロジックではフレームを平均しているので、Functionalsでも動作はしますが、
        # 意図した「フレームごとの特徴量」にはならないかもしれません。
        # もしフレームごとの特徴量が必須ならFeatureLevel.LLDを設定し、特徴量セットも変更してください。
        feature_level=opensmile.FeatureLevel.Functionals, 
        frame_step=0.02, # 20msフレームステップ
        frame_size=0.02  # 20msフレームサイズ
    )
    segment_length = 1 / smile.frame_step # frames per second (50 for 20ms step)
elif audio_model == 'mel':
    segment_length = 50 # 50 frames per second for mel spectrograms (with 20ms hop)
else:
    logger.info(f"No specific audio model set. Audio embeddings will not be generated.")


# パス設定
root_audio_input_path = os.path.join('dataset', 'diagnosis', 'train', 'audio')      # 音声ファイルのルートパス
text_transcriptions_csv_path = os.path.join('dataset', 'diagnosis', 'train', 'text_transcriptions.csv')  # 書き起こしデータのCSV

# ベース出力パス
base_output_path = os.path.join('dataset', 'diagnosis', 'train') 
text_embedding_output_root = os.path.join(base_output_path, 'text') 
audio_embedding_output_root = os.path.join(base_output_path, 'text') 

max_length = 200  # 最大トークン数


# メイン前処理関数
def preprocess_embeddings():
    logger.info(f"Starting embedding preprocessing for textual_model='{textual_model}', audio_model='{audio_model}'")

    # CSVファイルの存在確認
    if not os.path.exists(text_transcriptions_csv_path):
        logger.error(f"Error: {text_transcriptions_csv_path} not found!")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Looking for file at: {os.path.abspath(text_transcriptions_csv_path)}")
        return

    # 出力ディレクトリの作成
    if textual_model:
        for diagno in ['ad', 'cn']:
            os.makedirs(os.path.join(text_embedding_output_root, diagno), exist_ok=True)
    if audio_model:
        for diagno in ['ad', 'cn']:
            os.makedirs(os.path.join(audio_embedding_output_root, diagno), exist_ok=True)

    # CSVから書き起こしデータを読み込み
    df = pd.read_csv(text_transcriptions_csv_path, encoding='utf-8')

    # ポーズ情報の有無に応じて使用するカラムを選択
    transcription_column = 'transcription_pause' if pauses else 'transcription'

    # Unicode正規化（文字の正規化）
    df[transcription_column] = df[transcription_column].apply(lambda x: unicodedata.normalize("NFC", str(x)))

    completed_audios = 0

    # 各行（音声ファイル）を処理
    for index, row in df.iterrows():
        logger.info(f"------------------------------------------")
        logger.info(f"Processing {row['uid']}, {row['diagno']}")

        uid = row['uid']
        diagno = row['diagno']
        transcription = row[transcription_column]

        # --- ファイル名の構築 ---
        # 各構成要素が空文字列でない場合にのみ、'_'を付けて連結
        textual_file_part_suffix = f"_{textual_model_suffix}" if textual_model_suffix else ""
        pauses_file_part_suffix = pauses_data_suffix # pauses_data_suffix は既に '_pauses' または ''
        audio_file_part_suffix = f"_{audio_model_suffix}" if audio_model_suffix else ""

        # テキスト埋め込みファイルのパス
        current_text_embedding_path = os.path.join(text_embedding_output_root, diagno, 
                                                   f"{uid}{textual_file_part_suffix}{pauses_file_part_suffix}.pt")
        
        # 音声埋め込みファイルのパス (textual_model_suffix も考慮)
        current_audio_embedding_path = os.path.join(audio_embedding_output_root, diagno, 
                                                    f"{uid}{textual_file_part_suffix}{pauses_file_part_suffix}{audio_file_part_suffix}.pt")
        
        # --- 既に処理済みのファイルはスキップ ---
        # skip_text_embedding = textual_model and os.path.exists(current_text_embedding_path)
        # skip_audio_embedding = audio_model and os.path.exists(current_audio_embedding_path)

        # if skip_text_embedding and skip_audio_embedding:
        #     logger.info(f"Skipping {uid} - both embedding files already exist")
        #     completed_audios += 1
        #     continue
        # elif textual_model and not skip_text_embedding and audio_model and skip_audio_embedding:
        #     logger.info(f"Processing {uid} - Text embedding missing, audio exists. Generating text embedding.")
        # elif audio_model and not skip_audio_embedding and textual_model and skip_text_embedding:
        #     logger.info(f"Processing {uid} - Audio embedding missing, text exists. Generating audio embedding.")
        # elif not textual_model and not audio_model:
        #     logger.info(f"Neither textual nor audio model specified for {uid}. Skipping.")
        #     continue # 両方設定されていなければスキップ

        # --- テキスト埋め込みの処理 ---
        skip_text_embedding = False
        if textual_model and not skip_text_embedding:
            if not tokenizer or not model:
                logger.error(f"Textual model/tokenizer not loaded for {textual_model}. Cannot generate text embedding for {uid}.")
            else:
                try:
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
                    torch.save(last_hidden_states_text, current_text_embedding_path)
                    logger.info(f"Saved text embedding for {uid} to {current_text_embedding_path}")
                except Exception as e:
                    logger.error(f"Error processing text for {uid}: {e}", exc_info=True)
                    # エラーが発生しても音声処理は試みるので、continueしない

        # --- 音声埋め込みの処理 ---
        skip_audio_embedding = False
        if audio_model and not skip_audio_embedding:
            audio_path = os.path.join(root_audio_input_path, diagno, f"{uid}.wav")
            if not os.path.exists(audio_path):
                logger.error(f"Error: Audio file not found at {audio_path} for {uid}. Skipping audio embedding.")
                continue # 音声ファイルがない場合は音声埋め込みをスキップ

            try:
                features_audio = None
                processed_audio_tensor = None

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

                    features_audio = outputs_audio.last_hidden_state.squeeze(0).cpu()
                    
                    # NaN値の処理
                    if torch.isnan(features_audio).any():
                        logger.warning(f"NaN values in wav2vec2 features for {uid}. Replacing with 0.")
                        features_audio = torch.nan_to_num(features_audio, nan=0.0)

                # eGeMAPS特徴量の場合
                elif audio_model == 'egemaps':
                    y, sr = librosa.load(audio_path, sr=None) # Load with original sample rate
                    # opensmileでフレームレベルの特徴量を抽出
                    # smileオブジェクトは上記で frame_step, frame_size を設定済み
                    # process_new_chunkはLLDを返す場合があるが、Functionals設定の場合は1行になる。
                    # ここでFunctionalsが選択されているため、DataFrameは1行のみとなる。
                    # しかし、単語レベルのアライメントには複数フレームが必要。
                    # eGeMAPS Functionalsは発話全体の特徴量なので、単語レベルアライメントには不向きです。
                    # もしOpenSMILEで単語レベルアライメントを行う場合は、FeatureLevel.LLDとそれに対応する特徴量セットを使用し、
                    # 各フレームの特徴量を取得し、単語区間に該当するフレームを平均する必要があります。
                    # 現在のロジックでは、Functionalsを使うとfeatures_audioは(1, num_features)の形状になります。
                    # これを単語ごとに割り当てるのは無理があるため、注意が必要です。
                    # ここでは、Functionalsが発話レベルの特徴量であることを理解し、
                    # 後続の単語レベルアライメントが意味をなさなくなることを前提に進めます。
                    # もしくは、eGeMAPSでは単語レベルアライメントをスキップし、全体平均のみを使うべきです。

                    # ここでは、Functionalsを無理やりフレームに展開するのではなく、
                    # 発話レベルの単一特徴量として扱い、アライメントはスキップすることを検討すべきです。
                    # しかし、現在のコードの意図はアライメントを行うことなので、
                    # 単純にprocess_new_chunkの出力をfeatures_audioとして使用します。
                    features_df = smile.process_new_chunk(y, sr)
                    
                    if features_df.empty:
                        logger.warning(f"OpenSMILE returned empty features for {uid}. Skipping audio embedding.")
                        continue

                    # DataFrameをテンソルに変換
                    # ここでfeatures_audioは(1, num_features)の形状になるため、単語レベルアライメントでは問題が発生する
                    # 暫定的に、それを何度も繰り返してフレームに見せかけるか、最初の要素のみを使うか。
                    # 例えば、features_audioを必要な長さに複製するなど。
                    # しかし、それは特徴量として意味をなさないため、このパスでは単語レベルアライメントをスキップするのが適切。
                    # 以下では、このブロックの後の単語アライメントロジックが `textual_model and tokenizer and transcription` でガードされているため、
                    # textual_model がない場合、この問題は発生しません。
                    # textual_model がある場合、eGeMAPSのFunctionalsで単語アライメントを行うのは特徴量的に不適切です。
                    features_audio = torch.tensor(features_df.values).float().cpu()

                    # NaN値の処理
                    if torch.isnan(features_audio).any():
                        logger.warning(f"NaN values in eGeMAPS features for {uid}. Replacing with 0.")
                        features_audio = torch.nan_to_num(features_audio, nan=0.0)
                
                # メルスペクトログラム特徴量の場合
                elif audio_model == 'mel':
                    y, sr = librosa.load(audio_path, sr=None)
                    
                    win_length = int(0.02 * sr)  # 20msをサンプル数に変換
                    hop_length = int(0.02 * sr)  # 20ms（50セグメント/秒）
                    n_mels = 80  # メルフィルタバンクの数

                    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=win_length, hop_length=hop_length, n_mels=n_mels)
                    features_audio = torch.tensor(mel).float().permute(1,0).cpu() # (frames, n_mels)
                    
                    # NaN値の処理
                    if torch.isnan(features_audio).any():
                        logger.warning(f"NaN values in Mel features for {uid}. Replacing with 0.")
                        features_audio = torch.nan_to_num(features_audio, nan=0.0)
                
                # ここから単語レベルアライメントのロジック
                # テキスト情報がある場合のみアライメントを実行
                if textual_model and tokenizer and transcription: # tokenizerがロードされていることを確認
                    # processed_audio_tensor はmax_lengthで初期化
                    processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1]))
                    
                    # 音声特徴量の全体平均を最初の位置に設定 (CLSトークンの位置に相当)
                    # CLSトークンは通常0番目のインデックスに割り当てられる
                    processed_audio_tensor[0] = features_audio.mean(dim=0)

                    # トークン化とオフセットマッピングの取得
                    inputs_offset = tokenizer(
                        transcription,
                        return_tensors="pt",
                        return_offsets_mapping=True,  # トークン-オフセットマッピングを取得
                        padding="max_length",
                        truncation=True,
                        max_length=max_length
                    ) # .to(device) は不要、CPUで処理
                    
                    # オフセットマッピングと入力IDをCPUに
                    input_ids_cpu = inputs_offset["input_ids"].squeeze(0).cpu()
                    offset_mapping_cpu = inputs_offset["offset_mapping"].squeeze(0).cpu()

                    tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu.tolist())
                    
                    word_mapping = []
                    current_word_text = ""
                    current_token_indices = [] # トークン化されたシーケンスにおけるインデックス

                    # トークンを単語にグループ化
                    for i, (token, offset) in enumerate(zip(tokens, offset_mapping_cpu.tolist())):
                        start_char, end_char = offset

                        # 特殊トークン（[CLS], [SEP], [PAD]）をスキップ
                        if start_char == 0 and end_char == 0:
                            if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                                continue # 特殊トークンはスキップ
                        
                        # BERT/DistilBERTのサブワード接頭辞 "##" を処理
                        if token.startswith("##"):
                            current_word_text += token[2:]
                        else:
                            # 新しい単語の始まり、または最初の単語
                            if current_word_text: # 前の単語があれば保存
                                word_mapping.append((current_word_text, current_token_indices))
                            current_word_text = token # 新しい単語を開始
                        
                        # 特殊トークン以外のトークンインデックスを保存
                        current_token_indices.append(i) # トークン化されたシーケンス内のインデックス

                    # 最後の単語を保存
                    if current_word_text:
                        word_mapping.append((current_word_text, current_token_indices))
                    
                    # 単語レベルのタイムスタンプファイルのパス
                    # word_level_timestamps.csv は音声ファイルと同じディレクトリにあることが多い
                    word_level_timestamp_path = os.path.join(root_audio_input_path, diagno, f"{uid}.csv") 
                    
                    if not os.path.exists(word_level_timestamp_path):
                        logger.error(f"Error: Word-level timestamp CSV not found at {word_level_timestamp_path} for {uid}. Cannot perform word-level alignment.")
                        # ここでcontinueすると、音声埋め込み保存までスキップされる
                        # その場合、processed_audio_tensorをそのまま（アライメントなしで）保存するか、エラーとして処理するかを決める
                        # 現在のロジックでは、アライメント部分がエラーになり、processed_audio_tensorが未定義のままになるため、問題
                        # processed_audio_tensor = Noneとしておき、後続でNoneチェックする
                        processed_audio_tensor = None
                        
                    if processed_audio_tensor is not None: # タイムスタンプCSVが読み込めた場合のみ
                        # 単語レベルのタイムスタンプを読み込み
                        df_word_level = pd.read_csv(word_level_timestamp_path)
                        # カラム: ['word', 'start', 'end', 'probability']
                        words_timestamps = []
                        for _, data in df_word_level.iterrows():
                            words_timestamps.append((data['word'], data['start'], data['end']))

                        # アライメントロジック
                        word_map_idx = 0 # word_mapping (トークンから再構築された単語) のインデックス
                        timestamp_idx = 0 # words_timestamps (CSVからの単語) のインデックス

                        while word_map_idx < len(word_mapping) and timestamp_idx < len(words_timestamps):
                            mapped_word, mapped_token_indices = word_mapping[word_map_idx]
                            timestamp_word, start_time, end_time = words_timestamps[timestamp_idx]

                            cleaned_mapped_word = mapped_word.replace('Ġ', '').lower()
                            cleaned_timestamp_word = timestamp_word.replace('Ġ', '').lower()

                            # 句読点や特殊記号を簡易的に判断
                            # トークナイザーが生成する特殊トークン (例: [CLS], [SEP], [PAD]) もここに含まれる可能性がある
                            is_special_or_punctuation = not cleaned_mapped_word.isalnum() or \
                                                         mapped_token_indices[0] == 0 # CLSトークン (idx 0)
                            
                            # 単語のアライメント
                            if cleaned_mapped_word == cleaned_timestamp_word:
                                # タイムスタンプを使って音声セグメントを抽出
                                start_segment = math.floor(start_time * segment_length)
                                end_segment = math.ceil(end_time * segment_length)
                                
                                # セグメント範囲の調整
                                if end_segment <= start_segment: # 少なくとも1フレーム
                                    end_segment = start_segment + 1
                                if start_segment < 0: start_segment = 0
                                if end_segment > features_audio.shape[0]: end_segment = features_audio.shape[0]

                                # 実際に音声特徴量が取得できるか確認
                                if start_segment < end_segment:
                                    audio_features_segment = features_audio[start_segment:end_segment]
                                    avg_audio_feature = torch.clamp(audio_features_segment.mean(dim=0), min=-1e3, max=1e3)
                                else:
                                    # データがない場合はゼロ、または直前の単語の平均を使用
                                    avg_audio_feature = torch.zeros(features_audio.shape[1]) 
                                    logger.warning(f"No audio segment for word '{cleaned_mapped_word}' ({uid}). Using zero feature.")

                                # 関連するすべてのトークンインデックスに同じ音声特徴量を割り当てる
                                for tok_idx in mapped_token_indices:
                                    if 1 <= tok_idx < max_length: # CLSトークン(0)とPAD/SEP以外
                                        processed_audio_tensor[tok_idx] = avg_audio_feature
                                
                                word_map_idx += 1
                                timestamp_idx += 1
                            elif is_special_or_punctuation:
                                # 句読点や特殊トークンはタイムスタンプCSVにないことが多い
                                # これらのトークンには、例えば、全体の平均、直前の単語の平均、またはゼロを割り当てる。
                                # 現在、processed_audio_tensorはゼロ初期化されているため、何もしなければゼロになる。
                                # CLSトークン(0番目)は既に全体平均で埋められている。
                                for tok_idx in mapped_token_indices:
                                    if 1 <= tok_idx < max_length and tok_idx != 0: # CLSトークンは除く
                                        # 句読点トークンに対して特定の処理が必要な場合、ここにロジックを追加
                                        # 例えば、直前の単語の音声特徴量をコピーするなど
                                        pass # 現在はゼロのまま
                                word_map_idx += 1
                                # timestamp_idx は進めない (この句読点に対応する timestamp_word はないため)
                            elif len(cleaned_mapped_word) > len(cleaned_timestamp_word) and cleaned_mapped_word.startswith(cleaned_timestamp_word):
                                # テキストモデルが単語をさらに分割した場合（例: "running" -> "run", "##ning"）
                                # このケースは複雑。ここではtimestamp_wordを進めるが、mapped_wordは進めない (部分一致の次のトークンを見るため)
                                # これだとmapped_wordが永久に進まない可能性があるため、より複雑なアライメントロジックが必要
                                logger.warning(f"Partial text-word match: mapped='{cleaned_mapped_word}', timestamp='{cleaned_timestamp_word}'. Advancing timestamp.")
                                timestamp_idx += 1 
                                # mapped_wordはそのまま (次のタイムスタンプワードが残りのトークンにマッチするか見る)
                            else:
                                # 単語が一致しない場合（スペルミス、認識エラーなど）
                                # どちらか一方のインデックスを進めて、アライメントを続ける
                                logger.warning(f"Word mismatch: mapped='{cleaned_mapped_word}', timestamp='{cleaned_timestamp_word}'. Advancing timestamp.")
                                timestamp_idx += 1
                                # mapped_word_idx はそのまま。この mapped_word が次の timestamp_word にマッチするか見る

                        # アライメントできなかった残りのテキストトークン（主にPADトークンや不一致）はゼロのまま
                        # processed_audio_tensor の 0番目のインデックスは既に全体平均で埋められている。

                elif audio_model and not textual_model: # 音声のみの処理
                    # 単語レベルアライメントは不要。音声特徴量を直接、固定長にパディング/切り捨てして保存する
                    # features_audio は (フレーム数, 次元) の形状
                    if features_audio.shape[0] > max_length:
                        processed_audio_tensor = features_audio[:max_length].cpu()
                    else:
                        processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1])).cpu()
                        processed_audio_tensor[:features_audio.shape[0]] = features_audio
                    logger.info(f"Skipping word-level alignment for {uid} (audio-only mode). Saved fixed-length audio embedding.")
                else:
                    # 音声モデルが指定されていない、またはテキストモデルも指定されていない場合、ここには来ないはず
                    # あるいは、単語レベルアライメントができないためスキップされる
                    processed_audio_tensor = None
                    logger.warning(f"Could not prepare processed_audio_tensor for {uid} (possibly due to missing text/alignment issues).")

                if processed_audio_tensor is not None:
                    # NaN値の最終チェック
                    if torch.isnan(processed_audio_tensor).any():
                        logger.error(f"ERROR: NaN values in final processed_audio_tensor for {uid}. Replacing with 0.")
                        processed_audio_tensor = torch.nan_to_num(processed_audio_tensor, nan=0.0)
                    
                    torch.save(processed_audio_tensor, current_audio_embedding_path)
                    logger.info(f"Saved audio embedding for {uid} to {current_audio_embedding_path}")
                else:
                    logger.error(f"Failed to generate processed_audio_tensor for {uid}. Skipping saving audio embedding.")
            
            except Exception as e:
                logger.error(f"Error processing audio for {uid}: {e}", exc_info=True)
                # エラーが発生しても次のファイルの処理は試みる
                continue 

        completed_audios += 1

    logger.info(f"------------------------------------------")
    logger.info(f"Completed processing all audios listed in CSV: {completed_audios}")
    logger.info(f"Total entries in CSV: {len(df)}")


# メイン実行部分
if __name__ == "__main__":
    try:
        logger.info("Starting preprocessing embeddings...")
        preprocess_embeddings()
        logger.info("Preprocessing completed successfully!")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
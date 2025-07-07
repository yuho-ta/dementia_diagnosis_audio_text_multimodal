# =============================
# Whisperによる音声→テキスト前処理スクリプト
# - 音声ファイルからテキスト書き起こし・単語ごとの信頼度抽出
# - セグメンテーション情報で話者分離・ポーズ情報も考慮
# - 結果をCSVファイルとして保存
# =============================

import whisper
import os
import pandas as pd
import torch
import random
import re

# 英数字と一部記号以外を除去する関数
# テキストから不要な文字（日本語や記号など）を取り除く
# 例: "Hello! こんにちは。" → "Hello! "
def remove_non_english(text):
    return re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)

# Whisperの"turbo"モデルをロード
model = whisper.load_model("turbo")

# 音声ファイルのルートパス
root_path = os.path.join('dataset', 'diagnosis', 'train', 'audio')

# 診断カテゴリ（ad: アルツハイマー, cn: 正常）
diagnosis = ['ad', 'cn']
textual_data = os.path.join('dataset', 'diagnosis', 'train', 'text_transcriptions.csv')

# メイン前処理関数
def preprocess_whisper():
    # 結果を格納するデータフレームを作成
    df = pd.DataFrame(columns=['uid', 'diagno', 'transcription', 'transcription_pause', 'probablities'])

    # 各診断カテゴリごとにループ
    for diagno in diagnosis:
        diagno_path = os.path.join(root_path, diagno)

        for file in os.listdir(diagno_path):
            if file.endswith(".wav"):
                # 単語レベルのCSVパスを作成
                word_level_path = os.path.join(diagno_path, file).replace('.wav', '.csv').replace('audio', 'text')
                
                # 既に処理済みのファイルはスキップ
                if os.path.exists(word_level_path):
                    print(f'Skipping already processed: {file}')
                    
                    # 既存データをデータフレームに追加（未登録の場合）
                    uid = file.replace('.wav', '')
                    if not df[(df['uid'] == uid) & (df['diagno'] == diagno)].empty:
                        print(f'Data for {uid} already in dataframe')
                        continue
                    
                    # 既存CSVから書き起こしデータを復元
                    try:
                        word_level_df = pd.read_csv(word_level_path)
                        # 単語リストから書き起こし文を再構成
                        transcription = ''
                        transcription_pauses = ''
                        prev_start = 0.0
                        words = word_level_df['word'].tolist()
                        starts = word_level_df['start'].tolist() if 'start' in word_level_df.columns else [0.0]*len(words)
                        transcription_words = []
                        transcription_pauses_words = []
                        for i, word in enumerate(words):
                            transcription_words.append(word)
                            transcription_pauses_words.append(word)
                            if i > 0:
                                pause = starts[i] - prev_start
                                if pause > 2:
                                    transcription_pauses_words[-1] += ' ...'
                                elif pause > 1:
                                    transcription_pauses_words[-1] += ' .'
                                elif pause > 0.5:
                                    transcription_pauses_words[-1] += ' ,'
                            prev_start = starts[i]
                        transcription = ' '.join(transcription_words)
                        transcription_pauses = ' '.join(transcription_pauses_words)
                        probs = list(zip(words, word_level_df['probability'].tolist()))
                        df = df._append({
                            'uid': uid, 
                            'diagno': diagno, 
                            'transcription': remove_non_english(transcription), 
                            'transcription_pause': remove_non_english(transcription_pauses), 
                            'probablities': probs
                        }, ignore_index=True)
                        print(f'Added existing data for {uid}')
                        continue  # 既存データを追加したら次のファイルに進む
                    except Exception as e:
                        print(f'Error loading existing data for {uid}: {e}')
                        continue

                print('Processing:', file)
                
                audio_path = os.path.join(diagno_path, file)
                print(audio_path)
                # セグメンテーション（話者分離）CSVのパス
                segmentation_path = audio_path.replace('.wav', '.csv').replace('audio', 'segmentation')
                
                excluding_times = []

                # セグメンテーション情報があれば、話者が"INV"の区間を除外リストに追加
                if os.path.exists(segmentation_path):
                    df_segmentation = pd.read_csv(segmentation_path)
                    df_segmentation = df_segmentation[df_segmentation['speaker'] == 'INV']
                    for segment in df_segmentation.iterrows():
                        excluding_times += [(segment[1]['begin']/1000, segment[1]['end']/1000)]

                idx_exclude = 0
                # Whisperで音声認識し、単語ごとのタイムスタンプと確率を取得
                result = model.transcribe(audio_path, word_timestamps=True)

                probs = []
                print('Excluding times:', excluding_times)

                transcription = ''
                transcription_pauses = ''
                prev_start = 0.0

                # 単語ごとの情報を格納するデータフレーム
                pandas_word_level = pd.DataFrame(columns=['word', 'start', 'end', 'probability'])

                # 各セグメント・単語ごとに処理
                for segment in result['segments']:
                    for word in segment['words']:
                        # 除外区間の管理
                        if idx_exclude < len(excluding_times) and word['start'] >= excluding_times[idx_exclude][1]:
                            idx_exclude += 1

                        # 除外区間外の単語のみ処理
                        if idx_exclude >= len(excluding_times) or word['end'] < excluding_times[idx_exclude][0]:
                            transcription_pauses += word['word']
                            transcription += word['word']
                            # 単語をクリーンアップ
                            clean_word = remove_non_english(word['word'].replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower())
                            if clean_word != '':
                                pandas_word_level = pandas_word_level._append({'word': clean_word, 'start': word['start'], 'end': word['end'], 'probability': word['probability']}, ignore_index=True)
                            probs += [(clean_word, word['probability'])]
                            
                            # 前の単語との間隔（ポーズ）に応じて記号を挿入
                            if prev_start > 0.0:
                                pause = word['start'] - prev_start

                                if pause > 2:
                                    transcription_pauses += ' ...'
                                elif pause > 1:
                                    transcription_pauses += ' .'
                                elif pause > 0.5:
                                    transcription_pauses += ' ,'

                            prev_start = word['end']
                        else:
                            print('Excluding word:', word)

                        if idx_exclude < len(excluding_times) and word['end'] >= excluding_times[idx_exclude][1]:
                            idx_exclude += 1
                        
                print('Result:', result['text'])
                print('Transcription:', transcription)
                print('Transcription pauses:', transcription_pauses)
                print('Probs:', probs)

                # 単語ごとの情報をCSVに保存
                os.makedirs(os.path.dirname(word_level_path), exist_ok=True)
                pandas_word_level.to_csv(word_level_path, index=False)

                # 全体のデータフレームにも追加
                df = df._append({'uid': file.replace('.wav', ''), 'diagno': diagno, 'transcription': remove_non_english(transcription), 'transcription_pause': remove_non_english(transcription_pauses), 'probablities': probs}, ignore_index=True)
    # 全ファイル分の書き起こし結果をまとめてCSVに保存
    os.makedirs(os.path.dirname(textual_data), exist_ok=True)
    df.to_csv(textual_data, index=False)

# スクリプトを直接実行した場合に前処理を開始
preprocess_whisper()
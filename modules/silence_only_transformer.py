import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer
)
from pydub import AudioSegment
import io
from sklearn.model_selection import KFold
from torch import nn

# モデル設定（英語・多言語対応wav2vec2-large-xlsr-53）
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
NUM_LABELS = 2  # 2クラス分類に変更（ad/cn）

# ハイパーパラメータ
BATCH_SIZE = 1
GRAD_ACCUM = 4
EPOCHS = 15
LR = 3e-5
WARMUP_RATIO = 0.1

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データディレクトリ
data_dir = "dataset/diagnosis/train/silence_audio"

# dataリスト作成（mp3とwav両対応）
data = []
for label in ["ad", "cn"]:
    folder = os.path.join(data_dir, label)
    for file in os.listdir(folder):
        if file.endswith(".mp3") or file.endswith(".wav"):
            data.append({
                "path": os.path.join(folder, file),
                "label": 1 if label == "ad" else 0
            })
print(f"Total samples found: {len(data)}")
assert len(data) > 0, "No audio files found. Check data_dir and file paths."
# 音声ファイル読み込み関数（mp3対応）
def speech_file_to_array_fn(batch):
    if batch["path"].endswith(".mp3"):
        audio = AudioSegment.from_mp3(batch["path"])
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        speech_array, sampling_rate = torchaudio.load(wav_io)
    else:
        speech_array, sampling_rate = torchaudio.load(batch["path"])

    # 16kHzにリサンプリング
    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    batch["speech"] = resampler(speech_array.mean(dim=0)).view(-1).numpy()

    return batch

# DataFrame & Huggingface Dataset作成
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# 音声データ読み込み・加工をDatasetにマッピング
dataset = dataset.map(speech_file_to_array_fn)

# Feature Extractor準備
processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

# 前処理（processorで特徴量抽出）
def preprocess(batch):
    inputs = processor(batch["speech"], sampling_rate=16000, padding=True, return_tensors="pt")
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # バッチ次元(=1)を除去
    inputs["labels"] = batch["label"]
    return inputs

# k-fold cross validation設定
kf = KFold(n_splits=5, shuffle=True, random_state=42)

all_fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n### Fold {fold + 1}")

    # train/val データセット選択し前処理適用
    train_dataset = dataset.select(train_idx).map(preprocess, remove_columns=dataset.column_names)
    val_dataset = dataset.select(val_idx).map(preprocess, remove_columns=dataset.column_names)
    train_labels = [train_dataset[i]["labels"] for i in range(len(train_dataset))]
    val_labels = [val_dataset[i]["labels"] for i in range(len(val_dataset))]

    print(f"Fold {fold + 1} train label distribution:\n", pd.Series(train_labels).value_counts())
    print(f"Fold {fold + 1} val label distribution:\n", pd.Series(val_labels).value_counts())


    # モデル準備：2クラス分類、標準の分類ヘッド使用
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="single_label_classification",
    ).to(device)

    # TrainingArguments設定
    training_args = TrainingArguments(
        output_dir=f"./results_fold{fold+1}",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        save_total_limit=1,
        logging_dir=f"./logs_fold{fold+1}",
        logging_steps=10,
        # load_best_model_at_end=True,  ← 一時的にコメントアウト
        metric_for_best_model="accuracy"
    )

    # 評価指標
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}

    # Trainer作成（tokenizerは不要なので渡さない）
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=None  # → 削除 or processor=processor にする
    )

    # 学習開始
    trainer.train()

    # 評価
    eval_result = trainer.evaluate()
    eval_result["fold"] = fold + 1
    all_fold_results.append(eval_result)

# 全foldの結果表示
df_results = pd.DataFrame(all_fold_results)
print(df_results)

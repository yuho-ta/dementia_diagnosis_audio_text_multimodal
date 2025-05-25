<h1 align="center">CogniAlign: Word-Level Multimodal Speech Alignment with Gated Cross-Attention for Alzheimer’s Detection</h1>

## Overview

This repository contains the official implementation of the paper *“CogniAlign: Word-Level Multimodal Speech Alignment with Gated Cross-Attention for Alzheimer’s Detection”*.

The CogniAlign pipeline integrates Whisper-based audio transcription and DistilBert text embedding to perform word-level multimodal fusion using a Gated Cross-Attention Transformer. This enables precise alignment of spoken and textual features for Alzheimer's detection from interview data.


## 📊 Word-Level Fusion Architecture

<p align="center">
  <img src="imgs/word-level-fusion.svg" alt="Word-Level Fusion Architecture" width="600"/>
</p>

<p align="center">
  <em>The CogniAlign pipeline. Word-level timestamps are extracted using Whisper and aligned with DistilBert text embeddings. Gated cross-attention fuses both modalities before classification.</em>
</p>

---

## 📁 Project Structure

```
├── main.py                   # Entry point for training/evaluation
├── dataset.py                # Data loading, alignment, and batching
├── Preprocess                # Preprocess folder
├──── preprocesswhisper.py    # Whisper-based speech transcription
├──── preprocessembeddings.py # Preprocess data to obtain embeddings
├── utils.py                  # General-purpose utilities
├── model.py                  # Proposed Architectures Modules
```

---

## 🔧 Installation

```bash
git clone https://github.com/davidorp/CogniAlign
cd CogniAlign
```

Requirements include:
- `transformers`
- `torchaudio`
- `torch`
- `whisper`
- `wandb`
- `opensmile`
- `librosa`

---

## 📂 Dataset

This project uses the **ADReSSo Challenge dataset**, which provides audio recordings of spontaneous speech from individuals with Alzheimer’s Disease (AD) and Healthy Controls (HC). The dataset does **not include transcripts**—we generate them automatically using the Whisper speech recognition model.

Due to privacy and ethical restrictions, the ADReSSo dataset is not publicly available.  
To access the dataset, you must request permission from the original organizers:

👉 [Official ADReSSo Challenge page](https://dementia.talkbank.org/ADReSSo-2021/)

### 🗣️ Transcriptions

Transcriptions are generated using OpenAI’s [Whisper](https://github.com/openai/whisper), a robust multilingual speech recognition system that enables word-level alignment necessary for multimodal fusion.


## 🧪 Preprocessing

1. **Transcription Audio preprocessing (Whisper-based):**
   ```bash
   python preprocesswhisper.py
   ```

2. **Embeddings preprocessing:**
   ```bash
   python preprocessembeddings.py
   ```

The preprocessing scripts extract word-level timestamps and align them with spoken utterances and transcripts. Desired models have to be specified in the code

---

## 🧠 Model

`model.py` implements the **Gated Cross-Attention Fusion Transformer** for combining BERT (text) and Whisper (speech) embeddings at the word level. The fusion strategy is designed for early, late, or gated interaction depending on configuration.

---

## 🚀 Training

Run model training:
```bash
python main.py --conig config_file
```

Modify the configuration file for testing different models.

---

## 📬 Contact

For questions, contact: dortiz@dtic.ua.es

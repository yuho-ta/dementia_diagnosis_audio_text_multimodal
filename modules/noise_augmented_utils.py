#!/usr/bin/env python3
# =============================
# ノイズ追加分類器用ユーティリティ関数
# - 被験者ID抽出
# - ログ設定
# - シード設定
# =============================

import os
import re
import logging
from datetime import datetime
import torch
import numpy as np

def set_seed(seed=42):
    """再現性のためのシード設定"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # マルチGPUの場合
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging():
    """ログ設定"""
    log_filename = f"noise_augmented_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.DEBUG,  # DEBUGレベルに変更
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def extract_subject_id_from_filename(filename):
    """ファイル名から番号を抽出する（例：714-0_original.pt → 714-0）"""
    # ファイル名から番号部分を抽出
    match = re.match(r'(\d+-\d+)', filename)
    if match:
        return match.group(1)
    return None

def extract_par_id_from_cha_file(cha_file_path):
    """CHAファイルからPARのIDを抽出する"""
    try:
        with open(cha_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # @ID行でPARのIDを探す
        par_id_match = re.search(r'@ID:\s*eng\|Pitt\|PAR\|.*', content)
        if par_id_match:
            return par_id_match.group(0)
        return None
    except Exception as e:
        logging.warning(f"エラー: {cha_file_path} を読み込めませんでした: {e}")
        return None

def get_subject_id_from_cha_file(uid, diagno):
    """UIDと診断カテゴリからCHAファイルを探してPAR IDを抽出する"""
    # CHAファイルのパスを構築
    cha_file_path = os.path.join('dataset', 'diagnosis', 'train', 'segmentation', diagno, f"{uid}.cha")
    
    if os.path.exists(cha_file_path):
        par_id = extract_par_id_from_cha_file(cha_file_path)
        if par_id:
            return par_id
        else:
            logging.debug(f"CHA file {uid}.cha: no PAR ID found, using fallback")
    else:
        logging.debug(f"CHA file not found: {cha_file_path}")
    
    # CHAファイルが見つからない場合やPAR IDが抽出できない場合は、ファイル名から抽出
    fallback_id = uid.split('-')[0] if '-' in uid else uid
    logging.warning(f"CHA file not found or PAR ID extraction failed for {uid}, falling back to filename extraction: {fallback_id}")
    return fallback_id

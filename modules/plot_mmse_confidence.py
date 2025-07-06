import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob

def load_mismatch_data(log_dir):
    """
    mismatchファイルからデータを読み込む（ファイルごとに分離）
    """
    file_data = {}
    
    # 全てのmismatchファイルとconfidenceファイルを検索
    mismatch_pattern = os.path.join(log_dir, "*_best_mismatched_uids.txt")
    confidence_pattern = os.path.join(log_dir, "*_best_predictions_output.txt")
    mismatch_files = glob.glob(mismatch_pattern)
    confidence_files = glob.glob(confidence_pattern)
    
    # 両方のファイルタイプを処理
    files = mismatch_files + confidence_files
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        file_data[file_name] = []
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # ヘッダー行をスキップしてデータを読み込み
            data_lines = [line.strip() for line in lines if line.startswith('adrso')]
            
            for line in data_lines:
                parts = line.split(', ')
                if len(parts) >= 4:
                    uid = parts[0]
                    prob = float(parts[1])
                    mmse = int(parts[2]) if parts[2] != 'N/A' else np.nan
                    dx = parts[3]
                    
                    # confidenceファイルの場合はprobをそのまま、mismatchファイルの場合は0.5からの距離を使用
                    if '_best_confidence_output.txt' in file_path:
                        confidence = prob  # 予測確率をそのまま使用
                    else:
                        confidence = abs(prob - 0.5) * 2  # 0.5からの距離に変換
                    
                    file_data[file_name].append({
                        'UID': uid,
                        'Confidence': confidence,
                        'MMSE': mmse,
                        'Diagnosis': dx
                    })
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return file_data

def create_plots(file_data, output_dir='plots'):
    """
    ファイルごとにグラフを作成
    """
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 日本語フォント設定
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    for file_name, data_list in file_data.items():
        if not data_list:
            continue
            
        df = pd.DataFrame(data_list)
        
        # ファイル名から拡張子を除去
        if '_best_mismatched_uids.txt' in file_name:
            base_name = file_name.replace('_best_mismatched_uids.txt', '')
            file_type = 'mismatch'
        elif '_best_confidence_output.txt' in file_name:
            base_name = file_name.replace('_best_confidence_output.txt', '')
            file_type = 'confidence'
        else:
            base_name = file_name
            file_type = 'unknown'
        
        print(f"\n=== {base_name} の分析 ===")
        print(f"データ数: {len(df)}")
        print(f"CN (Healthy): {len(df[df['Diagnosis'] == 'cn'])}")
        print(f"AD (Dementia): {len(df[df['Diagnosis'] == 'ad'])}")
        
        # 1. MMSE vs Confidence の散布図
        plt.figure(figsize=(12, 8))
        
        # CNとADで色分け
        cn_data = df[df['Diagnosis'] == 'cn']
        ad_data = df[df['Diagnosis'] == 'ad']
        
        if file_type == 'mismatch':
            # mismatchファイルの場合：誤分類の説明
            plt.scatter(cn_data['MMSE'], cn_data['Confidence'], 
                       alpha=0.7, s=100, label='CN (Healthy but diagnosed as AD)', color='blue')
            plt.scatter(ad_data['MMSE'], ad_data['Confidence'], 
                       alpha=0.7, s=100, label='AD (Dementia but diagnosed as CN)', color='red')
        else:
            # confidenceファイルの場合：実際の診断ラベル
            plt.scatter(cn_data['MMSE'], cn_data['Confidence'], 
                       alpha=0.7, s=100, label='CN (Healthy)', color='blue')
            plt.scatter(ad_data['MMSE'], ad_data['Confidence'], 
                       alpha=0.7, s=100, label='AD (Dementia)', color='red')
        
        plt.xlabel('MMSE Score', fontsize=14)
        if file_type == 'confidence':
            plt.ylabel('Prediction Probability', fontsize=14)
        else:
            plt.ylabel('Confidence', fontsize=14)
        if file_type == 'mismatch':
            title_suffix = '(Mismatched Predictions)'
        elif file_type == 'confidence':
            title_suffix = '(All Predictions with Confidence)'
        else:
            title_suffix = '(Predictions)'
            
        plt.title(f'MMSE Score vs Model Confidence\n{base_name} {title_suffix}', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.axvline(x=26, color='gray', linestyle='--', alpha=0.7, label='MMSE=24 (Diagnosis boundary)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_name}_{file_type}_mmse_vs_confidence_scatter.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
def main():
    """
    メイン関数
    """
    # ログディレクトリのパス（必要に応じて変更）
    log_dir = './logs/distilbert_wav2vec2_P_gatedcross_mean'
    
    print("MMSEとConfidenceの関係を分析中...")
    
    # データを読み込み
    file_data = load_mismatch_data(log_dir)
    
    if not file_data:
        print("データが見つかりませんでした。")
        return
    
    print(f"読み込まれたファイル数: {len(file_data)}")
    
    # グラフを作成
    create_plots(file_data)
    
    print("\n分析完了！グラフは 'plots' ディレクトリに保存されました。")

if __name__ == "__main__":
    main() 
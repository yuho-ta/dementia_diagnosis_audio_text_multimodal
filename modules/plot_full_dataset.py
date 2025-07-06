import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_full_dataset(csv_path):
    """
    CSVファイルから全データを読み込む
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"データ読み込み成功: {len(df)}サンプル")
        return df
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return None

def create_full_dataset_plots(df, output_dir='plots_full_dataset'):
    """
    全データセットの様々なグラフを作成
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    cn_data = df[df['dx'] == 'cn']
    ad_data = df[df['dx'] == 'ad']
    
    print(f"CN (Healthy): {len(cn_data)}サンプル")
    print(f"AD (Dementia): {len(ad_data)}サンプル")
    
    # 1. MMSE分布のヒストグラム（診断別）
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.hist(cn_data['mmse'], bins=20, alpha=0.7, label='CN (Healthy)', color='blue', edgecolor='black')
    plt.xlabel('MMSE Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('MMSE Distribution - CN (Healthy)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.hist(ad_data['mmse'], bins=20, alpha=0.7, label='AD (Dementia)', color='red', edgecolor='black')
    plt.xlabel('MMSE Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('MMSE Distribution - AD (Dementia)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.hist([cn_data['mmse'], ad_data['mmse']], bins=20, alpha=0.7, 
             label=['CN (Healthy)', 'AD (Dementia)'], color=['blue', 'red'], edgecolor='black')
    plt.xlabel('MMSE Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('MMSE Distribution - Combined', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    cn_data['mmse'].plot(kind='density', label='CN (Healthy)', color='blue', linewidth=2)
    ad_data['mmse'].plot(kind='density', label='AD (Dementia)', color='red', linewidth=2)
    plt.xlabel('MMSE Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('MMSE Density Distribution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mmse_distributions.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. MMSEの箱ひげ図（診断別）
    plt.figure(figsize=(12, 8))
    
    data_for_box = [cn_data['mmse'], ad_data['mmse']]
    labels = ['CN (Healthy)', 'AD (Dementia)']
    
    box_plot = plt.boxplot(data_for_box, labels=labels, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightblue')
    box_plot['boxes'][1].set_facecolor('lightcoral')
    
    plt.ylabel('MMSE Score', fontsize=14)
    plt.title('MMSE Score Distribution by Diagnosis\n(Full Dataset)', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # 統計情報を表示
    cn_mean = cn_data['mmse'].mean()
    ad_mean = ad_data['mmse'].mean()
    cn_std = cn_data['mmse'].std()
    ad_std = ad_data['mmse'].std()
    
    plt.text(0.02, 0.98, 
             f'CN Mean: {cn_mean:.1f} ± {cn_std:.1f}\nAD Mean: {ad_mean:.1f} ± {ad_std:.1f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mmse_boxplot_full.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 診断境界の分析
    plt.figure(figsize=(15, 10))
    
    # MMSE=24の境界線付近の詳細分析
    plt.subplot(2, 2, 1)
    boundary_range = range(20, 30)
    cn_boundary = cn_data[cn_data['mmse'].isin(boundary_range)]
    ad_boundary = ad_data[ad_data['mmse'].isin(boundary_range)]
    
    plt.hist(cn_boundary['mmse'], bins=10, alpha=0.7, label='CN', color='blue', edgecolor='black')
    plt.hist(ad_boundary['mmse'], bins=10, alpha=0.7, label='AD', color='red', edgecolor='black')
    plt.axvline(x=26, color='gray', linestyle='--', alpha=0.8, label='MMSE=24 (Diagnosis boundary)')
    plt.xlabel('MMSE Score (20-29)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('MMSE Distribution Around Diagnosis Boundary', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MMSEスコアの範囲別患者数
    plt.subplot(2, 2, 2)
    mmse_ranges = ['0-10', '11-15', '16-20', '21-25', '26-30']
    cn_counts = []
    ad_counts = []
    
    for i, range_name in enumerate(mmse_ranges):
        start = i * 5 + 1 if i > 0 else 0
        end = (i + 1) * 5 if i < 4 else 30
        
        cn_count = len(cn_data[(cn_data['mmse'] >= start) & (cn_data['mmse'] <= end)])
        ad_count = len(ad_data[(ad_data['mmse'] >= start) & (ad_data['mmse'] <= end)])
        
        cn_counts.append(cn_count)
        ad_counts.append(ad_count)
    
    x = np.arange(len(mmse_ranges))
    width = 0.35
    
    plt.bar(x - width/2, cn_counts, width, label='CN (Healthy)', color='blue', alpha=0.7)
    plt.bar(x + width/2, ad_counts, width, label='AD (Dementia)', color='red', alpha=0.7)
    
    plt.xlabel('MMSE Score Range', fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    plt.title('Patient Distribution by MMSE Range', fontsize=14)
    plt.xticks(x, mmse_ranges)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mmse_diagnosis_boundary.png'), dpi=300, bbox_inches='tight')
    plt.show()

# 実行
if __name__ == "__main__":
    csv_path = "dataset/diagnosis/train/adresso-train-mmse-scores.csv"
    df = load_full_dataset(csv_path)
    if df is not None:
        create_full_dataset_plots(df)

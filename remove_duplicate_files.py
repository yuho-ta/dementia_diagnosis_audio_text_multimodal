import os
import re
import shutil
from collections import defaultdict

def extract_number_from_filename(filename):
    """ファイル名から番号を抽出する（例：714-0_silence_combined.mp3 → 714-0）"""
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
        print(f"エラー: {cha_file_path} を読み込めませんでした: {e}")
        return None

def find_duplicate_files():
    """重複するPAR IDのファイルを検索する"""
    base_path = "modules/dataset/diagnosis/train"
    silence_audio_path = os.path.join(base_path, "silence_audio")
    silence_features_path = os.path.join(base_path, "silence_features")
    segmentation_path = os.path.join(base_path, "segmentation")
    
    folders = ['ad', 'cn']
    par_id_to_files = defaultdict(list)
    
    for folder in folders:
        print(f"\n=== {folder.upper()} フォルダの処理 ===")
        
        # CHAファイルからPAR IDを取得
        cha_folder = os.path.join(segmentation_path, folder)
        if os.path.exists(cha_folder):
            cha_files = [f for f in os.listdir(cha_folder) if f.endswith('.cha')]
            
            for cha_file in cha_files:
                number = extract_number_from_filename(cha_file)
                if number:
                    cha_file_path = os.path.join(cha_folder, cha_file)
                    par_id = extract_par_id_from_cha_file(cha_file_path)
                    if par_id:
                        par_id_to_files[par_id].append({
                            'folder': folder,
                            'number': number,
                            'cha_file': cha_file,
                            'cha_path': cha_file_path
                        })
                        print(f"{number} → PAR ID: {par_id[:50]}...")
    
    return par_id_to_files

def remove_duplicate_files(par_id_to_files, keep_first=True):
    """重複ファイルを削除する（一つを残す）"""
    base_path = "modules/dataset/diagnosis/train"
    silence_audio_path = os.path.join(base_path, "silence_audio")
    silence_features_path = os.path.join(base_path, "silence_features")
    
    removed_files = []
    kept_files = []
    
    print("\n=== 重複ファイルの削除処理 ===")
    
    for par_id, files in par_id_to_files.items():
        if len(files) > 1:
            print(f"\n重複するPAR ID: {par_id[:50]}...")
            print(f"ファイル数: {len(files)}")
            
            # 最初のファイルを保持
            if keep_first:
                keep_file = files[0]
                files_to_remove = files[1:]
            else:
                # 最後のファイルを保持
                keep_file = files[-1]
                files_to_remove = files[:-1]
            
            print(f"保持するファイル: {keep_file['folder']}/{keep_file['cha_file']}")
            kept_files.append(keep_file)
            
            # 削除するファイルを処理
            for file_info in files_to_remove:
                folder = file_info['folder']
                number = file_info['number']
                
                # silence_audioフォルダのファイルを削除
                audio_file_pattern = f"{number}_silence_combined.mp3"
                audio_path = os.path.join(silence_audio_path, folder, audio_file_pattern)
                
                # silence_featuresフォルダのファイルを削除
                features_file_pattern = f"{number}_silence_features.npy"
                features_path = os.path.join(silence_features_path, folder, features_file_pattern)
                
                removed_file_info = {
                    'folder': folder,
                    'number': number,
                    'cha_file': file_info['cha_file'],
                    'audio_file': audio_file_pattern,
                    'features_file': features_file_pattern
                }
                
                # ファイルが存在する場合は削除
                if os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                        print(f"  削除: {folder}/{audio_file_pattern}")
                        removed_file_info['audio_deleted'] = True
                    except Exception as e:
                        print(f"  エラー: {folder}/{audio_file_pattern} の削除に失敗: {e}")
                        removed_file_info['audio_deleted'] = False
                else:
                    print(f"  存在しない: {folder}/{audio_file_pattern}")
                    removed_file_info['audio_deleted'] = False
                
                if os.path.exists(features_path):
                    try:
                        os.remove(features_path)
                        print(f"  削除: {folder}/{features_file_pattern}")
                        removed_file_info['features_deleted'] = True
                    except Exception as e:
                        print(f"  エラー: {folder}/{features_file_pattern} の削除に失敗: {e}")
                        removed_file_info['features_deleted'] = False
                else:
                    print(f"  存在しない: {folder}/{features_file_pattern}")
                    removed_file_info['features_deleted'] = False
                
                removed_files.append(removed_file_info)
    
    return removed_files, kept_files

def main():
    print("重複ファイルの検索を開始...")
    
    # 重複ファイルを検索
    par_id_to_files = find_duplicate_files()
    
    # 重複があるかチェック
    duplicates = {par_id: files for par_id, files in par_id_to_files.items() if len(files) > 1}
    
    if not duplicates:
        print("\n重複するPAR IDは見つかりませんでした。")
        return
    
    print(f"\n重複するPAR ID数: {len(duplicates)}")
    
    # 削除前の確認
    total_files_to_remove = sum(len(files) - 1 for files in duplicates.values())
    print(f"削除予定ファイル数: {total_files_to_remove}")
    
    # 削除処理の実行
    removed_files, kept_files = remove_duplicate_files(duplicates, keep_first=True)
    
    # 結果の表示
    print(f"\n=== 削除処理完了 ===")
    print(f"保持されたファイル数: {len(kept_files)}")
    print(f"削除されたファイル数: {len(removed_files)}")
    
    # 結果をファイルに保存
    output_file = "duplicate_removal_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== 重複ファイル削除結果 ===\n\n")
        
        f.write("保持されたファイル:\n")
        for file_info in kept_files:
            f.write(f"  {file_info['folder']}/{file_info['cha_file']}\n")
        
        f.write(f"\n削除されたファイル:\n")
        for file_info in removed_files:
            f.write(f"  {file_info['folder']}/{file_info['cha_file']}\n")
            if file_info.get('audio_deleted'):
                f.write(f"    - 削除: {file_info['folder']}/{file_info['audio_file']}\n")
            if file_info.get('features_deleted'):
                f.write(f"    - 削除: {file_info['folder']}/{file_info['features_file']}\n")
    
    print(f"\n結果を {output_file} に保存しました。")

if __name__ == "__main__":
    main() 
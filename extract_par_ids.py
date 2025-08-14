import os
import re
from collections import defaultdict

def extract_number_from_filename(filename):
    """ファイル名から番号を抽出する（例：714-0_silence_combined.mp3 → 714-0）"""
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
        print(f"エラー: {cha_file_path} を読み込めませんでした: {e}")
        return None

def main():
    base_path = "modules/dataset/diagnosis/train"
    silence_audio_path = os.path.join(base_path, "silence_audio")
    segmentation_path = os.path.join(base_path, "segmentation")
    
    # 処理するフォルダ（ad, cn）
    folders = ['ad', 'cn']
    
    # 番号とPARのIDのマッピング
    number_to_par_id = {}
    # 重複チェック用
    par_id_to_files = defaultdict(list)
    
    for folder in folders:
        print(f"\n=== {folder.upper()} フォルダの処理 ===")
        
        silence_folder = os.path.join(silence_audio_path, folder)
        cha_folder = os.path.join(segmentation_path, folder)
        
        if not os.path.exists(silence_folder) or not os.path.exists(cha_folder):
            print(f"フォルダが存在しません: {silence_folder} または {cha_folder}")
            continue
        
        # silence_audioフォルダのファイルを処理
        silence_files = [f for f in os.listdir(silence_folder) if f.endswith('.mp3')]
        
        for silence_file in silence_files:
            number = extract_number_from_filename(silence_file)
            if number:
                # 対応するCHAファイルを探す
                cha_file = f"{number}.cha"
                cha_file_path = os.path.join(cha_folder, cha_file)
                
                if os.path.exists(cha_file_path):
                    par_id = extract_par_id_from_cha_file(cha_file_path)
                    if par_id:
                        number_to_par_id[number] = par_id
                        par_id_to_files[par_id].append(f"{folder}/{cha_file}")
                        print(f"{number} → PAR ID: {par_id}")
                    else:
                        print(f"{number} → PAR IDが見つかりません")
                else:
                    print(f"{number} → 対応するCHAファイルが存在しません: {cha_file}")
            else:
                print(f"番号を抽出できませんでした: {silence_file}")
    
    # 重複チェック
    print("\n=== 重複チェック結果 ===")
    duplicates_found = False
    
    for par_id, files in par_id_to_files.items():
        if len(files) > 1:
            duplicates_found = True
            print(f"\n重複するPAR ID: {par_id}")
            print("ファイル:")
            for file_path in files:
                print(f"  - {file_path}")
    
    if not duplicates_found:
        print("重複するPAR IDは見つかりませんでした。")
    
    # 統計情報
    print(f"\n=== 統計情報 ===")
    print(f"処理したファイル数: {len(number_to_par_id)}")
    print(f"ユニークなPAR ID数: {len(par_id_to_files)}")
    
    # 結果をファイルに保存
    output_file = "par_id_extraction_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== PAR ID抽出結果 ===\n\n")
        for number, par_id in sorted(number_to_par_id.items()):
            f.write(f"{number}: {par_id}\n")
        
        f.write(f"\n=== 重複チェック結果 ===\n")
        for par_id, files in par_id_to_files.items():
            if len(files) > 1:
                f.write(f"\n重複するPAR ID: {par_id}\n")
                f.write("ファイル:\n")
                for file_path in files:
                    f.write(f"  - {file_path}\n")
    
    print(f"\n結果を {output_file} に保存しました。")

if __name__ == "__main__":
    main() 
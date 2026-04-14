import pandas as pd
import os

# 1. KHAI BÁO DANH SÁCH BỆNH HIẾM
rare_diseases = ['uvm', 'chol', 'acc', 'dlbc', 'meso', 'ov', 'paad', 'pcpg', 'tgct', 'ucs', 'thym', 'prad', 'thca']

for disease in rare_diseases:
    csv_path = f"F:\TEMP/{disease}/CSV/tcga_{disease}_survival.csv"
    out_dir = f"F:\TEMP/{disease}/CSV"
    
    try:
        # Đọc file survival gốc
        df = pd.read_csv(csv_path)
        
        # Lấy toàn bộ ID bệnh nhân ở cột thứ 3 (index = 2) và bỏ các dòng trống
        all_ids = df.iloc[:, 2].dropna().tolist()
        
        # Lọc bỏ các ID trùng lặp
        all_ids = list(dict.fromkeys(all_ids))
        total_count = len(all_ids)
        
        if total_count < 1:
            print(f"[CẢNH BÁO] Bệnh {disease.upper()} không có bệnh nhân nào!")
            continue

        # --- BẮT ĐẦU CHIẾN LƯỢC ĐÁNH LỪA (ZERO-SHOT) ---
        # BƯỚC 1: Đưa toàn bộ 100% bệnh nhân vào tập Test
        test_ids = all_ids
        
        # BƯỚC 2: Bốc đại bệnh nhân đầu tiên làm "hình nhân thế mạng" cho Train và Val
        dummy_id = all_ids[0]
        train_ids = [dummy_id]
        val_ids = [dummy_id]
        
        # Sử dụng pd.Series để ghép 3 cột khác chiều dài
        # Pandas tự động điền NaN vào các ô bị thiếu ở cột Train và Val
        split_df = pd.DataFrame({
            'train': pd.Series(train_ids),
            'val': pd.Series(val_ids),
            'test': pd.Series(test_ids)
        })

        # Lưu thành file splits_0.csv
        save_path = os.path.join(out_dir, 'splits_0.csv')
        split_df.to_csv(save_path, index=False)
        
        # In log thống kê
        print(f"[THÀNH CÔNG] {disease.upper():<6} | Tổng: {total_count:<4} ca -> Test (100%): {len(test_ids):<3} | Train (Dummy): {len(train_ids):<3} | Val (Dummy): {len(val_ids):<3}")
            
    except FileNotFoundError:
        print(f"[LỖI] Không tìm thấy file gốc: {csv_path}")
    except Exception as e:
        print(f"[LỖI] Xảy ra sự cố ở bệnh {disease.upper()}: {e}")
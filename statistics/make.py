import pandas as pd
import os
from sklearn.model_selection import train_test_split

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
        
        if total_count < 10:
            print(f"[CẢNH BÁO] Bệnh {disease.upper()} có quá ít ID ({total_count} ca). Không đủ để chia!")
            continue

        # --- BẮT ĐẦU CHIA TỈ LỆ ---
        # BƯỚC 1: Cắt 20% ra làm Test, 80% còn lại là Train+Val
        # random_state=42 giúp mỗi lần chạy code sẽ ra đúng 1 kết quả (không bị xáo trộn lung tung)
        train_val_ids, test_ids = train_test_split(all_ids, test_size=0.20, random_state=42)
        
        # BƯỚC 2: Từ 80% (Train+Val) đó, cắt tiếp 25% làm Val (Tương đương 20% của tổng ban đầu)
        # Suy ra Train còn lại sẽ là 60% tổng số.
        train_ids, val_ids = train_test_split(train_val_ids, test_size=0.25, random_state=42)
        
        # Sử dụng pd.Series để ghép 3 cột khác chiều dài
        # Pandas tự động điền NaN vào các ô bị thiếu ở cột Val và Test
        split_df = pd.DataFrame({
            'train': pd.Series(train_ids),
            'val': pd.Series(val_ids),
            'test': pd.Series(test_ids)
        })

        # Lưu thành file splits_0.csv
        save_path = os.path.join(out_dir, 'splits_0.csv')
        split_df.to_csv(save_path, index=False)
        
        # In log thống kê cực kỳ rõ ràng
        print(f"[THÀNH CÔNG] {disease.upper():<6} | Tổng: {total_count:<4} ca -> Train: {len(train_ids):<3} | Val: {len(val_ids):<3} | Test: {len(test_ids):<3}")
            
    except FileNotFoundError:
        print(f"[LỖI] Không tìm thấy file gốc: {csv_path}")
    except Exception as e:
        print(f"[LỖI] Xảy ra sự cố ở bệnh {disease.upper()}: {e}")
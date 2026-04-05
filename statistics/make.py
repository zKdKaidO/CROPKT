import pandas as pd
import os

# 1. KHAI BÁO DANH SÁCH BỆNH HIẾM
rare_diseases = ['uvm', 'chol', 'acc', 'dlbc', 'meso', 'ov', 'paad', 'pcpg', 'tgct', 'ucs', 'thym', 'prad', 'thca']

for disease in rare_diseases:
    csv_path = f"A:/CROPKT_DATASET/{disease}/CSV/tcga_{disease}_survival.csv"
    out_dir = f"A:/CROPKT_DATASET/{disease}/CSV"
    
    try:
        # Đọc file survival gốc
        df = pd.read_csv(csv_path)
        
        # Lấy toàn bộ ID bệnh nhân ở cột thứ 3 (index = 2) và bỏ các dòng trống
        all_ids = df.iloc[:, 2].dropna().tolist()
        
        # Lọc bỏ các ID trùng lặp ngay từ đầu (giúp terminal của bạn bớt in ra log cảnh báo)
        all_ids = list(dict.fromkeys(all_ids))
        
        if len(all_ids) == 0:
            print(f"[CẢNH BÁO] Không tìm thấy ID nào cho bệnh {disease}.")
            continue

        # Tập test: Chứa TOÀN BỘ bệnh nhân duy nhất
        test_ids = all_ids
        
        # Tập train: Chỉ lấy 1 ID đầu tiên làm "chim mồi"
        train_ids = [all_ids[0]]
        
        # Sử dụng pd.Series để ghép 2 cột khác chiều dài
        # Pandas tự động điền NaN vào phần thiếu của cột train
        split_df = pd.DataFrame({
            'train': pd.Series(train_ids),
            'test': pd.Series(test_ids)
        })

        # Chỉ lưu thành 1 file splits_0.csv duy nhất
        save_path = os.path.join(out_dir, 'splits_0.csv')
        split_df.to_csv(save_path, index=False)
        
        print(f"[THÀNH CÔNG] Đã tạo 1-Split cho {disease.upper()} | Test: {len(test_ids)} ca, Train mồi: 1 ca.")
            
    except FileNotFoundError:
        print(f"[LỖI] Không tìm thấy file gốc: {csv_path}")
    except Exception as e:
        print(f"[LỖI] Xảy ra sự cố ở bệnh {disease}: {e}")
import h5py
import numpy as np
import os

# Thay bằng đường dẫn đến 1 file .h5 bất kỳ của bạn
file_path = r"F:\TEMP\ACC\TCGA_ACC\TCGA-OR-A5J1-01Z-00-DX1.600C7D8C-F04C-4125-AF14-B1E76DC01A1E.h5"

file_name = os.path.basename(file_path)
print(f"=== BẮT ĐẦU NỘI SOI FILE: {file_name} ===\n")

try:
    with h5py.File(file_path, 'r') as h5f:
        # 1. KIỂM TRA METADATA (THUỘC TÍNH CHUNG CỦA FILE)
        print("1. CÁC THUỘC TÍNH CHUNG (ATTRIBUTES):")
        if len(h5f.attrs) == 0:
            print("   -> Không có thuộc tính chung nào được lưu.")
        else:
            for key, val in h5f.attrs.items():
                print(f"   - {key}: {val}")
        print("-" * 50)

        # 2. HÀM QUÉT TOÀN BỘ CẤU TRÚC BÊN TRONG
        print("2. CẤU TRÚC LƯU TRỮ BÊN TRONG (DATASETS & GROUPS):")
        
        def print_h5_structure(name, obj):
            # Nếu nó là Thư mục
            if isinstance(obj, h5py.Group):
                print(f"📁 Thư mục (Group) : /{name}")
            
            # Nếu nó là Dữ liệu thật (Dataset)
            elif isinstance(obj, h5py.Dataset):
                print(f"📊 Dữ liệu (Dataset): /{name}")
                print(f"   + Kích thước (Shape) : {obj.shape}")
                print(f"   + Kiểu dữ liệu (Type): {obj.dtype}")
                
                # In thử 3 giá trị đầu tiên để xem nó hình thù ra sao
                if len(obj.shape) >= 2: # Thường là (N, 1536) hoặc (N, 2)
                    sample = obj[0, :3] # Lấy patch đầu tiên, in 3 số đầu
                    print(f"   + Giá trị mẫu (Patch đầu, 3 số đầu): {sample} ...")
                elif len(obj.shape) == 1:
                    sample = obj[:3]
                    print(f"   + Giá trị mẫu: {sample} ...")
                print("   .")

        # Lệnh quét toàn bộ file
        h5f.visititems(print_h5_structure)
        print("-" * 50)
        
        # 3. MÔ PHỎNG LẠI CHO DỄ HIỂU
        print("3. TỔNG KẾT:")
        if 'features' in h5f:
            feats = h5f['features']
            n_patches = feats.shape[0] if feats.ndim == 2 else feats.shape[1]
            print(f"   -> Tiêu bản WSI này được cắt thành: {n_patches} patches.")
            print(f"   -> Mỗi patch được miêu tả bởi 1 vector gồm {feats.shape[-1]} chiều.")
            
        if 'coords' in h5f:
            print("   -> FILE NÀY CÓ LƯU TỌA ĐỘ (COORDS). Rất tốt để vẽ lại Heatmap!")

except Exception as e:
    print(f"Lỗi khi mở file: {e}")

"""
(env_cropkt) PS A:\CROPKT> python ./statistics/h5_read.py
=== BẮT ĐẦU NỘI SOI FILE: TCGA-OR-A5J1-01Z-00-DX1.600C7D8C-F04C-4125-AF14-B1E76DC01A1E.h5 ===

1. CÁC THUỘC TÍNH CHUNG (ATTRIBUTES):
   -> Không có thuộc tính chung nào được lưu.
--------------------------------------------------
2. CẤU TRÚC LƯU TRỮ BÊN TRONG (DATASETS & GROUPS):
📊 Dữ liệu (Dataset): /annots
   + Kích thước (Shape) : (1, 6351, 1)
   + Kiểu dữ liệu (Type): int64
   + Giá trị mẫu (Patch đầu, 3 số đầu): [[0]
 [0]
 [0]] ...
   .
📊 Dữ liệu (Dataset): /coords
   + Kích thước (Shape) : (1, 6351, 2)
   + Kiểu dữ liệu (Type): int64
   + Giá trị mẫu (Patch đầu, 3 số đầu): [[11776 40448]
 [11776 40960]
 [11776 41472]] ...
   .
📊 Dữ liệu (Dataset): /coords_patching
   + Kích thước (Shape) : (6351, 2)
   + Kiểu dữ liệu (Type): int64
   + Giá trị mẫu (Patch đầu, 3 số đầu): [11776 40448] ...
   .
📊 Dữ liệu (Dataset): /features
   + Kích thước (Shape) : (1, 6351, 1536)
   + Kiểu dữ liệu (Type): float32
   + Giá trị mẫu (Patch đầu, 3 số đầu): [[ 0.00146503 -0.09548201  0.7696172  ... -0.09871367 -0.02890113
   0.24746369]
 [-0.13309601 -0.01669721 -0.0555785  ... -0.14957611 -0.09514242
   0.61981946]
 [-0.7530204  -0.1751506   0.05452079 ... -0.10374744 -0.56208575
   0.42373383]] ...
   .
--------------------------------------------------
3. TỔNG KẾT:
   -> Tiêu bản WSI này được cắt thành: 6351 patches.
   -> Mỗi patch được miêu tả bởi 1 vector gồm 1536 chiều.
   -> FILE NÀY CÓ LƯU TỌA ĐỘ (COORDS). Rất tốt để vẽ lại Heatmap!
(env_cropkt) PS A:\CROPKT>
"""
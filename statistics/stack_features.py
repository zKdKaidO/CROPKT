# Dùng để gom tất cả những fold 0 của tất cả bệnh nguồn tfl cho 1 bệnh đích
import os
import torch

base_path = r"A:\CROPKT\result\transfer_features_rare_turmor"

target_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith("target_")]

print(f"🚀 TÌM THẤY {len(target_dirs)} BỆNH ĐÍCH: {target_dirs}\n")
print("-" * 60)

for target_dir_name in target_dirs:
    target_path = os.path.join(base_path, target_dir_name)
    print(f"[{target_dir_name.upper()}] Đang xử lý...")
    
    output_dir = os.path.join(target_path, "source_all-fold_0")
    os.makedirs(output_dir, exist_ok=True)
    
    # BƯỚC QUAN TRỌNG NHẤT: Chỉ lấy các thư mục kết thúc bằng "-fold_0"
    source_dirs = [d for d in os.listdir(target_path) 
                   if os.path.isdir(os.path.join(target_path, d)) 
                   and d.startswith("source_") 
                   and d.endswith("-fold_0") 
                   and d != "source_all-fold_0"]
    
    if not source_dirs:
        print("  -> [Bỏ qua] Không có chuyên gia fold_0 nào trong này.")
        print("-" * 60)
        continue
        
    print(f"  -> Tìm thấy {len(source_dirs)} chuyên gia fold_0: {source_dirs}")
    
    reference_source_path = os.path.join(target_path, source_dirs[0])
    patient_files = [f for f in os.listdir(reference_source_path) if f.endswith('.pt')]
    
    if not patient_files:
        print("  -> [Bỏ qua] Không có file .pt nào bên trong.")
        print("-" * 60)
        continue
        
    print(f"  -> Bắt đầu gom gói {len(patient_files)} bệnh nhân...")
    
    success_count = 0
    for pt_file in patient_files:
        patient_features = []
        is_valid = True
        
        for source_dir_name in source_dirs:
            file_path = os.path.join(target_path, source_dir_name, pt_file)
            
            if os.path.exists(file_path):
                feat = torch.load(file_path, map_location=torch.device('cpu'))
                patient_features.append(feat)
            else:
                print(f"    [Cảnh báo] Bệnh nhân {pt_file} thiếu kết quả từ chuyên gia {source_dir_name}")
                is_valid = False
                break 
                
        if is_valid and len(patient_features) > 0:
            stacked_tensor = torch.stack(patient_features, dim=0)
            save_path = os.path.join(output_dir, pt_file)
            torch.save(stacked_tensor, save_path)
            success_count += 1
            
    print(f"  -> ✅ Xong! Đã lưu thành công {success_count}/{len(patient_files)} file vào thư mục source_all-fold_0")
    print("-" * 60)

print("🎉 HOÀN THÀNH TOÀN BỘ QUÁ TRÌNH GOM GÓI CHO FOLD 0!")
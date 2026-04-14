import os
import shutil

# 1. Đường dẫn thư mục gốc
source_experts_dir = r"A:\CROPKT\saved_experts"
# Đường dẫn thư mục mới (bạn có thể đổi tên tùy ý)
target_experts_dir = r"A:\CROPKT\saved_experts_fold0"

print("🚀 BẮT ĐẦU QUÁ TRÌNH LỌC VÀ SAO CHÉP EXPERT FOLD_0...")
print("-" * 60)

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(target_experts_dir, exist_ok=True)

# 2. Quét tất cả các thư mục bên trong saved_experts
all_dirs = [d for d in os.listdir(source_experts_dir) if os.path.isdir(os.path.join(source_experts_dir, d))]

fold_0_dirs = [d for d in all_dirs if d.endswith("-fold_0")]

if not fold_0_dirs:
    print("❌ LỖI: Không tìm thấy bất kỳ thư mục fold_0 nào trong saved_experts!")
    exit()

print(f"✅ Tìm thấy {len(fold_0_dirs)} thư mục chứa chuyên gia Fold 0.")

count_success = 0

# 3. Tiến hành copy file
for expert_dir_name in fold_0_dirs:
    # Ví dụ: A:\CROPKT\saved_experts\blca_expert-data_tcga_blca-fold_0
    source_path = os.path.join(source_experts_dir, expert_dir_name)
    
    # Tìm file .pth trong thư mục này
    pth_files = [f for f in os.listdir(source_path) if f.endswith(".pth")]
    
    if not pth_files:
        print(f"  [Bỏ qua] Không tìm thấy file .pth nào trong: {expert_dir_name}")
        continue
    
    # Giả sử mỗi thư mục chỉ có 1 file .pth (ví dụ: train_model-last.pth)
    pth_filename = pth_files[0] 
    source_file_path = os.path.join(source_path, pth_filename)
    
    # 4. Tạo cấu trúc thư mục mới để model CROPKT nhận diện được
    # Model CROPKT cần cấu trúc: saved_experts_fold0_only / [Tên bệnh] / [File .pth]
    target_expert_subdir = os.path.join(target_experts_dir, expert_dir_name)
    os.makedirs(target_expert_subdir, exist_ok=True)
    
    target_file_path = os.path.join(target_expert_subdir, pth_filename)
    
    # Thực hiện copy đè (nếu đã có file)
    shutil.copy2(source_file_path, target_file_path)
    print(f"  -> Đã copy: {expert_dir_name}/{pth_filename}")
    count_success += 1

print("-" * 60)
print(f"🎉 HOÀN THÀNH! Đã sao chép thành công {count_success}/{len(fold_0_dirs)} chuyên gia.")
print(f"📁 Thư mục mới của bạn nằm tại: {target_experts_dir}")
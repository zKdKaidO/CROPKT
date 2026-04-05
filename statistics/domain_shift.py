import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

class DomainShiftAnalyzer:
    def __init__(self, base_dir, diseases):
        self.base_dir = base_dir
        self.diseases = diseases
        
        # Biến lưu trữ dữ liệu trên RAM sau khi nạp
        self.X = None 
        self.y = None

    def load_data(self):
        """Hàm dùng để quét file .h5 hoặc nạp siêu tốc từ CACHE"""
        all_features = []
        all_labels = []
        
        for disease in self.diseases:
            print(f"\n--- Đang xử lý bệnh: {disease.upper()} ---")
            cache_dir = os.path.join(self.base_dir, disease, "CACHE")
            cache_path = os.path.join(cache_dir, "median_features.npy")
            
            # 1. Đọc từ Cache nếu có
            if os.path.exists(cache_path):
                print(f"-> Tìm thấy cache tại {cache_path}. Đang nạp...")
                disease_feats_arr = np.load(cache_path)
                all_features.extend(disease_feats_arr)
                all_labels.extend([disease.upper()] * len(disease_feats_arr))
                print(f"-> Đã nạp xong {len(disease_feats_arr)} tiêu bản từ cache!")
                continue

            # 2. Nếu chưa có Cache thì đọc từ .h5
            print("-> Chưa có cache. Bắt đầu trích xuất từ các file .h5 gốc...")
            csv_path = os.path.join(self.base_dir, disease, "CSV", f"tcga_{disease}_survival.csv")
            h5_dir = os.path.join(self.base_dir, disease, f"TCGA_{disease}")
            disease_features_local = [] 

            try:
                df = pd.read_csv(csv_path)
                pathology_ids = df.iloc[:, 1].dropna().tolist()
                found_count = 0
                
                for pathology_id in pathology_ids:
                    h5_name = f"{pathology_id}.h5" if not str(pathology_id).endswith('.h5') else pathology_id
                    h5_path = os.path.join(h5_dir, h5_name)

                    if os.path.exists(h5_path):
                        try:
                            with h5py.File(h5_path, 'r') as h5f:
                                key = 'features' if 'features' in h5f else list(h5f.keys())[0]
                                feats = h5f[key][:]
                                
                                if feats.ndim == 3:
                                    feats = np.squeeze(feats, axis=0)
                                elif feats.ndim == 1:
                                    feats = feats.reshape(1, -1)
                                    
                                median_feat = np.median(feats, axis=0)
                                disease_features_local.append(median_feat)
                                found_count += 1
                        except Exception as e_file:
                            print(f"[CẢNH BÁO] Bỏ qua file {h5_name} do lỗi: {e_file}")
                            
                print(f"-> Đã trích xuất thành công: {found_count}/{len(pathology_ids)} files")
                
                # Lưu Cache
                if disease_features_local:
                    os.makedirs(cache_dir, exist_ok=True)
                    disease_feats_arr = np.array(disease_features_local)
                    np.save(cache_path, disease_feats_arr)
                    print(f"-> Đã lưu cache thành công vào {cache_path}")
                    
                    all_features.extend(disease_features_local)
                    all_labels.extend([disease.upper()] * len(disease_features_local))
                            
            except Exception as e:
                print(f"[LỖI CSV] Không thể xử lý bệnh {disease}: {e}") 

        # Đưa vào thuộc tính của Class
        self.X = np.array(all_features)
        self.y = np.array(all_labels)

        if self.X.shape[0] == 0:
            print("\n[LỖI NGHIÊM TRỌNG] Không thu thập được dữ liệu nào!")
            sys.exit(1)
        else:
            print(f"\n[HOÀN TẤT NẠP DỮ LIỆU] Tổng số tiêu bản: {self.X.shape[0]}")
            print(f"Kích thước ma trận đặc trưng: {self.X.shape}")

    def visualize_tsne(self):
        """Phương pháp 1: Vẽ biểu đồ t-SNE"""
        print("\nĐang chạy thuật toán t-SNE (có thể mất vài phút)...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(self.X)

        print("Đang vẽ biểu đồ...")
        plt.figure(figsize=(12, 9)) 
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=self.y, palette='tab10', s=60, alpha=0.8)
        plt.title("Trực quan hóa Domain Shift trên Không gian Đặc trưng (t-SNE)", fontsize=15, fontweight='bold')
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend(title="Domain (Loại bệnh)", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)

        save_img_path = "domain_shift_tsne.png"
        plt.savefig(save_img_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ tại: {save_img_path}")
        plt.show()

    def calculate_auc(self, source_disease):
        """Phương pháp 2: Đo lường Domain Shift bằng điểm số AUC"""
        source_disease = source_disease.upper()
        if source_disease not in self.y:
            print(f"Lỗi: Source '{source_disease}' không có trong dữ liệu đã nạp.")
            return

        print(f"\n=== TÍNH TOÁN DOMAIN SHIFT AUC SO VỚI SOURCE: {source_disease} ===")
        
        # Lấy dữ liệu của Source
        X_source = self.X[self.y == source_disease]
        y_source_clf = np.zeros(X_source.shape[0])
        
        results = []
        target_diseases = np.unique(self.y)
        
        for target in target_diseases:
            if target == source_disease:
                continue
                
            X_target = self.X[self.y == target]
            y_target_clf = np.ones(X_target.shape[0])
            
            X_combined = np.vstack((X_source, X_target))
            y_combined = np.hstack((y_source_clf, y_target_clf))
            
            clf = LogisticRegression(max_iter=1000, random_state=42)
            auc_scores = cross_val_score(clf, X_combined, y_combined, cv=5, scoring='roc_auc')
            
            results.append({'Target': target, 'AUC Score': np.mean(auc_scores)})
            
        df_results = pd.DataFrame(results).sort_values(by='AUC Score', ascending=True).reset_index(drop=True)
        print("Điểm số: 0.5 (Không lệch) -> 1.0 (Lệch hoàn toàn)")
        print(df_results.to_string(index=False))


# ---------------------------------------------------------
# XỬ LÝ LỆNH TỪ TERMINAL
# ---------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR]: Thiếu tham số phương thức cần chạy.")
        print("Cách dùng: python domain_shift.py [mode]")
        print("Các mode hỗ trợ:")
        print("  cache : Chỉ nạp dữ liệu và tạo cache")
        print("  tsne  : Vẽ biểu đồ t-SNE")
        print("  auc   : Tính điểm lệch pha AUC (Mặc định Source là BLCA)")
        sys.exit(1)

    mode = sys.argv[1].lower()
    
    # Khởi tạo class (Truyền danh sách bệnh và base_dir vào đây)
    base_dir = r"F:\TEMP" 
    # 'uvm', 'chol', 'acc', 'dlbc', 'meso', 'ov', 'paad', 'pcpg', 'tgct', 'ucs', 'thym', 'prad', 'thca', 
    # 'gbmlgg', 'hnsc', 'kipan', 'lihc', 'lung', 'sarc', 'skcm', 'stes', 'ucec', 'blca', 'brca', 'cesc', 'coadread'
    diseases_list = [
        'uvm', 'chol', 'acc', 'dlbc', 'meso', 'ov', 'paad', 'pcpg', 'tgct', 'ucs', 'thym', 'prad', 'thca',
        'blca'
    ]
    
    analyzer = DomainShiftAnalyzer(base_dir, diseases_list)

    # Bước bắt buộc: Nạp dữ liệu vào RAM
    analyzer.load_data()

    # Chạy các phương pháp tương ứng với terminal
    if mode == 'cache':
        print("\n[HOÀN TẤT] Dữ liệu đã được nạp/tạo cache thành công. Đã thoát chương trình.")
    elif mode == 'tsne':
        analyzer.visualize_tsne()
    elif mode == 'auc':
        # Bạn có thể thay đổi 'BLCA' thành nguồn khác nếu muốn
        analyzer.calculate_auc(source_disease='blca') 
    else:
        print(f"Lỗi: Không nhận diện được mode '{mode}'.")
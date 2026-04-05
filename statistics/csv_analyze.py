import os
import pandas as pd

base_dir = r"F:\TEMP"
diseases_list = [
    'uvm', 'chol', 'acc', 'dlbc', 'meso', 'ov', 'paad', 'pcpg', 'tgct', 'ucs', 'thym', 'prad', 'thca',
    'gbmlgg', 'hnsc', 'kipan', 'lihc', 'lung', 'sarc', 'skcm', 'stes', 'ucec', 'blca', 'brca', 'cesc', 'coadread'
]

class CSVAnalyze:
    def __init__(self, base_dir, diseases):
        self.base_dir = base_dir
        self.diseases = diseases
        self.statistics = [] # List lưu trữ dữ liệu để tạo bảng đẹp

    def load_and_analyze(self):
        print(f"[STATISTIC]: Đang nạp và phân tích {len(self.diseases)} file CSV...\n")
        
        for disease in self.diseases:
            # [SỬA LỖI 1]: Đã thêm đuôi .csv
            csv_path = os.path.join(self.base_dir, disease, "CSV", f"tcga_{disease}_survival.csv")
            
            if not os.path.exists(csv_path):
                print(f"[CẢNH BÁO] Không tìm thấy: {csv_path}")
                continue
                
            try:
                df = pd.read_csv(csv_path)
                
                # 1. Thống kê Số lượng (WSI vs Bệnh nhân)
                total_wsi = df.shape[0] # [SỬA LỖI 2]: Dùng ngoặc vuông
                # Kiểm tra xem có cột patient_id không, nếu có thì đếm số bệnh nhân độc nhất
                total_patients = df['patient_id'].nunique() if 'patient_id' in df.columns else total_wsi
                
                # 2. Thống kê Censored (Cột e)
                censored = (df['e'] == 0).sum()
                events = (df['e'] == 1).sum()
                ratio = (censored / total_wsi) * 100
                
                # 3. Thống kê Thời gian (Cột t) - Bỏ qua các giá trị NaN nếu có
                t_median = df['t'].median()
                t_max = df['t'].max()
                t_min = df['t'].min()
                
                # 4. Kiểm tra Dữ liệu rác (Missing values)
                missing_e = df['e'].isnull().sum()
                missing_t = df['t'].isnull().sum()
                
                # Đóng gói kết quả vào Dictionary
                self.statistics.append({
                    'Bệnh': disease.upper(),
                    'Số WSI': total_wsi,
                    'Bệnh nhân': total_patients,
                    'Sự kiện (Chết)': events,
                    'Censored': censored, # e = 0
                    'Tỷ lệ Ẩn (%)': round(ratio, 1),
                    'Median Time (Ngày)': round(t_median, 1),
                    'Max Time (Ngày)': round(t_max, 1),
                    'Lỗi NaN (e/t)': f"{missing_e}/{missing_t}"
                })
                
            except Exception as e:
                print(f"[LỖI] Xảy ra sự cố khi đọc file {disease.upper()}: {e}")

        self.display_report()

    def display_report(self):
        """Hàm in ra bảng báo cáo tổng thể bằng Pandas DataFrame"""
        if not self.statistics:
            print("\nKhông có dữ liệu để hiển thị.")
            return
            
        df_report = pd.DataFrame(self.statistics)
        
        # Có thể sort theo một cột nào đó cho dễ nhìn, ví dụ sort theo Tỷ lệ Ẩn
        df_report = df_report.sort_values(by='Tỷ lệ Ẩn (%)', ascending=False).reset_index(drop=True)
        
        print("="*105)
        print(f"{'BẢNG THỐNG KÊ TOÀN DIỆN DỮ LIỆU SINH TỒN TCGA':^105}")
        print("="*105)
        # to_string() in ra bảng rất đẹp trên terminal
        print(df_report.to_string(index=False))
        print("="*105)

# Khởi chạy
analyzer = CSVAnalyze(base_dir, diseases_list)
analyzer.load_and_analyze()
import torch
import torch.nn as nn

# ==========================================
# 1. BỘ LỌC ĐẢO NGƯỢC (Gradient Reversal Layer - GRL)
# Đóng vai trò là "Kẻ lừa đảo" che mắt Cảnh sát
# ==========================================
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # Khi luồng dữ liệu đi tiến (chuẩn đoán): Giữ nguyên mọi thứ, không làm gì cả
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Khi luồng dữ liệu đi lùi (học tập/cập nhật): Cố tình NHÂN ÂM (-alpha)
        # Việc này ép các layer phía trước phải học cách tạo ra đặc trưng 
        # làm cho ông Cảnh sát phía sau đoán SAI đi.
        output = grad_output.neg() * ctx.alpha
        return output, None

# ==========================================
# 2. ÔNG CẢNH SÁT (Domain Discriminator)
# ==========================================
class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(DomainDiscriminator, self).__init__()
        
        # Mạng Neural đơn giản (MLP) để phân loại 2 miền (Source vs Target)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1) # Output ra 1 số duy nhất
        )

    def forward(self, x, alpha=1.0):
        # Bước 1: Chạy qua bộ lọc lừa đảo (GRL)
        x_grl = GradientReversalFunction.apply(x, alpha)
        
        # Bước 2: Đưa vào mạng nơ-ron để Cảnh sát phân tích
        domain_preds = self.net(x_grl)
        
        return domain_preds
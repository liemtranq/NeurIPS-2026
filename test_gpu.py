import torch

print("-" * 30)
if torch.cuda.is_available():
    print("✅ Chúc mừng Liêm! ROCm đã nhận diện GPU thành công.")
    print(f"🔥 Tên GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Số lượng GPU: {torch.cuda.device_count()}")
    
    # Test thử một phép tính nhỏ trên MI300X
    x = torch.randn(1000, 1000).to("cuda")
    y = torch.mm(x, x.t())
    print("🚀 Phép tính ma trận trên MI300X: OK!")
else:
    print("❌ Lỗi rồi! PyTorch chưa nhận được GPU AMD.")
print("-" * 30)
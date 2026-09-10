import torch
import torch.nn as nn
import torch_npu  # noqa: F401  导入 torch_npu 后，torch.npu 命名空间才可用（CPU 机器上导入也不报错）


class RMSNorm(nn.Module):
    """
    RMSNorm（Root Mean Square Layer Normalization）
    与 LayerNorm 的区别：
      - LayerNorm 先减均值、再除以标准差，并带 bias
      - RMSNorm 只用均方根 (RMS) 归一化，不做中心化，也没有 bias
    公式: y = x / sqrt(mean(x^2) + eps) * weight
    优点：计算更少、精度几乎不损失，所以被 LLaMA / minimind 等模型采用
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        # 数值稳定项：避免除零（当 x 全为 0 时 mean(x^2)=0，若无 eps 会除以 0）
        self.eps = eps
        # 可学习的缩放参数 gamma，形状为 [dim]，初始化为全 1（初始时不改变数值）
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x: torch.Tensor):
        # x.pow(2).mean(-1, keepdim=True)：沿最后一维计算均方值，形状 [..., 1]，
        #   keepdim=True 保持维度以便广播回原形状
        # torch.rsqrt(...)：1/sqrt(v)，一步算出倒数平方根
        # x * rsqrt：把每个元素缩放到 RMS 为 1
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 1. x.float()：先转成 float32 计算，避免在 bf16/fp16 下平方求和溢出/精度损失
        # 2. self.weight * self.norm(...)：乘可学习参数 gamma（广播到最后一维）
        # 3. .type_as(x)：转回输入原来的 dtype，保证和其他模块 dtype 一致
        return (self.weight * self.norm(x.float())).type_as(x)


# ============================================================
# 测试用例：tensor 形状 [2, 3, 5]
# 含义可理解为：batch=2, seq_len=3, hidden_dim=5
# RMSNorm 沿最后一维（dim=5，即 hidden 维度）归一化
# ============================================================
def test_rms_norm(device):
    print(f"\n===== 在设备 {device} 上测试 RMSNorm =====")

    torch.manual_seed(42)  # 固定随机种子，保证每次运行结果可复现

    # 1. 构造输入 [2, 3, 5]
    x = torch.randn(2, 3, 5, device=device)
    print("输入 x 形状:", tuple(x.shape), ", dtype:", x.dtype)
    print("输入 x:")
    print(x)

    # 2. 创建 RMSNorm 模块并搬到目标设备（dim 必须等于最后一维大小 5）
    rms = RMSNorm(dim=5).to(device)
    print("\nweight (gamma) 初始值:", rms.weight.data)

    # 3. 前向计算
    y = rms(x)
    print("\n输出 y 形状:", tuple(y.shape), ", dtype:", y.dtype)
    print("输出 y:")
    print(y)

    # 4. 手动验证：用公式 y = x / sqrt(mean(x^2)+eps) * weight 逐步复算一遍
    #    由于 gamma 初始为全 1，结果应与 y 一致，用来确认实现正确
    x32 = x.float()
    rms_val = torch.sqrt(x32.pow(2).mean(-1, keepdim=True) + rms.eps)  # 均方根
    y_manual = (x32 / rms_val) * rms.weight.float()
    # 输出按输入 dtype 对齐后再比较，消除 float 计算的微小误差
    max_diff = (y.float() - y_manual.to(x.dtype).float()).abs().max().item()
    print(f"\n手动公式复算与模块输出的最大误差: {max_diff:.3e}")
    assert max_diff < 1e-5, "RMSNorm 输出与手动计算不一致！"
    print("✅ 数值校验通过")

    # 5. 检查归一化效果：归一化后（weight=1）每一行（最后一维）的均方值应接近 1
    #    即 RMS(y) ≈ 1
    row_rms = y.float().pow(2).mean(-1)  # 形状 [2, 3]，每个 (batch, seq) 位置的均方值
    print("每行均方值（应都接近 1）:")
    print(row_rms)
    assert torch.allclose(row_rms, torch.ones_like(row_rms), atol=1e-4), "归一化后 RMS 应接近 1"
    print("✅ 归一化校验通过")


if __name__ == "__main__":
    # 设备选择：优先 NPU 0（真机运行），NPU 不可用时回退 CPU
    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"NPU 可用: {torch.npu.is_available()}, 设备数量: {torch.npu.device_count()}")

    test_rms_norm(device)
    print("\n===== 所有测试完成 =====")

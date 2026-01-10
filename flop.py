import torch
import time
import numpy as np
from thop import profile
# 确保从你的模型文件中导入
from model import DualBranchDenoiser 

def benchmark_model():
    # 1. 初始化模型并移动到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBranchDenoiser(in_channels=3).to(device)
    model.eval()

    # 2. 准备输入数据 (SIDD 标准输入大小 256x256)
    input_size = (1, 3, 256, 256)
    dummy_input = torch.randn(input_size).to(device)

    # 3. 计算 Params 和 FLOPs
    # profile 会自动遍历模型层并计算浮点运算次数
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    # 4. 测量推理时间 (Inference Time)
    # GPU 测量必须进行 Warm-up 并使用同步，否则结果不准
    print("Warming up GPU...")
    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize() # 等待所有 CUDA 核函数执行完毕
    
    print("Measuring inference time...")
    repetitions = 300
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))
    
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    avg_time = np.sum(timings) / repetitions
    std_time = np.std(timings)

    # 5. 打印最终结果 (直接用于填表)
    print("\n" + "="*30)
    print("COMPUTATIONAL EFFICIENCY ANALYSIS")
    print("="*30)
    print(f"Device:         {torch.cuda.get_device_name(0)}")
    print(f"Input Size:     {input_size}")
    print(f"Parameters:     {params / 1e6:.4f} M")
    print(f"FLOPs:          {flops / 1e9:.4f} G")
    print(f"Avg Runtime:    {avg_time:.2f} ms")
    print(f"Std Deviation:  {std_time:.2f} ms")
    print(f"FPS:            {1000.0 / avg_time:.1f}")
    print("="*30)

if __name__ == "__main__":
    benchmark_model()
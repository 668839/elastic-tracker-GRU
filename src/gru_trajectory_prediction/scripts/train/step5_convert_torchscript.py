#!/usr/bin/env python3
"""
步骤5:转换为TorchScript - 用于C++部署
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from models.student_gru import StudentGRU
import argparse

def convert_to_torchscript(model, output_path, example_input):
    """
    将PyTorch模型转换为TorchScript
    """
    model.eval()
    
    # 方法1：Tracing（推荐用于无控制流的模型）
    try:
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(output_path)
        print(f"✓ Traced model saved to {output_path}")
        return traced_model
    except Exception as e:
        print(f"Tracing failed: {e}")
        print("Trying scripting...")
        
        # 方法2：Scripting（用于有控制流的模型）
        scripted_model = torch.jit.script(model)
        scripted_model.save(output_path)
        print(f"✓ Scripted model saved to {output_path}")
        return scripted_model

def test_torchscript(model_path, example_input):
    """测试TorchScript模型"""
    loaded_model = torch.jit.load(model_path)
    loaded_model.eval()
    
    with torch.no_grad():
        output = loaded_model(example_input)
    
    print(f"TorchScript output shape: {output.shape}")
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantized_path', required=True, help='Path to quantized model')
    parser.add_argument('--output_dir', default='../../models', help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cpu')
    
    # 加载量化模型
    print("Loading quantized model...")
    model = StudentGRU()
    checkpoint = torch.load(args.quantized_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 示例输入 (batch=1, seq_len=30, features=4)
    example_input = torch.randn(1, 30, 4)
    
    # 转换
    output_path = os.path.join(args.output_dir, 'best_student_quantized.pt')
    torchscript_model = convert_to_torchscript(model, output_path, example_input)
    
    # 测试
    print("\nTesting TorchScript model...")
    original_output = model(example_input)
    torchscript_output = test_torchscript(output_path, example_input)
    
    # 验证一致性
    diff = torch.abs(original_output - torchscript_output).max().item()
    print(f"Max difference: {diff:.6f}")
    
    if diff < 1e-4:
        print("✓ Conversion successful!")
    else:
        print("⚠ Large difference detected, check conversion")
    
    print(f"\n✓ Final model saved: {output_path}")
    print("This model can now be loaded in C++")

if __name__ == '__main__':
    main()
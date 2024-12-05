# Quantization in Depth  

![image](https://github.com/user-attachments/assets/5fb9bf72-58f4-455f-9630-15814725942e)


## Introduction

Quantization is a technique used in machine learning to reduce the memory and computational requirements of models by approximating high-precision weights and activations with lower-precision representations. This approach is especially critical for deploying models on edge devices with limited resources, as it improves efficiency while maintaining acceptable accuracy.

This Repo Contains, Explorations quantization in depth, covering various techniques, workflows, and implementations. Our work demonstrates how quantization can be applied effectively to PyTorch models, including using pre-trained weights from the Hugging Face Hub.

---

## Overview

### What is Quantization?
Quantization involves converting a full-precision tensor (typically 32-bit floating-point numbers) into a lower precision (e.g., 8-bit integers). This reduces the model's memory footprint and computation cost.

### Key Benefits:
- **Reduced Model Size**: Quantized models take up less memory.
- **Improved Inference Speed**: Smaller data types enable faster computation.
- **Energy Efficiency**: Essential for edge devices and mobile applications.

---

## Topics Explored

### 1. **Quantize and DeQuantize a Tensor**
- Quantization maps a tensor's values into a smaller range using a scale factor and zero point.
- Dequantization reverses this process to approximate the original values.

```python
import torch

# Quantizing and dequantizing a tensor
tensor = torch.tensor([1.0, 2.0, 3.0])
quantized_tensor = torch.quantize_per_tensor(tensor, scale=0.1, zero_point=10, dtype=torch.qint8)
dequantized_tensor = quantized_tensor.dequantize()
```
---

### 2. **Get the Scale and Zero Point**
- Scale and zero-point are essential parameters for quantization.
- Scale: Determines the step size between quantized values.
- Zero Point: Aligns the quantized values with the original range.

```python
scale = quantized_tensor.q_scale()
zero_point = quantized_tensor.q_zero_point()
```

---

### 3. **Symmetric vs Asymmetric Quantization**
- Symmetric: Quantized values are symmetric around zero. Simpler but less precise for values with non-zero means.
- Asymmetric: Uses a zero-point to handle values with non-zero means, providing better accuracy.

---

### 4. **Finer Granularity for More Precision**
Quantization precision can be improved using techniques such as:
- Per-Channel Quantization: Applies quantization parameters individually for each channel.
- Per-Group Quantization: Divides channels into groups for finer granularity.

---

### 5. **Quantizing Weights and Activations for Inference**
Both weights and activations are quantized to reduce computation costs during inference.
Quantize model weights

```
from torch.quantization import quantize_dynamic
quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

---

### 6. **Custom Build an 8-bit Quantizer**
We built a custom 8-bit quantizer in the helper.py script. This allows:

- Flexibility in choosing quantization schemes.
- Easy integration with custom models.

---

### 7. **Replace PyTorch Layers with Quantized Layers**
PyTorch layers (e.g., Linear, Conv2d) can be replaced with quantized versions to improve performance. For example:

```
from torch.quantization import QuantStub, DeQuantStub

class QuantizedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.model = model

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
```

---

### 8. **Quantize Any Open-Source PyTorch Model**
Models from Hugging Face can be quantized by:
- Loading pre-trained weights.
- Applying dynamic or static quantization.
- Saving and deploying the quantized model.

---

### 9. **Load Quantized Weights from Hugging Face Hub**
Quantized models can be directly loaded from the Hugging Face Hub for easy deployment.

---

### 10. **Weights Packing**
- Efficient storage of quantized weights:
- Packing 2-bit Weights: Significantly reduces storage.
- Unpacking 2-bit Weights: Converts back to original precision for computation.

---

### 11. **Beyond Linear Quantization**
Explored advanced quantization techniques like:
- Non-Linear Quantization: Adapts to distribution skewness.
- Mixed Precision Quantization: Combines multiple precisions for different layers.

---

### **Libraries Used**
- NumPy: For numerical computations.
- PyTorch: Core framework for building and quantizing models.
- Pandas: Data manipulation and analysis.
- Hugging Face: Loading and saving pre-trained models.
- Helper.py: Contains custom-built functions for quantization workflows.

### **Load a pre-trained model**
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")

### **Apply dynamic quantization**
quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

### **Save the quantized model**
quantized_model.save_pretrained("quantized-bert")

### **Applications of Quantization**
- Edge Devices: Run complex models on low-resource devices like smartphones.
- Inference Acceleration: Speed up predictions in production environments.
- Memory-Constrained Environments: Deploy models in memory-limited scenarios.

### Conclusion
Quantization is a game-changer for deploying deep learning models in resource-constrained environments. By applying techniques like dynamic quantization, per-channel quantization, and packing weights, we can significantly optimize performance while preserving accuracy. This project demonstrates practical applications of quantization for PyTorch models, leveraging tools like Hugging Face and custom scripts.


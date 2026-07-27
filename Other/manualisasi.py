import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# INPUT (4x5)
# =========================
input_data = torch.tensor([
    [-1.78, -1.26, -1, -0.97, -0.97],
    [1.73, 1.43, 1.07, 0.68, 0.37],
    [1.19, 0.99, 0.45, -0.07, 0.24],
    [0.53, 0.31, 0.17, 0.23, 0.04],
    [-0.93, -0.93, 0.96, -0.98, -1]
], dtype=torch.float32)

# reshape ke (batch, channel, H, W)
input_data = input_data.unsqueeze(0).unsqueeze(0)

class ManualCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv1: 1 input channel → 4 filter
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, bias=False)

        # Conv2: 4 channel → 8 filter
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, bias=False)

        # FC: 8 → 2
        self.fc = nn.Linear(8, 2, bias=False)

    def forward(self, x):
        z1 = self.conv1(x)
        a1 = F.relu(z1)

        z2 = self.conv2(a1)
        a2 = F.relu(z2)

        # flatten
        a2 = a2.view(a2.size(0), -1)  # (batch, 8)

        z3 = self.fc(a2) 
        y = F.softmax(z3, dim=1)

        return y, z1, z2, a2, z3
    
model = ManualCNN()

with torch.no_grad():
    model.conv1.weight[:] = torch.tensor([
        [[[0.6, 0.2, 0.9],
          [0.1, 0.4, 0.2],
          [0.5, -0.1, 0.6]]],

        [[[-0.2, 0, -0.4],
          [-0.3, -0.3, -0.5],
          [1, -0.9, -0.3]]],

        [[[-0.1, -0.4, 0.2],
          [-0.2, -0.4, 0],
          [0.9, 0, 0.6]]],

        [[[-0.5, -0.8, 0.7],
          [-0.7, 0.4, -0.7],
          [-0.7, 0.4, 0]]]
    ])
    
with torch.no_grad():
    model.conv2.weight[:] = torch.tensor([

        # ================= FILTER 1 =================
        [
            [[0.9, -0.4, -0.5],
             [0.7, -0.6, 0.5],
             [-0.4, 0.8, 0.5]],

            [[0.1, 0.3, 0.2],
             [-0.8, 0.5, -1],
             [0.6, -0.1, 0.7]],

            [[0.9, -0.9, 0.6],
             [0.3, -0.6, 0.5],
             [-0.4, 0.2, 0.9]],

            [[1, 1, 0.3],
             [-0.6, 0.8, 0.4],
             [-0.4, -0.4, 0.9]]
        ],

        # ================= FILTER 2 =================
        [
            [[-0.8, -0.9, 0.1],
             [0.8, 1, -0.5],
             [1, -0.7, 0.3]],

            [[-0.4, -0.1, -0.1],
             [0.4, 0.7, -0.8],
             [0.1, -0.7, 0.8]],

            [[0.2, 0.1, -0.5],
             [-0.2, 0.5, 0.1],
             [0.9, 0, -0.4]],

            [[-0.5, 0.9, -0.8],
             [-0.1, -0.4, 0.5],
             [0.4, 0.1, -0.5]]
        ],

        # ================= FILTER 3 =================
        [
            [[-0.9, -0.1, 0.4],
             [0.7, -0.6, 0.5],
             [-0.7, -0.8, 0.3]],

            [[-0.3, -0.7, 0],
             [0.4, -0.9, -0.7],
             [0, 0.5, -0.9]],

            [[-0.5, 0.8, -0.8],
             [0.7, -0.1, 0],
             [0.2, -0.9, -0.2]],

            [[0.9, -0.4, -0.4],
             [0, -0.1, 0.5],
             [-0.5, 0, 0.3]]
        ],

        # ================= FILTER 4 =================
        [
            [[0.3, -1, 0.1],
             [-0.7, 0.2, -0.3],
             [-0.5, 0.4, -0.1]],

            [[0.5, 1, 0],
             [-0.8, 0.3, -0.5],
             [-0.6, 0.3, -0.3]],

            [[0.4, -0.7, -0.5],
             [-0.6, 0.2, -0.3],
             [0.6, 0, -0.5]],

            [[-0.4, 0.9, -0.1],
             [-0.6, 0.9, 0.9],
             [0.6, 0.3, 0]]
        ],

        # ================= FILTER 5 =================
        [
            [[-0.2, 0.3, 0.7],
             [0.5, -0.6, -0.7],
             [0.6, 0.3, 0.5]],

            [[0.9, 0, 0.7],
             [-0.1, -0.5, -0.3],
             [0.9, -0.8, 0.6]],

            [[-0.4, 0.5, 0.3],
             [-0.6, -0.1, 0.3],
             [0, 0.8, -1]],

            [[0.5, 0.8, -0.4],
             [0.6, -0.7, 0.4],
             [-0.5, 0.4, 0.5]]
        ],

        # ================= FILTER 6 =================
        [
            [[0.7, 0.3, -0.5],
             [0.7, -0.2, -0.4],
             [-0.2, 0.5, -0.6]],

            [[-0.5, 0.6, 0.5],
             [0.6, -0.4, 0],
             [-0.4, 0.4, 0.4]],

            [[-0.8, 0.6, -0.5],
             [-0.5, 0.8, 0.6],
             [0, -0.4, 0]],

            [[0.7, -0.5, 1],
             [0.4, 0.8, -0.7],
             [1, -0.4, 0.7]]
        ],

        # ================= FILTER 7 =================
        [
            [[0.7, -0.7, 0.7],
             [0.9, -0.7, -0.1],
             [-0.3, 0.5, 0.3]],

            [[-0.2, 0.7, 0.3],
             [-0.4, -0.6, 0.6],
             [-0.3, -0.5, -0.9]],

            [[0.6, 0.8, 0.8],
             [-0.1, 0.3, -0.4],
             [-0.6, -0.8, 0.9]],

            [[0.7, -0.4, 0],
             [0.5, -0.5, 0.7],
             [0.1, 0.9, -0.1]]
        ],

        # ================= FILTER 8 =================
        [
            [[1, 0.4, -0.9],
             [0.7, 0.2, -0.5],
             [0.2, -0.8, 0.6]],

            [[0.9, 0.1, -0.8],
             [-0.1, -0.8, -0.9],
             [0.8, -0.7, 0.2]],

            [[0.4, -0.6, 0.2],
             [0.4, 0.4, -0.1],
             [-0.6, -0.3, -1]],

            [[1, 0.1, 0.2],
             [-0.7, -0.7, -0.8],
             [-0.6, 0.1, 0.9]]
        ]

    ], dtype=torch.float32)
    
with torch.no_grad():
    model.fc.weight[:] = torch.tensor([
        [0.3, 0.6, -1, -1, 0.6, -0.4, 1, 0.8],
        [0.7, 0.8, 0.4, -0.6, -0.2, -0.3, 0.8, 0.4]
    ]) 
    
y_true = torch.tensor([[1.0, 0.0]])
    
y_pred, z1, z2, flatten, z3 = model(input_data)

print("Output:", y_pred)
print("Conv1 sebelum ReLU:", z1)
print("Conv2 sebelum ReLU:", z2)
print("Flatten:", flatten)
print("FC sebelum ReLU:", z3)
print("softmax output:", torch.softmax(y_pred, dim=1))
print("Prediksi kelas:", torch.softmax(y_pred, dim=1).argmax(dim=1).item())

W_fc_old = model.fc.weight.data.clone()
conv2_old = model.conv2.weight.data.clone()
conv1_old = model.conv1.weight.data.clone()

def manual_backward(model, x, y_true, lr=0.001):
    y_pred, z1, z2, a2, z3 = model(x)

    # =====================
    # 1. OUTPUT ERROR
    # =====================
    y_true = y_true.float()
    delta_out = y_pred - y_true  # (1,2)

    # =====================
    # 2. FC GRAD
    # =====================
    grad_fc = delta_out.T @ a2  # (2x8)

    # update FC
    model.fc.weight.data -= lr * grad_fc

    # =====================
    # 3. DELTA CONV2
    # =====================
    delta_conv2 = model.fc.weight.data.T @ delta_out.T  # (8,1)
    delta_conv2 = delta_conv2.view(1, 8, 1, 1)

    # =====================
    # 4. RELU CONV2
    # =====================
    relu_mask2 = (z2 > 0).float()
    delta_conv2 = delta_conv2 * relu_mask2

    # =====================
    # 5. GRAD CONV2
    # =====================
    grad_conv2 = torch.zeros_like(model.conv2.weight)

    a1 = F.relu(z1)

    for f in range(8):
        for c in range(4):
            grad_conv2[f, c] = a1[0, c] * delta_conv2[0, f]

    model.conv2.weight.data -= lr * grad_conv2

    # =====================
    # 6. DELTA CONV1
    # =====================
    delta_conv1 = torch.zeros_like(a1)

    for f in range(8):
        for c in range(4):
            kernel = model.conv2.weight.data[f, c]
            rotated = torch.flip(kernel, [0,1])
            delta_conv1[0, c] += rotated * delta_conv2[0, f]

    # =====================
    # 7. RELU CONV1
    # =====================
    relu_mask1 = (z1 > 0).float()
    delta_conv1 = delta_conv1 * relu_mask1

    # =====================
    # 8. GRAD CONV1
    # =====================
    grad_conv1 = torch.zeros_like(model.conv1.weight)

    for f in range(4):
        for i in range(3):
            for j in range(3):
                region = x[0,0, i:i+3, j:j+3]
                grad_conv1[f, 0, i, j] = torch.sum(region * delta_conv1[0,f])

    model.conv1.weight.data -= lr * grad_conv1

    return {
        "delta_out": delta_out,
        "delta_conv2": delta_conv2,
        "delta_conv1": delta_conv1
    }

y_true = torch.tensor([[1.0, 0.0]])

result = manual_backward(model, input_data, y_true)

print(result["delta_out"])
print(result["delta_conv2"])
print(result["delta_conv1"])

print("=== FC ===")
print("Before:\n", W_fc_old)

print("After:\n", model.fc.weight.data)

print("\n=== CONV2 ===")
print("Before:\n", conv2_old)

print("After:\n", model.conv2.weight.data)

print("\n=== CONV1 ===")
print("Before:\n", conv1_old)

print("After:\n", model.conv1.weight.data)

import pandas as pd

def tensor_to_df(tensor, name):
    arr = tensor.detach().cpu().numpy().flatten()
    return pd.DataFrame({
        f"{name}_index": range(len(arr)),
        f"{name}_value": arr
    })
    
with pd.ExcelWriter("hasil_backprop.xlsx") as writer:

    # ===== DELTA =====
    tensor_to_df(result["delta_out"], "delta_out").to_excel(writer, sheet_name="delta_out", index=False)
    tensor_to_df(result["delta_conv2"], "delta_conv2").to_excel(writer, sheet_name="delta_conv2", index=False)
    tensor_to_df(result["delta_conv1"], "delta_conv1").to_excel(writer, sheet_name="delta_conv1", index=False)

    # ===== FC =====
    tensor_to_df(W_fc_old, "fc_before").to_excel(writer, sheet_name="fc_before", index=False)
    tensor_to_df(model.fc.weight.data, "fc_after").to_excel(writer, sheet_name="fc_after", index=False)

    # ===== CONV2 =====
    tensor_to_df(conv2_old, "conv2_before").to_excel(writer, sheet_name="conv2_before", index=False)
    tensor_to_df(model.conv2.weight.data, "conv2_after").to_excel(writer, sheet_name="conv2_after", index=False)

    # ===== CONV1 =====
    tensor_to_df(conv1_old, "conv1_before").to_excel(writer, sheet_name="conv1_before", index=False)
    tensor_to_df(model.conv1.weight.data, "conv1_after").to_excel(writer, sheet_name="conv1_after", index=False)

print("Berhasil export ke hasil_backprop.xlsx")


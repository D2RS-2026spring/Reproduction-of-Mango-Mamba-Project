import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import time

# ===================== 1. 基础设置 =====================
seed = 42;
torch.manual_seed(seed);
np.random.seed(seed);
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 2. 无量纲化参数定义 =====================
r1_phys, r2_phys, y3_phys, x4_phys = 0.019, 0.0125, 0.00325, 0.0265
Tf_phys, α_phys, qw_phys = 200.0, 3000.0, 450000.0
λ_ref = 55.0
L_char = r1_phys;
T0 = Tf_phys;
Delta_T_char = 250.0
r1, r2 = r1_phys / L_char, r2_phys / L_char
y3, x4 = y3_phys / L_char, x4_phys / L_char
θ0 = math.asin(y3 / r1);
x26 = r1 * math.cos(θ0)
Bi = α_phys * L_char / λ_ref;
Q = qw_phys * L_char / (λ_ref * Delta_T_char)
print(f"Calibrated Parameters: Bi = {Bi:.3f}, Q = {Q:.3f}")


# ===================== 3. 网络与辅助函数 =====================
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(nn.Linear(2, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256),
                                 nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))

    def forward(self, x, y): return self.net(torch.cat([x, y], dim=1))


def sample_points(num_interior=6000, num_boundary=800):
    points = {}
    x_interior, y_interior = [], []
    while len(x_interior) < num_interior:
        x_s, y_s = np.random.uniform(0, x4), np.random.uniform(-r1, r1)
        if (x_s ** 2 + y_s ** 2 >= r2 ** 2) and ((x_s ** 2 + y_s ** 2 <= r1 ** 2) or (x_s >= x26)) and (
                (x_s <= x26) or ((y_s >= -y3) and (y_s <= y3))) and (x_s <= x4):
            x_interior.append(x_s);
            y_interior.append(y_s)
    points['interior'] = (torch.tensor(x_interior, dtype=torch.float32, device=device).unsqueeze(1),
                          torch.tensor(y_interior, dtype=torch.float32, device=device).unsqueeze(1))

    def sampler_l1(n):
        return (torch.zeros(n, 1), torch.rand(n, 1) * (r1 - r2) + r2)

    def sampler_l2(n):
        theta = torch.rand(n, 1) * (np.pi / 2 - θ0) + θ0; return (r1 * torch.cos(theta), r1 * torch.sin(theta))

    def sampler_l3(n):
        return (torch.rand(n, 1) * (x4 - x26) + x26, torch.full((n, 1), y3))

    def sampler_l4(n):
        return (torch.full((n, 1), x4), torch.rand(n, 1) * (2 * y3) - y3)

    def sampler_l5(n):
        return (torch.rand(n, 1) * (x4 - x26) + x26, torch.full((n, 1), -y3))

    def sampler_l6(n):
        theta = torch.rand(n, 1) * (np.pi / 2 - θ0) - np.pi / 2; return (r1 * torch.cos(theta), r1 * torch.sin(theta))

    def sampler_l7(n):
        return (torch.zeros(n, 1), torch.rand(n, 1) * (r1 - r2) - r1)

    def sampler_l8(n):
        theta = torch.rand(n, 1) * np.pi - np.pi / 2; return (r2 * torch.cos(theta), r2 * torch.sin(theta))

    boundary_samplers = {'l1': sampler_l1, 'l2': sampler_l2, 'l3': sampler_l3, 'l4': sampler_l4, 'l5': sampler_l5,
                         'l6': sampler_l6, 'l7': sampler_l7, 'l8': sampler_l8}
    for name, sampler in boundary_samplers.items():
        points[name] = tuple(p.to(device) for p in sampler(num_boundary))
    return points


def compute_derivatives(model, x, y):
    x.requires_grad_(True);
    y.requires_grad_(True)
    T_star = model(x, y);
    grad = torch.autograd.grad(T_star.sum(), [x, y], create_graph=True);
    dT_dx, dT_dy = grad[0], grad[1]
    d2T_dx2 = torch.autograd.grad(dT_dx.sum(), x, create_graph=True)[0];
    d2T_dy2 = torch.autograd.grad(dT_dy.sum(), y, create_graph=True)[0]
    return dT_dx, dT_dy, d2T_dx2, d2T_dy2, T_star


# ===================== 5. 最终训练 =====================
def train_final(model, epochs=60000, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    mse_loss = nn.MSELoss()

    # 【【【 终极权重平衡！！！】】】
    loss_weights = {
        'pde': 10.0,  # 提高PDE权重，强制满足物理定律
        'bc_heat': 100.0,  # 保持热流为主要驱动力
        'bc_conv': 100.0,  # 保持对流为主要驱动力
        'bc_adia': 50.0  # 大幅提高绝热边界权重，防止模型“作弊”
    }

    print("=" * 80);
    print("开始训练 (V19 - 平衡权重最终版)...");
    print("=" * 80)
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad();
        points = sample_points()

        # PDE
        x_int, y_int = points['interior'];
        _, _, d2T_dx2, d2T_dy2, _ = compute_derivatives(model, x_int, y_int)
        loss_pde = mse_loss(d2T_dx2 + d2T_dy2, torch.zeros_like(d2T_dx2))

        # 绝热
        losses_adia = []
        for name in ['l1', 'l4', 'l5', 'l6', 'l7']:
            x, y = points[name];
            dT_dx, dT_dy, _, _, _ = compute_derivatives(model, x, y)
            if name in ['l1', 'l4', 'l7']:
                loss = mse_loss(dT_dx, torch.zeros_like(dT_dx))
            elif name == 'l5':
                loss = mse_loss(dT_dy, torch.zeros_like(dT_dy))
            elif name == 'l6':
                loss = mse_loss((x * dT_dx + y * dT_dy) / r1, torch.zeros_like(x))
            losses_adia.append(loss)
        loss_adia = sum(losses_adia)

        # 热流
        x2, y2 = points['l2'];
        dT_dx2, dT_dy2, _, _, _ = compute_derivatives(model, x2, y2);
        loss_bc2 = mse_loss((x2 * dT_dx2 + y2 * dT_dy2) / r1, Q * (y2 / r1))
        x3, y3b = points['l3'];
        _, dT_dy3, _, _, _ = compute_derivatives(model, x3, y3b);
        loss_bc3 = mse_loss(dT_dy3, torch.full_like(dT_dy3, Q))
        loss_heat = loss_bc2 + loss_bc3

        # 对流
        x8, y8 = points['l8'];
        dT_dx8, dT_dy8, _, _, T8_star = compute_derivatives(model, x8, y8);
        loss_conv = mse_loss((x8 * dT_dx8 + y8 * dT_dy8) / r2, Bi * T8_star)

        total_loss = (loss_weights['pde'] * loss_pde + loss_weights['bc_heat'] * loss_heat + loss_weights[
            'bc_conv'] * loss_conv + loss_weights['bc_adia'] * loss_adia)

        total_loss.backward();
        optimizer.step()
        if (epoch + 1) % 10000 == 0: scheduler.step()
        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}/{epochs} | Loss: {total_loss.item():.4e} | LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {time.time() - start_time:.2f}s")
    return model


# ===================== 6. 可视化 =====================
def visualize_results(model):
    model.eval()
    with torch.no_grad():
        nx, ny = 200, 200;
        x_s = torch.linspace(0, x4, nx, device=device);
        y_s = torch.linspace(-r1, r1, ny, device=device)
        X_s, Y_s = torch.meshgrid(x_s, y_s, indexing='ij');
        X_s_f, Y_s_f = X_s.flatten().unsqueeze(1), Y_s.flatten().unsqueeze(1)
        T_star_f = model(X_s_f, Y_s_f)
        T_star = T_star_f.reshape(nx, ny);
        T_phys = T_star * Delta_T_char + T0;
        X_phys, Y_phys = X_s * L_char, Y_s * L_char
        R2_s = X_s ** 2 + Y_s ** 2;
        mask = (R2_s >= r2 ** 2) & ((R2_s <= r1 ** 2) | (X_s >= x26)) & ((X_s <= x26) | ((Y_s >= -y3) & (Y_s <= y3)))
        T_phys_m = torch.where(mask, T_phys, torch.tensor(float('nan')))
        T_phys_np, X_phys_np, Y_phys_np = T_phys_m.cpu().numpy(), X_phys.cpu().numpy(), Y_phys.cpu().numpy()
        min_t, max_t = np.nanmin(T_phys_np), np.nanmax(T_phys_np)
    print(f"\n预测温度范围: Min={min_t:.2f} K, Max={max_t:.2f} K")
    plt.figure(figsize=(10, 8));
    plt.contourf(X_phys_np, Y_phys_np, T_phys_np.T, levels=100, cmap='jet')
    plt.colorbar(label='Temperature (K)');
    plt.xlabel('x (m)');
    plt.ylabel('y (m)');
    plt.title('PINN Prediction (V19 - Final)');
    plt.axis('equal');
    plt.tight_layout();
    plt.show()


if __name__ == "__main__":
    pinn_model = PINN().to(device)
    pinn_model = train_final(pinn_model, epochs=80000)
    visualize_results(pinn_model)
    torch.save(pinn_model.state_dict(), 'pinn_fin_tube_final.pth')
    print("\n训练完成！模型已保存至 pinn_fin_tube_final.pth")


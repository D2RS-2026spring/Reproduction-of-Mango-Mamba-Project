# ========================
# 生成：混淆矩阵 + Grad-CAM 热力图
# 直接运行，图片自动保存
# ========================

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

# 路径
BASE = "/content/Reproduction-of-Mango-Mamba-Project/Mango-Mamba-and-VN-MangoLeaf-main"
DATA_ROOT = "/content/Reproduction-of-Mango-Mamba-Project/VN-MangoLeaf"
sys.path.append(BASE)

from models.mango_mamba import mango_mamba_tiny
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型
model = mango_mamba_tiny(
    num_classes=7, window_size=[7,7,7,7], mixer_mode='hybrid',
    use_eca=True, use_grn=True, use_swiglu=True
).to(device)

weight_path = Path(BASE) / "three_phase_curriculum/phase2_vn_mangoleaf/mango_mamba-20251209-154445/model_best_inference.pth"
checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

# 类别名
CLASS_NAMES = ['Healthy', 'Anthracnose', 'Gall_Midge', 'Dieback', 'Sooty_Mold', 'Powdery_Mildew', 'Red_Rust']

# 图像预处理
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# 加载数据集
ds = ImageFolder(root=DATA_ROOT, transform=transform)
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

# ========================
# 1. 生成混淆矩阵
# ========================
print("正在生成混淆矩阵...")
all_preds = []
all_gts = []
with torch.no_grad():
    for x,y in tqdm(loader):
        x = x.to(device)
        pred = model(x)['logits'].argmax(1).cpu().numpy()
        all_preds.extend(pred)
        all_gts.extend(y.numpy())

cm = confusion_matrix(all_gts, all_preds, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

plt.figure(figsize=(10,8))
disp.plot(cmap=plt.cm.Blues, values_format='.2f', ax=plt.gca())
plt.title('MangoMamba - VN-MangoLeaf Confusion Matrix', fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("/content/Reproduction-of-Mango-Mamba-Project/confusion_matrix.png", dpi=300)
plt.close()
print("混淆矩阵已保存: /content/Reproduction-of-Mango-Mamba-Project/confusion_matrix.png")

# ========================
# 2. Grad-CAM 热力图
# ========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inp, out):
        self.activations = out

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x):
        self.model.zero_grad()
        out = self.model(x)['logits']
        class_idx = out.argmax(1).item()
        out[0, class_idx].backward()

        weights = torch.mean(self.gradients, dim=[2,3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.cpu().detach().numpy(), class_idx

# 选最后一个 stage 的 norm 层
grad_cam = GradCAM(model, model.norm)

# 每个类别取一张图生成 CAM
print("正在生成 Grad-CAM 热力图...")
os.makedirs("/content/Reproduction-of-Mango-Mamba-Project/gradcam", exist_ok=True)
class_to_idx = ds.class_to_idx
idx_to_class = {v:k for k,v in class_to_idx.items()}

for cls_name in CLASS_NAMES:
    idx = class_to_idx[cls_name]
    # 找第一张该类的图
    img_path = [s for s, l in ds.samples if l == idx][0]
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    cam, pred_idx = grad_cam(img_t)
    cam = cv2.resize(cam, (224,224))
    img_np = np.array(img.resize((224,224))) / 255.0

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap / 255.0
    overlay = heatmap * 0.4 + img_np * 0.6

    plt.figure(figsize=(6,5))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title(f'Grad-CAM: {cls_name} → Pred: {idx_to_class[pred_idx]}', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"/content/Reproduction-of-Mango-Mamba-Project/gradcam/gradcam_{cls_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

print("Grad-CAM 已保存到: /content/Reproduction-of-Mango-Mamba-Project/gradcam/")
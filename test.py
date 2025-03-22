import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model.wdnet import WDNet

# 定义超参数
imgs_dir = 'imgs'
results_dir = 'results'
model_path = 'wdnet64_st.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建结果文件夹
os.makedirs(results_dir, exist_ok=True)
Mean = [0.4970002770423889, 0.5053070783615112, 0.4676517844200134]
Std = [0.24092243611812592, 0.23609396815299988, 0.25256040692329407]
# 图像预处理
transform = transforms.Compose([
    transforms.Resize((480, 720)),
    transforms.ToTensor(),
    transforms.Normalize(mean=Mean, std=Std),
])

denormalize = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/Std[0], 1/Std[1], 1/Std[2]]),
    transforms.Normalize(mean=[-Mean[0], -Mean[1], -Mean[2]], std=[1., 1., 1.])
])

# 初始化并加载模型
model = WDNet(3,3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 单图像处理函数
def process_image(image_path, model, device, transform, denormalize):
    # 打开图像文件并转换为RGB模式
    image = Image.open(image_path).convert('RGB')
    # 将图像大小调整为720x480
    image_resized = image.resize((720, 480))  # Resize image to (480, 720)
    # 对调整大小的图像进行转换（例如归一化等），并添加一个维度以适应模型的输入要求，然后移动到指定的设备（CPU或GPU）
    image_tensor = transform(image_resized).unsqueeze(0).to(device)
    
    # 在不计算梯度的情况下执行模型推理
    with torch.no_grad():
        output = model(image_tensor)
        
    # 将模型输出进行反归一化处理，限制值在0到1之间，转换为numpy数组，并调整维度顺序
    output = denormalize(output.squeeze().cpu()).clamp(0, 1).numpy().transpose(1, 2, 0)
    # 将输出图像的像素值缩放到0到255之间，并转换为无符号8位整数类型
    output = (output * 255).astype(np.uint8)
    
    # 返回处理后的图像
    return output

# 遍历图像文件夹中的所有图像文件，并对每个图像文件进行处理
for img_file in os.listdir(imgs_dir):
    if img_file.endswith(('png', 'jpg', 'jpeg')):
        img_path = os.path.join(imgs_dir, img_file)
        result = process_image(img_path, model, device, transform, denormalize)
        result_image = Image.fromarray(result)
        result_image.save(os.path.join(results_dir, img_file))

print('Processing complete. Results saved in', results_dir)

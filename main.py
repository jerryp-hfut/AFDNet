import os
import random
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import numpy as np
from model.wdnet import WDNet
import threading

# 定义超参数
imgs_dir = 'imgs'
results_dir = 'results'
model_path = 'wdnet.pth'
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
model = WDNet(3, 3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 单图像处理函数
def process_image(image_path):
    # 打开指定路径的图像文件，并将其转换为RGB模式
    image = Image.open(image_path).convert('RGB')
    # 将图像调整为720x480的尺寸
    image_resized = image.resize((720, 480))
    # 将调整后的图像转换为张量，并添加一个维度以适应模型的输入要求，然后移动到指定的设备（如GPU）
    image_tensor = transform(image_resized).unsqueeze(0).to(device)
    
    # 在不计算梯度的情况下执行模型推理，以节省内存和计算资源
    with torch.no_grad():
        # 将图像张量输入模型，得到输出结果
        output = model(image_tensor)
    
    # 对模型输出进行处理：移除维度、移动到CPU、反归一化、限制值范围、转换为numpy数组，并调整维度顺序
    output = denormalize(output.squeeze().cpu()).clamp(0, 1).numpy().transpose(1, 2, 0)
    # 将输出图像的像素值缩放到0-255范围，并转换为无符号8位整数类型
    output = (output * 255).astype(np.uint8)
    # 将numpy数组转换为PIL图像对象并返回
    return Image.fromarray(output)

# GUI
class DerainGUI:
    def __init__(self, root):
        # 初始化方法，传入root参数，root是Tkinter的主窗口对象
        self.root = root
        # 设置主窗口的标题为"WDNet Derainer"
        self.root.title("WDNet Derainer")
        # 设置主窗口的大小为700x500像素
        self.root.geometry("700x500")

        # 创建一个LabelFrame，用于展示去雨演示，设置文本为"Derain Demo"，内边距为10
        self.demo_frame = ttk.LabelFrame(root, text="Derain Demo", padding=10)
        # 将LabelFrame放置在主窗口中，填充水平方向，左右边距为10，上下边距为5
        self.demo_frame.pack(fill="x", padx=10, pady=5)

        # 创建一个Label，用于显示"Before"文本，放置在demo_frame中，第0行第0列，左右边距为5
        self.before_label = ttk.Label(self.demo_frame, text="Before")
        self.before_label.grid(row=0, column=0, padx=5)
        # 创建一个Label，用于显示去雨前的图片，放置在demo_frame中，第1行第0列，左右边距为5
        self.before_img_label = ttk.Label(self.demo_frame)
        self.before_img_label.grid(row=1, column=0, padx=5)

        # 创建一个Label，用于显示"After"文本，放置在demo_frame中，第0行第1列，左右边距为5
        self.after_label = ttk.Label(self.demo_frame, text="After")
        self.after_label.grid(row=0, column=1, padx=5)
        # 创建一个Label，用于显示去雨后的图片，放置在demo_frame中，第1行第1列，左右边距为5
        self.after_img_label = ttk.Label(self.demo_frame)
        self.after_img_label.grid(row=1, column=1, padx=5)

        # 创建一个Button，用于触发展示演示的函数，文本为"Show Demo"，放置在demo_frame中，第2行，跨越2列，上下边距为5
        self.show_demo_button = ttk.Button(self.demo_frame, text="Show Demo", command=self.show_demo)
        self.show_demo_button.grid(row=2, column=0, columnspan=2, pady=5)

        # 创建一个LabelFrame，用于批量处理，设置文本为"Batch Processing"，内边距为10
        self.process_frame = ttk.LabelFrame(root, text="Batch Processing", padding=10)
        # 将LabelFrame放置在主窗口中，填充水平方向，左右边距为10，上下边距为5
        self.process_frame.pack(fill="x", padx=10, pady=5)

        # 创建一个Button，用于触发开始去雨的函数，文本为"Start Deraining"，放置在process_frame中，上下边距为5
        self.derain_button = ttk.Button(self.process_frame, text="Start Deraining", command=self.start_deraining)
        self.derain_button.pack(pady=5)

        # 创建一个Progressbar，用于显示处理进度，长度为400像素，模式为确定模式，放置在process_frame中，上下边距为5
        self.progress = ttk.Progressbar(self.process_frame, length=400, mode='determinate')
        self.progress.pack(pady=5)

        # 创建一个Label，用于显示处理状态，初始文本为空，放置在process_frame中，上下边距为5
        self.status_label = ttk.Label(self.process_frame, text="")
        self.status_label.pack(pady=5)

    def show_demo(self):
        # 获取imgs目录下所有以png、jpg或jpeg结尾的文件名
        img_files = [f for f in os.listdir(imgs_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
        # 如果没有找到图片文件，显示错误消息并返回
        if not img_files:
            messagebox.showerror("Error", "No images found in 'imgs' directory!")
            return

        # 随机选择一张图片
        random_img = random.choice(img_files)
        # 构建图片的完整路径
        img_path = os.path.join(imgs_dir, random_img)

        # 加载并显示示意图像（原图）
        original = Image.open(img_path).resize((300, 200))
        original_tk = ImageTk.PhotoImage(original)
        self.before_img_label.config(image=original_tk)
        self.before_img_label.image = original_tk  # Keep reference

        # 加载并显示示意图像（去雨结果）
        derained = process_image(img_path).resize((300, 200))
        derained_tk = ImageTk.PhotoImage(derained)
        self.after_img_label.config(image=derained_tk)
        self.after_img_label.image = derained_tk  # Keep reference

    def start_deraining(self):
        # 禁用去雨按钮，防止用户在处理过程中再次点击
        self.derain_button.config(state='disabled')
        # 创建一个新的线程来处理所有图像，使用daemon=True使得线程在主程序退出时自动结束
        threading.Thread(target=self.process_all_images, daemon=True).start()

    def process_all_images(self):
        # 获取imgs目录下所有以png、jpg、jpeg结尾的文件
        img_files = [f for f in os.listdir(imgs_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
        # 如果没有找到图片文件，显示错误消息并返回
        if not img_files:
            messagebox.showerror("Error", "No images found in 'imgs' directory!")
            self.derain_button.config(state='normal')
            return

        # 获取图片总数
        total_images = len(img_files)
        # 记录开始时间
        start_time = time.time()
        # 初始化已处理图片计数
        processed_count = 0

        # 遍历所有图片文件
        for img_file in img_files:
            # 获取图片的完整路径
            img_path = os.path.join(imgs_dir, img_file)
            # 处理图片
            result = process_image(img_path)
            # 保存处理后的图片到results目录
            result.save(os.path.join(results_dir, img_file))

            # 更新已处理图片计数
            processed_count += 1
            # 计算进度百分比
            progress_value = (processed_count / total_images) * 100
            # 更新进度条
            self.progress['value'] = progress_value

            # 计算已过去的时间
            elapsed_time = time.time() - start_time
            # 计算每秒处理的图片数
            images_per_sec = processed_count / elapsed_time if elapsed_time > 0 else 0
            # 计算剩余时间
            remaining_time = (total_images - processed_count) / images_per_sec if images_per_sec > 0 else 0

            # 更新状态标签显示
            self.status_label.config(text=f"Processed: {processed_count}/{total_images} | "
                                        f"Elapsed: {elapsed_time:.1f}s | "
                                        f"Remaining: {remaining_time:.1f}s | "
                                        f"Speed: {images_per_sec:.2f} img/s")
            # 更新界面
            self.root.update_idletasks()

        # 所有图片处理完成后，更新状态标签显示
        self.status_label.config(text="Processing complete!")
        # 恢复按钮状态
        self.derain_button.config(state='normal')

if __name__ == "__main__":
    root = tk.Tk()
    app = DerainGUI(root)
    root.mainloop()
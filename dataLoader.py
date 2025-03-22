import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DerainDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        # 初始化数据集，设置数据目录、数据集分割类型（训练或测试）和数据变换
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.image_pairs = self._make_dataset()  # 调用_make_dataset方法生成图像对列表

    def _make_dataset(self):
        # 生成图像对列表
        image_pairs = []
        if self.split == 'train':
            # 如果是训练集，设置对应的子目录和索引范围
            data_subdir = 'data_train'
            gt_subdir = 'gt_train'
            index_range = range(0, 9)
        elif self.split == 'test':
            # 如果是测试集，设置对应的子目录和索引范围
            data_subdir = 'data_test'
            gt_subdir = 'gt_test'
            index_range = range(0, 2)
        else:
            # 如果分割类型不是'train'或'test'，抛出异常
            raise ValueError("Invalid split name, expected 'train' or 'test'")

        for i in index_range:
            # 构建带雨图像和干净图像的路径
            rain_image_path = os.path.join(self.data_dir, 'data', data_subdir, f'{i}_rain.png')
            clean_image_path = os.path.join(self.data_dir, 'gt', gt_subdir, f'{i}_clean.png')
            if os.path.exists(rain_image_path) and os.path.exists(clean_image_path):
                # 如果图像文件存在，添加到图像对列表
                image_pairs.append((rain_image_path, clean_image_path))
            else:
                # 如果图像文件不存在，抛出文件未找到异常
                raise FileNotFoundError(f"Files not found: {rain_image_path}, {clean_image_path}")

        return image_pairs  # 返回图像对列表

    def __len__(self):
        # 返回数据集中图像对的数量
        return len(self.image_pairs)

    def __getitem__(self, idx):
        # 根据索引获取图像对，并进行必要的变换
        rain_image_path, clean_image_path = self.image_pairs[idx]
        rain_image = Image.open(rain_image_path).convert('RGB')  # 打开带雨图像并转换为RGB格式
        clean_image = Image.open(clean_image_path).convert('RGB')  # 打开干净图像并转换为RGB格式

        if self.transform:
            # 如果有变换操作
            # 同步的随机水平翻转
            if random.random() > 0.5:
                rain_image = transforms.functional.hflip(rain_image)  # 水平翻转带雨图像
                clean_image = transforms.functional.hflip(clean_image)  # 水平翻转干净图像

            # 其他变换（Resize、ToTensor、Normalize）
            rain_image = self.transform(rain_image)  # 对带雨图像进行变换
            clean_image = self.transform(clean_image)  # 对干净图像进行变换

        return rain_image, clean_image  # 返回变换后的带雨图像和干净图像

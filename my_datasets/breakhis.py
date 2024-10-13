import os
from .utils import Datum, DatasetBase
import random
# 用于生成图像描述的模板
template = ['A breast tumor tissue image of {}.']

# 定义新的类名映射
NEW_CNAMES = {
    'benign': 'benign breast tumor tissue',
    'malignant': 'malignant breast tumor tissue'
}

class Breakhis(DatasetBase):

    def __init__(self, root, num_shots=-1):
        self.dataset_dir = os.path.join(root, '40X')
        self.benign_dir = os.path.join(self.dataset_dir, 'benign')
        self.malignant_dir = os.path.join(self.dataset_dir, 'malignant')

        self.template = template

        # 将数据集分为训练集、验证集、测试集
        train, val, test = self.read_split()
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)

    def read_split(self):
        """
        读取本地的benign和malignant文件夹，划分为训练集、验证集和测试集
        假设数据集随机分为 70% 训练集, 15% 验证集, 15% 测试集
        """
        benign_images = self._read_images_from_folder(self.benign_dir, label=0, classname='benign')
        malignant_images = self._read_images_from_folder(self.malignant_dir, label=1, classname='malignant')

        # 合并所有的图片数据
        all_data = benign_images + malignant_images

        # 打乱数据
        random.shuffle(all_data)

        # 计算各个数据集的大小
        num_samples = len(all_data)
        train_size = int(0.8 * num_samples)
        val_size = int(0.1 * num_samples)
        test_size = num_samples - train_size - val_size

        # 划分数据集
        train_data = all_data[:train_size]
        val_data = all_data[train_size:train_size + val_size]
        test_data = all_data[train_size + val_size:]

        return train_data, val_data, test_data

    def _read_images_from_folder(self, folder, label, classname):
        """
        读取指定文件夹中的图片，并将每张图片包装为Datum对象
        Args:
            folder (str): 文件夹路径
            label (int): 图片的标签，benign为0，malignant为1
            classname (str): 类别名
        Returns:
            images (list): 包含Datum对象的列表
        """
        images = []
        for img_name in os.listdir(folder):
            if img_name.endswith('.png'):
                img_path = os.path.join(folder, img_name)
                # 使用NEW_CNAMES将类名映射为更详细的描述
                classname_mapped = NEW_CNAMES.get(classname, classname)
                datum = Datum(impath=img_path, label=label, classname=classname_mapped)
                images.append(datum)
        return images

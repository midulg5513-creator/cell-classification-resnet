import cv2
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from torch.nn import Dropout
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from collections import Counter
import torchvision.transforms as transforms
import pandas as pd

# --------------------------
# 1. 配置参数 
# --------------------------
TRAIN_DATA_DIR = "train"
VAL_DATA_DIR = "val"
TEST_DATA_DIR = "Test"  



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用设备: {DEVICE}")

IMAGE_SIZE = (32, 32)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

CACHE_DIR = "cache_balanced_data"
os.makedirs(CACHE_DIR, exist_ok=True)
DATA_CACHE_PATH = os.path.join(CACHE_DIR, "original_data.npz")
BALANCED_DATA_CACHE_PATH = os.path.join(CACHE_DIR, "balanced_data.npz")
LABEL_MAP_PATH = os.path.join(CACHE_DIR, "label_map.pkl")
MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_balanced_model.pth")
CSV_SAVE_PATH = os.path.join(CACHE_DIR, "submission.csv") 

BATCH_SIZE = 256
LEARNING_RATE = 0.0003
EPOCHS = 200
REGULARIZATION = 0.0001
EARLY_STOPPING_PATIENCE = 20
GRAD_CLIP = 1.0
NUM_WORKERS = 4
PIN_MEMORY = True

# --------------------------
# 2. 数据加载与平衡 (无需修改)
# --------------------------
def load_and_balance_data():
    if os.path.exists(BALANCED_DATA_CACHE_PATH) and os.path.exists(LABEL_MAP_PATH):
        print(f" 从缓存加载平衡后的训练数据...")
        data = np.load(BALANCED_DATA_CACHE_PATH)
        X_train_balanced = data["X_train"]
        y_train_balanced = data["y_train"]
        with open(LABEL_MAP_PATH, "rb") as f:
            label_map = pickle.load(f)
        return X_train_balanced, y_train_balanced, label_map

    if not os.path.exists(DATA_CACHE_PATH):
        print(f"📥 首次运行，正在从 {TRAIN_DATA_DIR} 加载所有数据...")
        X_images_list, y_labels = [], []
        def recursive_load(current_dir):
            img_files = [f for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if img_files:
                class_name = os.path.basename(current_dir)
                for img_name in img_files:
                    img_path = os.path.join(current_dir, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None: continue
                    if img.shape != IMAGE_SIZE:
                        img = cv2.resize(img, IMAGE_SIZE)
                    X_images_list.append(img)
                    y_labels.append(class_name)
            else:
                for subdir in os.listdir(current_dir):
                    subdir_path = os.path.join(current_dir, subdir)
                    if os.path.isdir(subdir_path): recursive_load(subdir_path)
        recursive_load(TRAIN_DATA_DIR)
        
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_labels)
        label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        
        X_images = np.array(X_images_list)
        np.savez(DATA_CACHE_PATH, X_images=X_images, y=y)
        with open(LABEL_MAP_PATH, "wb") as f:
            pickle.dump(label_map, f)
        print(f" 原始数据加载完毕，共 {len(X_images)} 张图片。")
    else:
        data = np.load(DATA_CACHE_PATH)
        X_images = data["X_images"]
        y = data["y"]
        with open(LABEL_MAP_PATH, "rb") as f:
            label_map = pickle.load(f)
        print(f"📌 从缓存加载原始数据，共 {len(X_images)} 张图片。")

    class_counts = Counter(y)
    target_count = max(800, int(np.median(list(class_counts.values()))))
    print(f"原始数据类别分布: {dict(class_counts)}")
    print(f"目标样本数: {target_count}")

    print("\n 开始平衡数据集...")
    X_train_balanced = []
    y_train_balanced = []
    for class_id in range(len(label_map)):
        class_indices = np.where(y == class_id)[0]
        num_samples = len(class_indices)
        
        if num_samples < target_count:
            additional_indices = np.random.choice(class_indices, size=target_count - num_samples, replace=True)
            balanced_indices = np.concatenate([class_indices, additional_indices])
        else:
            balanced_indices = np.random.choice(class_indices, size=target_count, replace=False)
            
        X_train_balanced.extend(X_images[balanced_indices])
        y_train_balanced.extend([class_id] * len(balanced_indices))
        
    X_train_balanced = np.array(X_train_balanced)
    y_train_balanced = np.array(y_train_balanced)
    
    print(f"平衡后数据类别分布: {dict(Counter(y_train_balanced))}")
    print(f" 数据集平衡完成，总样本数: {len(X_train_balanced)}")
    np.savez(BALANCED_DATA_CACHE_PATH, X_train=X_train_balanced, y_train=y_train_balanced)
    print(f" 平衡后的数据集已缓存到 {BALANCED_DATA_CACHE_PATH}")
    
    return X_train_balanced, y_train_balanced, label_map

def load_validation_data(label_encoder):
    print(f"\n📥 正在从 {VAL_DATA_DIR} 加载验证数据...")
    X_images_list, y_labels = [], []
    def recursive_load(current_dir):
        img_files = [f for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if img_files:
            class_name = os.path.basename(current_dir)
            for img_name in img_files:
                img_path = os.path.join(current_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                if img.shape != IMAGE_SIZE:
                    img = cv2.resize(img, IMAGE_SIZE)
                X_images_list.append(img)
                y_labels.append(class_name)
        else:
            for subdir in os.listdir(current_dir):
                subdir_path = os.path.join(current_dir, subdir)
                if os.path.isdir(subdir_path): recursive_load(subdir_path)
    recursive_load(VAL_DATA_DIR)
    
    X_images = np.array(X_images_list)
    y = label_encoder.transform(y_labels)
    print(f"✅ 验证数据加载完毕，共 {len(X_images)} 张图片。")
    return X_images, y

# --------------------------
# 3. 改进型模型 
# --------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels)) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        if self.downsample is not None: identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class EnhancedCellClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EnhancedCellClassifier, self).__init__()
        self.initial = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.res_blocks = nn.Sequential(ResBlock(32, 32), nn.MaxPool2d(2, 2), ResBlock(32, 64), nn.MaxPool2d(2, 2), ResBlock(64, 128), nn.MaxPool2d(2, 2), ResBlock(128, 256),Dropout(0.5), nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(nn.Dropout(0.7), nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Dropout(0.7), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def extract_features(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        return torch.flatten(x, 1)

# --------------------------
# 4. 训练与评估函数 
# --------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)
            total_train += inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train.double() / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc.cpu().numpy())

        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data)
                total_val += inputs.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val.double() / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc.cpu().numpy())
        
        scheduler.step(epoch_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> 新的最佳验证准确率: {best_val_acc:.4f}, 模型已保存。")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n 早停触发！已连续{EARLY_STOPPING_PATIENCE}轮验证损失未下降")
            break
            
    print(f"\n 训练完成！最佳验证准确率: {best_val_acc:.4f}")
    return model, train_losses, val_losses, train_accs, val_accs

# --------------------------
# 5. 可视化函数 
# --------------------------
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('模型训练曲线', fontsize=16)
    ax1.plot(train_losses, label='训练损失'), ax1.plot(val_losses, label='验证损失'), ax1.set_title('训练与验证损失'), ax1.set_xlabel('Epoch'), ax1.set_ylabel('Loss'), ax1.legend(), ax1.grid(True)
    ax2.plot(train_accs, label='训练准确率'), ax2.plot(val_accs, label='验证准确率'), ax2.set_title('训练与验证准确率'), ax2.set_xlabel('Epoch'), ax2.set_ylabel('Accuracy'), ax2.legend(), ax2.grid(True)
    plt.tight_layout(), plt.show()

def plot_tsne(model, data_loader, class_names, device):
    print("\n 正在提取验证集特征以进行TSNE可视化...")
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            features = model.extract_features(inputs.to(device))
            all_features.append(features.cpu().numpy()), all_labels.extend(labels.numpy())
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_jobs=-1)
    tsne_results = tsne.fit_transform(np.vstack(all_features))
    plt.figure(figsize=(12, 8))
    for i, class_name in enumerate(class_names):
        indices = np.array(all_labels) == i
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=class_name, alpha=0.7, s=30)
    plt.title('验证集特征TSNE可视化'), plt.xlabel('t-SNE Dimension 1'), plt.ylabel('t-SNE Dimension 2'), plt.legend(), plt.grid(True, linestyle='--', alpha=0.6), plt.tight_layout(), plt.show()

# --------------------------
# 6. 加载测试集并进行预测
# --------------------------
def load_test_data_for_prediction(test_dir):
    """加载测试集图片及其文件名，用于预测。"""
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"测试集目录未找到: {test_dir}")
        
    print(f"\n📥 正在从 {test_dir} 加载测试数据...")
    test_images_with_names = []
    def recursive_load(current_dir):
        entries = os.listdir(current_dir)
        img_files = [f for f in entries if os.path.isfile(os.path.join(current_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if img_files:
            for img_name in img_files:
                img_path = os.path.join(current_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                if img.shape != IMAGE_SIZE:
                    img = cv2.resize(img, IMAGE_SIZE)
                test_images_with_names.append((img, img_name))
        else:
            for subdir in entries:
                subdir_path = os.path.join(current_dir, subdir)
                if os.path.isdir(subdir_path):
                    recursive_load(subdir_path)
                    
    recursive_load(test_dir)
    print(f"测试数据加载完毕，共 {len(test_images_with_names)} 张图片。")
    return test_images_with_names

def predict_and_generate_csv(model, test_images_with_names, id_to_label, transform, device, save_path):
    """对测试集进行预测并生成指定格式的CSV文件。"""
    print("\n 开始对测试集进行预测并生成提交文件...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for img_np, img_name in test_images_with_names:
            # 预处理
            img_tensor = transform(img_np).unsqueeze(0) 
            img_tensor = img_tensor.to(device)
            
            # 预测
            outputs = model(img_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_label = id_to_label[predicted_idx.item()]
            
            predictions.append({'image_name': img_name, 'predicted_label': predicted_label})
            
            if len(predictions) % 1000 == 0:
                print(f"  已处理 {len(predictions)} 张图片...")

    # 生成DataFrame并保存
    df = pd.DataFrame(predictions)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n 提交文件已成功生成！")
    print(f"   文件路径: {os.path.abspath(save_path)}")
    print(f"   文件内容预览:\n", df.head())
# ==========================================================

# --------------------------
# 7. 主程序执行
# --------------------------
if __name__ == '__main__':
    # 1. 加载并平衡数据
    X_train_balanced, y_train_balanced, label_map = load_and_balance_data()
    class_names = list(label_map.keys())
    num_classes = len(class_names)
    
    # 2. 加载验证数据
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(class_names)
    X_val, y_val = load_validation_data(label_encoder)
    
    # 3. 数据预处理
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((X_train_balanced.mean()/255.0,), (X_train_balanced.std()/255.0,)), transforms.RandomAdjustSharpness(1.5, p=0.2), transforms.RandomAutocontrast(p=0.2)])
    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((X_train_balanced.mean()/255.0,), (X_train_balanced.std()/255.0,))])
    
    X_train_tensor = torch.stack([train_transform(img) for img in X_train_balanced])
    X_val_tensor = torch.stack([val_transform(img) for img in X_val])
    
    train_loader = DataLoader(TensorDataset(X_train_tensor, torch.LongTensor(y_train_balanced)), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(TensorDataset(X_val_tensor, torch.LongTensor(y_val)), batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    # 4. 初始化模型、损失函数和优化器
    model = EnhancedCellClassifier(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.4)

    # 5. 开始训练
    final_model, train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, DEVICE, EPOCHS)

    # 6. 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # 7. 加载最佳模型进行后续操作
    final_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    # 8. 绘制TSNE图
    plot_tsne(final_model, val_loader, class_names, DEVICE)

    # 9. 在验证集上评估
    final_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = final_model(inputs.to(DEVICE))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy()), all_labels.extend(labels.numpy())
    print("\n" + "="*60), print("📋 最佳模型在验证集上的分类报告:"), print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8)), sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names), plt.xlabel('预测标签'), plt.ylabel('真实标签'), plt.title('验证集混淆矩阵'), plt.tight_layout(), plt.show()

   
    # 10. 加载测试集数据
    if os.path.exists(TEST_DATA_DIR):
        test_images_with_names = load_test_data_for_prediction(TEST_DATA_DIR)
        
        # 11. 创建 ID 到 类别名 的映射
        id_to_label = {v: k for k, v in label_map.items()}
        
        # 12. 使用最佳模型进行预测并生成CSV
        predict_and_generate_csv(
            model=final_model,
            test_images_with_names=test_images_with_names,
            id_to_label=id_to_label,
            transform=val_transform, # 使用验证集的transform
            device=DEVICE,
            save_path=CSV_SAVE_PATH
        )
    else:
        print(f"\n⚠️ 竞赛测试集目录 '{TEST_DATA_DIR}' 不存在，跳过提交文件生成。")
    # ===========================================================
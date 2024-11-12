import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('data.csv')

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('Using device:', device)

# 设置特征和目标
features = df[['HHJSJDB_shifted1', 'HHJSJDC_shifted1', 'XYS1_shifted1', 'XYS2_shifted1']].values
target = df['close'].values

# 超参数
n_splits = 5  # k 折交叉验证折数
epochs = 100  # 每轮训练的 epoch 数

# 归一化特征和目标
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler_features.fit_transform(features)
target_scaled = scaler_target.fit_transform(target.reshape(-1, 1)).flatten()

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out

# 交叉验证
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
mse_scores = []

# 执行交叉验证
for train_index, val_index in kf.split(features_scaled):
    X_train, X_val = features_scaled[train_index], features_scaled[val_index]
    y_train, y_val = target_scaled[train_index], target_scaled[val_index]

    # 转换为张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    # 实例化模型，定义损失函数和优化器
    model = MLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    # 验证模型
    model.eval()
    with torch.no_grad():
        predictions = model(X_val_tensor)
        mse = criterion(predictions, y_val_tensor).item()  # 计算均方误差
        mse_scores.append(mse)

# 打印每折交叉验证的均方误差
print(f'MSE scores for each fold: {mse_scores}')
print(f'Average MSE: {np.mean(mse_scores)}')

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# read data
df = pd.read_csv('data.csv')
new_df = pd.read_csv('new_data.csv')

# set features and target
features = df[['alpha_3_shifted', 'alpha_6_shifted', 'alpha_9_shifted', 'alpha_17_shifted', 'alpha_25_shifted']].values
target = df['5d_diff'].values

new_features = new_df[['alpha_3_shifted', 'alpha_6_shifted', 'alpha_9_shifted', 'alpha_17_shifted', 'alpha_25_shifted']].values


# scale features and target 
scaled_features = MinMaxScaler(feature_range=(0, 1))
scaled_target = MinMaxScaler(feature_range=(0, 1))
features = scaled_features.fit_transform(features)
target = scaled_target.fit_transform(target.reshape(-1, 1)).flatten()

scaled_new_features = MinMaxScaler(feature_range=(0, 1))
new_features = scaled_new_features.fit_transform(new_features)
new_features_tensor = torch.tensor(new_features, dtype=torch.float32).to(device)


# set train/test data_set
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# convert to tensor
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# 定义DNN模型，输入层5个特征（五个因子），输出层1个特征（未来五日收益）
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(5, 256)  # 输入层到隐藏层1 
        self.dropout1 = nn.Dropout(0.1)  # 随机失活层1
        self.fc2 = nn.Linear(256, 128)  # 隐藏层1到隐藏层2 
        self.fc3 = nn.Linear(128, 1)  # 隐藏层4到输出层
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.dropout1(out)
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        return out
    

'''
# define elastic net
class ElasticNetLoss(nn.Module):
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        super(ElasticNetLoss, self).__init__()
        self.alpha = alpha  # 正则化强度
        self.l1_ratio = l1_ratio  # L1比率

    def forward(self, y_pred, y_true, model):
        mse_loss = nn.MSELoss()(y_pred, y_true)  # 计算均方误差
        l1_loss = sum(p.abs().sum() for p in model.parameters())  # L1损失
        l2_loss = sum(p.pow(2).sum() for p in model.parameters())  # L2损失
        return mse_loss + self.alpha * (self.l1_ratio * l1_loss + (1 - self.l1_ratio) * l2_loss)

'''



# initialize model, criterion, optimizer
model = DNN().to(device)
loss_function = nn.MSELoss()
# elastic_net_loss = ElasticNetLoss(alpha=0.01, l1_ratio=0.5)

# 选择优化器：梯度下降/随机梯度下降/Adagrad
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0)
#optimizer = optim.Adagrad(model.parameters(), lr=0.001)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# train model

epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = loss_function(output, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# evaluate model
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = loss_function(test_predictions, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# inverse transform predictions and actual values
test_predictions_cpu = test_predictions.cpu().detach().numpy()  
y_test_cpu = y_test.cpu().detach().numpy() 
test_predictions = scaled_target.inverse_transform(test_predictions_cpu)
y_test = scaled_target.inverse_transform(y_test_cpu)

win_rate = (test_predictions > 0).astype(int) == (y_test > 0).astype(int)

# print result
for i in range(len(y_test)):
    print(f'Predicted Price: {test_predictions[i][0]}, Actual Price: {y_test[i][0]}')
print(f'Win Rate: {win_rate.mean():.4f}')

error = np.sum(np.abs(test_predictions - y_test))
error_per_day = error / len(y_test)
print(f'Total Error: {error}, Error per day: {error_per_day}')


# predict new data
model.eval()
with torch.no_grad():
    predictions = model(new_features_tensor)
    predictions_cpu = predictions.cpu().detach().numpy()
    predictions = scaled_target.inverse_transform(predictions_cpu)

predicted_5d_diff = np.array([prediction[0] for prediction in predictions])
new_win_rate = (predicted_5d_diff> 0).astype(int) == (new_df['5d_diff'] > 0).astype(int)

# print result
for i in range(len(new_df['5d_diff'])):
    print(f'Predicted Price: {predictions[i][0]}, Actual Price: {new_df["5d_diff"][i]}')
print(f'Win Rate: {new_win_rate.mean():.4f}')



# plot result
plt.figsize = (12,6)
plt.subplot(121)
plt.title('Test')
plt.plot(test_predictions, label='Predictions')
plt.plot(y_test, label='Actual')
plt.subplot(122)
plt.title('New Data')
plt.plot(predictions, label='Predictions')
plt.plot(new_df['5d_diff'], label='Actual')
plt.xlabel('Time')
plt.ylabel('Price')

plt.legend()
plt.show()

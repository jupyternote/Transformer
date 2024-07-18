import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from Transformer import Transformer  # 确保您的Transformer文件在相应的路径下
from self_attention.give_mask import padding_mask, Mask

# 参数定义
seq_length = 28  
input_len = 1  # 每个像素值作为一个输入
hidden_len = 14
embedding_size = 28
stack_num = 6
head_num = 2
batch_size = 20
output_hidden = 28
output_out = 28  # 输出单个像素值

# 异常检测
if embedding_size % head_num != 0:
    print("头数量无法整除")
    exit()

model = Transformer(input_len, hidden_len, embedding_size, stack_num, head_num, batch_size, output_hidden, output_out)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

input1 = torch.rand(batch_size, seq_length, input_len, dtype=torch.float32).to(device)
input2 = torch.rand_like(input1).to(device)

mask1 = padding_mask(input1).to(device)
mask2 = Mask(input1).to(device)

# 数据加载和预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='D://data//mnist', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='D://data//mnist', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5

# 创建掩码
mask_basis = torch.ones(batch_size, seq_length, embedding_size, dtype=torch.float32).to(device)
PaddingMask = padding_mask(mask_basis).to(device)
AllMask = Mask(mask_basis).to(device)

def save_images(images, epoch, batch, prefix):
    images = images.view(images.size(0), 28, 28).cpu().numpy()
    for i in range(min(10, images.shape[0])):  # 保存前10张图片
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
        plt.savefig(f'output_images/{prefix}_epoch{epoch+1}_batch{batch+1}_img{i+1}.png')
        plt.close()

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        images = images.view(batch_size, 28, 28, 1)  # 将每张图片重塑为 (batch_size, 28, 28, 1)

        optimizer.zero_grad()
        
        loss = 0
        outputs = torch.zeros_like(images)
        temp = labels.unsqueeze(1).unsqueeze(2)

        # 使用expand扩展张量，参数顺序应为(20, 28, 1)
        input_left = temp.expand(-1, seq_length, -1).float().to(device)
        input_right = torch.rand_like(input_left, dtype=torch.float32).to(device)

        output_data = model(input_left, input_right, mask1, mask2)
        images = images.view(20, 28, 28)
        loss += criterion(output_data, images)
        
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        if (i + 1) % 1000 == 0:
            save_images(output_data, epoch, i, "train")
        exit()
    # 保存模型
    torch.save(model.state_dict(), f'output_images/model_epoch{epoch+1}.pth')

    # model.eval()
    # with torch.no_grad():
    #     for images, labels in test_loader:
    #         images = images.to(device)
    #         images = images.view(batch_size, 28, 28, 1)
            
    #         outputs = torch.zeros_like(images)
    #         for j in range(28):
    #             input_data = images[:, j, :, :]
    #             output_data = model(input_data)
    #             outputs[:, j, :, :] = output_data

    #         save_images(outputs, epoch, i, 'test')
    #         break  # 只保存一个batch的图片

    # print(f'Epoch [{epoch+1}/{num_epochs}] completed')

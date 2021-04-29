import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import argparse
import os
from tensorboardX import SummaryWriter

#net1
class CNN_net1(torch.nn.Module):
    def __init__(self):
        super(CNN_net1, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 5)
        # in_channels = 3,out_channels = 10,kernel_size = 5, stride = 1, padding = 0
        self.pool = torch.nn.MaxPool2d(2, 2)
        # kernel_size = 2，stride = 2
        self.conv2 = torch.nn.Conv2d(10, 20, 3)
        self.fc1 = torch.nn.Linear(20 * 6 * 6, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 6 * 6)  # 卷积结束后将多层图片平铺batchsize行16*5*5列，每行为一个sample，16*5*5个特征
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#print(cnn_net1)
cnn_net1 = CNN_net1()
SumWriter = SummaryWriter(log_dir="CNN1/CNN_123")
dummyinput = torch.rand(12,3,32,32)
SumWriter.add_graph(cnn_net1, dummyinput ,verbose = True)

# 超参数设置
EPOCH = 200  #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128      #批处理尺寸(batch_size)
LR = 0.01        #学习率


#net2
class CNN_net2(torch.nn.Module):
    def __init__(self):
        super(CNN_net2, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 5)
        # in_channels = 3,out_channels = 10,kernel_size = 5, stride = 1, padding = 0
        self.pool = torch.nn.MaxPool2d(2, 2)
        # kernel_size = 2，stride = 2
        self.conv2 = torch.nn.Conv2d(10, 20, 3)
        self.fc1 = torch.nn.Linear(20 * 6 * 6, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.sigmoid(self.conv1(x)))
        x = self.pool(torch.sigmoid(self.conv2(x)))
        x = x.view(-1, 20 * 6 * 6)  # 卷积结束后将多层图片平铺batchsize行16*5*5列，每行为一个sample，16*5*5个特征
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

cnn_net2 = CNN_net2()


#net3
class CNN_net3(torch.nn.Module):
    def __init__(self):
        super(CNN_net3, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 5)
        # in_channels = 3,out_channels = 10,kernel_size = 5, stride = 1, padding = 0
        self.pool = torch.nn.MaxPool2d(2, 2)
        # kernel_size = 2，stride = 2
        self.conv2 = torch.nn.Conv2d(10, 20, 3)
        self.fc1 = torch.nn.Linear(20 * 6 * 6, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(-1, 20 * 6 * 6)  # 卷积结束后将多层图片平铺batchsize行16*5*5列，每行为一个sample，16*5*5个特征
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

cnn_net3 = CNN_net3()


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
args = parser.parse_args()



# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) #训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
net1 = CNN_net1().to(device)
net2 = CNN_net2().to(device)
net3 = CNN_net3().to(device)
# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer1 = optim.SGD(net1.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

print_step = 100 # 100次迭代后输出损失
# 训练
#net1
if __name__ == "__main__":
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    best_acc1 = 70  #2 初始化best test accuracy
    print("Start Training, CNN_net1!")  # 定义遍历数据集的次数
    with open("acc1.txt", "w") as f_1:
        with open("log1.txt", "w")as f2_1:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net1.train()
                sum_loss1 = 0.0
                correct1 = 0.0
                total1 = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer1.zero_grad()

                    # forward + backward
                    outputs1 = net1(inputs)
                    loss1 = criterion(outputs1, labels)
                    loss1.backward()
                    optimizer1.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss1 += loss1.item()

                    #test_loss_all.append()

                    _, predicted1 = torch.max(outputs1.data, 1)
                    total1 += labels.size(0)
                    correct1 += predicted1.eq(labels.data).cpu().sum()
                    niter1 = i + 1 + epoch * length
                    train_acc1 = 100. * correct1 / total1
                    train_loss1 = sum_loss1 / (i+1)
                    print('[epoch:%d, iter:%d] Loss: %.03f_1 | Acc: %.3f_1%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss1 / (i + 1), 100. * correct1 / total1))
                    f2_1.write('%03d  %05d |Loss: %.03f_1 | Acc: %.3f_1%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss1 / (i + 1), 100. * correct1 / total1))
                    f2_1.write('\n')
                    f2_1.flush()
                    if niter1 % print_step == 0:
                        # 为日志添加损失函数
                        SumWriter.add_scalar("train_loss_net1", train_loss1 , niter1)
                        testinputs1_im = vutils.make_grid(inputs, nrow=12)  # (数据预处理？)
                        SumWriter.add_image("input image sample_net1", testinputs1_im, niter1)
                        testout1_im = vutils.make_grid(outputs1, nrow=128)
                        SumWriter.add_image("output image sample_net1", testout1_im, niter1)

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct1 = 0
                    total1 = 0
                    for data in testloader:
                        net1.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs1 = net1(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted1 = torch.max(outputs1.data, 1)
                        total1 += labels.size(0)
                        correct1 += (predicted1 == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct1 / total1))
                    acc1 = 100. * correct1 / total1
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net1.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f_1.write("EPOCH=%03d  |Accuracy= %.3f_1%%" % (epoch + 1,acc1))
                    f_1.write('\n')
                    f_1.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc1 > best_acc1:
                        f3_1 = open("best_acc1.txt", "w")
                        f3_1.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc1))
                        f3_1.close()
                        best_acc1 = acc1
                SumWriter.add_scalars("train/test_acc_epoch_net1", {'train_acc': train_acc1,'test_acc':acc1}, epoch)
                SumWriter.add_scalars("train_acc_epoch",{'net_1':train_acc1},epoch)
                SumWriter.add_scalars("test_acc_epoch", {'net_1': acc1}, epoch)
                SumWriter.add_scalars("train_loss_epoch", {'net_1':train_loss1}, epoch)
                testout1_im = vutils.make_grid(outputs1, nrow=12)
                SumWriter.add_image("output image sample epoch_net1", testout1_im,global_step=epoch, dataformats='CHW')
            print("Training Finished, TotalEPOCH=%d" % EPOCH)

#net2

optimizer2 = optim.SGD(net2.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

if __name__ == "__main__":
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    best_acc2 = 70  #2 初始化best test accuracy
    print("Start Training, CNN_net0!")  # 定义遍历数据集的次数
    with open("acc2.txt", "w") as f:
        with open("log2.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net2.train()
                sum_loss2 = 0.0
                correct2 = 0.0
                total2 = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer2.zero_grad()

                    # forward + backward
                    outputs2 = net2(inputs)
                    loss2 = criterion(outputs2, labels)
                    loss2.backward()
                    optimizer2.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss2 += loss2.item()

                    #test_loss_all.append()

                    _, predicted2 = torch.max(outputs2.data, 1)
                    total2 += labels.size(0)
                    correct2 += predicted2.eq(labels.data).cpu().sum()
                    niter2 = i + 1 + epoch * length
                    train_acc2 = 100. * correct2 / total2
                    train_loss2 = sum_loss2 / (i+1)
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss2 / (i + 1), 100. * correct2 / total2))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss2 / (i + 1), 100. * correct2 / total2))
                    f2.write('\n')
                    f2.flush()
                    if niter2 % print_step == 0:
                        # 为日志添加损失函数
                        SumWriter.add_scalar("train_loss_net2", train_loss2 , niter2)
                        testinputs2_im = vutils.make_grid(inputs, nrow=12)  # (数据预处理？)
                        SumWriter.add_image("input image sample_net2", testinputs2_im, niter2)
                        testout2_im = vutils.make_grid(outputs2, nrow=128)
                        SumWriter.add_image("output image sample_net2", testout2_im, niter2)

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct2 = 0
                    total2 = 0
                    for data in testloader:
                        net2.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs2 = net2(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted2 = torch.max(outputs2.data, 1)
                        total2 += labels.size(0)
                        correct2 += (predicted2 == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct2 / total2))
                    acc2 = 100. * correct2 / total2
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net2.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d  |Accuracy= %.3f%%" % (epoch + 1,acc2))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc2 > best_acc2:
                        f3 = open("best_acc2.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc2))
                        f3.close()
                        best_acc2 = acc2
                SumWriter.add_scalars("train/test_acc_epoch_net2",{'train_acc':train_acc2,'test_acc':acc2},epoch)
                SumWriter.add_scalars("train_acc_epoch", {'net_2': train_acc2}, epoch)
                SumWriter.add_scalars("test_acc_epoch", {'net_2': acc2}, epoch)
                SumWriter.add_scalars("train_loss_epoch", {'net_2':train_loss2}, epoch)
                testout2_im = vutils.make_grid(outputs2, nrow=12)
                SumWriter.add_image("output image sample epoch_net2", testout2_im,global_step=epoch, dataformats='CHW')
            print("Training Finished, TotalEPOCH=%d" % EPOCH)

#net3
optimizer3 = optim.SGD(net3.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
if __name__ == "__main__":
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    best_acc3 = 70  #2 初始化best test accuracy
    print("Start Training, CNN_net3!")  # 定义遍历数据集的次数
    with open("acc3.txt", "w") as f_3:
        with open("log3.txt", "w")as f2_3:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net3.train()
                sum_loss3 = 0.0
                correct3 = 0.0
                total3 = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer3.zero_grad()

                    # forward + backward
                    outputs3 = net3(inputs)
                    loss3 = criterion(outputs3, labels)
                    loss3.backward()
                    optimizer3.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss3 += loss3.item()

                    #test_loss_all.append()

                    _, predicted3 = torch.max(outputs3.data, 1)
                    total3 += labels.size(0)
                    correct3 += predicted3.eq(labels.data).cpu().sum()
                    niter3 = i + 1 + epoch * length
                    train_acc3 = 100. * correct3 / total3
                    train_loss3 = sum_loss3 / (i+1)
                    print('[epoch:%d, iter:%d] Loss: %.03f_3 | Acc: %.3f_3%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss3 / (i + 1), 100. * correct3 / total3))
                    f2_3.write('%03d  %05d |Loss: %.03f_3 | Acc: %.3f_3%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss3 / (i + 1), 100. * correct3 / total3))
                    f2_3.write('\n')
                    f2_3.flush()
                    if niter3 % print_step == 0:
                        # 为日志添加损失函数
                        SumWriter.add_scalar("train_loss_net3", train_loss3 , niter3)
                        testinputs3_im = vutils.make_grid(inputs, nrow=12)  # (数据预处理？)
                        SumWriter.add_image("input image sample_net3", testinputs3_im, niter3)
                        testout3_im = vutils.make_grid(outputs3, nrow=128)
                        SumWriter.add_image("output image sample_net3", testout3_im, niter3)

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct3 = 0
                    total3 = 0
                    for data in testloader:
                        net3.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs3 = net3(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted3 = torch.max(outputs3.data, 1)
                        total3 += labels.size(0)
                        correct3 += (predicted3 == labels).sum()
                    print('测试分类准确率为：%.3f_3%%' % (100 * correct3 / total3))
                    acc3 = 100. * correct3 / total3
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net3.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f_3.write("EPOCH=%03d  |Accuracy= %.3f%%" % (epoch + 1,acc3))
                    f_3.write('\n')
                    f_3.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc3 > best_acc3:
                        f3_3 = open("best_acc3.txt", "w")
                        f3_3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc2))
                        f3_3.close()
                        best_acc3 = acc3
                SumWriter.add_scalars("train/test_acc_epoch_net3",{'train_acc':train_acc3,'test_acc':acc3},epoch)
                SumWriter.add_scalars("train_acc_epoch", {'net_3': train_acc3}, epoch)
                SumWriter.add_scalars("test_acc_epoch", {'net_3': acc3}, epoch)
                SumWriter.add_scalars("train_loss_epoch", {'net_3':train_loss3}, epoch)
                testout3_im = vutils.make_grid(outputs3, nrow=12)
                SumWriter.add_image("output image sample epoch_net3", testout3_im,global_step=epoch, dataformats='CHW')
            print("Training Finished, TotalEPOCH=%d" % EPOCH)


#

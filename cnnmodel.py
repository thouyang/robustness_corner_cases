import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, n_class):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 3*8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(24, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, n_class)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def train(self,TrainD,num_epochs=20, learning_rate=0.0002):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = self.cuda()
        loss_criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        total_step = len(TrainD)
        loss_list = []
        acc_list = []
        ### training the model
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(TrainD):
                # Run the forward pass
                outputs = model(images.to(device))

                # loss = lossf(outputs, labels.to(device),w.to(device)).cuda()
                loss = loss_criterion(outputs, labels.to(device))
                loss_list.append(loss.item())

                # Backprop and perform Adam optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the accuracy
                total = labels.size(0)
                _, predicted = torch.max(outputs.data.cpu(), 1)
                correct = (predicted == labels).sum().item()
                acc_list.append(correct / total)

                if (i + 1) % 10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                                  (correct / total) * 100))


import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import time
import copy

from torch.optim import lr_scheduler
from torchvision import datasets
from efficientnet_pytorch import EfficientNet
from torch.utils.tensorboard import SummaryWriter

batch_size = 16
epochs = 30
data_dir = r'C:\Users\user\Desktop\car_crush\091.차량 외관 영상 데이터\01.데이터\1.Training\brand/'
writer = SummaryWriter(r'C:\Users\user\Desktop\car_crush\091.차량 외관 영상 데이터\01.데이터\1.Training\brand')

image_size = 224

data_transforms = {'train': transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        transforms.Resize((image_size,image_size))
    ])}

image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms['train'])
                            for x in ['train','test']} 

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=10)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

since = time.time()

best_model_weights = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 10)

    for phase in ['train', 'test']:
        if phase == 'train':
            model.train()

        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        # writer.add_graph('epoch loss', epoch_loss, epoch)
        # writer.add_graph('epoch acc', epoch_acc, epoch)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        if phase == 'test' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            
            
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

torch.save(best_model_weights, r'C:\Users\user\Desktop\car_crush\091.차량 외관 영상 데이터\01.데이터\1.Training\brand\best_weights_b2_brand_9.pth')

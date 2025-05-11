
from neural_network import FamilyCnn
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader


def train_model(path,epo):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # randomly flip image
    transforms.RandomRotation(degrees=15),   # rotate within ±15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # color variation
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # randomly zoom in
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],       
                         std=[0.229, 0.224, 0.225])
])

    dataset=datasets.ImageFolder(path,transform=transform)
    loader=DataLoader(dataset,batch_size=8,shuffle=True)

    #img,label=dataset[0]
    #print(f"Image shape:{img.shape}")
    #print(f"label:{label}")

    idx_to_class={v:k for k,v in dataset.class_to_idx.items()}

    #train the model with cuda gpu
    device=torch.device("cuda" if torch.cuda.is_available()else "cpu")
    model=FamilyCnn(num_classes=len(dataset.class_to_idx)).to(device)
    criterian=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)

    epoch=epo
    for epochs in range(epoch):
        running_loss=0
        correct=0
        total=0
        for images,label in loader:
        #  print(images.shape)

            images,label=images.to(device),label.to(device)
            optimizer.zero_grad()

            output=model(images)
            loss=criterian(output,label)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()
            _,predicted=torch.max(output,1)
            total+=label.size(0)
            correct+=(predicted==label).sum().item()

        epoch_loss=running_loss/len(loader)
        accuracy=100*correct/total

        print(f"Epoch[{epochs+1}/{epoch}],loss:{epoch_loss},Accuracy:{accuracy}")
        #print("Training completed")

        torch.save(model.state_dict(), "/content/drive/MyDrive/Soul vision ai trained data/family_cnn.pth")
        torch.save(idx_to_class, "/content/drive/MyDrive/Soul vision ai trained data/idx_to_class.pth")

    print("Training completed")

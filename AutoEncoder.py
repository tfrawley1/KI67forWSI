import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader 
from torchvision.utils import save_image 

num_epochs = 5


# Define the autoencoder neural network 
class Autoencoder(nn.Module):
    def __init__(self): 
        super(Autoencoder, self).__init__() 
        self.encoder = nn.Sequential( 
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            ) 
        self.decoder = nn.Sequential( 
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(), 
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid() 
            ) 
        
    def forward(self, x): 
        x = self.encoder(x) 
        x = self.decoder(x) 
        return x
    
    def encode(self, x):
        x = self.encoder(x)
        return x
        
# Define the training loop 
def train(model, dataloader, criterion, optimizer, device): 
    model.train() 
    train_loss = 0 
    for i, (images, labels) in enumerate(dataloader): 
        images = images.to(device) 

        outputs = model(images) 
        loss = criterion(outputs, images) 

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 
        train_loss += loss.item() 
    
    return train_loss / len(dataloader.dataset) 
    
# Define the testing loop 
def test(model, dataloader, criterion, device): 
    model.eval() 
    test_loss = 0 
    with torch.no_grad(): 
        for i, data in enumerate(dataloader): 
            inputs, _ = data 
            inputs = inputs.to(device) 
            outputs = model(inputs) 
            loss = criterion(outputs, inputs) 
            test_loss += loss.item() 
    return test_loss / len(dataloader.dataset) 
        
# Set the device (CPU or GPU) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Load the CIFAR10 dataset 
transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) 
    ]) 

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) 
trainloader = DataLoader(trainset, batch_size=128, shuffle=True) 
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) 
testloader = DataLoader(testset, batch_size=128, shuffle=False) 

# Create the model, loss function, and optimizer 
model = Autoencoder().to(device) 
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

# Train the model 
for epoch in range(num_epochs): 
    train_loss = train(model, trainloader, criterion, optimizer, device)
    test_loss = test(model, testloader, criterion, device) 
    print(f'Epoch: {epoch+1}, train_loss = {train_loss}, test_loss = {test_loss}')
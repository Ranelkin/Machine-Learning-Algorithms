import os
import torch
from torch import nn
import torch.utils.data 
from torchvision import datasets, transforms
import logging


logger = logging.getLogger("nn")
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
logger.info(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() 
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        
        return self.linear_relu_stack(x)
 
def random_split(): 
    pass
    transform = transforms.Compose(
        [
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])
        ]
    )
    
    training_data = datasets.MNIST(root="data", download=True, train=True, transform=transform)
    training_data, val_data = torch.utils.data.random_split(
        training_data, [50000, 10000]
    )
    
    test_data = datasets.MNIST(root="data", download=True, train=False, transform=transform)
    
    
class Train: 
    def __init__(self, device, model: NeuralNetwork, loss, data, **kwargs) -> None:
        self.device = device 
        self.model = model
        self.loss = loss 
        self.trainset = data['train']
        self.valset = data['val']
        self.model.to(device)

        
        self.defaults = {
            'batch_size': 128, 
            'num_train_samples': 10000, 
            'num_val_samples': None, 
            'optim': 'SGD', 
            'optim_params': {
                'lr': 0.01,
                'momentum': 0.9, 
                'weight_decay': 5e-4
            }, 
            'print_every': 1, 
            'scheduler': 'MultiStepLR', 
            'scheduler_params': {
                'milestones': [20, 40, 60, 80, 100]
                
            },
            'verbose': False         
        }
        
        for key, value in self.defaults.items(): 
            self.__dict__[key] = kwargs.pop(key, value)
        
        self.optimizer = (
            getattr(torch.optim, self.optim)(
            model.parameters(), **self.optim_params
            )
        )
        
        self.scheduler = (
            getattr(torch.optim.lr_scheduler, self.scheduler)(
                self.optimizer, **self.scheduler_params 
            )
        )
        self.epoch = 0
        self.num_epochs = 0
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        
    def train(self, num_epochs=10): 
        
        self.num_epochs += num_epochs
        
        trainloader = torch.utils.data.DataLoader(
            self.trainset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
    
        best_val_acc = 0
        best_params = None 
        
        for epoch in range(num_epochs): 
            self.epoch += 1
            loss_history = torch.empty(len(trainloader))
            
            self.model.train()
            
            for i, (inputs, labels) in enumerate(trainloader): 
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                
                loss = self.loss(outputs, labels)
                loss.backward()
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                loss_history[i] = loss.item()
        
        train_acc = self.evaluate(
            self.trainset,
            num_samples=self.num_train_samples
        )
        self.train_acc_history.append(train_acc)
        
        train_loss = torch.mean(loss_history).item()
        self.train_loss_history.append(train_loss)
        
        val_acc, val_loss = self.evaluate(
            self.valset,
            num_samples=self.num_val_samples,
            compute_loss=True
        )
        self.val_acc_history.append(val_acc)
        self.val_loss_history.append(val_loss)
        
        self.scheduler.step()
        
        if val_acc > best_val_acc: 
            best_params = self.model.state_dict().copy()
            best_val_acc = val_acc
            
        if self.verbose and epoch % self.print_every == 0: 
            print(
                f'Epoch: {self.epoch:3}/{self.num_epochs}', 
                f'Train loss: {self.train_loss_history[-1]:.5f}', 
                f'Val loss: {self.val_loss_history[-1]:.5f}',
                f'Train accuracy: {train_acc:.1f} % ', 
                f'Val accuracy: {val_acc:.1f} %'
            )
        self.model.load_state_dict(best_params)
        
        return {
            'train_loss': self.train_loss_history,
            'train_accuracy': self.train_acc_history, 
            'val_loss': self.val_loss_history, 
            'val_accuracy': self.val_acc_history
        }
    def evaluate(self, dataset, num_samples=None, compute_loss=False): 
        
        if num_samples is not None and num_samples < len(dataset): 
            dataset, _ = torch.utils.data.random_split(
                dataset, 
                [num_samples, len(dataset) - num_samples]
            )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=2,
        )
        total, correct = 0, 0 
        
        if compute_loss: 
            loss_history = torch.empty(len(dataloader))
        
        self.model.eval()
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader): 
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                
                predicted = torch.argmax(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
        if compute_loss: 
            loss_history[i] = self.loss(outputs, labels).item()
        
        accuracy = 100 * correct/total
        
        if compute_loss: 
            loss = torch.mean(loss_history).item()
            
            return accuracy, loss

        return accuracy
    
    
if __name__ == '__main__': 
    PATH = './model'
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])
    ])
    
    train_data = datasets.MNIST(
        root='./data/model', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    train_data, val_data = torch.utils.data.random_split(
        train_data, [50000, 10000]
    )
    
    test_data = datasets.MNIST(
        root='./datasets', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    
    learning_rates = [5e-2, 1e-2, 5e-3, 1e-3, 1e-4]
    
    best_model = None 
    best_lr = None 
    best_val_acc = 0
    
    num_epochs = 20 
    
    for lr in learning_rates:
        state_dict = torch.load(PATH, weights_only=False)
        model = NeuralNetwork()  # Create a new model instance
        if state_dict is not None:
            model.load_state_dict(state_dict)  # Load weights into the model
            
        trainer = Train(
            device,
            model, 
            data = {
                'train': test_data, 
                'val': val_data
            }, 
            loss= nn.CrossEntropyLoss(),
            optim='SGD',
            optim_params={
                'lr': lr, 
                'momentum': 0.9, 
                'weight_decay': 5e-4
            },
            verbose=True
        )
        history= trainer.train(num_epochs)
        
        val_acc = torch.max(torch.tensor(history['val_accuracy'])).item()
        
        if val_acc > best_val_acc: 
            
            best_model = model 
            best_lr = lr
            best_val_acc = val_acc
        torch.save(best_model.state_dict(), PATH)
    print(f'Best learning rate: {best_lr}')
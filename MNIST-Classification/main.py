import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import MNIST
from model import LeNet5, CustomMLP

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """
    trn_loss = 0
    acc = 0
    
    for i, (data, label) in tqdm(enumerate(trn_loader), total=len(trn_loader)):
        data = data.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, label)
        trn_loss += loss.data
        accuracy = output.data.max(1)[1].eq(label.data).sum()/len(data) * 100
        acc += accuracy.data
        
        loss.backward()
        optimizer.step()
        
    trn_loss = trn_loss / len(trn_loader)
    acc = acc / len(trn_loader)
            
    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    tst_loss = 0
    acc = 0
    
    for i, (data, label) in tqdm(enumerate(tst_loader), total=len(tst_loader)):
        data = data.to(device)
        label = label.to(device)
        
        with torch.no_grad():    
            output = model(data)
        loss = criterion(output, label)
        tst_loss += loss.data
        accuracy = output.data.max(1)[1].eq(label.data).sum()/len(data) * 100
        acc += accuracy.data
        
    tst_loss = tst_loss / len(tst_loader)
    acc = acc / len(tst_loader)
        
    return tst_loss, acc


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    # hyperparameter
    epochs = 30
    batch_size = 64
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print('>> Loading Dataset & Dataloader ...')
    train_dataset = MNIST('/data/MNIST/data/train')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4)
    test_dataset = MNIST('/data/MNIST/data/test')
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4)
    
    print('>> Loading Model ...')
    lenet = LeNet5().to(device)
    mlp = CustomMLP().to(device)
    
    print('>> Loading Optimizer & Criterion ...')
    opt_lenet = torch.optim.AdamW(lenet.parameters(), lr=learning_rate, weight_decay=0.01)
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    print('>> Training & Testing LeNet-5 ...')
    # train & test lenet
    total_lenet_trn_loss = []
    total_lenet_trn_acc = []
    total_lenet_tst_loss = []
    total_lenet_tst_acc = []
    
    # Early stopping config
    min_val_loss = 100_000
    patient = 0
    
    for epoch in range(epochs):
        lenet_trn_loss, lenet_trn_acc = train(model=lenet, trn_loader=train_dataloader, criterion=criterion, device=device, optimizer=opt_lenet)
        total_lenet_trn_loss.append(lenet_trn_loss.item())
        total_lenet_trn_acc.append(lenet_trn_acc.item())
        print(f'{epoch+1} Epoch train loss: ', lenet_trn_loss.item())
        print(f'{epoch+1} Epoch train accuracy: ', lenet_trn_acc.item())
        lenet_tst_loss, lenet_tst_acc = test(model=lenet, tst_loader=test_dataloader, criterion=criterion, device=device)
        
        # for early stopping
        if min_val_loss > lenet_tst_loss:
            min_val_loss = lenet_tst_loss
        else:
            if patient > 5:
                print('Early stopping: Training ENDs')
                break
            else:
                patient += 1
                print(f'Early stopping: Patient {patient} ({5-patient} left)')
                
        total_lenet_tst_loss.append(lenet_tst_loss.item())
        total_lenet_tst_acc.append(lenet_tst_acc.item())
        print(f'{epoch+1} Epoch test loss: ', lenet_tst_loss.item())
        print(f'{epoch+1} Epoch test accuracy: ', lenet_tst_acc.item())
        
    print('>> Training & Testing CustomMLP ...')
    # train & test mlp
    total_mlp_trn_loss = []
    total_mlp_trn_acc = []
    total_mlp_tst_loss = []
    total_mlp_tst_acc = []
    
    for epoch in range(epochs):
        mlp_trn_loss, mlp_trn_acc = train(model=mlp, trn_loader=train_dataloader, criterion=criterion, device=device, optimizer=opt_mlp)
        total_mlp_trn_loss.append(mlp_trn_loss.item())
        total_mlp_trn_acc.append(mlp_trn_acc.item())
        print(f'{epoch+1} Epoch train loss: ', mlp_trn_loss.item())
        print(f'{epoch+1} Epoch train accuracy: ', mlp_trn_acc.item())
        mlp_tst_loss, mlp_tst_acc = test(model=mlp, tst_loader=test_dataloader, criterion=criterion, device=device)
        total_mlp_tst_loss.append(mlp_tst_loss.item())
        total_mlp_tst_acc.append(mlp_tst_acc.item())
        print(f'{epoch+1} Epoch test loss: ', mlp_tst_loss.item())
        print(f'{epoch+1} Epoch test accuracy: ', mlp_tst_acc.item())
        
    print('>> Plotting Loss & Accuracy')
    # plot: lenet
    plt.plot(range(len(total_lenet_trn_loss)), total_lenet_trn_loss, marker='o', label='LeNet-5 train')
    plt.plot(range(len(total_lenet_tst_loss)), total_lenet_tst_loss, marker='o', label='LeNet-5 test')
    plt.plot(range(len(total_mlp_trn_loss)), total_mlp_trn_loss, marker='o', label='CustomMLP train')
    plt.plot(range(len(total_mlp_tst_loss)), total_mlp_tst_loss, marker='o', label='CustomMLP test')
    plt.title('Loss curve', pad=20)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('./assets/loss_curve.png')
    plt.clf()
    
    plt.plot(range(len(total_lenet_trn_acc)), total_lenet_trn_acc, marker='o', label='LeNet-5 train')
    plt.plot(range(len(total_lenet_tst_acc)), total_lenet_tst_acc, marker='o', label='LeNet-5 test')
    plt.plot(range(len(total_mlp_trn_acc)), total_mlp_trn_acc, marker='o', label='CustomMLP train')
    plt.plot(range(len(total_mlp_tst_acc)), total_mlp_tst_acc, marker='o', label='CustomMLP test')
    plt.title('Accuracy', pad=20)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('./assets/acc_curve.png')
    
    print('>> Done')

if __name__ == '__main__':
    main()
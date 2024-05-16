import os
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler

from dataset import Shakespeare
from model import CharRNN, CharLSTM
from sample import generate

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

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
    """
    trn_loss = 0
    
    for i, (data, label) in tqdm(enumerate(trn_loader), total=len(trn_loader)):
        data = data.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        
        hidden = model.init_hidden(data.shape[0]).to(device)
        output, _ = model(data, hidden)
        loss = criterion(output.view(-1, output.shape[-1]), label.view(-1))
        trn_loss += loss.data
        
        loss.backward()
        optimizer.step()
        
    trn_loss = trn_loss / len(trn_loader)
            
    return trn_loss

def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    val_loss = 0
    
    for i, (data, label) in tqdm(enumerate(val_loader), total=len(val_loader)):
        data = data.to(device)
        label = label.to(device)
        
        with torch.no_grad():    
            hidden = model.init_hidden(data.shape[0]).to(device)
            output, _ = model(data, hidden)
        loss = criterion(output.view(-1, output.shape[-1]), label.view(-1))
        val_loss += loss.data
        
    val_loss = val_loss / len(val_loader)
        
    return val_loss

def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    # cofig
    epochs = 30
    batch_size = 128
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # dataset & dataloader
    print('> > Loading Dataset & Dataloader ...')
    dataset = Shakespeare('./shakespeare_train.txt')
    
    num_data = len(dataset)     
    indices = list(range(num_data))
    np.random.shuffle(indices)
    split = int(np.floor(0.1 * num_data))
    trn_idx, val_idx = indices[split:], indices[:split]
    
    trn_sampler = SubsetRandomSampler(trn_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    trn_loader = torch.utils.data.DataLoader(dataset, 
                                               batch_size=batch_size,
                                               shuffle=False,
                                               sampler=trn_sampler,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size,
                                            shuffle=False,
                                            sampler=val_sampler,
                                            num_workers=0)
    
    # model
    print('> > Loading Model ...')
    rnn = CharRNN().to(device)
    lstm = CharLSTM().to(device)
    
    print('> > Loading Optimizer & Criterion ...')
    opt_rnn = torch.optim.AdamW(rnn.parameters(), lr=learning_rate, weight_decay=0.1)
    opt_lstm = torch.optim.AdamW(lstm.parameters(), lr=learning_rate, weight_decay=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    
    print('> > Training & Testing RNN ...')
    total_rnn_trn_loss = []
    total_rnn_val_loss = []    
    
    min_val_loss = 100_000
    for epoch in range(epochs):
        trn_loss = train(model=rnn, 
                         trn_loader=trn_loader, 
                         criterion=criterion, 
                         device=device, 
                         optimizer=opt_rnn)
        val_loss = validate(model=rnn, 
                         val_loader=val_loader, 
                         criterion=criterion, 
                         device=device)
        # for early stopping
        if min_val_loss > val_loss and epoch % 5 == 0:
            min_val_loss = val_loss
            
            if not os.path.exists('./ckpt'):
                os.makedirs('./ckpt')
            torch.save(rnn, f'./ckpt/{epoch}_loss{val_loss}_rnn.ckpt')

        total_rnn_trn_loss.append(trn_loss.item())
        total_rnn_val_loss.append(val_loss.item())
        print(f'{epoch+1} Epoch Train loss: ', trn_loss.item())
        print(f'{epoch+1} Epoch Val loss: ', val_loss.item())

    print('> > RNN Sampling')
    model_ckpt = [i for i in os.listdir('ckpt') if 'rnn' in i][-1]
    model = torch.load(os.path.join('./ckpt', model_ckpt), map_location='cpu')
    seeds = ["s",
             "I",
             "A",
             "t",
             "y"]
    temperature = [10.0, 1.0, 0.1]
    for seed in seeds:
        for temp in temperature:
            print(f"------------------------------------ Temperature {temp}--------------------------------")
            print(f"Input: {seed}")
            print(f"Output: {generate(model, seed, temp, dataset.vocab, 150)}")
            print("----------------------------------------------------------------------------------------")
        print("")
            
    print('> > Training & Testing LSTM ...')
    total_lstm_trn_loss = []
    total_lstm_val_loss = []    
    
    min_val_loss = 100_000
    
    for epoch in range(epochs):
        trn_loss = train(model=lstm, 
                         trn_loader=trn_loader, 
                         criterion=criterion, 
                         device=device, 
                         optimizer=opt_lstm)
        val_loss = validate(model=lstm, 
                         val_loader=val_loader, 
                         criterion=criterion, 
                         device=device)
        # for early stopping
        if min_val_loss > val_loss and epoch % 5 == 0:
            min_val_loss = val_loss
            torch.save(lstm, f'./ckpt/{epoch}_loss{val_loss}_lstm.ckpt')

        total_lstm_trn_loss.append(trn_loss.item())
        total_lstm_val_loss.append(val_loss.item())
        print(f'{epoch+1} Epoch Train loss: ', trn_loss.item())
        print(f'{epoch+1} Epoch Val loss: ', val_loss.item())
    
    print('> > LSTM Sampling')
    model_ckpt = [i for i in os.listdir('ckpt') if 'lstm' in i][-1]
    model = torch.load(os.path.join('./ckpt', model_ckpt), map_location='cpu')
    seeds = ["s",
             "I",
             "A",
             "t",
             "y"]
    temperature = [10.0, 1.0, 0.1]
    for seed in seeds:
        for temp in temperature:
            print(f"------------------------------------ Temperature {temp}--------------------------------")
            print(f"Input: {seed}")
            print(f"Output: {generate(model, seed, temp, dataset.vocab, 150)}")
            print("----------------------------------------------------------------------------------------")
        print("")
            
    print('>> Plotting Loss')
    plt.plot(range(len(total_rnn_trn_loss)), total_rnn_trn_loss, marker='o', label='RNN train')
    plt.plot(range(len(total_rnn_val_loss)), total_rnn_val_loss, marker='o', label='RNN validation')
    plt.plot(range(len(total_lstm_trn_loss)), total_lstm_trn_loss, marker='o', label='LSTM train')
    plt.plot(range(len(total_lstm_val_loss)), total_lstm_val_loss, marker='o', label='LSTM validation')
    plt.title('Loss curve', pad=20)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('./assets/loss_curve.png')
    plt.clf()

def fix_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


if __name__ == '__main__':
    fix_seed(42)
    main()

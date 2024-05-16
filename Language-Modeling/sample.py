import torch
import torch.nn as nn
from tqdm import tqdm
from model import CharRNN, CharLSTM

def generate(model, seed_characters, temperature, *args):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
        temperature: T
        args: other arguments if needed

    Returns:
        samples: generated characters
    """
    model.eval()
    model.cuda()
    
    vocab = args[0]
    length = args[1]
    
    idx_to_char = {i: ch for ch, i in vocab.items()}
    hidden = model.init_hidden(1).cuda()
    
    for i in range(length):
        input = torch.tensor([[vocab[seed_characters[-1]]]])
        input = input.cuda()
        if isinstance(model, CharRNN):
            output, hidden = model(input, hidden)
        else:
            if i == 0:
                h = hidden
                output, (h, c) = model(input, h)
            else:
                output, (h, c) = model(input, h)
        
        try:
            output = output.div(temperature).exp().cpu()
            char_idx = torch.multinomial(output.view(-1, output.shape[-1]), 1)[0]
            char = idx_to_char[char_idx.item()]
            seed_characters += char
        except:
            print(f'ERROR - (seed, temp) - ({seed_characters}, {temperature})')
            break
    samples = seed_characters
    return samples

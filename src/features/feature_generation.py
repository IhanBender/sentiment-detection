import torch
from torch.autograd import Variable
from unidecode import unidecode

def char_tensor(string):
    string = unidecode(string)
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = ord(string[c])
        except:
            print(c)
            raise
    return Variable(tensor)
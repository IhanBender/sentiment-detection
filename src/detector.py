import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "features/"))
from feature_generation import char_tensor
sys.path.append(os.path.join(os.path.dirname(__file__), "models/"))
from net import Net

class Detector():
    def __init__(self):
        self.net = Net()
        if torch.cuda.is_available():
            self.net = self.net.cuda()
            self.net.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "../data/models/net_9.pth")))
        else:
            self.net.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "../data/models/net_9.pth")), map_location="cpu")
            self.net.cpu()
            
        self.net.eval()

    def _get_score(self, prediction):
        hiscore, higrade = 1, 0
        for score, e in enumerate(list(prediction.detach()[0])):
            if e.item() > higrade:
                hiscore = score+1
                higrade = e.item()

        if higrade == 0:
            return -1
        return hiscore

    def evaluate_text(self, text):
        tensor = char_tensor(text)
        if torch.cuda.is_avaiable():
            tensor = tensor.cuda()
        pred = self.net(tensor)
        score = self._get_score(pred)
        
        if score == -1:
            return "Not conclusive sentiment detected."
        elif score <= 3:
            return "Negative sentiment detected."
        else:
            return "Positive sentiment detected."
        

def main():
    detector = Detector()
    print("Write your text to find out if it has positive or negative sentiment (type empty text to quit)")
    while True:
        input_text = input()
        if input_text == "":
            break
        print(self.detector.evaluate_text(input_text))


if __name__ == '__main__':
    main()

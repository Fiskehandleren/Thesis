import torch.nn as nn
import argparse


class AR_Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(AR_Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.fc3 = nn.Linear(hidden_dim, hidden_dim*2) 

        # Non-linearity
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim*2, output_dim) 
        # Linear function (readout)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.fc2(out)
        return out
    
    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=256)
        parser.add_argument("--output_dim", type=int, default=8)
        return parser
    
class AR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AR, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim) 
        print("AR model created with input_dim: ", input_dim, " and output_dim: ", output_dim)
    def forward(self, x):
        out = self.fc1(x)
        return out

class AR_Net_multi(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(AR_Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.fc3 = nn.Linear(hidden_dim, hidden_dim*2) 

        # Non-linearity
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim*2, output_dim) 
        # Linear function (readout)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.fc2(out)
        return out
    
    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=256)
        parser.add_argument("--output_dim", type=int, default=8)
        return parser
    
class AR_multi(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AR, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim) 
        print("AR model created with input_dim: ", input_dim, " and output_dim: ", output_dim)
    def forward(self, x):
        out = self.fc1(x)
        return out
    
    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--output_dim", type=int, default=8)
        return parser
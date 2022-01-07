import torch
import torch.nn as nn
from ConvLSTM2D import ConvLSTM2D

class ConvLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size, device):

        super().__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.frame_size = frame_size
        self.device = device
                
        self.convLSTMcell1 = ConvLSTM2D(in_channels, out_channels, 
        kernel_size, padding, activation, frame_size)
    
    def forward(self, X):

        # X is a frame sequence (batch_size, num_channels, seq_len, height, width) # Before
        # X is now -> (batch_size, seq_len, height * width * 3) # Update

        # Get the dimensions
     
        height, width = self.frame_size
        
        batch_size, _, seq_len, _, _ = X.size()
       
        device = self.device
        
        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len, 
        height, width, dtype=torch.float, device=device)
        
        # Initialize Hidden State
        H =  torch.zeros(batch_size, self.out_channels, 
        height, width, dtype=torch.float, device=device)

        # Initialize Cell Input
        C = torch.zeros(batch_size,self.out_channels, 
        height, width, dtype=torch.float, device=device)
        
        
        # Unroll over time steps
        for time_step in range(seq_len):
            
            
            H, C = self.convLSTMcell1(X[:,:, time_step], H, C)    
            output[:,:, time_step] = H
       
        return output

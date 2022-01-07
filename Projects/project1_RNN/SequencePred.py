import torch
import torch.nn as nn

from ConvLSTM import ConvLSTM

class Seq2Seq(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding, 
    activation, frame_size, num_layers, device):

        super().__init__()
        
        self.frame_size = frame_size
        
        self.future_step = 5
        self.device = device
        self.convlstm1 = ConvLSTM(in_channels=num_channels, out_channels=num_kernels, 
                                  kernel_size=kernel_size, padding=padding, 
                                  activation=activation, frame_size=frame_size, device=device)
        
        self.batchnorm = nn.BatchNorm3d(num_features=num_kernels, dtype=torch.float)
        
        
        self.convlstm2 = ConvLSTM(in_channels=num_kernels, out_channels=num_kernels*2, 
                                  kernel_size=kernel_size, padding=padding, 
                                  activation=activation, frame_size=frame_size, device=device)  
        
        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(in_channels=num_kernels*2, out_channels=num_channels,kernel_size=(3, 3), padding=(1, 1),
                             dtype=torch.float)
        
    def forward(self, X):

        # Forward propagation through all the layers
        h, w = self.frame_size
        device = self.device
        
        batch = X.size(0) 
        X = X.view(batch, 3, 5, h, w) #.float() #.to(dev).float()
        
       # print("X dev", X.get_device())
        outputs = torch.zeros(batch, 3, 5, h, w)#, dtype=torch.float) #, device=device)
        
        for frame in range(self.future_step):
            output = self.convlstm1(X)
            output = self.convlstm2(output)    
            
            output = self.conv(output[:, :, -1])
            out = output
            X = torch.cat([X, output[:, :, None]], dim=2)
            outputs[:, :, frame] = out
            
            
        return nn.Sigmoid()(outputs.view(batch, -1, h, w).double())
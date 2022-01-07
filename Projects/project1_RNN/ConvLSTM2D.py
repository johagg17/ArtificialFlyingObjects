import torch
import torch.nn as nn

class ConvLSTM2D(nn.Module):

    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super().__init__()  

        self.activation = torch.tanh 
        
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels, 
            out_channels=4 * out_channels, 
            kernel_size=kernel_size, 
            padding=padding, dtype=torch.float)           

        self.W_ci = nn.Parameter(torch.zeros(out_channels, *frame_size, dtype=torch.float))
        self.W_co = nn.Parameter(torch.zeros(out_channels, *frame_size, dtype=torch.float))
        self.W_cf = nn.Parameter(torch.zeros(out_channels, *frame_size, dtype=torch.float))

    def forward(self, X, H_prev, C_prev):
        
        
        
        print(H_prev.get_device())
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))
        
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)
        
        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev )
        
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev )
        
        # Current Cell output
        C = forget_gate*C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C )

        # Current Hidden State
        H = output_gate * self.activation(C)
        
      #  print("New Hidden state H is computed")

        return H, C
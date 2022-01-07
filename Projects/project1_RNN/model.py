import pytorch_lightning as pl
import torch.nn as nn
import torchvision

from torch.utils.tensorboard import SummaryWriter

class NextFiveFramePredictor(pl.LightningModule): 
    
    def __init__(self, model, optimizer, criterion, **hparams):
        
        super().__init__()
        
        self.model = model
        self.opt = optimizer['type'](model.parameters(), **optimizer["args"])
        
        self.loss = criterion
   
        
    def forward(self, x):
        #X = X.view(batch, 3, 5, h, w).float()
        
        return self.model(x)
        
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        dev = y.get_device()
        
        output = self.forward(x)
       
        y = y.float()
        output = output.to(dev).float()
        
        
        loss_ = self.loss(output, y)
        
        self.log('Loss_Train',loss_)
               
        return {'loss': loss_}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        dev = y.get_device()
        
        output = self.forward(x)
      
        
        y = y.float()
        output = output.to(dev).float()
        
        loss_ = self.loss(output, y)
        
        self.log('Loss_Validation',loss_)
        
        return {'loss': loss_}
    
    def test_step(self, batch, batch_idx):
        """

        :param batch: dict:
        :param batch_idx: int:

        """
        x, y = batch
        
       # print(batch_idx)
        dev = y.get_device()
        
        preds = self.forward(x).to(dev)
        
        

        loss_ = self.loss(preds,y)
          
        self.log('loss_Test',loss_)
        
        return {'loss_Test':loss_, 'preds':preds}
    
    def configure_optimizers(self):
        """ """
        # Note: dont use list if only one item.. Causes silent crashes
        #optimizer = torch.optim.Adam(self.model.parameters())
        return self.opt
import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=(1,1), 
                 dilation=1, activation=nn.ReLU, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            dilation=dilation,
            padding=(kernel_size[0]//2, kernel_size[1]//2),
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.bn(x)
        return x

class Conv2dResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=(kernel_size//2)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),        
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size//2)),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.residual_block(x)
        out = self.relu(out+x)
        return out

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

class ECAPA_TDNN(nn.Module):

    def __init__(self, C=512):

        super(ECAPA_TDNN, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug()
        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug=False):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

class ECAPA_emAndco(nn.Module):

    def __init__(self, C=512):

        super(ECAPA_emAndco, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 384)
        self.bn6 = nn.BatchNorm1d(192)
        self.fc7 = nn.Linear(192, 4)

    def forward(self, x, aug=False):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x1 = x[:,:192]
        x2 = x[:,192:]
        x1 = self.bn6(x1)
        x2 = self.bn6(x2)
        x2 = self.fc7(x2)

        return x1, x2
    
class ECAPA_CNN_TDNN(nn.Module):
    def __init__(self, C=512):
        super(ECAPA_CNN_TDNN, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug()

        # CNN
        self.cnn_block1 = Conv2dBlock(1, 128, kernel_size=(3,3), stride=(2,1))
        self.cnn_res_block1 = Conv2dResBlock(128, 128, 128)
        self.cnn_res_block2 = Conv2dResBlock(128, 128, 128)
        self.cnn_block2 = Conv2dBlock(128, 128, kernel_size=(3,3), stride=(2,1))

        # ECAPA_TDNN
        self.conv1  = nn.Conv1d(128*20, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x, aug=False):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug:
                x = self.specaug(x)

        x = x.unsqueeze(1) 
        x = self.cnn_block1(x)
        x = self.cnn_res_block1(x)
        x = self.cnn_res_block2(x)
        x = self.cnn_block2(x)

        x = rearrange(x, 'b c h w -> b (c h) w')

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

class CRNN(nn.Module):
    def __init__(self, num_classes=6, input_channels=1):
        super(CRNN, self).__init__()
        
        self.zero_pad = nn.ZeroPad2d((0, 0, 0, 0))
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3))
        
        self.conv3 = nn.Conv2d(32, 128, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3))
        
        self.dropout = nn.Dropout(0.5)
        
        # 400 -> 398 (conv1) -> 396 (conv2) -> 132 (pool1) -> 
        # 130 (conv3) -> 128 (conv4) -> 42 (pool2)
        # output: (batch_size, 64, 42, 20)
        # input of LSTM 64 * 20 = 1280
        self.lstm = nn.LSTM(input_size=1280, hidden_size=40, batch_first=True)
        
        self.pool1d = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(840, num_classes)  
        
    def forward(self, x):
        x = self.zero_pad(x)  # (batch_size, 1, 400, 201)
        
        # CNN layers
        x = self.conv1(x)     # (batch_size, 64, 398, 199)
        x = self.relu(x)
        x = self.conv2(x)     # (batch_size, 32, 396, 197)
        x = self.relu(x)
        x = self.pool1(x)     # (batch_size, 32, 132, 65)
        
        x = self.conv3(x)     # (batch_size, 128, 130, 63)
        x = self.relu(x)
        x = self.conv4(x)     # (batch_size, 64, 128, 61)
        x = self.relu(x)
        x = self.pool2(x)     # (batch_size, 64, 42, 20)
        
        x = self.dropout(x)
        
        # Prepare for LSTM
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)
        
        # LSTM layer
        x, _ = self.lstm(x)   # (batch_size, 42, 40)
        
        # Final processing
        x = x.transpose(1, 2) # (batch_size, 40, 42)
        x = self.pool1d(x)    # (batch_size, 40, 21)
        x = self.flatten(x)   # (batch_size, 840)
        x = self.fc(x)        # (batch_size, num_classes)
        
        return x

class CRNN_feature(nn.Module):
    def __init__(self, num_classes=192, input_channels=1):
        super(CRNN_feature, self).__init__()
        
        self.zero_pad = nn.ZeroPad2d((0, 0, 0, 0))
        self.relu = nn.ReLU(inplace=True)
        

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3))
        
        self.conv3 = nn.Conv2d(32, 128, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3))
        
        self.dropout = nn.Dropout(0.5)
        
        self.lstm = nn.LSTM(input_size=1280, hidden_size=40, batch_first=True)

        self.pool1d = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(840, num_classes)
        
    def forward(self, x):
        x = self.zero_pad(x)  # (batch_size, 1, 400, 201)
        
        x = self.conv1(x)     # (batch_size, 64, 398, 199)
        x = self.relu(x)
        x = self.conv2(x)     # (batch_size, 32, 396, 197)
        x = self.relu(x)
        x = self.pool1(x)     # (batch_size, 32, 132, 65)
        
        x = self.conv3(x)     # (batch_size, 128, 130, 63)
        x = self.relu(x)
        x = self.conv4(x)     # (batch_size, 64, 128, 61)
        x = self.relu(x)
        x = self.pool2(x)     # (batch_size, 64, 42, 20)
        
        x = self.dropout(x)
        
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)
        
        x, _ = self.lstm(x)   # (batch_size, 42, 40)
        
        x = x.transpose(1, 2) # (batch_size, 40, 42)
        x = self.pool1d(x)    # (batch_size, 40, 21)
        x = self.flatten(x)   # (batch_size, 840)
        x = self.fc1(x)        # (batch_size, num_classes)

        return x

def initialize_model(num_classes, input_channels=1):
    model = CRNN(num_classes=num_classes, input_channels=input_channels).to('cuda:0')
    
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(weights_init)
    return model


if __name__ == "__main__":
    model = initialize_model(num_classes=6, input_channels=1)
    
    sample_input = torch.randn(2, 1, 400, 201).to('cuda:0')
    
    output = model(sample_input)
    print(f"Output shape: {output.shape}")
import torch
import torch.nn as nn
import time

class RecurrentProp(nn.Module):

    def __init__(self, input_size):
        super(RecurrentProp, self).__init__()
        self.input_size = input_size

        self.weight = torch.nn.Parameter(torch.eye(self.input_size))

    def forward(self, data):
        N, C, H, W = data.shape
        assert C == self.input_size, "%d vs %d" % (C, self.input_size)
        previous = torch.zeros((self.input_size))
        # left to right
        output0 = torch.tensor(data)
        for x in range(1, W):
            data1 = output0[:, :, :, x-1]
            # N, C, H --> N, H, C
            data1 = torch.transpose(data1, 1, 2).contiguous()
            data1 = data1.view(-1, self.input_size)
            tmp = torch.mm(data1, self.weight)
            tmp = torch.reshape(tmp, (N, H, C))
            # N, H, C --> N, C, H
            tmp = torch.transpose(tmp, 1, 2).contiguous()
            output0[:, :, :, x] += tmp
        
        # right to left
        output1 = torch.tensor(data)
        for x in range(W-2, -1, -1):
            data1 = output1[:, :, :, x+1]
            # N, C, H --> N, H, C
            data1 = torch.transpose(data1, 1, 2).contiguous()
            data1 = data1.view(-1, self.input_size)
            tmp = torch.mm(data1, self.weight)
            tmp = torch.reshape(tmp, (N, H, C))
            # N, H, C --> N, C, H
            tmp = torch.transpose(tmp, 1, 2).contiguous()
            output1[:, :, :, x] += tmp
            
        # top down
        output2 = torch.tensor(data)
        for y in range(1, H):
            # data of previous row
            data1 = output2[:, :, y-1, :]
            # N, C, W --> N, W, C
            data1 = torch.transpose(data1, 1, 2).contiguous()
            data1 = data1.view(-1, self.input_size)
            tmp = torch.mm(data1, self.weight)
            tmp = torch.reshape(tmp, (N, H, C))
            # N, W, C --> N, C, W
            tmp = torch.transpose(tmp, 1, 2).contiguous()
            output2[:, :, y, :] += tmp
        
         # bottom up
        output3 = torch.tensor(data)
        for y in range(H-2, -1, -1):
            # data of previous row
            data1 = output3[:, :, y+1, :]
            # N, C, W --> N, W, C
            data1 = torch.transpose(data1, 1, 2).contiguous()
            data1 = data1.view(-1, self.input_size)
            tmp = torch.mm(data1, self.weight)
            tmp = torch.reshape(tmp, (N, H, C))
            # N, W, C --> N, C, W
            tmp = torch.transpose(tmp, 1, 2).contiguous()
            output3[:, :, y, :] += tmp

        return torch.cat((output0, output1, output2, output3), dim=1)

class IRNN(nn.Module):
    def __init__(self, input_size):
        
        super(IRNN, self).__init__()

        self.horizontal = nn.RNN(input_size=input_size, hidden_size=input_size//2, bidirectional=True)
        self.vertical = nn.RNN(input_size=input_size, hidden_size=input_size//2, bidirectional=True)

    def forward(self, data):
        
        N, C, H, W = data.shape
        data_h = data.permute(3, 0, 2, 1).contiguous().view(W, N*H, C)
        data_v = data.permute(2, 0, 3, 1).contiguous().view(H, N*W, C)

        output_h = self.horizontal(data_h)[0]
        output_v = self.vertical(data_v)[0]

        output_h = output_h.view(W, N, H, C).permute(1, 3, 2, 0).contiguous()
        output_v = output_h.view(H, N, W, C).permute(1, 3, 0, 2).contiguous()

        print(output_h.shape)


if __name__ == "__main__":

    x = torch.zeros(1, 128, 500, 500).cuda()
    
    irnn0 = RecurrentProp(128).cuda()
    irnn1 = IRNN(128).cuda()

    start = time.time()
    for i in range(10):
        irnn0(x)
    print(time.time() - start)

    start = time.time()
    for i in range(10):
        irnn1(x)
    print(time.time() - start)

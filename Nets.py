import torch
from torch.nn.parameter import Parameter
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import pdb
from functional import reset_normal_param, LinearWeightNorm


# class Discriminator(nn.Module):
#     def __init__(self, input_dim=96*96*3, output_dim=500):
#         super(Discriminator, self).__init__()
#         self.input_dim = input_dim
#         self.layers = torch.nn.ModuleList([
#             nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=2),
#             # nn.ReLU(),
#             nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, stride=2),
#             # nn.ReLU(),
#             nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7)
#             # nn.ReLU()
#         ])
#
#         # self.layers = torch.nn.ModuleList([
#         #     LinearWeightNorm(input_dim, 2500),
#         #     LinearWeightNorm(2500, 2000),
#         #     LinearWeightNorm(2000, 1000),
#         #     LinearWeightNorm(1000, 1000),
#         #     LinearWeightNorm(1000, 1000)]
#         # )
#         self.final = LinearWeightNorm((14**2) * 16, output_dim, weight_scale=1)
#         # for layer in self.layers:
#         #    reset_normal_param(layer, 0.1)
#         # reset_normal_param(self.final, 0.1, 5)
#
#     def forward(self, x, feature = False, cuda = True):
#         # x = x.view(-1, self.input_dim)
#         noise = torch.randn(x.size()) * 0.3 if self.training else torch.Tensor([0])
#         if cuda:
#             noise = noise.cuda()
#         x = x + Variable(noise, requires_grad = False)
#         # for i in range(len(self.layers)):
#         #     m = self.layers[i]
#         #     x_f = F.relu(m(x))
#         #     noise = torch.randn(x_f.size()) * 0.5 if self.training else torch.Tensor([0])
#         #     if cuda:
#         #         noise = noise.cuda()
#         #     x = (x_f + Variable(noise, requires_grad = False))
#         for i in range(len(self.layers)):
#             m = self.layers[i]
#             x = F.relu(m(x))
#             noise = torch.randn(x.size()) *0.5 if self.training else torch.Tensor([0])
#             if cuda:
#                 noise = noise.cuda()
#             x = x + Variable(noise, requires_grad = False)
#
#         x = x.view(-1, 14*14*16)
#         if feature:
#             return x, self.final(x)
#         return self.final(x)
#
#
# class Generator(nn.Module):
#     def __init__(self, z_dim, output_dim = 96*96*3):
#         super(Generator, self).__init__()
#         self.z_dim = z_dim
#         self.fc1 = nn.Linear(z_dim, 2500, bias=False)
#         self.bn1 = nn.BatchNorm1d(2500, affine=False, eps=1e-6, momentum=0.5)
#         # self.bn1.track_running_stats=1
#         self.fc2 = nn.Linear(2500, 5000, bias=False)
#         self.bn2 = nn.BatchNorm1d(5000, affine=False, eps=1e-6, momentum=0.5)
#         # self.bn2.track_running_stats=1
#         self.fc3 = LinearWeightNorm(5000, output_dim, weight_scale=1)
#         self.bn1_b = Parameter(torch.zeros(2500))
#         self.bn2_b = Parameter(torch.zeros(5000))
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         # reset_normal_param(self.fc1, 0.1)
#         # reset_normal_param(self.fc2, 0.1)
#         # reset_normal_param(self.fc3, 0.1)
#
#     def forward(self, batch_size, cuda = True):
#         x = Variable(torch.rand(batch_size, self.z_dim), requires_grad=False)
#         if cuda:
#             x = x.cuda()
#         x = F.softplus(self.bn1(self.fc1(x)) + self.bn1_b)
#         x = F.softplus(self.bn2(self.fc2(x)) + self.bn2_b)
#         x = F.softplus(self.fc3(x))
#         return x


# class Discriminator(nn.Module):
#     def __init__(self, output_units = 10):
#         super(Discriminator, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 100)
#         self.fc2 = nn.Linear(100, output_units)

#     def forward(self, x, feature = False, cuda = False):
#         x = F.leaky_relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.leaky_relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x_f = self.fc1(x)
#         x = F.leaky_relu(x_f)
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return x if not feature else x_f


class Discriminator(nn.Module):
   def __init__(self, nc = 3, ndf = 16, output_units = 1000):
       super(Discriminator, self).__init__()
       self.ndf = ndf
       self.main = nn.Sequential(
           # state size. (nc) x 96 x 96
           nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=7, stride=2, bias=False),
           nn.BatchNorm2d(ndf),

           nn.LeakyReLU(0.2, inplace=True),
           # state size. (ndf) x 45 x 45
           nn.Conv2d(ndf, ndf * 2, 7, 2, bias=False),
           nn.BatchNorm2d(ndf * 2),
           nn.LeakyReLU(0.2, inplace=True),
           # state size. (ndf*2) x 20 x 20
           nn.Conv2d(ndf * 2, ndf * 4, 7, 2, bias=False),
           nn.BatchNorm2d(ndf * 4),
           nn.LeakyReLU(0.2, inplace=True),
           # state size. (ndf*4) x 7 x 7
           # nn.Conv2d(ndf * 4, ndf * 4, 3, 1, bias=False),
           # state size. (ndf*4) x 7 x 7
       )

       self.final = nn.Linear(self.ndf * 4 * 7 * 7, output_units, bias=False)

   def forward(self, x, feature = False, cuda = False):
       x_f = self.main(x).view(-1, self.ndf * 4 * 7 * 7)
       return x_f if feature else self.final(x_f)


class Generator(nn.Module):
   def __init__(self, z_dim, ngf = 4, output_dim = 96*96*3):
       super(Generator, self).__init__()
       self.z_dim = z_dim
       self.main = nn.Sequential(
           # input is Z, going into a convolution
           nn.ConvTranspose2d(z_dim, ngf * 4, 4, 1, 0, bias=False),
           nn.BatchNorm2d(ngf * 4),
           nn.ReLU(True),
           # state size. (ngf*8) x 4 x 4
           nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
           nn.BatchNorm2d(ngf * 2),
           nn.ReLU(True),
           # state size. (ngf*4) x 8 x 8
           nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
           nn.BatchNorm2d(ngf),
           nn.ReLU(True),
           # state size. (ngf*2) x 16 x 16
           nn.ConvTranspose2d(ngf, 1, 4, 2, 3, bias=False),
           # state size. (ngf) x 32 x 32
           nn.Sigmoid()
       )

   def forward(self, batch_size, cuda = False):
       x = Variable(torch.rand(batch_size, self.z_dim, 1, 1), requires_grad = False, volatile = not self.training)
       if cuda:
           x = x.cuda()
       return self.main(x)

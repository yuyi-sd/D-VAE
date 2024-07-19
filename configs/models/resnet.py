import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.bn1 = nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.bn1 = nn.Identity()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def cam_register(self, layer = 'layer4'):
        self.cam_feed_forward_features = None
        self.cam_backward_features = None
        self.cam_grad_layer = layer
        self.cam_register_hooks(self.cam_grad_layer)

    def cam_register_hooks(self, cam_grad_layer):
        def cam_forward_hook(module, input, output):
            self.cam_feed_forward_features = output

        def cam_backward_hook(module, grad_input, grad_output):
            self.cam_backward_features = grad_output[0]
        
        cam_gradient_layer_found = False
        for idx, m in self.named_modules():
            if idx == cam_grad_layer:
                m.register_forward_hook(cam_forward_hook)
                m.register_backward_hook(cam_backward_hook)
                print("Register cam forward hook !")
                print("Register cam backward hook !")
                cam_gradient_layer_found = True
                break
        if not cam_gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001,
            momentum=0.1,
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionResnetV1(nn.Module):
    def __init__(self, num_classes=10575, face_features=512, dropout_prob=0.6):
        super().__init__()
        self.num_classes = num_classes

        # Define layers
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, face_features, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)
        self.fc = nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        if self.training:
            return self.fc(x)
        else:
            return F.normalize(x, p=2, dim=1)



class ResNetAE(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, stem_end_block = 13):
        super(ResNetAE, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.num_blocks = num_blocks

        self.stem_end_block = stem_end_block
        if self.stem_end_block == 5:
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  
                nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(32), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1),  
                nn.InstanceNorm2d(16), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1), 
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )
        elif self.stem_end_block in [7,9]:
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(64), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  
                nn.InstanceNorm2d(32), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1), 
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )
        elif self.stem_end_block in [11,13]:
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(128), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.InstanceNorm2d(64), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1), 
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )
        else:
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), 
                nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
                nn.InstanceNorm2d(256), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(128), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_stem(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        if self.stem_end_block == 5:
            stem_out = out
        elif self.stem_end_block == 7:
            out = self.layer2[0](out)
            stem_out = out
        elif self.stem_end_block == 9:
            out = self.layer2(out)
            stem_out = out
        elif self.stem_end_block == 11:
            out = self.layer2(out)
            out = self.layer3[0](out)
            stem_out = out
        elif self.stem_end_block == 13:
            out = self.layer2(out)
            out = self.layer3(out)
            stem_out = out
        elif self.stem_end_block == 15:
            out = self.layer2(out)
            out = self.layer3(out)
            stem_out = self.layer4[0](out)
        elif self.stem_end_block == 17:
            out = self.layer2(out)
            out = self.layer3(out)
            stem_out = self.layer4(out)
        return stem_out

    def forward_main_branch(self, stem_out):
        if self.stem_end_block == 5:
            stem_out = self.layer2(stem_out)
            stem_out = self.layer3(stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 7:
            stem_out = self.layer2[1](stem_out)
            stem_out = self.layer3(stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 9:
            stem_out = self.layer3(stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 11:
            stem_out = self.layer3[1](stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 13:
            x = self.layer4(stem_out)
        elif self.stem_end_block == 15:
            x = self.layer4[1](stem_out)
        elif self.stem_end_block == 17:
            x = stem_out

        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def forward_ae_branch(self, stem_out):
        # print(stem_out.shape)
        # x = self.decoder1(stem_out)
        # print(x.shape)
        # x = self.decoder2(x)
        # print(x.shape)
        # rec = self.decoder3(x)
        # print(rec.shape)
        rec = self.decoder(stem_out)
        return rec

    def forward(self, x):
        eps = torch.randn(x.size()).cuda() * 0.062
        x = x + eps
        stem_out = self.forward_stem(x)
        logits = self.forward_main_branch(stem_out)
        rec = self.forward_ae_branch(stem_out)
        return logits, rec


def ResNet18AE(num_classes=10, stem_end_block=15):
    return ResNetAE(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, stem_end_block=stem_end_block)




class ResNetVAE(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, stem_end_block = 13, latent_dim = 128):
        super(ResNetVAE, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.num_blocks = num_blocks
        self.latent_dim = latent_dim

        self.stem_end_block = stem_end_block
        print (self.stem_end_block)
        if self.stem_end_block == 5:
            modules = []
            in_channels = 64
            # hidden_dims = [128,256,512]
            hidden_dims = [128]

            # self.encoder = nn.Identity()
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size= 3, stride= 2, padding  = 1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
                in_channels = h_dim
            self.encoder = nn.Sequential(*modules)

            self.fc_mu = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.fc_var = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.post_conv = nn.Conv2d(latent_dim, in_channels, kernel_size= 1, stride= 1, padding  = 0)

            # self.decoder1 = nn.Sequential(
            #     nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  
            #     nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1), 
            #     nn.InstanceNorm2d(32), 
            #     nn.ReLU(),
            # )
            # self.decoder2 = nn.Sequential(
            #     nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1),  
            #     nn.InstanceNorm2d(16), 
            #     nn.ReLU(),
            # )
            # self.decoder3 = nn.Sequential(
            #     nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1), 
            #     nn.Sigmoid(),
            # )
            # self.decoder = nn.Sequential(
            #     self.decoder1, self.decoder2, self.decoder3
            # )
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(64), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  
                nn.InstanceNorm2d(32), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1), 
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )
        elif self.stem_end_block in [7,9]:
            modules = []
            in_channels = 128
            hidden_dims = [256,512]
            # for h_dim in hidden_dims:
            #     modules.append(
            #         nn.Sequential(
            #             nn.Conv2d(in_channels, out_channels=h_dim,
            #                       kernel_size= 3, stride= 2, padding  = 1),
            #             nn.BatchNorm2d(h_dim),
            #             nn.LeakyReLU())
            #     )
            #     in_channels = h_dim
            # self.encoder = nn.Sequential(*modules)
            # self.fc_mu = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            # self.fc_var = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)

            # self.post_conv = nn.Conv2d(latent_dim, in_channels, kernel_size= 1, stride= 1, padding  = 0)

            # self.decoder1 = nn.Sequential(
            #     nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), 
            #     nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            #     nn.InstanceNorm2d(256), 
            #     nn.ReLU(),
            # )
            # self.decoder2 = nn.Sequential(
            #     nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
            #     nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1), 
            #     nn.InstanceNorm2d(128), 
            #     nn.ReLU(),
            # )
            # self.decoder3 = nn.Sequential(
            #     nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  
            #     nn.Sigmoid(),
            # )
            # self.decoder = nn.Sequential(
            #     self.decoder1, self.decoder2, self.decoder3
            # )

            self.encoder = nn.Identity()
            self.fc_mu = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.fc_var = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.post_conv = nn.Conv2d(latent_dim, in_channels, kernel_size= 1, stride= 1, padding  = 0)
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(64), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  
                nn.InstanceNorm2d(32), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1), 
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )
        elif self.stem_end_block in [11,13]:
            in_channels = 256
            self.encoder = nn.Identity()
            self.fc_mu = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.fc_var = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.post_conv = nn.Conv2d(latent_dim, in_channels, kernel_size= 1, stride= 1, padding  = 0)

            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(128), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.InstanceNorm2d(64), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1), 
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )
        else:
            in_channels = 512
            self.encoder = nn.Identity()
            self.fc_mu = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.fc_var = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.post_conv = nn.Conv2d(latent_dim, in_channels, kernel_size= 1, stride= 1, padding  = 0)

            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), 
                nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
                nn.InstanceNorm2d(256), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(128), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_stem(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        if self.stem_end_block == 5:
            stem_out = out
        elif self.stem_end_block == 7:
            out = self.layer2[0](out)
            stem_out = out
        elif self.stem_end_block == 9:
            out = self.layer2(out)
            stem_out = out
        elif self.stem_end_block == 11:
            out = self.layer2(out)
            out = self.layer3[0](out)
            stem_out = out
        elif self.stem_end_block == 13:
            out = self.layer2(out)
            out = self.layer3(out)
            stem_out = out
        elif self.stem_end_block == 15:
            out = self.layer2(out)
            out = self.layer3(out)
            stem_out = self.layer4[0](out)
        elif self.stem_end_block == 17:
            out = self.layer2(out)
            out = self.layer3(out)
            stem_out = self.layer4(out)
        return stem_out

    def forward_main_branch(self, stem_out):
        if self.stem_end_block == 5:
            stem_out = self.layer2(stem_out)
            stem_out = self.layer3(stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 7:
            stem_out = self.layer2[1](stem_out)
            stem_out = self.layer3(stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 9:
            stem_out = self.layer3(stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 11:
            stem_out = self.layer3[1](stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 13:
            x = self.layer4(stem_out)
        elif self.stem_end_block == 15:
            x = self.layer4[1](stem_out)
        elif self.stem_end_block == 17:
            x = stem_out

        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def forward_ae_branch(self, stem_out):
        # print(stem_out.shape)
        # x = self.decoder1(stem_out)
        # print(x.shape)
        # x = self.decoder2(x)
        # print(x.shape)
        # rec = self.decoder3(x)
        # print(rec.shape)
        mu, log_var = self.encode(stem_out)
        z = self.reparameterize(mu, log_var)
        z = self.post_conv(z)
        rec = self.decoder(z)
        return rec, mu, log_var

    def forward(self, x):
        # eps = torch.randn(x.size()).cuda() * 0.093 # best for cifar10, rem
        # eps = torch.randn(x.size()).cuda() * 0.062 # best for cifar10, rem
        eps = torch.randn(x.size()).cuda() * 0.0
        x = x + eps
        stem_out = self.forward_stem(x)
        logits = self.forward_main_branch(stem_out)
        rec, mu, log_var = self.forward_ae_branch(stem_out)
        return logits, rec, mu, log_var

    def purify_with_mean(self, x):
        # eps = torch.randn(x.size()).cuda() * 0.093 # best for cifar10, rem
        # eps = torch.randn(x.size()).cuda() * 0.062 # best for cifar10, rem
        eps = torch.randn(x.size()).cuda() * 0.0
        x = x + eps
        stem_out = self.forward_stem(x)
        logits = self.forward_main_branch(stem_out)
        mu, log_var = self.encode(stem_out)
        z = mu
        z = self.post_conv(z)
        rec = self.decoder(z)
        return logits, rec, mu, log_var

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    # def decode(self, z: Tensor) -> Tensor:
    #     """
    #     Maps the given latent codes
    #     onto the image space.
    #     :param z: (Tensor) [B x D]
    #     :return: (Tensor) [B x C x H x W]
    #     """
    #     result = self.decoder_input(z)
    #     result = result.view(-1, 512, 2, 2)
    #     result = self.decoder(result)
    #     result = self.final_layer(result)
    #     return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    # def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
    #     mu, log_var = self.encode(input)
    #     z = self.reparameterize(mu, log_var)
    #     return  [self.decode(z), input, mu, log_var]


def ResNet18VAE(num_classes=10, stem_end_block=15, latent_dim = 128):
    return ResNetVAE(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, stem_end_block=stem_end_block, latent_dim = latent_dim)


def yuv_rgb(YUV):
    R = YUV[:,0,:,:] + 1.4075 * YUV[:,2,:,:]
    R= R.reshape(R.shape[0],1,R.shape[1],R.shape[2])
    G = YUV[:,0,:,:] - 0.3455 * (YUV[:,1,:,:]) - 0.7169 * (YUV[:,2,:,:])
    G= G.reshape(G.shape[0],1,G.shape[1],G.shape[2])
    B = YUV[:,0,:,:] + 1.779 * (YUV[:,1,:,:])
    B= B.reshape(B.shape[0],1,B.shape[1],B.shape[2])
    RGB= torch.cat([R,G,B],1)
    return RGB




class RResNetVAE(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, stem_end_block = 13, latent_dim = 128):
        super(RResNetVAE, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.num_blocks = num_blocks
        self.latent_dim = latent_dim
        self.epsilon = 12.0/255

        self.stem_end_block = stem_end_block
        print (self.stem_end_block)
        if self.stem_end_block == 5:
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  
                nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(32), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1),  
                nn.InstanceNorm2d(16), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1), 
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )
        elif self.stem_end_block in [7,9]:
            modules = []
            in_channels = 128
            hidden_dims = [256,512]
            # for h_dim in hidden_dims:
            #     modules.append(
            #         nn.Sequential(
            #             nn.Conv2d(in_channels, out_channels=h_dim,
            #                       kernel_size= 3, stride= 2, padding  = 1),
            #             nn.BatchNorm2d(h_dim),
            #             nn.LeakyReLU())
            #     )
            #     in_channels = h_dim
            # self.encoder = nn.Sequential(*modules)
            # self.fc_mu = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            # self.fc_var = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)

            # self.post_conv = nn.Conv2d(latent_dim, in_channels, kernel_size= 1, stride= 1, padding  = 0)

            # self.decoder1 = nn.Sequential(
            #     nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), 
            #     nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            #     nn.InstanceNorm2d(256), 
            #     nn.ReLU(),
            # )
            # self.decoder2 = nn.Sequential(
            #     nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
            #     nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1), 
            #     nn.InstanceNorm2d(128), 
            #     nn.ReLU(),
            # )
            # self.decoder3 = nn.Sequential(
            #     nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  
            #     nn.Sigmoid(),
            # )
            # self.decoder = nn.Sequential(
            #     self.decoder1, self.decoder2, self.decoder3
            # )

            self.encoder = nn.Identity()
            self.fc_mu = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.fc_var = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.post_conv = nn.Conv2d(latent_dim, in_channels, kernel_size= 1, stride= 1, padding  = 0)
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(64), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  
                nn.InstanceNorm2d(32), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1), 
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )
        elif self.stem_end_block in [11,13]:
            in_channels = 256
            self.encoder = nn.Identity()
            self.fc_mu = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.fc_var = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.post_conv = nn.Conv2d(latent_dim, in_channels, kernel_size= 1, stride= 1, padding  = 0)

            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(128), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.InstanceNorm2d(64), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1), 
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )
        else:
            in_channels = 512
            self.encoder = nn.Identity()
            self.fc_mu = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.fc_var = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.post_conv = nn.Conv2d(latent_dim, in_channels, kernel_size= 1, stride= 1, padding  = 0)

            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), 
                nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
                nn.InstanceNorm2d(256), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(128), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_stem(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        if self.stem_end_block == 5:
            stem_out = out
        elif self.stem_end_block == 7:
            out = self.layer2[0](out)
            stem_out = out
        elif self.stem_end_block == 9:
            out = self.layer2(out)
            stem_out = out
        elif self.stem_end_block == 11:
            out = self.layer2(out)
            out = self.layer3[0](out)
            stem_out = out
        elif self.stem_end_block == 13:
            out = self.layer2(out)
            out = self.layer3(out)
            stem_out = out
        elif self.stem_end_block == 15:
            out = self.layer2(out)
            out = self.layer3(out)
            stem_out = self.layer4[0](out)
        elif self.stem_end_block == 17:
            out = self.layer2(out)
            out = self.layer3(out)
            stem_out = self.layer4(out)
        return stem_out

    def forward_main_branch(self, stem_out):
        if self.stem_end_block == 5:
            stem_out = self.layer2(stem_out)
            stem_out = self.layer3(stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 7:
            stem_out = self.layer2[1](stem_out)
            stem_out = self.layer3(stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 9:
            stem_out = self.layer3(stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 11:
            stem_out = self.layer3[1](stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 13:
            x = self.layer4(stem_out)
        elif self.stem_end_block == 15:
            x = self.layer4[1](stem_out)
        elif self.stem_end_block == 17:
            x = stem_out

        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def forward_ae_branch(self, x, stem_out):
        mu, log_var = self.encode(stem_out)
        z = self.reparameterize(mu, log_var)
        z = self.post_conv(z)
        # rec = self.epsilon * torch.tanh(self.decoder(z)) + x
        rec = self.decoder(z).clamp(-self.epsilon,self.epsilon) + x
        return rec, mu, log_var

    def forward(self, x):
        stem_out = self.forward_stem(x)
        logits = self.forward_main_branch(stem_out)
        rec, mu, log_var = self.forward_ae_branch(x, stem_out)
        return logits, rec, mu, log_var


    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

def RResNet18VAE(num_classes=10, stem_end_block=15, latent_dim = 128):
    return RResNetVAE(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, stem_end_block=stem_end_block, latent_dim = latent_dim)





def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class vconv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(vconv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        latent_dim = ch_out
        self.encoder = nn.Identity()
        self.fc_mu = nn.Conv2d(ch_out, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
        self.fc_var = nn.Conv2d(ch_out, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
        self.post_conv = nn.Conv2d(latent_dim, ch_out, kernel_size= 1, stride= 1, padding  = 0)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def forward(self,x):
        x = self.conv(x)
        mu, log_var = self.encode(x)
        x = self.reparameterize(mu, log_var)
        x = self.post_conv(x)
        return x, mu, log_var


class U_Net(nn.Module):
    def __init__(self, epsilon = 0.03, img_ch=3,output_ch=1, multiplier = 1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv3 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv4 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv5 = conv_block(ch_in=int(512 * multiplier),ch_out=int(1024 * multiplier))

        self.Up5 = up_conv(ch_in=int(1024 * multiplier),ch_out=int(512 * multiplier))
        self.Up_conv5 = conv_block(ch_in=int(1024 * multiplier), ch_out=int(512 * multiplier))

        self.Up4 = up_conv(ch_in=int(512 * multiplier),ch_out=int(256 * multiplier))
        self.Up_conv4 = conv_block(ch_in=int(512 * multiplier), ch_out=int(256 * multiplier))
        
        self.Up3 = up_conv(ch_in=int(256 * multiplier),ch_out=int(128 * multiplier))
        self.Up_conv3 = conv_block(ch_in=int(256 * multiplier), ch_out=int(128 * multiplier))
        
        self.Up2 = up_conv(ch_in=int(128 * multiplier),ch_out=int(64 * multiplier))
        self.Up_conv2 = conv_block(ch_in=int(128 * multiplier), ch_out=int(64 * multiplier))

        self.Conv_1x1 = nn.Conv2d(int(64 * multiplier),output_ch,kernel_size=1,stride=1,padding=0)
        self.epsilon = epsilon

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        # d = F.tanh(d1) * self.epsilon
        d = d1
        return d


class VU_Net(nn.Module):
    def __init__(self, epsilon = 0.03, img_ch=3,output_ch=3, multiplier = 1):
        super(VU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = vconv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = vconv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv3 = vconv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv4 = vconv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv5 = vconv_block(ch_in=int(512 * multiplier),ch_out=int(1024 * multiplier))

        self.Up5 = up_conv(ch_in=int(1024 * multiplier),ch_out=int(512 * multiplier))
        self.Up_conv5 = vconv_block(ch_in=int(1024 * multiplier), ch_out=int(512 * multiplier))

        self.Up4 = up_conv(ch_in=int(512 * multiplier),ch_out=int(256 * multiplier))
        self.Up_conv4 = vconv_block(ch_in=int(512 * multiplier), ch_out=int(256 * multiplier))
        
        self.Up3 = up_conv(ch_in=int(256 * multiplier),ch_out=int(128 * multiplier))
        self.Up_conv3 = vconv_block(ch_in=int(256 * multiplier), ch_out=int(128 * multiplier))
        
        self.Up2 = up_conv(ch_in=int(128 * multiplier),ch_out=int(64 * multiplier))
        self.Up_conv2 = vconv_block(ch_in=int(128 * multiplier), ch_out=int(64 * multiplier))

        self.Conv_1x1 = nn.Conv2d(int(64 * multiplier),output_ch,kernel_size=1,stride=1,padding=0)
        self.epsilon = epsilon

    def forward(self,x):
        # encoding path
        x1, mu1, log_var1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2, mu2, log_var2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3, mu3, log_var3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4, mu4, log_var4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5, mu5, log_var5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5, mu6, log_var6 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4, mu7, log_var7 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3, mu8, log_var8 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2, mu9, log_var9 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        # d = F.tanh(d1) * self.epsilon
        d = d1
        mu = [mu1,mu2,mu3,mu4,mu5,mu6,mu7,mu8,mu9]
        log_var = [log_var1,log_var2,log_var3,log_var4,log_var5,log_var6,log_var7,log_var8,log_var9]
        return d, zip(mu,log_var)


def VUNet(multiplier = 1):
    return VU_Net(multiplier = multiplier)



class V_AE(nn.Module):
    def __init__(self, epsilon = 0.03, img_ch=3,output_ch=3, multiplier = 1):
        super(V_AE,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = vconv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = vconv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv3 = vconv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        # self.Conv4 = vconv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        # self.Conv5 = vconv_block(ch_in=int(512 * multiplier),ch_out=int(1024 * multiplier))

        # self.Up5 = up_conv(ch_in=int(1024 * multiplier),ch_out=int(512 * multiplier))
        # self.Up_conv5 = conv_block(ch_in=int(512 * multiplier), ch_out=int(512 * multiplier))

        # self.Up4 = up_conv(ch_in=int(512 * multiplier),ch_out=int(256 * multiplier))
        # self.Up_conv4 = conv_block(ch_in=int(256 * multiplier), ch_out=int(256 * multiplier))
        
        self.Up3 = up_conv(ch_in=int(256 * multiplier),ch_out=int(128 * multiplier))
        self.Up_conv3 = conv_block(ch_in=int(128 * multiplier), ch_out=int(128 * multiplier))
        
        self.Up2 = up_conv(ch_in=int(128 * multiplier),ch_out=int(64 * multiplier))
        self.Up_conv2 = conv_block(ch_in=int(64 * multiplier), ch_out=int(64 * multiplier))

        self.Conv_1x1 = nn.Conv2d(int(64 * multiplier),output_ch,kernel_size=1,stride=1,padding=0)
        self.epsilon = epsilon

    def forward(self,x):
        # encoding path
        x1, mu1, log_var1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2, mu2, log_var2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3, mu3, log_var3 = self.Conv3(x3)

        # x4 = self.Maxpool(x3)
        # x4, mu4, log_var4 = self.Conv4(x4)

        # x5 = self.Maxpool(x4)
        # x5, mu5, log_var5 = self.Conv5(x5)

        # # decoding + concat path
        # d5 = self.Up5(x5)
        # # d5 = torch.cat((x4,d5),dim=1)
        
        # d5 = self.Up_conv5(d5)
        
        # d4 = self.Up4(d5)
        # # d4 = torch.cat((x3,d4),dim=1)
        # d4 = self.Up_conv4(d4)

        # d3 = self.Up3(d4)
        d3 = self.Up3(x3)
        # d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        # d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        # d = F.tanh(d1) * self.epsilon
        d = d1
        mu = [mu1,mu2,mu3]
        log_var = [log_var1,log_var2,log_var3]
        return d, zip(mu,log_var)


def VAE(multiplier = 1):
    return V_AE(multiplier = multiplier)



class ResNetNAE(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, stem_end_block = 13):
        super(ResNetNAE, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.num_blocks = num_blocks

        self.stem_end_block = stem_end_block
        if self.stem_end_block == 5:
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  
                nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(32), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1),  
                nn.InstanceNorm2d(16), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1), 
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )
        elif self.stem_end_block in [7,9]:
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(64), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  
                nn.InstanceNorm2d(32), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1), 
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )
        elif self.stem_end_block in [11,13]:
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(128), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.InstanceNorm2d(64), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1), 
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )
        else:
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), 
                nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
                nn.InstanceNorm2d(256), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(128), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_stem(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        if self.stem_end_block == 5:
            stem_out = out
        elif self.stem_end_block == 7:
            out = self.layer2[0](out)
            stem_out = out
        elif self.stem_end_block == 9:
            out = self.layer2(out)
            stem_out = out
        elif self.stem_end_block == 11:
            out = self.layer2(out)
            out = self.layer3[0](out)
            stem_out = out
        elif self.stem_end_block == 13:
            out = self.layer2(out)
            out = self.layer3(out)
            stem_out = out
        elif self.stem_end_block == 15:
            out = self.layer2(out)
            out = self.layer3(out)
            stem_out = self.layer4[0](out)
        elif self.stem_end_block == 17:
            out = self.layer2(out)
            out = self.layer3(out)
            stem_out = self.layer4(out)
        return stem_out

    def forward_main_branch(self, stem_out):
        if self.stem_end_block == 5:
            stem_out = self.layer2(stem_out)
            stem_out = self.layer3(stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 7:
            stem_out = self.layer2[1](stem_out)
            stem_out = self.layer3(stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 9:
            stem_out = self.layer3(stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 11:
            stem_out = self.layer3[1](stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 13:
            x = self.layer4(stem_out)
        elif self.stem_end_block == 15:
            x = self.layer4[1](stem_out)
        elif self.stem_end_block == 17:
            x = stem_out

        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def forward_ae_branch(self, stem_out):
        # print(stem_out.shape)
        # x = self.decoder1(stem_out)
        # print(x.shape)
        # x = self.decoder2(x)
        # print(x.shape)
        # rec = self.decoder3(x)
        # print(rec.shape)
        rec = self.decoder(stem_out)
        return rec

    def forward(self, x, noise_level = None):
        # eps = torch.randn(x.size()).cuda() * 0.062
        # x = x + eps
        x
        stem_out = self.forward_stem(x)
        n, c, h, w = stem_out.shape
        # stem_out = stem_out / max(stem_out.max(), 1e-12)
        stem_out = torch.nn.functional.normalize(stem_out.reshape(n, c, h*w), dim=2).reshape(n, c, h, w)
        if noise_level:
            stem_out = stem_out + torch.randn(stem_out.size()).cuda() * noise_level
        logits = self.forward_main_branch(stem_out)
        rec = self.forward_ae_branch(stem_out)
        return logits, rec


def ResNet18NAE(num_classes=10, stem_end_block=15):
    return ResNetNAE(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, stem_end_block=stem_end_block)




class ResNetChroma_VAE(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, stem_end_block = 13, latent_dim = 128):
        super(ResNetChroma_VAE, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.num_blocks = num_blocks
        self.latent_dim = latent_dim

        self.stem_end_block = stem_end_block
        print (self.stem_end_block)
        if self.stem_end_block == 5:
            modules = []
            in_channels = 64
            # hidden_dims = [128,256,512]
            hidden_dims = [128]

            # self.encoder = nn.Identity()
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size= 3, stride= 2, padding  = 1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
                in_channels = h_dim
            self.encoder = nn.Sequential(*modules)

            self.fc_mu = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.fc_var = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.post_conv = nn.Conv2d(latent_dim, in_channels, kernel_size= 1, stride= 1, padding  = 0)

            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(64), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  
                nn.InstanceNorm2d(32), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1), 
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )

            self.linear = nn.Linear(64 * block.expansion, num_classes)
        
        elif self.stem_end_block in [7,9]:
            in_channels = 128
            latent_cls_dim = 4
            self.encoder = nn.Identity()
            self.fc_mu = nn.Conv2d(in_channels, latent_dim + latent_cls_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.fc_var = nn.Conv2d(in_channels, latent_dim + latent_cls_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.post_conv = nn.Conv2d(latent_dim + latent_cls_dim, in_channels, kernel_size= 1, stride= 1, padding  = 0)
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(64), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  
                nn.InstanceNorm2d(32), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1), 
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )

            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.linear = nn.Linear(latent_cls_dim, num_classes)

        elif self.stem_end_block in [11,13]:
            in_channels = 256
            latent_cls_dim = 4
            self.encoder = nn.Identity()
            self.fc_mu = nn.Conv2d(in_channels, latent_dim + latent_cls_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.fc_var = nn.Conv2d(in_channels, latent_dim + latent_cls_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.post_conv = nn.Conv2d(latent_dim + latent_cls_dim, in_channels, kernel_size= 1, stride= 1, padding  = 0)

            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(128), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.InstanceNorm2d(64), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1), 
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.linear = nn.Linear(latent_cls_dim, num_classes)
            
        else:
            in_channels = 512
            latent_cls_dim = 4
            self.encoder = nn.Identity()
            self.fc_mu = nn.Conv2d(in_channels, latent_dim + latent_cls_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.fc_var = nn.Conv2d(in_channels, latent_dim + latent_cls_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.post_conv = nn.Conv2d(latent_dim + latent_cls_dim, in_channels, kernel_size= 1, stride= 1, padding  = 0)

            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), 
                nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
                nn.InstanceNorm2d(256), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(128), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(latent_cls_dim, num_classes)
            

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_stem(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        if self.stem_end_block == 5:
            stem_out = out
        elif self.stem_end_block == 7:
            out = self.layer2[0](out)
            stem_out = out
        elif self.stem_end_block == 9:
            out = self.layer2(out)
            stem_out = out
        elif self.stem_end_block == 11:
            out = self.layer2(out)
            out = self.layer3[0](out)
            stem_out = out
        elif self.stem_end_block == 13:
            out = self.layer2(out)
            out = self.layer3(out)
            stem_out = out
        elif self.stem_end_block == 15:
            out = self.layer2(out)
            out = self.layer3(out)
            stem_out = self.layer4[0](out)
        elif self.stem_end_block == 17:
            out = self.layer2(out)
            out = self.layer3(out)
            stem_out = self.layer4(out)
        return stem_out

    def forward_cls_branch(self, stem_out):
        result = self.encoder(stem_out)
        mu = self.fc_mu(result)[:,self.latent_dim:,:,:]
        stem_out = mu
        x = self.pooling(stem_out)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def forward_ae_branch(self, stem_out):
        # print(stem_out.shape)
        # x = self.decoder1(stem_out)
        # print(x.shape)
        # x = self.decoder2(x)
        # print(x.shape)
        # rec = self.decoder3(x)
        # print(rec.shape)
        mu, log_var = self.encode(stem_out)
        z = self.reparameterize(mu, log_var)
        z = self.post_conv(z)
        rec = self.decoder(z)
        return rec, mu, log_var

    def forward(self, x):
        # eps = torch.randn(x.size()).cuda() * 0.093 # best for cifar10, rem
        # eps = torch.randn(x.size()).cuda() * 0.062 # best for cifar10, rem
        eps = torch.randn(x.size()).cuda() * 0.0
        x = x + eps
        stem_out = self.forward_stem(x)
        logits = self.forward_cls_branch(stem_out)
        rec, mu, log_var = self.forward_ae_branch(stem_out)
        return logits, rec, mu, log_var

    def purify(self, x):
        eps = torch.randn(x.size()).cuda() * 0.0
        x = x + eps
        stem_out = self.forward_stem(x)
        result = self.encoder(stem_out)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        z = self.reparameterize(mu, log_var)
        z[:,:self.latent_dim,:,:] = torch.randn_like(z[:,:self.latent_dim,:,:])
        z = self.post_conv(z)
        rec = self.decoder(z)
        return rec

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


def ResNet18CVAE(num_classes=10, stem_end_block=15, latent_dim = 128):
    return ResNetChroma_VAE(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, stem_end_block=stem_end_block, latent_dim = latent_dim)


class ResNetDVAE(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, stem_end_block = 13, latent_dim = 128):
        super(ResNetDVAE, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.num_blocks = num_blocks
        self.latent_dim = latent_dim
        self.sample_wise = True
        self.use_y = True
        self.spatial_emb = False
        self.num_classes = num_classes

        self.stem_end_block = stem_end_block
        print (self.stem_end_block)
        if self.stem_end_block == 5:
            in_channels = 64
            self.encoder = nn.Identity()
            self.fc_mu = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.fc_var = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.post_conv = nn.Conv2d(latent_dim, in_channels, kernel_size= 1, stride= 1, padding  = 0)
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(64), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  
                nn.InstanceNorm2d(32), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1), 
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )
        elif self.stem_end_block in [7,9]:
            in_channels = 128
            self.encoder = nn.Identity()
            self.fc_mu = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.fc_var = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.post_conv = nn.Conv2d(latent_dim, in_channels, kernel_size= 1, stride= 1, padding  = 0)
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(64), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  
                nn.InstanceNorm2d(32), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1), 
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )
            if self.spatial_emb:
                self.emb_p = nn.Embedding(num_classes, 128 * 16 * 16)
            else:
                self.emb_p = nn.Embedding(num_classes, 128)
            self.decoder_p1 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(64), 
                nn.ReLU(),
            )
            self.decoder_p2 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  
                nn.InstanceNorm2d(32), 
                nn.ReLU(),
            )
            self.decoder_p3 = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1), 
                # nn.Sigmoid(),
            )
            self.decoder_p = nn.Sequential(
                self.decoder_p1, self.decoder_p2, self.decoder_p3
            )
        elif self.stem_end_block in [11,13]:
            in_channels = 256
            self.encoder = nn.Identity()
            self.fc_mu = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.fc_var = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.post_conv = nn.Conv2d(latent_dim, in_channels, kernel_size= 1, stride= 1, padding  = 0)

            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(128), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.InstanceNorm2d(64), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1), 
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )
        else:
            in_channels = 512
            self.encoder = nn.Identity()
            self.fc_mu = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.fc_var = nn.Conv2d(in_channels, latent_dim, kernel_size= 1, stride= 1, padding  = 0)
            self.post_conv = nn.Conv2d(latent_dim, in_channels, kernel_size= 1, stride= 1, padding  = 0)

            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), 
                nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
                nn.InstanceNorm2d(256), 
                nn.ReLU(),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(128), 
                nn.ReLU(),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                self.decoder1, self.decoder2, self.decoder3
            )
            if self.spatial_emb:
                self.emb_p = nn.Embedding(num_classes, 128 * 28 * 28)
            else:
                self.emb_p = nn.Embedding(num_classes, 128)
            self.decoder_p1 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), 
                nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
                nn.InstanceNorm2d(256), 
                nn.ReLU(),
            )
            self.decoder_p2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
                nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1), 
                nn.InstanceNorm2d(128), 
                nn.ReLU(),
            )
            self.decoder_p3 = nn.Sequential(
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  
                # nn.Sigmoid(),
            )
            self.decoder_p = nn.Sequential(
                self.decoder_p1, self.decoder_p2, self.decoder_p3
            )
    def set_spatial_emb(self):
        if self.spatial_emb:
            self.emb_p = nn.Embedding(self.num_classes, 128 * 16 * 16)
        else:
            self.emb_p = nn.Embedding(self.num_classes, 128)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_stem(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        if self.stem_end_block == 5:
            stem_out = out
        elif self.stem_end_block == 7:
            out = self.layer2[0](out)
            stem_out = out
        elif self.stem_end_block == 9:
            out = self.layer2(out)
            stem_out = out
        elif self.stem_end_block == 11:
            out = self.layer2(out)
            out = self.layer3[0](out)
            stem_out = out
        elif self.stem_end_block == 13:
            out = self.layer2(out)
            out = self.layer3(out)
            stem_out = out
        elif self.stem_end_block == 15:
            out = self.layer2(out)
            out = self.layer3(out)
            stem_out = self.layer4[0](out)
        elif self.stem_end_block == 17:
            out = self.layer2(out)
            out = self.layer3(out)
            stem_out = self.layer4(out)
        return stem_out

    def forward_main_branch(self, stem_out):
        if self.stem_end_block == 5:
            stem_out = self.layer2(stem_out)
            stem_out = self.layer3(stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 7:
            stem_out = self.layer2[1](stem_out)
            stem_out = self.layer3(stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 9:
            stem_out = self.layer3(stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 11:
            stem_out = self.layer3[1](stem_out)
            x = self.layer4(stem_out)
        elif self.stem_end_block == 13:
            x = self.layer4(stem_out)
        elif self.stem_end_block == 15:
            x = self.layer4[1](stem_out)
        elif self.stem_end_block == 17:
            x = stem_out

        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def forward_ae_branch(self, stem_out):
        mu, log_var = self.encode(stem_out)
        z1 = self.reparameterize(mu, log_var)
        z = self.post_conv(z1)
        rec = self.decoder(z)
        return rec, mu, log_var, z

    def forward(self, x, y):
        # eps = torch.randn(x.size()).cuda() * 0.093 # best for cifar10, rem
        # eps = torch.randn(x.size()).cuda() * 0.062 # best for cifar10, rem
        eps = torch.randn(x.size()).cuda() * 0.0
        x = x + eps
        stem_out = self.forward_stem(x)
        logits = self.forward_main_branch(stem_out)
        rec, mu, log_var, z = self.forward_ae_branch(stem_out)
        if self.spatial_emb:
            Embedding = self.emb_p(y).reshape(stem_out.shape)
        else:
            Embedding = self.emb_p(y)[:,:,None,None].repeat(1, 1, stem_out.shape[2], stem_out.shape[3])
        if self.sample_wise:
            if self.use_y:
                perturbation = self.decoder_p(Embedding + stem_out)
            else:
                perturbation = self.decoder_p(Embedding + z)
        else:
            perturbation = self.decoder_p(Embedding)
        return logits, rec, mu, log_var, perturbation

    def purify_with_mean(self, x):
        # eps = torch.randn(x.size()).cuda() * 0.093 # best for cifar10, rem
        # eps = torch.randn(x.size()).cuda() * 0.062 # best for cifar10, rem
        eps = torch.randn(x.size()).cuda() * 0.0
        x = x + eps
        stem_out = self.forward_stem(x)
        logits = self.forward_main_branch(stem_out)
        mu, log_var = self.encode(stem_out)
        z = mu
        z = self.post_conv(z)
        rec = self.decoder(z)
        return logits, rec, mu, log_var

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

def ResNet18DVAE(num_classes=10, stem_end_block=15, latent_dim = 128):
    return ResNetDVAE(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, stem_end_block=stem_end_block, latent_dim = latent_dim)
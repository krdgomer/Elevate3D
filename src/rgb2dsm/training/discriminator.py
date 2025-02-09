import torch
from torch import nn

class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels,out_channels,4,stride,padding=1, bias=False,padding_mode="reflect")),
            nn.InstanceNorm2d(out_channels,affine=True),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2)
        )

    def forward(self,x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=2,features=[64,128,256,512]):
        super().__init__()
        self.initial= nn.Sequential(
            nn.Conv2d(1+1,features[0],kernel_size=4,stride=2,padding=1,padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels,feature,stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels,1,kernel_size=4,stride=1,padding=1,padding_mode="reflect"
            )
        )


        self.model = nn.Sequential(*layers)

    def forward(self,x,y):
        x = torch.cat([x,y], dim=1)
        x = self.initial(x)
        return self.model(x)

def test():
    x=torch.randn((1,1,512,512))
    y=torch.randn((1,1,512,512))
    model = Discriminator()
    preds = model(x,y)
    print(preds.shape)


if __name__ == "__main__":
    test()
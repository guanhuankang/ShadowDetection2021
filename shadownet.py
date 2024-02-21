import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, in_channels, 1, bias=False), nn.BatchNorm2d(in_channels)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        block1 = F.relu(self.block1(x) + x, True)
        block2 = self.block2(block1)
        return block2


class SoftPred(nn.Module):
    def __init__(self, in_channels):
        super(SoftPred, self).__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )
    def forward(self, x):
        return self.conv_1x1(x)


class LocalShadowDetector(nn.Module):
    def __init__(self, size=(320,320)):
        super(LocalShadowDetector, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d( 64, 32, kernel_size=(1,1), bias=False, groups=32 ), 
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d( 32, 32, kernel_size=(7,7), padding=(3, 3), bias=False, groups=8 ),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False, groups=32)
        )
        self.pred = nn.Conv2d(32, 1, 1, bias=False)
        self.size = size
    
    def forward(self, x):
        return self.pred( F.interpolate(self.block(x), size=self.size, mode="bilinear") + \
            F.interpolate(self.conv(x), size=self.size, mode="bilinear"))

class ShadowNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnext101_32x8d(pretrained=True)
        
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        trans_img = torchvision.transforms.Normalize(mean, std)
        trans_back = torchvision.transforms.Normalize(-mean/std, 1.0/std)
        self.trans_imgs = lambda x: torch.cat([trans_img(x[_]).unsqueeze(0) for _ in range(x.shape[0])], dim=0)
        self.trans_backs = lambda x: torch.cat([trans_back(x[_]).unsqueeze(0) for _ in range(x.shape[0])], dim=0)
        
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )
        self.layer1 = nn.Sequential(
            backbone.maxpool,
            backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.reduction4 = nn.Sequential(
            nn.Conv2d( 2048, 512, 3, padding=1, bias=False ), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d( 512, 32, 1, bias = False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.reduction3 = nn.Sequential(
            nn.Conv2d( 1024, 512, 3, padding=1, bias=False ), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d( 512, 32, 1, bias = False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.reduction2 = nn.Sequential(
            nn.Conv2d( 512, 256, 3, padding=1, bias=False ), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d( 256, 32, 1, bias = False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.reduction1 = nn.Sequential(
            nn.Conv2d( 256, 128, 3, padding=1, bias=False ), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d( 128, 32, 1, bias = False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.reduction0 = nn.Sequential(
            nn.Conv2d( 64, 64, 3, padding=1, bias=False ), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d( 64, 32, 1, bias = False), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.DRR = nn.Sequential(
            nn.Conv2d( 3, 64, kernel_size=(7,7), padding=(3,3), bias=False ), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d( 64, 32, kernel_size=(3,3), padding=(1,1), bias=False ), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d( 32, 1, 1, bias=False )
        )

        self.fusion3 = ConvBlock(64, 32)
        self.fusion2 = ConvBlock(64, 32)
        self.fusion1 = ConvBlock(96, 32)
        self.fusion0 = ConvBlock(128, 32)

        self.pred3 = SoftPred(32)
        self.pred2 = SoftPred(32)
        self.pred1 = SoftPred(32)
        self.pred0 = SoftPred(32)

        self.LSDForIntersectionBranch0 = LocalShadowDetector()
        self.LSDForIntersectionBranch1 = LocalShadowDetector()
        self.LSDForIntersectionBranch2 = LocalShadowDetector()
        self.LSDForIntersectionBranch3 = LocalShadowDetector()

        self.LSDForDivergenceBranch0 = LocalShadowDetector()
        self.LSDForDivergenceBranch1 = LocalShadowDetector()
        self.LSDForDivergenceBranch2 = LocalShadowDetector()
        self.LSDForDivergenceBranch3 = LocalShadowDetector()

        self.fuseIntersection = nn.Conv2d(4, 1, 1, bias=False)
        self.fuseDivergence = nn.Conv2d(4, 1, 1, bias=False)
        self.fuseNonDark = nn.Conv2d(4, 1, 1, bias=False)

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        df0 = self.reduction0(layer0)
        df1 = self.reduction1(layer1)
        df2 = self.reduction2(layer2)
        df3 = self.reduction3(layer3)
        df4 = self.reduction4(layer4)

        ff3 = self.fusion3( torch.cat( (
            F.interpolate(df4, size=df3.shape[2:4], mode="bilinear"),
            df3
        ), dim = 1 ) )
        ff2 = self.fusion2( torch.cat( (
            F.interpolate(ff3, size=df2.shape[2:4], mode="bilinear"),
            df2
        ), dim = 1 ) )
        ff1 = self.fusion1( torch.cat( (
            F.interpolate(ff3, size=df1.shape[2:4], mode="bilinear"),
            F.interpolate(ff2, size=df1.shape[2:4], mode="bilinear"),
            df1
        ), dim = 1 ) )
        ff0 = self.fusion0( torch.cat( (
            F.interpolate(ff3, size=df0.shape[2:4], mode="bilinear"),
            F.interpolate(ff2, size=df0.shape[2:4], mode="bilinear"),
            F.interpolate(ff1, size=df0.shape[2:4], mode="bilinear"),
            df0
        ), dim = 1 ) )
        
        scoreMap3 = self.pred3( F.interpolate(ff3, size=x.shape[2:4], mode="bilinear") )
        scoreMap2 = self.pred2( F.interpolate(ff2, size=x.shape[2:4], mode="bilinear") )
        scoreMap1 = self.pred1( F.interpolate(ff1, size=x.shape[2:4], mode="bilinear") )
        scoreMap0 = self.pred0( F.interpolate(ff0, size=x.shape[2:4], mode="bilinear") )

        darkRegion = self.DRR( x )

        DarkRegion = torch.sign( torch.relu(darkRegion) )
        ScoreMap = torch.sign( torch.relu(scoreMap0) )
        Divergence = torch.relu(DarkRegion - ScoreMap)
        NonDarkMap = 1.0 - torch.sign(ScoreMap + DarkRegion)

        globalContextCue0 = torch.cat((df0,ff0),dim=1)
        globalContextCue1 = torch.cat((df1,ff1),dim=1)
        globalContextCue2 = torch.cat((df2,ff2),dim=1)
        globalContextCue3 = torch.cat((df3,ff3),dim=1)

        dive0 = self.LSDForDivergenceBranch0(globalContextCue0)
        dive1 = self.LSDForDivergenceBranch1(globalContextCue1)
        dive2 = self.LSDForDivergenceBranch2(globalContextCue2)
        dive3 = self.LSDForDivergenceBranch3(globalContextCue3)

        isec0 = self.LSDForIntersectionBranch0(globalContextCue0)
        isec1 = self.LSDForIntersectionBranch1(globalContextCue1)
        isec2 = self.LSDForIntersectionBranch2(globalContextCue2)
        isec3 = self.LSDForIntersectionBranch3(globalContextCue3)

        divergence = self.fuseDivergence( torch.cat( (
            dive0, dive1, dive2, dive3
        ), dim = 1 ) ) * Divergence

        intersection = self.fuseIntersection( torch.cat( ( 
            isec0, isec1, isec2, isec3
         ), dim = 1) ) * ScoreMap ## can be filter to be intersection

        nondark = self.fuseNonDark( torch.cat((
            scoreMap0,scoreMap1,scoreMap2,scoreMap3
        ),dim=1) ) * NonDarkMap

        final = divergence + intersection + nondark

        return {
            "prediction":final,
            "darkRegion":darkRegion
        }


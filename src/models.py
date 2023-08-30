import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src import utils


class Unet(nn.Module):
    """
    Our modified Unet :
    Use of padding to keep size of input in output easily.
    Use of batchnorm2d after Conv2d
    """

    def __init__(self) -> None:
        super().__init__()

        self.down_block1 = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(3, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down_block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.down_block3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.down_block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout2d(0.2),
            nn.Conv2d(256, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.middleU = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout2d(0.2),
            nn.Conv2d(512, 1024, kernel_size=3, padding="same"),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding="same"),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.2),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
        )

        self.up_block1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
        )

        self.up_block2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        )

        self.up_block3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        )

        self.up_block4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down_block1(x)

        x2 = self.down_block2(x1)

        x3 = self.down_block3(x2)

        x4 = self.down_block4(x3)

        x_middle = self.middleU(x4)

        xup0_1 = torch.cat((x4, x_middle), dim=1)
        xup1 = self.up_block1(xup0_1)

        xup1_2 = torch.cat((x3, xup1), dim=1)
        xup2 = self.up_block2(xup1_2)

        xup2_3 = torch.cat((x2, xup2), dim=1)
        xup3 = self.up_block3(xup2_3)

        xup3_4 = torch.cat((x1, xup3), dim=1)
        xup4 = self.up_block4(xup3_4)

        return xup4

    def visu(
        self,
        x: torch.Tensor,
        im1: int = 0,
        im2: int = 0,
        im3: int = 0,
        im4: int = 0,
        im_middle: int = 0,
        imu1: int = 0,
        imu2: int = 0,
        imu3: int = 0,
    ) -> None:
        self.eval()
        with torch.no_grad():
            x1 = self.down_block1(x)

            x2 = self.down_block2(x1)

            x3 = self.down_block3(x2)

            x4 = self.down_block4(x3)

            x_middle = self.middleU(x4)

            xup0_1 = torch.cat((x4, x_middle), dim=1)
            xup1 = self.up_block1(xup0_1)

            xup1_2 = torch.cat((x3, xup1), dim=1)
            xup2 = self.up_block2(xup1_2)

            xup2_3 = torch.cat((x2, xup2), dim=1)
            xup3 = self.up_block3(xup2_3)

            xup3_4 = torch.cat((x1, xup3), dim=1)
            xup4 = self.up_block4(xup3_4)

            fig, ax = plt.subplots(2, 5, figsize=(34, 13))

            ax = ax.flatten()

            utils.imshow(x[0], ax[0])
            utils.imshow(x1[0][im1], ax[1])
            utils.imshow(x2[0][im2], ax[2])
            utils.imshow(x3[0][im3], ax[3])
            utils.imshow(x4[0][im4], ax[4])
            utils.imshow(x_middle[0][im_middle], ax[5])
            utils.imshow(xup1[0][imu1], ax[6])
            utils.imshow(xup2[0][imu2], ax[7])
            utils.imshow(xup3[0][imu3], ax[8])
            utils.imshow(xup4[0].argmax(0), ax[9])


class UnetVanilla(nn.Module):
    """
    Vanilla Unet as define in the paper : https://arxiv.org/pdf/1505.04597.pdf
    """

    def __init__(self) -> None:
        super().__init__()
        self.down_block1 = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(3, 64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down_block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, kernel_size=3),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.down_block3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, kernel_size=3),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.down_block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout2d(0.2),
            nn.Conv2d(256, 512, kernel_size=3),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.middleU = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout2d(0.2),
            nn.Conv2d(512, 1024, kernel_size=3),
            # nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3),
            # nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.2),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
        )

        self.up_block1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
        )

        self.up_block2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        )

        self.up_block3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        )

        self.up_block4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down_block1(x)

        x2 = self.down_block2(x1)

        x3 = self.down_block3(x2)

        x4 = self.down_block4(x3)

        x_middle = self.middleU(x4)

        x4cut = x4[:, :, 4:-4, 4:-4]
        xup0_1 = torch.cat((x4cut, x_middle), dim=1)
        xup1 = self.up_block1(xup0_1)

        x3cut = x3[:, :, 16:-17, 16:-17]
        xup1_2 = torch.cat((x3cut, xup1), dim=1)
        xup2 = self.up_block2(xup1_2)

        x2cut = x2[:, :, 41:-41, 41:-41]
        xup2_3 = torch.cat((x2cut, xup2), dim=1)
        xup3 = self.up_block3(xup2_3)

        x1cut = x1[:, :, 90:-90, 90:-90]
        xup3_4 = torch.cat((x1cut, xup3), dim=1)
        xup4 = self.up_block4(xup3_4)

        adapt_size = nn.Upsample(x.shape[2])

        x_final = adapt_size(xup4)

        return x_final

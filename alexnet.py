import torch


class AlexNet(torch.nn.Module):
    """
    The AlexNet module.
    """

    def __init__(self, num_classes=1000):

        # Mandatory call to super class module.
        super(AlexNet, self).__init__()

        # Defining the feature extraction layers.
        self.feature_extractor = torch.nn.Sequential(

            # Layer 1 - Convolution Layer - Nx3x224x224 -> Nx96x55x55
            torch.nn.Conv2d(in_channels=3, out_channels=96,
                            kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),

            # Layer 2 - Max Pooling Layer - Nx96x55x55 -> Nx96x27x27
            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            # Layer 3 - Convolution Layer - Nx96x27x27 -> Nx256x27x27
            torch.nn.Conv2d(in_channels=96, out_channels=256,
                            kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),

            # Layer 4 - Max Pooling Layer - Nx256x27x27 -> Nx256x13x13
            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            # Layer 5 - Convolution Layer - Nx256x13x13 -> Nx384x13x13
            torch.nn.Conv2d(in_channels=256, out_channels=384,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),

            # Layer 6 - Convolution Layer - Nx384x13x13 -> Nx384x13x13
            torch.nn.Conv2d(in_channels=384, out_channels=384,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),

            # Layer 7 - Convolution Layer - Nx384x13x13 -> Nx256x13x13
            torch.nn.Conv2d(in_channels=384, out_channels=256,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),

            # Layer 8 - Max Pooling Layer - Nx256x13x13 -> Nx256x6x6
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Defining the classification layers.
        self.classifier = torch.nn.Sequential(

            # Layer 9 - Fully Connected - Nx1x9216 -> Nx1x4096
            torch.nn.Linear(in_features=256*6*6, out_features=4096),
            torch.nn.ReLU(inplace=True),

            # Layer 9 - Fully Connected - Nx256x6x6 -> Nx1x10
            torch.nn.Linear(in_features=4096, out_features=4096),
            torch.nn.ReLU(inplace=True),

            # Layer 9 - Fully Connected - Nx256x6x6 -> Nx1xnum_classes
            torch.nn.Linear(in_features=4096, out_features=num_classes),
            torch.nn.Softmax(),
        )

    def forward(self, x):

        # Forward pass through the feature extractor - Nx3x224x224 -> Nx256x6x6
        x = self.feature_extractor(x)

        # Flattening the feature map - Nx256x6x6 -> Nx1x9216
        x = torch.flatten(x, 1)

        # Forward pass through the classifier - Nx1x9216 -> Nx1xnum_classes
        return self.classifier(x)

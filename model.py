import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2023)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias, 0.1)



class LossFn:
    def __init__(self):
        # loss function
        # self.land_factor = landmark_factor
        self.loss_landmark = nn.MSELoss()

    def landmark_loss(self,gt_label,gt_landmark,pred_landmark):
        pred_landmark = torch.squeeze(pred_landmark)
        gt_landmark = torch.squeeze(gt_landmark)

        return self.loss_landmark(valid_pred_landmark,valid_gt_landmark)
    

class ONet(nn.Module):

    def __init__(self,is_train=False, use_cuda=True):
        super(ONet, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda
        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1 48x48 -> 46x46
            nn.PReLU(),  # prelu1 
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1 46x46 -> 23x23
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2 23x23 -> 21x21
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2 21x21 -> 10x10
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3 10x10 -> 8x8
            nn.PReLU(), # prelu3
            nn.MaxPool2d(kernel_size=2,stride=2), # pool3 8x8 -> 4x4
            nn.Conv2d(64,128,kernel_size=2,stride=1), # conv4
            nn.PReLU() # prelu4
        )
        self.conv4 = nn.Conv2d(64,128,kernel_size=2,stride=1) 
        self.conv5 = nn.Linear(128*2*2, 256)  # conv5
        self.prelu5 = nn.PReLU()  # prelu5
        # detection
        self.conv6_1 = nn.Linear(256, 1)
        # bounding box regression
        self.conv6_2 = nn.Linear(256, 4)
        # lanbmark localization
        self.conv6_3 = nn.Linear(256, 136)
        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)
        # detection
        det = torch.sigmoid(self.conv6_1(x))
        box = self.conv6_2(x)
        landmark = self.conv6_3(x)
        if self.is_train is True:
            return landmark
        #landmard = self.conv5_3(x)
        return landmark
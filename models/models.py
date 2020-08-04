import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Xception.xception import xception
import torchvision
from torchsummary import summary


def return_pytorch_xception(pretrained=True):
    model = xception(pretrained=False)
    if pretrained:
        model.fc = model.last_linear
        del model.last_linear
        state_dict = torch.load('/home/kooyunmo/workspace/private/GAN-detector/models/Xception/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model.load_state_dict(state_dict)
        model.last_linear = model.fc
        del model.fc
    return model


class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """
    def __init__(self, modelchoice, num_out_classes, dropout=0.0):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice == 'xception':
            # return pretrained xception model without fc layer.
            self.model = return_pytorch_xception()
            # replace fc
            num_filters = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_filters, num_out_classes)
            else:
                print("Using dropout (dropout rate: ", dropout, ")")
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_filters, num_out_classes)
                )
        elif modelchoice == 'resnet101' or modelchoice == 'resnet50' or modelchoice == 'resnet18':
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True)
            if modelchoice == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained=True)
            if modelchoice == 'resnet101':
                self.model = torchvision.models.resnet101(pretrained=True)

            # replace fc
            num_filters = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_filters, num_out_classes)
            else:
                print("Using dropout (dropout rate: ", dropout, ")")
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_filters, num_out_classes)
                )
        else:
            raise Exception('Choose valid model, e.g. resnet50')
        
    def set_trainable_up_to(self, boolean, layername="Conv2d_4a_3x3"):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param boolean:
        :param layername: depends on network, for inception e.g. Conv2d_4a_3x3
        :return:
        """
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise Exception('Layer({}) not found, cant finetune!'.format(
                    layername))
        else:
            if self.modelchoice == 'xception':
                # Make fc trainable
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True

            else:
                # Make fc trainable
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x

def model_selection(modelname, num_out_classes,
                    dropout=None):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    if modelname == 'xception':
        return TransferModel(modelchoice='xception',
                             num_out_classes=num_out_classes), 299, \
               True, ['image'], None
    elif modelname == 'resnet18' or modelname == 'resnet50' or modelname == 'resnet101':
        return TransferModel(modelchoice=modelname, dropout=dropout,
                             num_out_classes=num_out_classes), \
               224, True, ['image'], None
    else:
        raise NotImplementedError(modelname)


if __name__ == '__main__':
    model, image_size, *_ = model_selection('xception', num_out_classes=2)
    print(model)
    model = model.cuda()
    input_s = (3, image_size, image_size)
    print(summary(model, input_s))
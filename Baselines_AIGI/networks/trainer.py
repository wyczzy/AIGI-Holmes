import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.resnet_new import resnet50_new
from networks.models import Model
from networks.base_model import BaseModel, init_weights
import pdb
from models import get_model




class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            if opt.trainmode == "NPR":
                self.model = resnet50(pretrained=False, num_classes=1)
            elif opt.trainmode == "CNNDetection":
                self.model = resnet50_new(pretrained=False, num_classes=1)
            elif opt.trainmode == "rine":
                self.model = Model(['ViT-L/14', 1024], 2, 128, self.device)

            elif opt.trainmode == "lora" and opt.modelname.startswith("CLIP:"):
                self.model = get_model(opt.modelname, opt)
                # import pdb; pdb.set_trace()
                torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
                params = []
                for name, p in self.model.named_parameters():
                    if name == "fc.weight" or name == "fc.bias" or 'lora_' in name or name == "fc1.weight" or name == "fc1.bias" or name == "fc2.weight" or name == "fc2.bias":
                        params.append(p)
                        p.requires_grad = True
                        # names.append(name)
                    else:
                        p.requires_grad = False
                        
        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                # import pdb; pdb.set_trace()
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to("cuda:{}".format(opt.gpu_ids[0]))
 

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.9
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/0.9} to {param_group["lr"]}')
        print('*'*25)
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        if self.opt.trainmode == "rine":
            self.loss = self.loss_fn(self.output[0].squeeze(1), self.label)
        else:
            self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


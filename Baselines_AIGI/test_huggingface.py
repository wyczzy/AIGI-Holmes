import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np
from validate import validate, validate4huggingface
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger
from transformers import CLIPVisionModel, CLIPImageProcessor

import random
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


# test config
vals = ['Show-o',
 'Janus-Pro-7B',
 'LlamaGen',
 'Infinity',
 'Janus',
 'VAR',
 'FLUX',
 'PixArt-XL',
 'SD35-L',
 'Janus-Pro-1B']

multiclass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# vals = [
#         'progan',
#         'stylegan', 'biggan', 'cyclegan',
#         'stargan', 'gaugan',
#         'stylegan2',
#         'whichfaceisreal',
#         'ADM',
#         'Glide',
#     'Midjourney','stable_diffusion_v_1_4','stable_diffusion_v_1_5','VQDM','wukong','DALLE2'
# ]


# multiclass = [
#               1,
#               1, 0, 1,
#               0, 0,
#               1,
#               0,
#               0,
#               0,
#               0,0,0,0,0,0
#               ]

# vals = ['LOKI']
# multiclass = [0]

# vals = ['test']
# multiclass = [0]
# vals = ['playground_v2.5',
#  'Backdoor_Imagenet',
#  'LIIF_SR4_224',
#  'AdvAtk_Imagenet',
#  'DALLE2',
#  'lama_224',
#  'deeperforensics_faceOnly',
#  'Control_COCO',
#  'FaceForensics++',
#  'SD3',
#  'SDXL',
#  'IF',
#  'SGXL',
#  'SD2SuperRes_SR4_224',
#  'dalle3',
#  'DFDC',
#  'COCO',
#  'DiffusionDB',
#  'GLIDE',
#  'DataPoison_Imagenet',
#  'SD2Inpaint_224',
#  'flickr30k_224',
#  'SD2']
# multiclass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]



def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True

    return val_opt

import torch
import torch.nn as nn
from PIL import Image

class AIGCDetector(nn.Module):
    def __init__(self, visual_encoder, proj, fc):
        super().__init__()
        self.visual_encoder = visual_encoder.eval()  # 保证eval模式
        self.proj = proj
        self.fc = fc

    @torch.no_grad()
    def forward(self, image):
        """
        pil_image: PIL.Image, 单张图片
        返回: pred (float), 0或1
        """
        # 特征提取
        feature = self.visual_encoder(image)["pooler_output"] @ self.proj
        # 分类
        pred = self.fc(feature)
        return pred  # 返回二值和原始分数

if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.classes = ""
    opt.dataroot = "/path/to/dataset0215"
    # opt.dataroot = "/path/to/AntifakePrompt_sd3_train"
    seed_torch(100)
#     Testdataroot = os.path.join(opt.dataroot, 'test')
    # opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    # opt.dataroot = "/path/to/datasetx4"
#     Testdataroot = "/path/to/autogressive_data/TestSet"
    # Testdataroot = "/path/to/AntifakePrompt_sd3_test_23"
    # Testdataroot = "/path/to/Chameleon"
    Testdataroot = "/path/to/autogressive_data/TestSet"

    opt.checkpoints_dir = "./checkpoints_new_0623"
    opt.trainmode = "lora"
    opt.modelname = 'CLIP:ViT-L/14@336px'
    opt.name = opt.modelname
    opt.data_aug = True
    opt.loadSize = 384
    opt.cropSize = 336
    opt.batch_size = 64
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.name), exist_ok=True)
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)) )
    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)
    data_loader = create_dataloader(opt)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    pretrained_model_name_or_path = "./aigcllmdetectvisual0628"
    visual_encoder = CLIPVisionModel.from_pretrained(pretrained_model_name_or_path)
    visual_encoder = visual_encoder.cuda()
    fc = nn.Linear(in_features=768, out_features=1, bias=True)
    path = "./aigcllmdetectvisual0628"

    proj_path = os.path.join(path, "proj.pth")
    fc_path = os.path.join(path, "fc.pth")
    fc_st = torch.load(fc_path, map_location="cpu")
    proj = torch.load(proj_path, map_location="cpu")
    fc.load_state_dict(fc_st)
    proj = proj.to(visual_encoder.device)
    fc = fc.to(visual_encoder.device)

    model = AIGCDetector(visual_encoder, proj, fc)
    
#     model = Trainer(opt)
#     model.train()
# #     import ipdb; ipdb.set_trace()
# #     st = torch.load("debug.pth", map_location="cpu")
# #     model.model.model.visual.load_state_dict(st, strict=False)
    
#     model.model.load_state_dict(torch.load("./checkpoints_new_0613/CLIP:ViT-L/14@336px/model_epoch_0.99_1.00.pth", map_location="cpu")['fc'], strict=False)
#     model.model.load_state_dict(torch.load("./checkpoints_new_0613/CLIP:ViT-L/14@336px/model_epoch_0.99_1.00.pth", map_location="cpu")['lora'], strict=False)
#     model.save_networks('debug')
    # import ipdb; ipdb.set_trace()
    
    def testmodel():
        print('*'*25);accs = [];aps = []
        Testopt.trainmode = opt.trainmode
        Testopt.modelname = opt.modelname
        Testopt.noise_type = None
        Testopt.noise_ratio = None
        Testopt.loadSize = 384
        Testopt.cropSize = 336
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        for v_id, val in enumerate(vals):
            Testopt.dataroot = '{}/{}'.format(Testdataroot, val)
            Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']
            Testopt.no_resize = False
            Testopt.no_crop = False
            Testopt.batch_size = opt.batch_size
            acc, ap, r_acc, f_acc, y_true, y_pred = validate4huggingface(model, Testopt)
            accs.append(acc);aps.append(ap)
            print("({} {:10}) acc: {:.1f}; ap: {:.1f}, racc: {:.1f}, facc: {:.1f};".format(v_id, val, acc*100, ap*100, r_acc*100, f_acc*100))
        print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        return np.mean(accs), np.mean(aps)
#     model.train()
    # model.eval();testmodel();
    model.eval();testmodel()
    # model.train()

    
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys, traceback
import h5py
import os.path
import pytorch_DIW_scratch


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def __init__(self, opt, _isTrain):
        BaseModel.initialize(self, opt)

        # isTrain = True

        self.isTrain = _isTrain
        # define tensors
        self.input = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        # self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   # opt.fineSize, opt.fineSize)

        # load/define networks
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    # opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids)


        # This is DIW network
        if _isTrain:
            print("======================================  DIW NETWORK TRAIN FROM SCRATH =======================")
            # model = pytorch_DIW.pytorch_DIW

            model = pytorch_DIW_scratch.pytorch_DIW_scratch
            model = model.cuda()
            self.netG = torch.nn.parallel.DataParallel(model, device_ids = [0])
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        else:
            model = pytorch_DIW_scratch.pytorch_DIW_scratch    
            # model = model.cuda()
            model = torch.nn.parallel.DataParallel(model, device_ids = [0])
            # model_parameters = self.load_network(model, 'G', '_best_vanila_DIWs_full_onMake3D_iteration2')
            # model_parameters = self.load_network(model, 'G', '_best_vanila_MG')
            model_parameters = torch.load("/floyd/home/best_generalization_net_G.pth")
            model.load_state_dict(model_parameters)
            self.netG = model.cuda()
        # end of DIW network

        # self.load_network(self.netG, 'G', '_best_bilateral_both_0to36_regularizer')
        self.old_lr = opt.lr
        self.netG.train()

        if True:            
            self.criterion_joint = networks.JointLoss()
            self.criterionGAN = networks.GANLoss(opt.gan_mode)
            # reflecntance consistncy 
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))


            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            # sys.exit()
            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            if _isTrain:
                networks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self, input, targets, is_DIW):
        self.num_pair = input.size(0)
        # combined_data = torch.cat( (data['img_1'], data['img_2']),0 )
        self.input.resize_(input.size()).copy_(input)
        self.targets = targets
        self.is_DIW = is_DIW


    def forward(self):
        # print("We are Forwarding !!")
        self.input_images = Variable(self.input.cuda())
        self.prediction_d = self.netG.forward(self.input_images)
        self.prediction_d = self.prediction_d.squeeze(1)

    #get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self, current_epoch):

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Combined loss
        self.loss_joint = self.criterion_joint(self.input_images, self.prediction_d, self.targets, self.is_DIW , current_epoch)
        print("loss is %f "%self.loss_joint)
        # sys.exit()
        if self.loss_joint <= 0:
            return 

        self.loss_joint_var = self.criterion_joint.get_loss_var()

        self.lossg = self.loss_joint_var + self.loss_G_GAN
        self.loss_joint_var.backward()


    def optimize_parameters(self, current_epoch):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.optimizer_G.zero_grad()
        self.backward_G(current_epoch)
        self.optimizer_G.step()

    def switch_to_train(self):
        self.netG.train()

    def switch_to_eval(self):
        self.netG.eval()

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        # self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        # for param_group in self.optimizer_D.param_groups:
            # param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr






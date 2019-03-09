import torch
import torch.nn as nn
import torch.sparse
from torch.autograd import Variable
import functools
import numpy as np
import sys
from torch.autograd import Function
import math
from scipy import misc 
###############################################################################
# Functions
###############################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or  classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

# def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[]):
#     netG = None
#     use_gpu = len(gpu_ids) > 0
#     norm_layer = get_norm_layer(norm_type=norm)
#
#     if use_gpu:
#         assert(torch.cuda.is_available())
#
#     if which_model_netG == 'resnet_9blocks':
#         netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
#     elif which_model_netG == 'resnet_6blocks':
#         netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
#     elif which_model_netG == 'unet_128':
#         netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
#     elif which_model_netG == 'unet_256':
#         # netG = SingleUnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
#         # netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
#         netG = MultiUnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
#     else:
#         print('Generator model name [%s] is not recognized' % which_model_netG)
#     if len(gpu_ids) > 0:
#         netG.cuda(device_id=gpu_ids[0])
#     netG.apply(weights_init)
#     return netG


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator
    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a discriminator
    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.
        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()
        self.w_data = 1.0
        self.w_grad = 0.5
        self.w_sm = 2.0
        self.w_od =  0.5
        self.w_od_auto = 0.2
        self.w_sky = 0.1
        # self.h_offset = [0,0,0,1,1,2,2,2,1]
        # self.w_offset = [0,1,2,0,2,0,1,2,1]
        self.total_loss = None

    def Ordinal_Loss(self, prediction_d, targets, input_images):
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        n_point_total = 0

        batch_input = torch.exp(prediction_d)

        ground_truth_arr = Variable(targets['ordinal'].cuda(), requires_grad = False)
        # sys.exit()
        for i in range(0, prediction_d.size(0)):
            gt_d = ground_truth_arr[i]
            n_point_total = n_point_total + gt_d.size(0)
            # zero index!!!!
            x_A_arr = targets['x_A'][i]
            y_A_arr = targets['y_A'][i]
            x_B_arr = targets['x_B'][i]
            y_B_arr = targets['y_B'][i]

            inputs = batch_input[i,:,:]

            # o_img = input_images[i,:,:,:].data.cpu().numpy()
            # o_img = np.transpose(o_img, (1,2,0))

            # store_img = inputs.data.cpu().numpy()
            # misc.imsave(targets['path'][i].split('/')[-1] + '_p.jpg', store_img)
            # misc.imsave(targets['path'][i].split('/')[-1] + '_o.jpg', o_img)

            z_A_arr = inputs[y_A_arr ,x_A_arr]
            z_B_arr = inputs[y_B_arr ,x_B_arr]

            inner_loss = torch.mul(-gt_d, (z_A_arr   - z_B_arr) ) 

            if inner_loss.data[0] > 5:
                print('DIW difference is too large !!!!')
                # inner_loss = torch.mul(-gt_d, (torch.log(z_A_arr)   - torch.log(z_B_arr) ) )  
                return 5

            ordinal_loss = torch.log(1 + torch.exp(inner_loss ))

            total_loss = total_loss + torch.sum(ordinal_loss)


        if total_loss.data[0] != total_loss.data[0]:
            print("SOMETHING WRONG !!!!!!!!!!", total_loss.data[0])
            sys.exit()

        return total_loss/n_point_total

    def sky_loss(self, prediction_d, targets, i):
        tau = 4
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        gt_d = 1
        # inverse depth 
        inputs = torch.exp(prediction_d)
        x_A_arr = targets['sky_x'][i,0]
        y_A_arr = targets['sky_y'][i,0]
        x_B_arr = targets['depth_x'][i,0]
        y_B_arr = targets['depth_y'][i,0]

        z_A_arr = inputs[y_A_arr ,x_A_arr]
        z_B_arr = inputs[y_B_arr ,x_B_arr]

        inner_loss = -gt_d * (z_A_arr   - z_B_arr)  

        if inner_loss.data[0] > tau:
            print("sky prediction reverse")
            inner_loss = -gt_d * (torch.log(z_A_arr)   - torch.log(z_B_arr) )  

        ordinal_loss = torch.log(1 + torch.exp(inner_loss ))
        return torch.sum(ordinal_loss)

    def Ordinal_Loss_AUTO(self, prediction_d, targets, i):
        tau = 1.2
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        # n_point_total = 0

        inputs = torch.exp(prediction_d)
        gt_d = targets['ordinal'][i,0]

        x_A_arr = targets['x_A'][i,0]
        y_A_arr = targets['y_A'][i,0]
        x_B_arr = targets['x_B'][i,0]
        y_B_arr = targets['y_B'][i,0]

        z_A_arr = inputs[y_A_arr ,x_A_arr]
        z_B_arr = inputs[y_B_arr ,x_B_arr]

        # A is close, B is further away
        inner_loss = -gt_d * (z_A_arr   - z_B_arr)  

        ratio = torch.div(z_A_arr, z_B_arr)

        if ratio.data[0] > tau:
            print("DIFFERNCE IS TOO LARGE, REMOVE OUTLIERS!!!!!!")
            return 1.3873
        else:
            ordinal_loss = torch.log(1 + torch.exp(inner_loss ))
            return torch.sum(ordinal_loss)

    def GradientLoss(self, log_prediction_d, mask, log_gt):
        N = torch.sum(mask)
        log_d_diff = log_prediction_d - log_gt
        log_d_diff = torch.mul(log_d_diff, mask)

        v_gradient = torch.abs(log_d_diff[0:-2,:] - log_d_diff[2:,:])
        v_mask = torch.mul(mask[0:-2,:], mask[2:,:])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(log_d_diff[:,0:-2] - log_d_diff[:,2:])
        h_mask = torch.mul(mask[:,0:-2], mask[:,2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        gradient_loss = gradient_loss/N

        return gradient_loss

    def Data_Loss(self, log_prediction_d, mask, log_gt):
        N = torch.sum(mask)
        log_d_diff = log_prediction_d - log_gt
        log_d_diff = torch.mul(log_d_diff, mask)
        s1 = torch.sum( torch.pow(log_d_diff,2) )/N 
        s2 = torch.pow(torch.sum(log_d_diff),2)/(N*N)  
        data_loss = s1 - s2
        
        return data_loss

    def Data_Loss_test(self,prediction_d, targets):
        mask = targets['mask'].cuda()
        d_gt = targets['gt'].cuda()
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        k = 0.5
        for i in range(0, mask.size(0)):
            # number of valid pixels
            N = torch.sum(mask[i,:,:],0)
            d_log_gt = torch.log(d_gt[i,:,:])
            log_d_diff = prediction_d[i,:,:] - d_log_gt
            log_d_diff = torch.cmul(log_d_diff, mask)

            data_loss = (torch.sum(torch.pow(log_d_diff,2))/N - torch.pow(torch.sum(log_d_diff),2)/(N*N)  )
 
            total_loss = total_loss + data_loss

        return total_loss/mask.size(0)


    def __call__(self, input_images, prediction_d,targets, is_DIW, current_epoch):
        # num_features_d = 5

        # prediction_d_un = prediction_d.unsqueeze(1)
        prediction_d_1 = prediction_d[:,::2,::2]
        prediction_d_2 = prediction_d_1[:,::2,::2]
        prediction_d_3 = prediction_d_2[:, ::2,::2]

        mask_0 = Variable(targets['mask_0'].cuda(), requires_grad = False)
        d_gt_0 = torch.log(Variable(targets['gt_0'].cuda(), requires_grad = False))
        
        mask_1 = Variable(targets['mask_1'].cuda(), requires_grad = False)
        d_gt_1 = torch.log(Variable(targets['gt_1'].cuda(), requires_grad = False))

        mask_2 = Variable(targets['mask_2'].cuda(), requires_grad = False)
        d_gt_2 = torch.log(Variable(targets['gt_2'].cuda(), requires_grad = False))

        mask_3 = Variable(targets['mask_3'].cuda(), requires_grad = False)
        d_gt_3 = torch.log(Variable(targets['gt_3'].cuda(), requires_grad = False))

        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        count = 0 

        for i in range(0, mask_0.size(0)):
            # print(i, targets['has_ordinal'][i, 0]) 
            if targets['has_ordinal'][i, 0] > 0.1:
                continue 
            else:
                total_loss += self.w_data * self.Data_Loss(prediction_d[i,:,:], mask_0[i,:,:], d_gt_0[i,:,:]) 
                total_loss += self.w_grad * self.GradientLoss(prediction_d[i,:,:] , mask_0[i,:,:], d_gt_0[i,:,:]) 
                total_loss += self.w_grad * self.GradientLoss(prediction_d_1[i,:,:] , mask_1[i,:,:], d_gt_1[i,:,:])
                total_loss += self.w_grad * self.GradientLoss(prediction_d_2[i,:,:], mask_2[i,:,:], d_gt_2[i,:,:])
                total_loss += self.w_grad * self.GradientLoss(prediction_d_3[i,:,:], mask_3[i,:,:], d_gt_3[i,:,:])
                count += 1

        if count == 0:
            count = 1

        total_loss = total_loss/count

        self.total_loss = total_loss

        return total_loss.data[0]


    def get_loss_var(self):
        return self.total_loss

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


import h5py
import torch.utils.data as data
import pickle
import PIL
import numpy as np
import torch
from scipy import misc

from PIL import Image
import os
import math, random
import os.path
import sys, traceback
from skimage.transform import resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(list_dir):
    # subgroup_name1 = "/dataset/image_list/"
    file_name = list_dir + "imgs_MD.p"
    file_name_1 = open( file_name, "rb" )
    images_list = pickle.load( file_name_1)
    file_name_1.close()

    file_name_t= list_dir + "targets_MD.p"
    file_name_2 = open( file_name_t, "rb" )
    targets_list = pickle.load(file_name_2)
    file_name_2.close()
    return images_list, targets_list


def default_loader(path, targets_path, input_height, input_width, is_fliped):
    # read images
    # print('path  ', path)
    sfm_image = misc.imread(path)
    resized_sfm_img = misc.imresize(sfm_image, (input_height,input_width))

    color_rgb = np.zeros((input_height,input_width,3))
    resized_sfm_img = resized_sfm_img / 255.0
    prob = random.random()

    if len(sfm_image.shape) == 2:
        color_rgb[:,:,0] = resized_sfm_img.copy()
        color_rgb[:,:,1] = resized_sfm_img.copy()
        color_rgb[:,:,2] = resized_sfm_img.copy()
    else:
        color_rgb = resized_sfm_img.copy()

    # flip for DA 
    if prob > 0.5 and is_fliped:
        color_rgb = np.fliplr(color_rgb)

    color_rgb = np.transpose(color_rgb, (2,0,1))

    # read targets
    hdf5_file_read = h5py.File(targets_path,'r')
    gt = hdf5_file_read.get('/targets/gt_depth')
    gt = np.transpose(gt, (1,0))

    mask = hdf5_file_read.get('/targets/mask')
    mask = np.array(mask)
    mask = np.transpose(mask, (1,0))    

    sky_map = hdf5_file_read.get('/targets/sky_map')
    sky_map = np.array(sky_map)
    sky_map = np.transpose(sky_map, (1,0))   

    if prob > 0.5 and is_fliped:
        gt = np.fliplr(gt)
        mask = np.fliplr(mask)
        sky_map = np.fliplr(sky_map)

    color_rgb = np.ascontiguousarray(color_rgb)
    gt = np.ascontiguousarray(gt)
    mask = np.ascontiguousarray(mask)
    sky_map = np.ascontiguousarray(sky_map)

    hdf5_file_read.close()

    # print(sky_map.shape)
    # sys.exit()

    return color_rgb, gt, mask, sky_map


# class ImageFolder(data.Dataset):

#     def __init__(self, root, list_dir, transform=None, 
#                  loader=default_loader):
#         imgs = make_dataset(list_dir)
#         if len(imgs) == 0:
#             raise(RuntimeError("Found 0 images in: " + root + "\n"
#                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

#         self.root = root
#         self.imgs = imgs
#         self.transform = transform
#         self.return_paths = return_paths
#         self.loader = loader

#     def __getitem__(self, index):
#         img_path = self.imgs[index]
#         img_1, img_2 = self.loader(img_path)
#         # if self.transform is not None:
#         img_1 = self.transform(img_1)
#         img_2 = self.transform(img_2)

#         return img_1, img_2
#         # if self.return_paths:
#         #     return img, path
#         # else:
#         #     return img

#     def __len__(self):
#         return len(self.imgs)


class ImageFolder(data.Dataset):

    def __init__(self, root, list_dir, input_height, input_width, transform=None, 
                 loader=default_loader, is_flip = True):
        # load image list from hdf5
        img_list , targets_list = make_dataset(list_dir)
        if len(img_list) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        # img_list_1, img_list_2 = selfshuffle_dataset(img_list)
        self.root = root
        self.list_dir = list_dir
        self.img_list = img_list
        self.targets_list = targets_list
        self.transform = transform
        self.loader = loader
        self.input_height = input_height
        self.input_width = input_width
        self.is_flip = is_flip

    def load_MD_ORD(self, img_path, targets_path, input_height, input_width):
        # img_path = '//phoenix/S6/zl548/DIW/DIW_train_val/136eaa4fe4498925375efd1aff4172249f0057b9.thumb'

        hdf5_file_read = h5py.File(targets_path,'r')
        ordinal_map = hdf5_file_read.get('/targets/ordinal_map')
        ordinal_map = np.array(ordinal_map)
        ordinal_map = np.transpose(ordinal_map, (1,0))   

        sky_map = hdf5_file_read.get('/targets/sky_map')
        sky_map = np.array(sky_map)
        sky_map = np.transpose(sky_map, (1,0))   

        hdf5_file_read.close()

        DIW_image = misc.imread(img_path)    
        resized_diw_img = misc.imresize(DIW_image, (input_height,input_width))
        color_rgb = np.zeros((input_height,input_width,3))
        resized_diw_img = resized_diw_img / 255.0
        if len(DIW_image.shape) == 2:
            color_rgb[:,:,0] = resized_diw_img.copy()
            color_rgb[:,:,1] = resized_diw_img.copy()
            color_rgb[:,:,2] = resized_diw_img.copy()
        else:
            color_rgb = resized_diw_img.copy()

        color_rgb = np.transpose(color_rgb, (2,0,1))

        [y_A_arr, x_A_arr]  = np.where(ordinal_map>0)
        [y_B_arr, x_B_arr] = np.where(ordinal_map < 0)

        if y_A_arr.shape[0] < 100 or y_B_arr.shape[0] < 100:
            return color_rgb, 0,0,0,0,0

        c_idx = random.randint(0, y_A_arr.shape[0]-1)
        f_idx = random.randint(0, y_B_arr.shape[0]-1)

        y_A = y_A_arr[c_idx]
        x_A = x_A_arr[c_idx]
        y_B = y_B_arr[f_idx]
        x_B = x_B_arr[f_idx]

        return color_rgb, sky_map, y_A, x_A ,y_B, x_B, -1

    # def loader_5000(self, targets_path, input_height, input_width, is_fliped):

    #     hdf5_file_read = h5py.File(targets_path,'r')
    #     color_rgb = hdf5_file_read.get('/targets/img')
    #     color_rgb = np.transpose(color_rgb, (2, 1,0))

    #     hdf5_file_read = h5py.File(targets_path,'r')
    #     gt = hdf5_file_read.get('/targets/gt_depth')
    #     gt = np.transpose(gt, (1,0))

    #     mask = hdf5_file_read.get('/targets/mask')
    #     mask = np.array(mask)
    #     mask = np.transpose(mask, (1,0))    

    #     sky_map = hdf5_file_read.get('/targets/sky_map')
    #     sky_map = np.array(sky_map)
    #     sky_map = np.transpose(sky_map, (1,0))   

        
    #     prob = random.random()
    #     if prob > 0.5 and is_fliped:
    #         gt = np.fliplr(gt)
    #         mask = np.fliplr(mask)
    #         sky_map = np.fliplr(sky_map)
    #         color_rgb = np.fliplr(color_rgb)

    #     color_rgb = np.transpose(color_rgb, (2, 0, 1))


    #     color_rgb = np.ascontiguousarray(color_rgb)
    #     gt = np.ascontiguousarray(gt)
    #     mask = np.ascontiguousarray(mask)
    #     sky_map = np.ascontiguousarray(sky_map)

    #     hdf5_file_read.close()

    #     return color_rgb, gt, mask, sky_map


    def __getitem__(self, index):
        # 00xx/1/
        targets_1 = {}
        # targets_1['L'] = []
        targets_1['path'] = []
        sparse_path_1s = []
        # for i in range(0, len(self.img_list[index])):
        img_path_suff = self.img_list[index]
        targets_path_suff = self.targets_list[index]

        img_path_1 = self.root + img_path_suff
        targets_path_1 = self.root + targets_path_suff

        folder_name = img_path_suff.split('/')[-4]
        # sparse_path_1d = self.root + "/phoenix/S6/zl548/MD_sparse_hdf5/" + folder_name + "/" + targets_path_1.split('/')[-1]

        hdf5_file_read = h5py.File(targets_path_1,'r')
        has_ordinal = hdf5_file_read.get('/targets/has_ordinal')
        has_ordinal = np.array(has_ordinal)
        hdf5_file_read.close()

        if has_ordinal[0] > 0.1:
            # print("we are in ordinal mode")
            mask = np.zeros( (self.input_height, self.input_width) )
            gt = np.zeros( (self.input_height, self.input_width) )
            img, sky_map, y_A, x_A ,y_B, x_B, ordinal = self.load_MD_ORD(img_path_1, targets_path_1, self.input_height, self.input_width)
            targets_1['has_ordinal'] = torch.FloatTensor([1])
        else:
            img, gt, mask, sky_map = self.loader(img_path_1, targets_path_1, self.input_height, self.input_width, self.is_flip)
            y_A = x_A = y_B = x_B = ordinal = 0
            targets_1['has_ordinal'] = torch.FloatTensor([0])


        targets_1['path'] = targets_path_suff
        # multi-scale
        targets_1['mask_0'] = torch.from_numpy(mask).float()
        mask_1 = mask[::2,::2]
        targets_1['mask_1'] = torch.from_numpy(mask_1).float()
        mask_2 = mask_1[::2,::2]
        targets_1['mask_2'] = torch.from_numpy(mask_2).float()
        mask_3 = mask_2[::2,::2]
        targets_1['mask_3'] = torch.from_numpy(mask_3).float()


        targets_1['gt_0'] = torch.from_numpy(gt).float()
        gt_1 = gt[::2,::2]
        targets_1['gt_1'] = torch.from_numpy(gt_1).float()
        gt_2 = gt_1[::2,::2]
        targets_1['gt_2'] = torch.from_numpy(gt_2).float()
        gt_3 = gt_2[::2,::2]
        targets_1['gt_3'] = torch.from_numpy(gt_3).float()


        final_img = torch.from_numpy(img).contiguous().float()

        targets_1['x_A'] = torch.LongTensor([long(x_A)])
        targets_1['y_A'] = torch.LongTensor([long(y_A)])
        targets_1['x_B'] = torch.LongTensor([long(x_B)])
        targets_1['y_B'] = torch.LongTensor([long(y_B)])
        targets_1['ordinal'] = torch.FloatTensor([ordinal])

        targets_1['sky_labels'] = torch.FloatTensor([0])
        [y_sky, x_sky] = np.where(sky_map>0.999)
        [y_f, x_f] = np.where(sky_map<0.001)

        targets_1['sky_y']= torch.LongTensor([long(1000)])
        targets_1['sky_x']= torch.LongTensor([long(1000)])
        targets_1['depth_x']= torch.LongTensor([long(1000)])
        targets_1['depth_y']= torch.LongTensor([long(1000)])

        if y_sky.shape[0] > 3000 and y_f.shape[0] > 3000:
            rad_idx = random.randint(0,y_sky.shape[0]-1)
            rad_idx_f = random.randint(0,y_f.shape[0]-1)

            targets_1['sky_y']= torch.LongTensor([long(y_sky[rad_idx])])
            targets_1['sky_x']= torch.LongTensor([long(x_sky[rad_idx])])
            targets_1['depth_y']= torch.LongTensor([long(y_f[rad_idx_f])])
            targets_1['depth_x']= torch.LongTensor([long(x_f[rad_idx_f])])

            targets_1['sky_labels'] = torch.FloatTensor([1])


        return final_img, targets_1
        # return final_img, targets_1, sparse_path_1r, sparse_path_1s

    def __len__(self):
        return len(self.img_list)



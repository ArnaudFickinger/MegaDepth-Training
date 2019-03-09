import random
import numpy as np
import torch.utils.data
# import torchvision.transforms as transforms
from data.base_data_loader import BaseDataLoader
from data.image_folder import ImageFolder
from data.image_folder import DIWImageFolder

from pdb import set_trace as st
# pip install future --upgrade
from builtins import object
from PIL import Image
import sys
import h5py

# torch.manual_seed(1)

class PairedData(object):
    def __init__(self, data_loader, flip):
        self.data_loader = data_loader
        # self.fineSize = fineSize
        # self.max_dataset_size = max_dataset_size
        self.flip = flip
        # st()
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
    

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    # def sparse_loader(self, sparse_path, num_features):
    #     # print("sparse_path  ", sparse_path)
    #     # sys.exit()
    #     hdf5_file_sparse = h5py.File(sparse_path,'r')
    #     B_arr = []
    #     data_whole = hdf5_file_sparse.get('/sparse/mn')
    #     mn = np.array(data_whole)
    #     mn = np.transpose(mn, (1,0))
    #     m = int(mn[0][0])
    #     n = int(mn[1][0])
    #     # print(m, n)
    #     data_whole = hdf5_file_sparse.get('/sparse/S')
    #     S_coo = np.array(data_whole)
    #     S_coo = np.transpose(S_coo, (1,0))
    #     S_coo = torch.transpose(torch.from_numpy(S_coo),0,1)

    #     # print(S_coo[:,0:2])
    #     # print(torch.FloatTensor([3, 4]))
    #     S_i = S_coo[0:2,:].long()
    #     S_v = S_coo[2,:].float()
    #     S = torch.sparse.FloatTensor(S_i, S_v, torch.Size([m+2,n]))

    #     # print(S)
    #     # sys.exit()
    #     for i in range(num_features+1):
    #         data_whole = hdf5_file_sparse.get('/sparse/B'+str(i) )
    #         B_coo = np.array(data_whole)
    #         B_coo = np.transpose(B_coo, (1,0))
    #         B_coo = torch.transpose(torch.from_numpy(B_coo),0,1)
    #         B_i = B_coo[0:2,:].long()
    #         B_v = B_coo[2,:].float()

    #         B_mat = torch.sparse.FloatTensor(B_i, B_v, torch.Size([m+2,m+2]))
    #         B_arr.append(B_mat)

    #     # print(B_arr)
    #     # sys.exit()

    #     data_whole = hdf5_file_sparse.get('/sparse/N')
    #     N = np.array(data_whole)
    #     N = np.transpose(N, (1,0))
    #     N = torch.from_numpy(N)

    #     # print(N)
    #     # sys.exit()
    #     hdf5_file_sparse.close()
    #     # sys.exit()
    #     return S, B_arr, N 

    def __next__(self):
        self.iter += 1
        # if self.iter > self.max_dataset_size:
            # raise StopIteration

        # img_1, img_2, target_1, target_2 = next(self.data_loader)
        final_img, target_1 = next(self.data_loader_iter)

        # print(target_1['ordinal'][0,0])
        # sys.exit()
        # target_1['dS'] = []    
        # target_1['dB_list'] = []     
        # target_1['dN'] = []
        # for i in range(final_img.size(0)):
            # SS_1, SB_list_1, SN_1  = self.sparse_loader(sparse_path_1s[i][0], 2)
            # dS_1, dB_list_1, dN_1  = self.sparse_loader(sparse_path_1d[i], 5)
            # target_1['dS'].append(dS_1)
            # target_1['dB_list'].append(dB_list_1)
            # target_1['dN'].append(dN_1)
        # print(final_img.size())
        # print(target_1['x_A'])
        # print(target_1['ordinal'])
        # print("we are good")
        # sys.exit()
        return {'img_1': final_img, 'target_1': target_1}


class AlignedDataLoader(BaseDataLoader):
    def __init__(self,_root, _list_dir, _input_height, _input_width, _is_flip, _shuffle):
        transform = None
        dataset = ImageFolder(root=_root, \
                list_dir =_list_dir, input_height = _input_height, input_width = _input_width, transform=transform, is_flip = _is_flip)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size= 32, shuffle= _shuffle, num_workers=int(3))

        self.dataset = dataset
        flip = False
        self.paired_data = PairedData(data_loader, flip)

    def name(self):
        return 'AlignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return len(self.dataset)


class DIW_DataLoader(BaseDataLoader):
    def __init__(self,_root, _list_dir, _input_height, _input_width,):
        transform = None
        dataset = DIWImageFolder(root=_root, \
                list_dir =_list_dir, input_height = _input_height, input_width = _input_width,transform=transform)
        # data_loader = torch.utils.data.DataLoader(dataset, batch_size= 32, shuffle= True, num_workers=int(1))
        self.dataset = dataset
        # flip = False
        # self.paired_data = PairedData(data_loader, flip)

    def name(self):
        return 'DIW_DataLoader'

    # def load_data(self):
        # return self.paired_data
    def get_next(self, idx_list):
        # print("get next  ", idx_list)
        final_img, target_1 = self.dataset.load_data(idx_list)
        return {'img_1': final_img, 'target_1': target_1}

    def __len__(self):
        return len(self.dataset)

def CreateDataLoader(_root, _list_dir, _input_height, _input_width, is_flip = True, shuffle =  True):
    data_loader = None
    from data.aligned_data_loader import AlignedDataLoader
    data_loader = AlignedDataLoader(_root, _list_dir, _input_height, _input_width, is_flip, shuffle)
    # if opt.align_data > 0:
    #     from data.aligned_data_loader import AlignedDataLoader
    #     data_loader = AlignedDataLoader()
    # else:
    #     from data.unaligned_data_loader import UnalignedDataLoader
    #     data_loader = UnalignedDataLoader()
    # print(data_loader.name())
    # data_loader.initialize(opt)
    return data_loader

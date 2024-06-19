import warnings


class DefaultConfig(object):
    load_img_path = None  
    load_txt_path = None


    #nus-wide
    training_size = 10500
    query_size = 2085
    database_size = 193749
    batch_size = 128
    img_dir =
    imgname_mat_dir =
    tag_mat_dir =
    label_mat_dir =



    #nus-wide
    max_epoch = 100
    alpha=1
    beta=1.4
    gamma = 0.1
    bit = 64
    y_dim=1000
    label_dim=21
    lr = 10 ** (-1.5)




    use_gpu = True
    valid = True


    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = DefaultConfig()

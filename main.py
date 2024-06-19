import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
from torch.utils.data import DataLoader
from config import opt

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm
from models.tnet import *
from utils import calc_map_k
import torch.nn.functional as Funtional1
from dataset.Mirflckr25K.dataset import DatasetMirflckr25KTrain,DatasetMirflckr25KValid
from valid import valid

import scipy.io as sio
import time
from dataset.Mirflckr25K.dataset_nus import get_single_datasets_nus

def train():


    lnet = LabelNet()
    inet = ImageNet()
    tnet = TextNet()

    if opt.use_gpu:
        inet = inet.cuda()
        tnet = tnet.cuda()
        lnet = lnet.cuda()



    #nus-wide
    train_data,valid_data =get_single_datasets_nus(opt.img_dir, opt.imgname_mat_dir,opt.tag_mat_dir,opt.label_mat_dir, batch_size=opt.batch_size, train_num=opt.training_size, query_num=opt.query_size)

    num_train = len(train_data)
    train_L = train_data.get_all_label()

    F_buffer = torch.randn(num_train, opt.bit)  
    G_buffer = torch.randn(num_train, opt.bit)  

    if opt.use_gpu:
        # train_L = train_L.cuda()
        F_buffer = F_buffer.cuda()
        G_buffer = G_buffer.cuda()

    
    B = torch.sign(F_buffer + G_buffer)

    batch_size = opt.batch_size  # 128

    lr_lab = [np.power(0.1, x) for x in np.arange(2.0, MAX_ITER, 0.5)]
    lr_img = [np.power(0.1, x) for x in np.arange(4.5, MAX_ITER, 0.5)]
    lr_txt = [np.power(0.1, x) for x in np.arange(3.5, MAX_ITER, 0.5)]

    lnet_opt = torch.optim.Adam(lnet.parameters(), lr=lr_lab[0])
    inet_opt = torch.optim.Adam(inet.parameters(), lr=lr_img[0])
    tnet_opt = torch.optim.Adam(tnet.parameters(), lr=lr_txt[0])


    var = {}
    var['lr_lab'] = lr_lab
    var['lr_img'] = lr_img
    var['lr_txt'] = lr_txt

    var['batch_size'] = batch_size
    var['F'] = np.random.randn(num_train, opt.bit).astype(np.float32)
    var['G'] = np.random.randn(num_train, opt.bit).astype(np.float32)
    var['H'] = np.random.randn(num_train, opt.bit).astype(np.float32)
    var['FG'] = np.random.randn(num_train, SEMANTIC_EMBED).astype(np.float32)
    var['B'] = np.sign(alpha_v * var['F'] + alpha_v * var['G'] + eta * var['H'])



    max_mapi2t = max_mapt2i = 0.
    lossResult=np.zeros([2,4,opt.max_epoch])
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=4,drop_last=True)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)


    for epoch in range(Epoch):
        results = {}
        results['loss_labNet'] = []
        results['loss_imgNet'] = []
        results['loss_txtNet'] = []
        results['Loss_D'] = []
        results['mapl2l'] = []
        results['map i2i'] = []
        results['mapt2t'] = []

        print('++++++++Start train lab_net++++++++')
        train_data.txt_load()
        train_data.re_random_item()
            for idx in range(2):     #2
            lr_lab_Up = var['lr_lab'][epoch:]
            lr_lab = lr_lab_Up[idx]
            for train_labNet_k in range(k_lab_net // (idx + 1)):
                adjust_learning_rate(lnet_opt, lr_lab)
                print('update label_net')
                F = var['F']
                G = var['G']
                H = var['H']
                B = var['B']
                loss_total = 0.0
                for data in tqdm(train_loader):
                    ind = data['index'].numpy()
                    sample_L = data['label'].numpy()
                    label =sample_L
                    label = label.reshape([label.shape[0], 1, 1, label.shape[1]])
                    S = calc_neighbor(train_L.numpy(), sample_L)
                    hsh_L = lnet(torch.from_numpy(label).cuda())
                    H[ind, :] = hsh_L.detach().cpu().numpy()
                    S_cuda = torch.from_numpy(S).cuda()
                    B_cuda = torch.from_numpy(B[ind, :]).cuda()
                    theta_FL = 1.0 / 2 * torch.from_numpy(F).cuda().mm(hsh_L.transpose(1, 0))
                    Loss_pair_Hsh_FL = nn.functional.mse_loss(S_cuda.mul(theta_FL), nn.functional.softplus(theta_FL),
                                                              reduction='sum')
                    theta_GL = 1.0 / 2 * torch.from_numpy(G).cuda().mm(hsh_L.transpose(1, 0))
                    Loss_pair_Hsh_GL = nn.functional.mse_loss(S_cuda.mul(theta_GL), nn.functional.softplus(theta_GL),
                                                              reduction='sum')
                    Loss_quant_L = nn.functional.mse_loss(B_cuda, hsh_L, reduction='sum')
                    loss_l = (Loss_pair_Hsh_FL + Loss_pair_Hsh_GL) + eta * Loss_quant_L
                    loss_total += float(loss_l.detach().cpu().numpy())

                    lnet_opt.zero_grad()
                    loss_l.backward()
                    lnet_opt.step()

                train_labNet_loss = loss_total
                var['B'] = np.sign(alpha_v * var['F'] + alpha_t * var['G'] + eta * var['H'])
                results['loss_labNet'].append(train_labNet_loss)
                print('---------------------------------------------------------------')
                print('...epoch: %3d, loss_labNet: %3.3f' % (epoch, train_labNet_loss))
                print('---------------------------------------------------------------')
                if train_labNet_k > 1 and (results['loss_labNet'][-1] - results['loss_labNet'][-2]) >= 0:
                    break

            print('++++++++Starting Train txt_net++++++++')
            train_data.txt_load()
            train_data.re_random_item()
            for idx in range(3):            #3
                lr_txt_Up = var['lr_txt'][epoch:]
                lr_txt = lr_txt_Up[idx]
                for train_txtNet_k in range(k_txt_net// (idx + 1)):   #k_txt_net
                    adjust_learning_rate(tnet_opt, lr_txt)
                    print('update text_net')
                    G = var['G']
                    H = var['H']
                    FG = var['FG']
                    B = var['B']
                    loss_total = 0.0
                    for data in tqdm(train_loader):
                        ind = data['index'].numpy()
                        sample_L = data['label'].numpy()
                        text = data['txt'].numpy()
                        S = calc_neighbor(train_L.numpy(), sample_L)

                        in_aff, out_aff = affinity_tag_multi(sample_L, sample_L)
                        fea_T, hsh_T, lab_T = tnet(torch.from_numpy(text).cuda(), in_aff, out_aff)
                        G[ind, :] = hsh_T.detach().cpu().numpy()
                        FG[ind, :] = fea_T.detach().cpu().numpy()
                        S_cuda = torch.from_numpy(S).cuda()
                        B_cuda = torch.from_numpy(B[ind, :]).cuda()
                        theta_MH = 1.0 / 2 * torch.from_numpy(H).cuda().mm(hsh_T.transpose(1, 0))
                        Loss_pair_Hsh_MH = nn.functional.mse_loss(S_cuda.mul(theta_MH),
                                                                  nn.functional.softplus(theta_MH),
                                                                  reduction='sum')
                        theta_MM = 1.0 / 2 * torch.from_numpy(G).cuda().mm(hsh_T.transpose(1, 0))
                        Loss_pair_Hsh_MM = nn.functional.mse_loss(S_cuda.mul(theta_MM),
                                                                  nn.functional.softplus(theta_MM),
                                                                  reduction='sum')
                        Loss_quant_T = nn.functional.mse_loss(B_cuda, hsh_T, reduction='sum')
                        Loss_label_T = nn.functional.mse_loss(train_L[ind, :].cuda(), lab_T,
                                                              reduction='sum')
                        loss_t = (Loss_pair_Hsh_MH + Loss_pair_Hsh_MM) \
                                 + alpha_v * Loss_quant_T \
                                 + gamma * Loss_label_T
                        loss_total += float(loss_t.detach().cpu().numpy())
                        tnet_opt.zero_grad()
                        loss_t.backward()
                        tnet_opt.step()

                    train_txtNet_loss = loss_total
                    var['B'] = np.sign(alpha_v * var['F'] + alpha_v * var['G'] + eta * var['H'])
                    if train_txtNet_k % 2 == 0:
                        # train_txtNet_loss = self.calc_loss(self.alpha_t, var['G'], var['H'], var['B'], Sim)
                        results['loss_txtNet'].append(train_txtNet_loss)
                        print('---------------------------------------------------------------')
                        print('...epoch: %3d, Loss_txtNet: %s' % (epoch, train_txtNet_loss))
                        print('---------------------------------------------------------------')
                    if train_txtNet_k > 2 and (results['loss_txtNet'][-1] - results['loss_txtNet'][-2]) >= 0:
                        break

            # train image net
            train_data.img_load()
            train_data.re_random_item()
            for idx in range(3):  # 3
                lr_img_Up = var['lr_img'][epoch:]
                lr_img = lr_img_Up[idx]
                for train_imgNet_k in range(k_img_net // (idx + 1)):  # k_img_net
                    adjust_learning_rate(inet_opt, lr_img)
                    print('update image_net')
                    F = var['F']
                    H = var['H']
                    FG = var['FG']
                    B = var['B']
                    loss_total = 0.0

                    for data in tqdm(train_loader):
                        ind = data['index'].numpy()
                        sample_L = data['label'].numpy()
                        image = data['img']
                        S = calc_neighbor(train_L.numpy(), sample_L)
                        fea_I, hsh_I, lab_I = inet(image.cuda())
                        F[ind, :] = hsh_I.detach().cpu().numpy()
                        fea_T_real = torch.from_numpy(FG[ind, :]).cuda()
                        S_cuda = torch.from_numpy(S).cuda()
                        B_cuda = torch.from_numpy(B[ind, :]).cuda()
                        theta_MH = 1.0 / 2 * torch.from_numpy(H).cuda().mm(hsh_I.transpose(1, 0))
                        Loss_pair_Hsh_MH = nn.functional.mse_loss(S_cuda.mul(theta_MH),
                                                                  nn.functional.softplus(theta_MH),
                                                                  reduction='sum')
                        theta_MM = 1.0 / 2 * torch.from_numpy(F).cuda().mm(hsh_I.transpose(1, 0))
                        Loss_pair_Hsh_MM = nn.functional.mse_loss(S_cuda.mul(theta_MM),
                                                                  nn.functional.softplus(theta_MM),
                                                                  reduction='sum')
                        Loss_quant_I = nn.functional.mse_loss(B_cuda, hsh_I, reduction='sum')
                        Loss_label_I = nn.functional.mse_loss(train_L[ind, :].cuda(), lab_I,
                                                              reduction='sum')

                        loss_i = (Loss_pair_Hsh_MH + Loss_pair_Hsh_MM) \
                                 + alpha_v * Loss_quant_I \
                                 + gamma * Loss_label_I \

                        loss_total += float(loss_i.detach().cpu().numpy())

                        inet_opt.zero_grad()
                        loss_i.backward()
                        inet_opt.step()

                    train_imgNet_loss = loss_total
                    var['B'] = np.sign(alpha_v * var['F'] + alpha_v * var['G'] + eta * var['H'])
                    if train_imgNet_k % 2 == 0:
                        # train_imgNet_loss = self.calc_loss(self.alpha_v, var['F'], var['H'], var['B'], Sim)
                        results['loss_imgNet'].append(train_imgNet_loss)
                        print('---------------------------------------------------------------')
                        print('...epoch: %3d, loss_imgNet: %3.3f' % (epoch, train_imgNet_loss))
                        print('---------------------------------------------------------------')
                    if train_imgNet_k > 2 and (results['loss_imgNet'][-1] - results['loss_imgNet'][-2]) >= 0:
                        break



            with torch.no_grad():
                if opt.valid:
                    mapi2t, mapt2i = valid(opt, inet, tnet, valid_data)
                    if mapt2i + mapi2t >= max_mapt2i + max_mapi2t:
                        max_mapi2t = mapi2t
                        max_mapt2i = mapt2i
                    print(
                        '...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f;MAX_MAP(i->t): %3.4f, MAX_MAP(t->i): %3.4f' % (
                            epoch + 1, mapi2t, mapt2i, max_mapi2t, max_mapt2i))

                condition_dir = './result-mi-%f-sigma-%f-bit-%d' % (hyper_mi, hyper_sigma, opt.bit)
                if not os.path.exists(condition_dir):
                    os.mkdir(condition_dir)

                save_dir_name = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
                cur_dir_path = os.path.join(condition_dir, save_dir_name)
                os.mkdir(cur_dir_path)


                with open(os.path.join(cur_dir_path, 'map.txt'), 'a') as f:
                    f.write('==================================================\n')
                    f.write('...test map: map(i->t): %3.4f, map(t->i): %3.4f\n' % (mapi2t, mapt2i))
                    f.write('...best map: bestmap(i->t): %3.4f, bestmap(t->i): %3.4f\n' %(max_mapi2t, max_mapt2i))
                    f.write('...hash:  %d\n' % (opt.bit))
                    f.write('==================================================\n')



def calc_neighbor(label_1, label_2):
    Sim = (np.dot(label_1, label_2.transpose()) > 0).astype(np.float32)*0.999
    return Sim


def myNormalization(X):
    x1=torch.sqrt(torch.sum(torch.pow(X, 2),1)).unsqueeze(1)
    return X/x1

def calc_inner(X1,X2):
    X1=myNormalization(X1)
    X2=myNormalization(X2)
    X=torch.matmul(X1,X2.t())  # [-1,1]
   
    return X
if __name__ == '__main__':
    train()

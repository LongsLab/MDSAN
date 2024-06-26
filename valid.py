from torch import nn
from utils import calc_map_k
from torch.utils.data import DataLoader
from dataset.Mirflckr25K.dataset import DatasetMirflckr25KValid
from tqdm import tqdm
import torch
from setting import affinity_tag_multi

def valid(opt, img_model: nn.Module, txt_model: nn.Module, dataset: DatasetMirflckr25KValid, return_hash = False):
    # get query img and txt binary code
    dataset.query()
    qB_img = get_img_code(opt, img_model,  dataset)
    qB_txt = get_txt_code(opt, txt_model, dataset)
    query_label = dataset.get_all_label()
    # get retrieval img and txt binary code
    dataset.retrieval()
    rB_img = get_img_code(opt, img_model, dataset)
    rB_txt = get_txt_code(opt, txt_model, dataset)
    retrieval_label = dataset.get_all_label()
    mAPi2t = calc_map_k(qB_img, rB_txt, query_label, retrieval_label)
    mAPt2i = calc_map_k(qB_txt, rB_img, query_label, retrieval_label)
    if return_hash:
        return mAPi2t, mAPt2i, qB_img.cpu(), qB_txt.cpu(), rB_img.cpu(), rB_txt.cpu(), query_label, retrieval_label
    return mAPi2t, mAPt2i

def get_img_code(opt, img_model: nn.Module, dataset: DatasetMirflckr25KValid, isPrint=False):
    dataset.img_load()
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, num_workers=4, pin_memory=True)
    B_img = torch.zeros(len(dataset), opt.bit, dtype=torch.float)
    if opt.use_gpu:
        B_img = B_img.cuda()
    for data in tqdm(dataloader):
        index = data['index'].numpy()  # type: np.ndarray
        img = data['img']  # type: torch.Tensor
        if opt.use_gpu:
            img = img.cuda()
        fea_I, hsh_I, lab_I, fea_T_pred, mu_I, log_sigma_I = img_model(img)
        B_img[index, :] = hsh_I.data
        if isPrint:
            print(B_img[index, :])
    B_img = torch.sign(B_img)
    return B_img.cpu()


def get_txt_code(opt, txt_model: nn.Module, dataset: DatasetMirflckr25KValid, isPrint=False):
    dataset.txt_load()
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, num_workers=8, pin_memory=True)
    B_txt = torch.zeros(len(dataset), opt.bit, dtype=torch.float)
    #
    # #相似矩阵验证
    # train_L = train_data.get_all_label()

    if opt.use_gpu:
        B_txt = B_txt.cuda()
    for data in tqdm(dataloader):
        index = data['index'].numpy()  # type: np.ndarray
        txt = data['txt']  # type: torch.Tensor
        txt = txt.float()

        sample_L = data['label'].numpy()

        in_aff, out_aff = affinity_tag_multi(sample_L, sample_L)



        if opt.use_gpu:
            txt = txt.cuda()

        fea_T, hsh_T, lab_T = txt_model(txt, in_aff, out_aff)
        B_txt[index, :] = hsh_T.data
        if isPrint:
            print(B_txt[index, :])
    B_txt = torch.sign(B_txt)
    return B_txt.cpu()

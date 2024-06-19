import numpy as np
import scipy.io
import torch

# environment and parameters
checkpoint_path = './checkpoint'

SEMANTIC_EMBED = 512
MAX_ITER = 100
batch_size = 64


DATABASE_SIZE = 18015
TRAINING_SIZE = 10000
QUERY_SIZE = 2000

Epoch = 50
k_lab_net = 10
k_img_net = 15
k_txt_net = 15

bit = 64
# hyper here




def calc_hammingDist(B1, B2):
	q = B2.shape[1]
	distH = 0.5 * (q - np.dot(B1, B2.transpose()))
	return distH

def calc_map(qB, rB, query_L, retrieval_L):

	num_query = query_L.shape[0]
	map = 0
	for iter in range(num_query):
		gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
		tsum = int(np.sum(gnd))
		if tsum == 0:
			continue
		hamm = calc_hammingDist(qB[iter, :], rB)
		ind = np.argsort(hamm)
		gnd = gnd[ind]
		count = np.linspace(1, tsum, tsum)
		tindex = np.asarray(np.where(gnd == 1)) + 1.0
		map = map + np.mean(count / tindex)
	map = map / num_query
	return map


from torch.autograd import Variable

def affinity_tag_multi(tag1: np.ndarray, tag2: np.ndarray):
    tag1 = torch.tensor(tag1)
    tag2 = torch.tensor(tag2)
    aff = torch.matmul(tag1, tag2.T)
    affnty = aff
    col_sum = affnty.sum(axis=1)[:, np.newaxis]
    row_sum = affnty.sum(axis=0)
    out_affnty = affnty / col_sum
    in_affnty = (affnty / row_sum).t()
    out_affnty = Variable(torch.Tensor(out_affnty)).cuda()
    in_affnty = Variable(torch.Tensor(in_affnty)).cuda()
    return in_affnty, out_affnty
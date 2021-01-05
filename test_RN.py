import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
from datasets.sklarge_RN import TestDataset
import torch
from Ada_LSN.model import Network
from Ada_LSN.genotypes import geno_inception as geno
import os
import time
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scipy.io as scio
from google.colab.patches import cv2_imshow
import cv2

gpu_id = 0
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.set_device(gpu_id)
net = Network(64, 5, [0, 1, 2, 3], geno).cuda(0).eval()
net.load_state_dict(torch.load('/content/drive/MyDrive/IP - 7th sem/SK-SMALL/ADA_LSN/weights/skel_11000.pth', map_location=lambda storage, loc: storage))

root = '/content/drive/MyDrive/IP - 7th sem/SK-SMALL/SK506/images/test'
# files = './OriginalSKLARGE/test.lst'

dataset = TestDataset(root)
dataloader = list(DataLoader(dataset, batch_size=1))


def plot_single_scale(scale_lst, size):
    pylab.rcParams['figure.figsize'] = size, size / 2
    plt.figure()
    for i, image in enumerate(scale_lst):
        image = image[0,0].cpu().detach().numpy().astype(np.float32)
        cv2_imshow(image)
        s = plt.subplot(1, 5, i + 1)
        plt.imshow(1 - image, cmap=cm.Greys_r)
        cv2_imshow(1-image)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout()


idx = 5
inp, fname, H, W = dataloader[idx]
# print(inp.shape, fname, H, W)
inp = Variable(inp.cuda(gpu_id))
# print(inp)
out, loss = net(inp)
scale_lst = [out]
plot_single_scale(scale_lst, 22)
plt.show()

output_dir = '/content/drive/MyDrive/IP - 7th sem/SK-SMALL/ADA_LSN/output_images/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
start_time = time.time()
tep = 1
for inp, fname, H, W in dataloader:
    inp = Variable(inp.cuda(gpu_id))
    out,loss = net(inp)
    fileName = output_dir + fname[0] + '.mat'
    tep += 1
    out_np = out[0, 0].cpu().detach().numpy()
    out_resize = cv2.resize(out_np, (W.item(), H.item()), interpolation=cv2.INTER_LINEAR)
    # cv2.imwrite(fileName, out_resize)
    scio.savemat(fileName, {'sym': out_resize})
    print('{} size{} resize{}'.format(fileName, out.size(2) * out.size(3), out_resize.shape[0] * out_resize.shape[1]))
diff_time = time.time() - start_time
print('Detection took {:.3f}s per image'.format(diff_time / len(dataloader)))

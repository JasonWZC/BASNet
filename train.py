import numpy as np
np.set_printoptions(threshold=np.inf)
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from assess import hist_sum, compute_metrics
from index2one_hot import get_one_hot
from poly import adjust_learning_rate_poly

# ========================Methods=============================# Select the .py file of the model from the model folder and enter the main function
from model.BASNet import BASNet

# ==========================Net===============================# Main function parameter settings
net = BASNet(pretrained=False, normal_init=True).cuda()

# ========================Dataload============================# Load the dataset (Data augmentation is optional)
from load_TEST import BuildingChangeDataset

# ================Implementation Details======================# Training parameter settings
Epoch = 200                                             # Number of iterations
lr = 0.001                                              # learning rate
n_class = 2                                             # Categories
F1_max = 0.80                                           # Starting weight storage point (Save storage space)
batch_size = 4                                         # Batch size（GPU memory related）

root = r'./summaryTEST/BASNet/'                        # Store the root directory of experimental results


import warnings
warnings.filterwarnings('ignore')

train_data = BuildingChangeDataset(mode='train')
data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = BuildingChangeDataset(mode='test')
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# ====================Loss & optimizer =======================# Select the loss function and optimizer
criterion = nn.BCEWithLogitsLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=lr)

# ====================visualization =========================# Progress visualization
with open(root +'/train.txt', 'a') as f:
    for epoch in range(Epoch):
        torch.cuda.empty_cache()

        new_lr = adjust_learning_rate_poly(optimizer, epoch, Epoch, lr, 0.9)
        print('lr:', new_lr)

        _train_loss = 0

        _hist = np.zeros((n_class, n_class))

        net.train()
        for before, after, change in tqdm(data_loader, desc='epoch{}'.format(epoch), ncols=100):
            before = before.cuda()
            after = after.cuda()

            change = change.squeeze(dim=1).long()
            change_one_hot = get_one_hot(change, 2).permute(0, 3, 1, 2).contiguous().cuda()

            optimizer.zero_grad()

            pred = net(before, after)
            loss_pred = criterion(pred, change_one_hot)
            loss = loss_pred

            loss.backward()
            optimizer.step()
            _train_loss += loss.item()

            label_pred = F.softmax(pred, dim=1).max(dim=1)[1].data.cpu().numpy()
            label_true = change.data.cpu().numpy()

            hist = hist_sum(label_true, label_pred, 2)

            _hist += hist

        miou, oa, kappa, precision, recall, iou, F1 = compute_metrics(_hist)

        trainloss = _train_loss / len(data_loader)

        print('Epoch:', epoch, ' |train loss:', trainloss, ' |train oa:', oa,  ' |train iou:', iou, ' |train F1:', F1)
        f.write('Epoch:%d|train loss:%0.04f|train miou:%0.04f|train oa:%0.04f|train kappa:%0.04f|train precision:%0.04f|train recall:%0.04f|train iou:%0.04f|train F1:%0.04f' % (
                epoch, trainloss, miou, oa, kappa, precision, recall, iou, F1))
        f.write('\n')
        f.flush()

        with torch.no_grad():
            with open(root + '/test.txt', 'a') as f1:
                torch.cuda.empty_cache()

                _test_loss = 0

                _hist = np.zeros((n_class, n_class))

                k = 0

                net.eval()
                for before, after, change in tqdm(test_data_loader, desc='epoch{}'.format(epoch), ncols=100):
                    before = before.cuda()
                    after = after.cuda()
                    change = change.squeeze(dim=1).long()
                    change_one_hot = get_one_hot(change, 2).permute(0, 3, 1, 2).contiguous().cuda()

                    pred = net(before, after)

                    loss = criterion(pred, change_one_hot)

                    _test_loss += loss.item()

                    label_pred = F.softmax(pred, dim=1).max(dim=1)[1].data.cpu().numpy()
                    label_true = change.data.cpu().numpy()

                    hist = hist_sum(label_true, label_pred, 2)

                    _hist += hist

                miou, oa, kappa, precision, recall, iou, F1 = compute_metrics(_hist)

                testloss = _test_loss / len(test_data_loader)

                print('Epoch:', epoch, ' |test loss:', testloss, ' |test oa:', oa,  ' |test iou:', iou, ' |test F1:', F1)
                f1.write('Epoch:%d|test loss:%0.04f|test miou:%0.04f|test oa:%0.04f|test kappa:%0.04f|test precision:%0.04f|test recall:%0.04f|test iou:%0.04f|test F1:%0.04f' % (
                    epoch, testloss, miou, oa, kappa, precision, recall, iou, F1))
                f1.write('\n')
                f1.flush()

            if F1 > F1_max:

                save_path = root + 'F1_{:.4f}_epoch_{}.pth'.format(F1, epoch)
                torch.save(net.state_dict(), save_path)

                F1_max = F1

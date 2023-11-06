import argparse
import os
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

import image.utils as utils
from image.model import Model
from image.CurricularLoss import CurricularLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def train(net, data_loader, train_optimizer, loss_function, temperature):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

    for it, [pos_1, pos_2, _] in enumerate(train_bar):
        pos_1, pos_2 = pos_1.to(device, non_blocking=True), pos_2.to(device, non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        loss = loss_function(out_1, out_2, batch_size, temperature)

        loss.backward()
        train_optimizer.step()
        train_optimizer.zero_grad()

        total_num += batch_size

        total_loss += loss.item() * batch_size

        train_bar.set_description(
            'Train Epoch: [{}/{}] Loss: {:.4f} t:{:.4f}'
                .format(epoch, epochs,
                        total_loss / total_num,
                        loss_function.t.item()))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.to(device, non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        if 'cifar' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        elif 'stl' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device)
        elif 'svhn' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device)

        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('KNN Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))
    torch.cuda.empty_cache()
    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--root', type=str, default='../data', help='Path to data directory')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=400, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset_name', default='cifar10', type=str, help='Choose loss function')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    dataset_name = args.dataset_name

    # data prepare
    train_data, memory_data, test_data = utils.get_dataset(dataset_name, root=args.root)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                              drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim).to(device)
    model = nn.DataParallel(model, device_ids=[0, 1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    c = len(memory_data.classes)
    # print('# Classes: {}'.format(c))

    momentum = 0.999
    regular = 1e-3

    loss_function = CurricularLoss(device, momentum, regular, if_cuda=torch.cuda.is_available())
    if torch.cuda.is_available():
        loss_function = loss_function.to(device)

    best_acc = 0

    # training loop
    for epoch in range(1, epochs + 1):

        train_loss = train(model, train_loader, optimizer, loss_function, temperature)

        if epoch % 20 == 0:
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
            if test_acc_1 > best_acc:
                best_acc = test_acc_1
                save_dict = {
                    'ACC': best_acc,
                    'MODEL': model.state_dict(),
                    'OPTIMIZER': optimizer.state_dict(),
                    't': loss_function.t}
                torch.save(save_dict, '/a%.4f_e%d.pth' % (best_acc, epoch))

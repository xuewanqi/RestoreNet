import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import ResNet18, Transformer
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric


def get_sample_from_pool(label_list, way, number, pool):
    index = label_list[:way]
    g = pool[index]
    g = g[:, :number, :]
    return g.reshape(-1, 512)


def find_similar_gallery(x, g, shot, way, number=4):
    similarity_matrix = euclidean_metric(x, g)  
    _, index = torch.topk(similarity_matrix, 5, dim=1)
    index = index[:, 1:]
    choosen_gallery = g[index.data].view(
        shot, way, number, -1).transpose(1, 2).contiguous()
    return choosen_gallery.view(shot * number, way, -1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load', default='./Network Params/')
    parser.add_argument('--datapath', default='./miniImageNet')
    parser.add_argument('--batch', type=int, default=10000)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=30)
    parser.add_argument('--augnumber', type=int, default=4)

    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)

    dataset = MiniImageNet(args.datapath, 'test')
    sampler = CategoriesSampler(dataset.label,
                                args.batch, args.way, args.shot + args.query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=4, pin_memory=True)

    # Create feature extractor
    model = ResNet18().cuda()
    model.load_state_dict(torch.load(args.load + 'FeatureExtractor.pth'))
    model.eval()

    # Create transformer
    trans = Transformer(512).cuda()
    trans.load_state_dict(torch.load(args.load + 'Transformer.pth'))
    trans.eval()

    # Load pre-computed images features(extracted by 'model') in test split for self-training
    # It is to avoid unnecessary computation
    g = torch.load('./GalleryPool')

    # Create performance recorder
    ave_acc = Averager()  # Baseline (DEML-ProtoNets)
    trans_ave_acc = Averager()  # RestoreNet
    ag_ave_acc = Averager()  # Baseline with Self-training
    trans_ag_ave_acc = Averager()  # RestoreNet with self-training

    # Set skip-connection rate
    p = 0.5

    for i, batch in enumerate(loader, 1):
        data, l = [_.cuda() for _ in batch]

        # Prepare gallery images(unlabeled data) for current episode to do
        # self-training
        gallery = get_sample_from_pool(l, args.way, 30, g)

        k = args.way * args.shot
        data_shot, data_query = data[:k], data[k:]

        x = model(data_shot)
        y = model(data_query)

        data = None
        data_shot = None
        data_query = None

        x = x.reshape(args.shot, args.way, -1)

        proto = x.mean(dim=0)

        # Portotypes after transformation(RestoreNet)
        trans_proto = trans(proto)
        trans_proto = (1 - p) * proto + p * trans_proto

        # Prototypes aftr self-training
        most_similar_gallery = find_similar_gallery(
            proto, gallery, 1, args.way, args.augnumber)

        ag_proto = torch.cat(
            (proto.unsqueeze(0), most_similar_gallery), 0).mean(dim=0)

        # Applying transformation on self-training prototypes
        trans_ag_proto = trans(ag_proto)
        trans_ag_proto = (1 - p) * ag_proto + p * trans_ag_proto

        index = np.random.choice(
            args.way * args.query, args.way * args.query, False).tolist()
        y = y[index]

        logits = euclidean_metric(y, proto)
        trans_logits = euclidean_metric(y, trans_proto)
        ag_logits = euclidean_metric(y, ag_proto)
        trans_ag_logits = euclidean_metric(y, trans_ag_proto)

        label = torch.arange(args.way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)
        label = label[index]

        # Calcutate accuracy
        acc = count_acc(logits, label)
        trans_acc = count_acc(trans_logits, label)
        ag_acc = count_acc(ag_logits, label)
        trans_ag_acc = count_acc(trans_ag_logits, label)

        # Add accuracy to performance recorders
        ave_acc.add(acc)
        trans_ave_acc.add(trans_acc)
        ag_ave_acc.add(ag_acc)
        trans_ag_ave_acc.add(trans_ag_acc)

        print('batch {}: Baseline is {:.2f}({:.2f}), RestoreNet is {:.2f}({:.2f}), Self-training is {:.2f}({:.2f}), Self+RestoreNet is {:.2f}({:.2f})'.format(
            i,
            ave_acc.item() * 100, acc * 100,
            trans_ave_acc.item() * 100, trans_acc * 100,
            ag_ave_acc.item() * 100, ag_acc * 100,
            trans_ag_ave_acc.item() * 100, trans_ag_acc * 100))

    print('95% confidence intervals: Baseline is {:.2f}, RestoreNet is {:.2f}, Self-training is {:.2f}, Self+RestoreNet is {:.2f}'.format(
            ave_acc.stat(),
            trans_ave_acc.stat(),
            ag_ave_acc.stat(),
            trans_ag_ave_acc.stat()))

if __name__ == '__main__':
    main()

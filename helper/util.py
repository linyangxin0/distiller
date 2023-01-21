from __future__ import print_function

import clip
import torch
import numpy as np
from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_features(model, preprocess, dataset):
    all_features = []

    with torch.no_grad():
        for images in dataset:
            image = transforms.ToPILImage()(images.cpu())
            image_input = preprocess(image).unsqueeze(0).to(device)
            features = model.encode_image(image_input)
            all_features.append(features)

    return all_features


def clip_validate(data, model, preprocess, train_dataset, is_feat):
    teacher_preds = torch.tensor([], device=device)
    text_features = torch.cat(
        [clip.tokenize(f"a photo of a {c}") for c in train_dataset.dataset.classes]).to(device)
    text_features = model.encode_text(text_features)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    image_features = get_features(model, preprocess, data)

    for image_feature in image_features:
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_feature @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(len(train_dataset.dataset.classes))
        teacher_preds = torch.cat([teacher_preds, torch.unsqueeze(values, 0)], dim=0)

    if is_feat:
        return [], teacher_preds
    else:
        return teacher_preds


if __name__ == '__main__':
    pass

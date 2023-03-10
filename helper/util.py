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
    all_features = torch.tensor([]).to(device)
    for images in dataset:
        t_image = transforms.ToPILImage()(images.cpu())
        image_input = preprocess(t_image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image_input)
        all_features = torch.cat((all_features, features), 0)

    return all_features


validate_directory = {}


def clip_validate(data, model, preprocess, train_dataset, is_feat, indexes=torch.Tensor([])):
    teacher_preds = torch.tensor([], device=device)

    if is_feat:
        if indexes[0].item() in validate_directory:
            for index in indexes:
                index = index.item()
                teacher_preds = torch.cat([teacher_preds, torch.unsqueeze(validate_directory[index], 0)], dim=0)
        else:
            text_input = torch.cat([clip.tokenize(f"a photo of a {c}") for c in train_dataset.dataset.classes]).to(
                device)
            with torch.no_grad():
                text_features = model.encode_text(text_input)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            image_features = get_features(model, preprocess, data)

            for image_feature, index in zip(image_features, indexes):
                index = index.item()
                image_feature = torch.tensor(data=[image_feature.tolist()], dtype=torch.float16).to(device)
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_feature @ text_features.T)
                values = similarity[0]
                teacher_preds = torch.cat([teacher_preds, torch.unsqueeze(values, 0)], dim=0)
                validate_directory[index] = values

    else:
        text_input = torch.cat([clip.tokenize(f"a photo of a {c}") for c in train_dataset.dataset.classes]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        image_features = get_features(model, preprocess, data)
        for image_feature in image_features:
            image_feature = torch.tensor(data=[image_feature.tolist()], dtype=torch.float16).to(device)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_feature @ text_features.T)
            values = similarity[0]
            teacher_preds = torch.cat([teacher_preds, torch.unsqueeze(values, 0)], dim=0)

    if is_feat:
        return [], teacher_preds
    else:
        return teacher_preds


# if __name__ == '__main__':
#     # feat_t, logit_t = clip_validate(data=input, model=model_t, preprocess=preprocess,
#     #                                 train_dataset=train_loader, is_feat=True)
#     pass

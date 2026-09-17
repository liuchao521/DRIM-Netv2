
'''
Created by: Zhiqing Guo
Institutions: Xinjiang University
Email: guozhiqing@xju.edu.cn
Copyright (c) 2023
'''
import os
import torch
import time
import platform
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from network.MainNet_IHR_V2 import MainNet
from torch.utils.data import WeightedRandomSampler
from network.data import *
from network.transform import Data_Transforms
from datetime import datetime

# from thop import profile
# from ptflops import get_model_complexity_info
import torch.distributed as dist

from network.log_record import *
# from network.pipeline import *s
from network.utils import setup_seed, cal_metrics, plot_ROC
print('-'*20)
print("PyTorch version:{}".format(torch.__version__))
print("Python version:{}".format(platform.python_version()))
print("cudnn version:{}".format(torch.backends.cudnn.version()))
print("GPU name:{}".format(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
print("GPU number:{}".format(torch.cuda.device_count()))
print('-'*20)
import os

# os.environ['RANK'] = str(1)  # 当前进程的排名
# os.environ['WORLD_SIZE'] = str(10)  # 总进程数
import torch.nn.functional as F
# import torch.nn as nn
def convert_syncbn_to_bn(module):
    """Recursively convert all SyncBatchNorm layers to BatchNorm2d."""
    for name, child in module.named_children():
        if isinstance(child, nn.SyncBatchNorm):
            setattr(module, name, nn.BatchNorm2d(child.num_features))
        else:
            convert_syncbn_to_bn(child)
    return module
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
        """
        alpha: 真实样本的权重（越高越关注真实图）
        gamma: 难易样本调制因子，越高越聚焦困难样本
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # shape: [B]
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def alignment_loss(embeddings, labels):
    """Pull same-label samples together on the unit hypersphere."""
    positive_mask = (labels[:, None] == labels[None, :]).triu(diagonal=1)
    pairs = positive_mask.nonzero(as_tuple=False)
    if pairs.numel() == 0:
        return embeddings.new_zeros(())
    return (embeddings[pairs[:, 0]] - embeddings[pairs[:, 1]]).pow(2).sum(dim=1).mean()


def uniformity_loss(embeddings, temperature=2.0):
    """Spread normalized embeddings over the hypersphere."""
    if embeddings.size(0) < 2:
        return embeddings.new_zeros(())
    return torch.pdist(embeddings, p=2).pow(2).mul(-temperature).exp().mean().clamp_min(1e-6).log()


def hypersphere_regularization(fused_embeddings, inc_embeddings, labels, reg_target):
    """Compute alignment/uniformity on fused, inconsistency, or both embeddings."""
    if reg_target == 'fused':
        loss_align = alignment_loss(fused_embeddings, labels)
        loss_uniform = uniformity_loss(fused_embeddings)
    elif reg_target == 'inc':
        loss_align = alignment_loss(inc_embeddings, labels)
        loss_uniform = uniformity_loss(inc_embeddings)
    elif reg_target == 'both':
        loss_align = 0.5 * (
            alignment_loss(fused_embeddings, labels)
            + alignment_loss(inc_embeddings, labels)
        )
        loss_uniform = 0.5 * (
            uniformity_loss(fused_embeddings)
            + uniformity_loss(inc_embeddings)
        )
    else:
        raise ValueError(f'Unsupported reg_target: {reg_target}')
    return loss_align, loss_uniform


def main():

    args = parse.parse_args()

    name = args.name
    train_txt_path = args.train_txt_path
    valid_txt_path = args.valid_txt_path
    continue_train = args.continue_train
    epoches = args.epoches
    batch_size = args.batch_size
    model_name = args.model_name
    model_path = args.model_path
    num_classes = args.num_classes
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    torch.autograd.set_detect_anomaly(False)
    #     os.mkdir(output_path)
    output_path = args.out_path
    os.makedirs(output_path, exist_ok=True)
    log_path = os.path.join(output_path)
    
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    print('Training datetime: ', time_str)
    
    torch.backends.cudnn.benchmark = True

    # -----create train&val data----- #
    train_data = SingleInputDataset(txt_path=train_txt_path, train_transform=Data_Transforms['train'])
    valid_data = SingleInputDataset(txt_path=valid_txt_path, valid_transform=Data_Transforms['val'])
    targets = train_data.get_labels()
    class_sample_count = torch.tensor([(targets == 0).sum(), (targets == 1).sum()], dtype=torch.float)

    # 计算每个类别的反向权重
    weights_per_class = 1.0 / class_sample_count
    sample_weights = weights_per_class[targets]  # 每个样本的权重

    # 构建采样器
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )



    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
    )
    valid_loader = DataLoader(
        dataset=valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
    )
    model = MainNet(num_classes)
    model = convert_syncbn_to_bn(model)

    # from thop import profile
    # x = torch.rand(1, 3, 224, 224)
    # flops, params = profile(model, inputs=(x, ))
    # print('Params: %2fM' % (params / 1e6)) # 打印模型的参数量，单位为百万
    # print('FLOPs: %2fGFLOPs' % (flops / 1e9)) # 打印FLOPs，单位为Giga FLOPs
    # exit()
   
    if continue_train:
        if not model_path:
            raise ValueError('--model_path is required with --continue_train')
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    dino_ln_params, other_params = [], []
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "rgb_extract_feature.dino.dinov3" in param_name:
            dino_ln_params.append(param)
        else:
            other_params.append(param)
    optimizer = optim.Adam(
        [
            {"params": other_params, "lr": args.lr},
            {"params": dino_ln_params, "lr": args.dino_lr},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    print(f"Trainable parameters: other={sum(p.numel() for p in other_params)}, DINO-LN={sum(p.numel() for p in dino_ln_params)}")
    # -----define the learning rate scheduler----- #
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
       # -----------------------------------------define the train & val----------------------------------------- #
    best_acc = 0.0
    best_auc = 0.0
    time_open = time.time()

    for epoch in range(epoches):
        label_val_list = []
        predict_val_list = []
        total_train_samples = 0.0
        correct_tra = 0.0
        sum_loss_tra = 0.0

        print('\nEpoch {}/{}'.format(epoch+1, epoches))
        print('-'*10)

        # -----Training----- #
        for i, data in enumerate(train_loader):
            img_train, labels_train = data
            img_train = img_train.to(device)
            labels_train = labels_train.to(device)  # labels_train.size(0) = batchsize = 64

            optimizer.zero_grad(set_to_none=True)
            model=model.train()
            loss_weight1 =1 # 可以根据需要调整
            outputs, embeddings, inc_embeddings, inc_logits = model(
                img_train,
                return_embedding=True,
                return_inc_embedding=True,
            )
            pre_tra = outputs
            pre_tra = pre_tra.to(device)
           # the average loss of a batch
            loss_tra1 = criterion(pre_tra, labels_train)
            loss_inc_cls = criterion(inc_logits, labels_train)
            # loss_tra2 = criterion1(feature_fusion, recon)
            loss_align, loss_uniform = hypersphere_regularization(
                embeddings,
                inc_embeddings,
                labels_train,
                args.reg_target,
            )
            total_loss = (
                loss_tra1
                + args.inc_cls_weight * loss_inc_cls
                + args.align_weight * loss_align
                + args.uniform_weight * loss_uniform
            )

            sum_loss_tra  = total_loss.detach() * labels_train.size(0) + sum_loss_tra
            
            # prediction
            _, pred = torch.max(pre_tra.data, 1)
    
            total_loss.backward()
            optimizer.step()                                        
            correct_tra += (pred == labels_train).squeeze().sum().cpu().numpy()
            total_train_samples += labels_train.size(0)
            eps = 1e-8
            lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
            if i % 100 == 99:
                inc_std = inc_embeddings.std(dim=0, unbiased=False).mean().item()
                inc_dist = torch.pdist(inc_embeddings.detach(), p=2).mean().item()
                print(
                    "Training: Epoch[{:0>1}/{:0>1}] Iteration[{:0>1}/{:0>1}] "
                    "Loss:{:.4f} CE:{:.4f} IncCE:{:.4f} Reg:{} Align:{:.6f} "
                    "Uniform:{:.6f} IncStd:{:.6f} IncDist:{:.6f} Acc:{:.2%}".format(
                        epoch + 1, epoches, i + 1, len(train_loader),
                        float(sum_loss_tra / total_train_samples), loss_tra1.item(),
                        loss_inc_cls.item(), args.reg_target,
                        loss_align.item(), loss_uniform.item(),
                        inc_std, inc_dist,
                        correct_tra / total_train_samples,
                    )
                )

        # -----Validating----- #
        if epoch % 1 == 0:
            sum_loss_val = 0.0
            correct_val = 0.0
            total_valid_samples = 0.0

            model.eval()
            
            with torch.no_grad():
                for i, data in enumerate(valid_loader):
                    img_valid, labels_valid = data

                    img_valid = img_valid.to(device)
                    labels_valid = labels_valid.to(device)
                    outputsval = model(img_valid)
                    # pre_val, feature_fusion, recon = outputsval
                    pre_val = outputsval
                    pre_val = pre_val.to(device)

                    loss_weight1 =1 # 可以根据需要调整
                    loss_weight2 = 0  # 可以根据需要调整
                    # the average loss of a batch
                    loss_val1 = criterion(pre_val, labels_valid)
                    # loss_val2 = criterion1(feature_fusion, recon)
                    total_loss = loss_weight1 * loss_val1 
                    # sum_loss_val += loss_val.item() * labels_valin.size(0)
                    sum_loss_val += total_loss.item() * labels_valid.size(0)            
                    # prediction
                    _, pred = torch.max(pre_val.data, 1)
                   # the number of all validating sample
                    total_valid_samples += labels_valid.size(0)
                    # the correct number of prediction
                    correct_val += (pred == labels_valid).squeeze().sum().cpu().numpy()
                    # prepare for ROC
                    pre_val_abs = torch.nn.functional.softmax(pre_val, dim=1)
                    pred_abs_temp = torch.zeros(pre_val_abs.size()[0])
                    for m in range(pre_val_abs.size()[0]):
                        pred_abs_temp[m] = pre_val_abs[m][1]
                    label_val_list.extend(labels_valid.detach().cpu().numpy())
                    predict_val_list.extend(pred_abs_temp.detach().cpu().numpy())
      
                # acc
                epoch_acc = correct_val / total_valid_samples
                
                # auc
                if num_classes == 2:
                    ap_score, epoch_auc, epoch_eer, TPR_2, TPR_3, TPR_4 = cal_metrics(label_val_list, predict_val_list)
                    
                    # save the results
                    save_acc(epoch_acc, ap_score, epoch_auc, epoch_eer, TPR_2, TPR_3, TPR_4, epoch, log_path)
                    
                    if epoch_auc > best_auc:
                        best_auc = epoch_auc
                        torch.save(model.state_dict(), os.path.join(output_path, "best_auc.pkl"))
                    
                else:
                    ap_score = 0
                    epoch_auc = 0
                    epoch_eer = 0
                    TPR_2 = 0
                    TPR_3 = 0
                    TPR_4 = 0
                    # save the results
                    save_acc(epoch_acc, ap_score, epoch_auc, epoch_eer, TPR_2, TPR_3, TPR_4, epoch, log_path)
                
                print("Validating: Epoch[{:0>1}/{:0>1}] Acc:{:.2%} Auc:{:.2%}".format(epoch + 1, epoches, epoch_acc, epoch_auc))
                
                # select the best accuracy and save the best pretrained model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    if multiple_gpus:
                        best_model_wts = model.module.state_dict()
                        torch.save(best_model_wts, os.path.join(output_path, str(epoch+1) + str(best_acc) +"best.pkl"))
                    else:
                        best_model_wts = model.state_dict()
                        torch.save(best_model_wts, os.path.join(output_path, str(epoch+1) + str(best_acc) +"best.pkl"))
                if (epoch+1 ) /5 == 0:
                    if multiple_gpus:
                        torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch+1) + 'ff+_' + model_name))
                    else:
                        torch.save(model.state_dict(), os.path.join(output_path, str(epoch+1) + 'ff++' + model_name))
                    
            # update learning rate 
            scheduler.step()    # for other strategy

        # -----save the pretrained model----- #
        if (epoch+1 ) %5 == 0:
            if multiple_gpus:
                torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch+1) + 'ff++_' + model_name))
            else:
                torch.save(model.state_dict(), os.path.join(output_path, str(epoch+1) + 'ff++_' + model_name))
    
    # -----print the results----- #
    print('-'*20)        
    print('Best_accuracy:', best_acc)
    print('Best_AUC:', best_auc)
    # print time
    time_end = time.time() - time_open
    print('All time: ', time_end)
if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--name', '-n', type=str, default='drimnetpp')
    parse.add_argument('--train_txt_path', '-tp', type=str, required=True)
    parse.add_argument('--valid_txt_path', '-vp', type=str, required=True)
    parse.add_argument('--out_path', type=str, default='outputs/drimnetpp')
    parse.add_argument('--batch_size', '-bz', type=int, default=64)
    parse.add_argument('--epoches', '-e', type=int, default=40)
    parse.add_argument('--model_name', '-mn', type=str, default='drimnetpp.pkl')
    parse.add_argument('--continue_train', action='store_true')
    parse.add_argument('--model_path', '-mp', type=str, default=None)
    parse.add_argument('--num_classes', '-nc', type=int, default=2)
    parse.add_argument('--device', type=str, default='cuda:0')
    parse.add_argument('--num_workers', type=int, default=8)
    parse.add_argument('--pin_memory', action='store_true')
    parse.add_argument('--persistent_workers', action='store_true')
    parse.add_argument('--seed', default=7, type=int)
    parse.add_argument('--lr', default=5e-4, type=float)
    parse.add_argument('--dino_lr', default=3e-4, type=float)
    parse.add_argument('--align_weight', default=0.05, type=float)
    parse.add_argument('--uniform_weight', default=0.05, type=float)
    parse.add_argument('--inc_cls_weight', default=0.1, type=float)
    parse.add_argument(
        '--reg_target',
        default='inc',
        choices=['inc', 'fused', 'both'],
        help='Which embedding receives alignment/uniformity regularization.',
    )
    
    multiple_gpus = False
    gpus = [0,1]

    label_val_list = []
    predict_val_list = []
    
    main()

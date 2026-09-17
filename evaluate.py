import os
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from network.data import TestDataset
from network.transform import Data_Transforms
from network.plot_roc import plot_ROC
from network.utils import setup_seed, cal_metrics
# from network.models.efficientnet.fourth5 import cranet as MainNet
from network.MainNet_IHR_V2 import MainNet

# ---------------- Device 更健壮 ----------------
def get_device(d='cuda:5'):
    if torch.cuda.is_available():
        try:
            idx = int(str(d).split(':')[-1])
            _ = torch.cuda.get_device_properties(idx)  # 若无此卡会抛异常
            return torch.device(d)
        except Exception:
            print(f"[WARN] invalid device {d}, fallback to cuda:0")
            return torch.device('cuda:0')
    return torch.device('cpu')

def convert_syncbn_to_bn(module: nn.Module):
    """单卡推理把 SyncBN 转 BN 更稳"""
    mod = module
    if isinstance(module, nn.SyncBatchNorm):
        mod = nn.BatchNorm2d(
            module.num_features, eps=module.eps, momentum=module.momentum,
            affine=module.affine, track_running_stats=module.track_running_stats
        )
        if module.affine:
            with torch.no_grad():
                mod.weight.data.copy_(module.weight.data)
                mod.bias.data.copy_(module.bias.data)
        mod.running_mean = module.running_mean
        mod.running_var  = module.running_var
        mod.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        setattr(mod, name, convert_syncbn_to_bn(child))
    return mod

# ---- BACC / 阈值工具 ----
def compute_bacc(labels_t: torch.Tensor, probs_t: torch.Tensor, t: float):
    pred = (probs_t >= t).long()
    tp = ((pred == 1) & (labels_t == 1)).sum().item()
    tn = ((pred == 0) & (labels_t == 0)).sum().item()
    fp = ((pred == 1) & (labels_t == 0)).sum().item()
    fn = ((pred == 0) & (labels_t == 1)).sum().item()
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    bacc = 0.5 * (tpr + tnr)
    acc  = (tp + tn) / max(1, tp + tn + fp + fn)
    return bacc, acc, (tp, tn, fp, fn)

def scan_best_threshold(labels_t: torch.Tensor, probs_t: torch.Tensor, metric='bacc'):
    ths = torch.linspace(0, 1, 1001, device=probs_t.device)
    best_val, best_t, best_conf = -1.0, 0.5, None
    for t in ths:
        bacc, acc, conf = compute_bacc(labels_t, probs_t, float(t))
        val = bacc if metric == 'bacc' else acc
        if val > best_val:
            best_val, best_t, best_conf = val, float(t), conf
    return best_t, best_val, best_conf

def main():
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', '-bz', type=int, default=64)
    parse.add_argument('--model_dir', '-md', type=str, required=True,
                       help='Checkpoint file or directory containing checkpoints')
    parse.add_argument('--test_txt_path', '-tp', type=str, required=True)

    parse.add_argument('--num_classes', '-nc', type=int, default=2)
    parse.add_argument('--device', type=str, default='cuda:0')
    parse.add_argument('--seed', type=int, default=7)
    parse.add_argument('--select_by', type=str, default='auc', choices=['auc', 'acc', 'bacc'],
                      help='选最优模型的指标')
    # 新增：排查稳定性参数
    parse.add_argument('--num_workers', type=int, default=8, help='排查时先用0，确认OK再调大')
    parse.add_argument('--pin_memory', action='store_true', help='需要就带上该 flag')
    parse.add_argument('--persistent_workers', action='store_true', help='需要就带上该 flag')
    parse.add_argument('--recursive', action='store_true', help='递归扫描 model_dir 下子目录的权重')
    parse.add_argument('--results_file', type=str, default='outputs/evaluation.txt')
    args = parse.parse_args()

    # ------- 关键信息打印 -------
    print(f"[INFO] device request = {args.device}, cuda_count={torch.cuda.device_count()}")
    print(f"[INFO] test_txt_path = {args.test_txt_path}")
    print(f"[INFO] model_dir     = {args.model_dir}")
    print(f"[INFO] num_workers={args.num_workers}, pin_memory={args.pin_memory}, persistent_workers={args.persistent_workers}, recursive={args.recursive}")

    setup_seed(args.seed)
    device = get_device(args.device)
    torch.backends.cudnn.benchmark = True

    # 测试集：评测时不要打乱
    test_data = TestDataset(txt_path=args.test_txt_path, test_transform=Data_Transforms['test'])
    print(f"[INFO] test samples  = {len(test_data)}")
    if len(test_data) == 0:
        print("[ERROR] test set is empty! 检查 txt 路径/格式/是否可读")
        return

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,           # 先用0排查，多卡/稳定后再提高
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers
    )

    # 找到所有权重
    if os.path.isfile(args.model_dir):
        model_paths = [args.model_dir]
    elif not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f'Checkpoint path does not exist: {args.model_dir}')
    elif args.recursive:
        model_paths = []
        for root, _, files in os.walk(args.model_dir):
            for f in files:
                if f.endswith(('.pkl', '.pth', '.pt')):
                    model_paths.append(os.path.join(root, f))
        model_paths.sort()
    else:
        model_paths = sorted([os.path.join(args.model_dir, f)
                              for f in os.listdir(args.model_dir)
                              if f.endswith(('.pkl', '.pth', '.pt'))])
    print(f"[INFO] found {len(model_paths)} checkpoints")
    for p in model_paths[:10]:
        print("  -", p)
    if not model_paths:
        print(f"[WARN] No checkpoints found in {args.model_dir}")
        return

    best_model_path = None
    best_score = -1.0

    for model_path in model_paths:
        print(f"\n==== Testing model: {model_path} ====")

        # 按训练时签名构建模型
        # model = MainNet(model_name1='efficientnet-b4', advprop1=True, num_classes1=args.num_classes)
        model = MainNet(num_classes=2)
        model = convert_syncbn_to_bn(model)  # 单卡更稳
        model = model.to(device)

        # 加载权重（strict=False 兼容“只保存了部分模块”的情况）
        ckpt = torch.load(model_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing or unexpected:
            print(f"[load_state_dict] missing={len(missing)}, unexpected={len(unexpected)}")

        model.eval()

        # ------ 单样本 sanity check ------
        try:
            sample = next(iter(test_loader))
            imgs_s, _ = sample
            imgs_s = imgs_s[:1].to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                _ = model(imgs_s)
            print("[INFO] single forward ok")
        except StopIteration:
            print("[ERROR] test_loader empty (iterator).")
            return
        except Exception as e:
            print(f"[ERROR] single forward failed: {repr(e)}")
            return

        labels_all, probs_all = [], []
        n_total, n_correct_argmax = 0, 0

        with torch.no_grad():
            use_amp = torch.cuda.is_available()
            for imgs, labels in test_loader:
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # with torch.cuda.amp.autocast(enabled=use_amp):
                with torch.no_grad():
                    logits = model(imgs)                     # [B, C]
                    probs  = F.softmax(logits, dim=1)       # [B, C]

                # argmax ACC（Top-1）
                pred = probs.argmax(1)
                n_total += labels.size(0)
                n_correct_argmax += (pred == labels).sum().item()

                # 二分类：保存正类概率
                pos = probs[:, 1] if args.num_classes == 2 else probs.max(1).values
                labels_all.extend(labels.detach().cpu().tolist())
                probs_all.extend(pos.detach().cpu().tolist())

        # —— 统计指标 —— #
        labels_t = torch.tensor(labels_all, dtype=torch.long)
        probs_t  = torch.tensor(probs_all,  dtype=torch.float32)

        ap, auc, eer, TPR_2, TPR_3, TPR_4 = cal_metrics(labels_all, probs_all)

        bacc05, acc05, conf05 = compute_bacc(labels_t, probs_t, 0.5)
        t_bacc, best_bacc, conf_bacc = scan_best_threshold(labels_t, probs_t, metric='bacc')
        t_acc,  best_acc,  conf_acc  = scan_best_threshold(labels_t, probs_t, metric='acc')

        acc_argmax = n_correct_argmax / max(1, n_total)

        print(f"AUC={auc:.4f} | AP={ap:.4f} | EER={eer:.4f}")
        print(f"ACC@argmax={acc_argmax:.4f} | ACC@0.5={acc05:.4f} | BACC@0.5={bacc05:.4f} conf={conf05}")
        print(f"Best BACC={best_bacc:.4f} @ t={t_bacc:.3f} conf={conf_bacc}")
        print(f"Best ACC ={best_acc:.4f}  @ t={t_acc:.3f}  conf={conf_acc}")

        # 画 ROC（保持你的接口）
        plot_ROC(
            labels_all,
            probs_all,
            acc05,
            args.test_txt_path,
            model_path,
            output_path=args.results_file,
        )

        # 选择“最佳模型”的标准
        pick = {'auc': auc, 'acc': best_acc, 'bacc': best_bacc}[args.select_by]
        if pick > best_score:
            best_score = pick
            best_model_path = model_path

    if best_model_path:
        print(f"\n>>> Best model by [{args.select_by.upper()}] is:\n{best_model_path}\nscore={best_score:.4f}")
    else:
        print("\nNo valid model found.")

if __name__ == '__main__':
    main()

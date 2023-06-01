#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
"""

import argparse
from tqdm import tqdm
import pickle
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.append("../scripts")
from utils import make_directory, make_logger, get_run_info, count_items
from transformers import BertForTokenClassification, BertTokenizer, BertConfig, AutoModelForTokenClassification, AutoTokenizer, AutoConfig, get_polynomial_decay_schedule_with_warmup
from transformers.modeling_outputs import ModelOutput
import bp_dataset
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch.cuda.amp import autocast, GradScaler

import logging
logger = logging.getLogger(__name__)


@autocast()
@torch.no_grad()
def predict(model: BertForTokenClassification, loader: DataLoader, desc=None):
    r"""
    Return:
    ---
    all_pre : listd
    all_label : list
    au : floatc
    ap : float
    f1 : float
    num_pos : int
    num_total : int
    """
    model.eval()
    device = next(model.parameters()).device
    all_pred, all_label = list(), list()
    for it, (ids, label, _) in enumerate(tqdm(loader, desc=desc)):
        ids = ids.to(device)
        logits = model.forward(ids).logits[:, 1:-1, 1] # (B, S, 2)
        logits = logits.detach().cpu().reshape(-1)
        label = label.reshape(-1)
        k = torch.where(label >= 0)[0]
        label = label[k].numpy() 
        logits = torch.sigmoid(logits[k].float()).numpy()
        all_pred.append(logits)
        all_label.append(label)
    all_pred = np.concatenate(all_pred)
    all_label = np.concatenate(all_label)
    auc = roc_auc_score(all_label, all_pred)
    ap = average_precision_score(all_label, all_pred)
    f1 = f1_score(all_label, all_pred > 0.5)
    return all_pred, all_label, auc, ap, f1, np.where(all_label > 0)[0].shape[0], len(all_label)


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dnabert", action="store_true")
    p.add_argument("--no-pretrain", action="store_true")
    p.add_argument("--freeze-bert", type=int)
    p.add_argument("-n", "--n-fold", type=int, default=5)
    p.add_argument("-m", "--model-path", required=True)
    p.add_argument("-b", "--batch-size", type=int, default=8, help="batch size")
    p.add_argument("-lr", default=1e-5, type=float, help="learning rate")
    p.add_argument("--patience", type=int, default=10, help="patience in early stopping")
    p.add_argument("--num-workers", type=int, default=8, help="num_workers in dataloader")
    p.add_argument('-d', "--device", required=False, help="device")
    p.add_argument('-o', "--outdir", required=True, help="output directory")
    p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    args = get_args().parse_args()

    args.outdir = make_directory(args.outdir)
    logger = make_logger(filename=os.path.join(args.outdir, "train.log"))
    logger.info(get_run_info(argv=sys.argv, args=args))

    if args.dnabert:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        ds = bp_dataset.BranchPointData(tokenizer=tokenizer, dnabert_mode=True)
    else:
        ds = bp_dataset.BranchPointData()
    logger.info("dataset: {}".format(ds))
    if args.no_pretrain:
        config = AutoConfig.from_pretrained(args.model_path)
        config.num_labels = 2
        model = AutoModelForTokenClassification(config)
    else:
        model = AutoModelForTokenClassification.from_pretrained(args.model_path, num_labels=2) 
        if args.freeze_bert is not None:
            for n, p in model.bert.embeddings.named_parameters():
                p.requires_grad = False
                logger.info("freeze: {}".format(n))
            for n, p in model.bert.encoder.named_parameters():
                if int(n.split('.')[1]) < args.freeze_bert:
                    logger.info("freeze: {}".format(n))
                    p.requires_grad = False


    logger.info("{}".format(model))

    grouped_inds = list()
    splits = GroupKFold(n_splits=args.n_fold)
    for t1, t2 in splits.split(X=range(len(ds)), groups=ds.chroms): # split by chromosome
        grouped_inds.append(t2)
    for fold in range(args.n_fold):
        logger.info("Fold{} (n={}/{}): {}".format(fold, len(ds) - len(grouped_inds[fold]), len(grouped_inds[fold]), count_items(ds.chroms[grouped_inds[fold]])))
        fold_outdir = make_directory("{}/fold{}".format(args.outdir, fold))
        model.save_pretrained(fold_outdir)
    del model


    device = torch.device("cuda")

    best_ap = -1
    wait = 0

    # demo = ds[0]
    # pickle.dump(demo, open("./{}/demo.data.pkl".format(args.outdir), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
    for epoch in range(200):
        if epoch > 0:
            val_ap = np.mean(cv_ap)
        else:
            val_ap = None

        cv_auc = list()
        cv_ap = list()
        cv_f1 = list()
        cv_test_pred = list()
        cv_test_label = list()
        for fold in range(args.n_fold):
            all_inds = grouped_inds[fold:] + grouped_inds[:fold]
            test_inds = all_inds[0]
            val_inds = all_inds[1]
            train_inds = np.concatenate(all_inds[2:])
            train_loader = DataLoader(
                Subset(ds, indices=train_inds),
                batch_size=args.batch_size,
                drop_last=True,
                num_workers=args.num_workers,
                shuffle=True
            )
            val_loader = DataLoader(
                Subset(ds, indices=val_inds),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            test_loader = DataLoader(
                Subset(ds, indices=test_inds),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

            fold_outdir = "{}/fold{}".format(args.outdir, fold)
            ckpt = "{}/checkpoint.fold{}.pt".format(fold_outdir, fold)
            epoch_loss = 0
            epoch_ap = list()
            epoch_auc = list()
            if epoch == 0:
                model = AutoModelForTokenClassification.from_pretrained(fold_outdir)
                model = model.to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=1e-6
                )
                scaler = GradScaler()
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=1, mode="max", min_lr=1e-7)
            else:
                del model, optimizer, scaler
                model = AutoModelForTokenClassification.from_pretrained(fold_outdir) # will be reload in model.load_state_dict
                model = model.to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=1e-6
                )
                scaler = GradScaler()
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=1, mode="max", min_lr=1e-7)

                d = torch.load(ckpt)
                model.load_state_dict(d['model'])
                optimizer.load_state_dict(d['optimizer'])
                scaler.load_state_dict(d['scaler'])
                scheduler.load_state_dict(d['scheduler'])

                scheduler.step(val_ap)


            model.train()
            
            pbar = tqdm(train_loader, desc="Epoch{}-{}".format(epoch, fold), total=len(train_loader))
            for it, (ids, label, _) in enumerate(pbar):
                if epoch == 0 and fold == 0:
                    torch.save((ids, label), "{}/demo.pt".format(args.outdir))
                ids = ids.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                with autocast():
                    logits = model.forward(ids).logits[:, 1:-1]
                    # loss = F.cross_entropy(logits.reshape(-1, 2), label.reshape(-1), label_smoothing=0.01)
                    loss = F.cross_entropy(logits.reshape(-1, 2), label.reshape(-1))
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                
                pbar.set_postfix_str("loss/lr={:.5e}/{:.3e}".format(epoch_loss / (it + 1), optimizer.param_groups[-1]['lr']))
            
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, ckpt)
            torch.save(model.state_dict(), "{}/model_weights.tmp.pt".format(fold_outdir))
            # model.save_pretrained(fold_outdir)

            _, _, auc, ap, f1, num_pos, num_all = predict(model, val_loader, desc="validating")
            logger.info("Validation{}-{}(AUC/AP/F1): {:.4f} {:.4f} {:.4f} ({}/{}={:.3f})".format(epoch, fold, auc, ap, f1, num_pos, num_all, num_pos / num_all))
            cv_auc.append(auc)
            cv_ap.append(ap)
            cv_f1.append(f1)

            test_pred, test_label, auc, ap, f1, num_pos, num_all = predict(model, test_loader, desc="test")
            logger.info("Test{}-{}(AUC/AP/F1): {:.4f} {:.4f} {:.4f} ({}/{}={:.3f})".format(epoch, fold, auc, ap, f1, num_pos, num_all, num_pos / num_all))
            cv_test_pred.append(test_pred.astype(np.float16))
            cv_test_label.append(test_label.astype(np.int8))

        logger.info("CV-results(epoch={})(AUC/AP/F1): {:.4f} {:.4f} {:.4f}".format(epoch, np.mean(cv_auc), np.mean(cv_ap), np.mean(cv_f1)))

        if np.mean(cv_ap) > best_ap:
            best_ap = np.mean(cv_ap)
            for fold in range(args.n_fold):
                fold_outdir = "{}/fold{}".format(args.outdir, fold)
                shutil.copy("{}/model_weights.tmp.pt".format(fold_outdir), "{}/pytorch_model.bin".format(fold_outdir))
                with open("{}/test_results.txt".format(fold_outdir), 'w') as out:
                    for label, score in zip(cv_test_label[fold], cv_test_pred[fold]):
                        out.write("{:d}\t{:.4f}\n".format(label, score))
            logger.info("best models saved\n")
            wait = 0
        else:
            wait += 1
            logger.info("wait: {}\n".format(wait))
            if wait >= args.patience:
                logger.info("early stopped!")
                break



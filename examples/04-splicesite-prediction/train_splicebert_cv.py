#!/usr/bin/env python3

import argparse
import shutil
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
sys.path.append("../scripts")
from utils import make_directory, make_logger, get_run_info
from sklearn.model_selection import StratifiedKFold, GroupKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.cuda.amp import autocast, GradScaler
import spliceator_data

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--no-pretrain", action="store_true")
    p.add_argument('-m', "--model-path", required=True)
    p.add_argument('-p', "--positive", required=True)
    p.add_argument('-n', "--negative", required=True)
    p.add_argument('-o', "--outdir", required=True)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("-b", "--batch-size", type=int, default=16)
    # p.add_argument("--shift", type=int, default=0)
    # p.add_argument("--freeze-bert", action="store_true")
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("-lr", type=float, default=5E-5)
    p.add_argument("--debug", action="store_true", help="debug mode")
    p.add_argument('--seed', type=int, default=2020)
    return p

@torch.no_grad()
@autocast()
def test_model(model: AutoModelForSequenceClassification, loader: DataLoader):
    r"""
    Return:
    auc : float
    f1 : float
    pred : list
    true : list
    """
    model.eval()
    pred, true = list(), list()
    for it, (ids, mask, label) in enumerate(tqdm(loader, desc="predicting", total=len(loader))):
        ids = ids.to(device)
        mask = mask.to(device)
        # score = torch.softmax(model.forward(ids, attention_mask=mask).logits, dim=1)[:, 1].detach().cpu().numpy()
        score = torch.sigmoid(model.forward(ids, attention_mask=mask).logits.squeeze(1)).detach().cpu().numpy()
        del ids
        label = label.numpy()
        pred.append(score.astype(np.float16))
        true.append(label.astype(np.float16))
    pred = np.concatenate(pred)
    true = np.concatenate(true)
    auc_list = roc_auc_score(true.T, pred.T)
    f1 = f1_score(true.T, pred.T > 0.5)
    return auc_list, f1, pred, true

if __name__ == "__main__":
    args = get_args().parse_args()

    args.outdir = make_directory(args.outdir)
    logger = make_logger(filename=os.path.join(args.outdir, "train.log"), level="DEBUG" if args.debug else "INFO")
    logger.info(get_run_info(argv=sys.argv, args=args))

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    ds = spliceator_data.SpliceatorDataset(
        positive=args.positive,
        negative=args.negative,
        tokenizer=tokenizer,
        max_len=400,
        # shift=args.shift
    )

    splits = list()
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    for _, inds in splitter.split(np.arange(len(ds)), y=ds.labels):
        splits.append(inds)

    best_auc = -1
    best_epoch = -1
    fold_ckpt = dict()
    
    for epoch in range(200):
        epoch_val_auc = list()
        epoch_val_f1 = list()
        epoch_test_auc = list()
        epoch_test_f1 = list()

        for fold in range(10):
            if args.debug and fold > 0:
                logger.warning("skip fold {}".format(fold))
                continue
            ## setup folder
            fold_outdir = make_directory(os.path.join(args.outdir, "fold{}".format(fold)))
            ckpt = os.path.join(fold_outdir, "checkpoint.pt")
            fold_ckpt[fold] = ckpt

            ## setup dataset
            all_inds = splits[fold:] + splits[:fold]
            train_inds = np.concatenate(all_inds[3:])
            val_inds = all_inds[0]
            test_inds = np.concatenate(all_inds[1:3])

            if args.debug:
                train_inds = np.random.permutation(train_inds)[:100]
                val_inds = np.random.permutation(val_inds)[:100]
                test_inds = np.random.permutation(test_inds)[:100]

            train_loader = DataLoader(
                Subset(ds, indices=train_inds),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=args.num_workers
            )
            val_loader = DataLoader(
                Subset(ds, indices=val_inds),
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            test_loader = DataLoader(
                Subset(ds, indices=test_inds),
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )

            ## setup model, optimizer & scaler
            if epoch > 0 or (args.resume and os.path.exists(ckpt)):
                if epoch > 0:
                    del model, optimizer , scaler
                d = torch.load(ckpt)
                # logger.info("load ckpt: {}".format(ckpt))
                if args.no_pretrain:
                    config = AutoConfig.from_pretrained(args.model_path)
                    config.num_labels = 1
                    model = AutoModelForSequenceClassification(config).to(device)
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=1).to(device)
                model.load_state_dict(d["model"])
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=1E-6
                )
                optimizer.load_state_dict(d["optimizer"])
                scaler = GradScaler()
                scaler.load_state_dict(d["scaler"])
                if epoch == 0:
                    trained_epochs = d.get("epoch", -1) + 1
            else:
                if args.no_pretrain:
                    config = AutoConfig.from_pretrained(args.model_path)
                    config.num_labels = 1
                    model = AutoModelForSequenceClassification(config).to(device)
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=1).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=1E-6
                )
                torch.save((train_inds, val_inds, test_inds), "{}/split.pt".format(args.outdir))
                scaler = GradScaler()
                trained_epochs = 0

            # if args.freeze_bert:
            #     for p in model.bert.embeddings.parameters():
            #         p.requires_grad = False
            #     for p in model.bert.encoder.parameters():
            #         p.requires_grad = False


            model.train()

            ## train
            pbar = tqdm(train_loader, 
                        total=len(train_loader), 
                        desc="Epoch{}-{}".format(epoch +  trained_epochs, fold)
                    )
            epoch_loss = 0
            for it, (ids, mask, label) in enumerate(pbar):
                ids, mask, label = ids.to(device), mask.to(device), label.to(device).float()
                optimizer.zero_grad()
                with autocast():
                    logits = model.forward(ids, attention_mask=mask).logits.squeeze(1)
                    # if torch.isnan(logits).sum() > 0:
                    #     raise ValueError("NaN in logits: {}".format(torch.isnan(logits).sum()))
                    loss = F.binary_cross_entropy_with_logits(logits, label).mean()
                    # if torch.isnan(loss).sum() > 0:
                    #     raise ValueError("NaN in loss: {}".format(torch.isnan(loss).sum()))

                scaler.scale(loss).backward()
                scaler.step(optimizer) #0.step()
                scaler.update()
                # for n, p in model.named_parameters():
                #     if torch.isnan(p).sum() > 0:
                #         raise ValueError("NaN in weight {}: {}".format(n, torch.isnan(p).sum()))
                
                epoch_loss += loss.item()

                pbar.set_postfix_str("loss/lr={:.4f}/{:.2e}".format(
                    epoch_loss / (it + 1), optimizer.param_groups[-1]["lr"]
                ))
            
            ## validate
            val_auc, val_f1, val_score, val_label = test_model(model, val_loader)
            torch.save((val_score, val_label), os.path.join(fold_outdir,  "val.pt"))
            epoch_val_auc.append(val_auc)
            epoch_val_f1.append(val_f1)
            test_auc, test_f1, test_score, test_label = test_model(model, test_loader)
            torch.save((test_score, test_label), os.path.join(fold_outdir,  "test.pt"))
            epoch_test_auc.append(test_auc)
            epoch_test_f1.append(test_f1)
            logger.info("validate/test({}-{})AUC/F1: {:.4f} {:.4f} {:.4f} {:.4f}".format(epoch, fold, val_auc, val_f1, test_auc, test_f1))

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch
            }, ckpt)
        
        logger.info("Epoch{}validation&test(AUC/F1): {:.4f} {:.4f} {:.4f} {:.4f}".format(
            epoch,
            np.mean(epoch_val_auc),
            np.mean(epoch_val_f1),
            np.mean(epoch_test_auc),
            np.mean(epoch_test_f1)
        ))

        if np.mean(epoch_val_auc) > best_auc:
            best_auc = np.mean(epoch_val_auc)
            best_epoch = epoch
            for fold in range(10):
                ckpt = fold_ckpt[fold]
                shutil.copy2(ckpt, "{}.best_model.pt".format(ckpt))
            wait = 0
            logger.info("model saved\n")
        else:
            wait += 1
            logger.info("wait{}\n".format(wait))
            if wait >= args.patience:
                break


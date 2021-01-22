#coding=utf-8

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import random
from tqdm import tqdm, trange
import pdb
import warnings
warnings.filterwarnings('ignore')

import numpy as np

import torch
from torch.nn import CrossEntropyLoss, MSELoss, MultiLabelSoftMarginLoss, BCEWithLogitsLoss

from run_classifier_dataset_utils import compute_metrics, fix_label, fix_14_label

def hier_pred_eval(args, _e, category_map, logger, model, dataloader, device, task_name, eval_type='valid'):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    out_label_ids = None
    ddd = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]

    for input_ids, input_mask, segment_ids, category_label_ids, sentiment_label_ids in dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        category_label_ids = category_label_ids.to(device)
        sentiment_label_ids = sentiment_label_ids.to(device)

        with torch.no_grad():
            if args.model_type != 'Baseline_1' and args.model_type != 'AddOD':
                tmp_eval_loss, c_loss, s_loss, category_logits, sentiment_logits = model(_e, category_map, input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                    cate_labels=category_label_ids, senti_labels=sentiment_label_ids)
            else:
                logit, tmp_eval_loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, senti_labels=sentiment_label_ids)
                logits = logit.detach().cpu().numpy()
                category_logits = (np.argmax(logits, axis=-1)<=2).astype(int)
                sentiment_logits = (logits.argmax(axis=-1)[...,None] == np.arange(logits.shape[-1]-1)).astype(int)
        # create eval loss and other metric required by the task

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        logits = []
        if args.model_type != 'Baseline_1' and args.model_type != 'AddOD':
            category_logits = category_logits.detach().cpu().numpy()
            sentiment_logits = sentiment_logits.detach().cpu().numpy()

            for i in range(len(category_logits)):
                tmp = []
                for j in range(len(category_logits[i])):
                    if category_logits[i][j] < 0.5:
                        tmp = np.append(tmp, ddd[-1])
                    else:
                        tmp = np.append(tmp, ddd[np.argmax(sentiment_logits[i][j])])
                logits.append(tmp)
        else:
            logits = list(sentiment_logits.reshape(sentiment_logits.shape[0], -1))

        label_ids = []

        sentiment_label_ids = torch.argmax(sentiment_label_ids, dim=-1)
        for i in range(len(category_label_ids)):
            tmp = []
            for j in range(len(category_label_ids[i])):
                if category_label_ids[i][j] == 0:
                    tmp = np.append(tmp, ddd[-1])
                else:
                    tmp = np.append(tmp, ddd[sentiment_label_ids[i][j]])

            label_ids.append(tmp)
        if len(preds) == 0:
            preds.append(logits)
            out_label_ids = label_ids
        else:
            preds[0] = np.append(
                preds[0], logits, axis=0)
            out_label_ids = np.append(
                out_label_ids, label_ids, axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]

    preds = (preds > 0).astype(int)
    out_label_ids = (out_label_ids > 0).astype(int)

    result = compute_metrics(task_name, preds, out_label_ids)
    # pdb.set_trace()

    logger.info("***** Valid results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    if eval_type == 'test':
        if task_name != 'csc14':
            if task_name != 'jcsc14':
                preds, _ = fix_label(args.domain_type, 'sentiment', preds, out_label_ids)
            else:
                preds, _ = fix_14_label(preds, out_label_ids)
        fix_result = compute_metrics(task_name, preds, out_label_ids)
        # pdb.set_trace()
        logger.info("***** Category Valid results *****")
        for key in sorted(fix_result.keys()):
            logger.info("  %s = %s", key, str(fix_result[key]))
        return result, fix_result

    return result

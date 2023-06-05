import os
import sys
### sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy
import json
import glob
import time
import random
import pickle
import logging
import argparse
import datasets
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (
    Dataset,
    RandomSampler,
    SequentialSampler,
    DataLoader,
)
from utils.metric_logger import TensorboardLogger

import transformers
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
)
transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)
log_json = []

class RobertaForClusterIdentification(RobertaForSequenceClassification):
    def __init__(self,
                 config):
        super().__init__(config)
        self.ci_classifier = torch.nn.Linear(self.roberta.config.hidden_size, 2)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels = None,
        mode = None,
        args = None,
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # pooled_output = outputs[1]

        # pooled_output = self.dropout(pooled_output)
        # logits = self.ci_classifier(pooled_output)
        logits = self.classifier(outputs[0])

        if mode == 'train':
            loss_fct = CrossEntropyLoss()
        else:
            loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output)

class RedditThreadDataset(Dataset):
    def __init__(self,
                 args,
                 tokenizer,
                 mode):
        super(RedditThreadDataset, self).__init__()
        assert mode in ['train', 'val', 'test']

        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.examples = []

        def comment_to_string(comment, sep):
            string = f"{comment['author_anon']}: {comment['body']}".replace('\n',' ').replace('\r', ' ')
            if comment['replies'] == []:
                return string
            for reply in comment['replies']:
                string += sep
                string += comment_to_string(reply, sep)
            return string

        with open(f'/gallery_getty/jaewoo.ahn/multimodal-thread-sum/data/{mode}_edited_csums.json', 'r') as fp:
            data = json.load(fp)['threads']

        with open('/gallery_tate/keighley.overbay/thread-summarization/data/image_captions_blip2_caption_best.json', 'r') as fp:
            imgcaps = json.load(fp)

        num_pos, num_neg = 0, 0
        for example_idx, example in enumerate(data):
            sub_id = example['submission_id']
            original_post = example['raw_caption']
            comment_id2comments = {}
            for comment in example['comments']:
                comment_id2comments[comment['comment_id']] = comment_to_string(comment, "")
            cluster_id2comments = {}
            for cluster_id, comments in example['clusters_auto'].items():
                cluster_comments = [comment_id2comments[x] for x in comments['comments']]
                cluster_id2comments[cluster_id] = ' '.join(cluster_comments)
                if cluster_id in example['csums_with_ids'].keys():
                    label = 1
                    num_pos += 1
                else:
                    label = 0
                    num_neg += 1
                self.examples.append((imgcaps[sub_id], original_post, cluster_id2comments[cluster_id], label, sub_id, '-'.join(comments['comments']), mode))
        print(f'num. of {mode} dataset: {len(self.examples)}')
        print(f'num. of positive: {num_pos}, num. of negative: {num_neg}')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        # TODO: max 5 clusters
        caption, original_post, cluster_comment, label, sub_id, cluster_id, mode = self.examples[index]

        inputs = self.tokenizer(f"Original Post: {original_post} Image: {caption}. {cluster_comment}",
                                truncation=True,
                                padding='max_length',
                                max_length=self.args.max_seq_length)
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']

        assert len(input_ids) == self.args.max_seq_length
        assert len(attention_mask) == self.args.max_seq_length

        feature = [
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
            sub_id,
            cluster_id,
        ]

        return feature

def main():
    parser = argparse.ArgumentParser()

    ## Required (or pre-defined) params
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--model_type", default='roberta', type=str)
    parser.add_argument("--model_name_or_path", default='', type=str,
                        help="Path to pre-trained model or shortcut name")

    ## Configs
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision instead of 32-bit")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run test on the test set.")
    parser.add_argument("--tokenizer_name", default="",
                        type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--config_name", default="",
                        type=str, help="Pretrained config name or path if not the same as model_name")

    # Misc: other params (model, input, etc)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--logging_steps', type=int, default=100, help="Log every X updates steps.")
    parser.add_argument('--save_epoch', type=int, default=5, help="Save checkpoint every X epochs.")
    parser.add_argument('--save_after_epoch', type=int, default=-1, help="Save checkpoint after epoch.")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        raise NotImplementedError
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method="env://"
        )
        args.n_gpu = 1
    args.device = device

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # set seed
    set_seed(args)

    # Output config
    os.makedirs(args.output_dir, exist_ok=True)

    # Load saved checkpoint
    recover_args = {'global_step': 0, 'step': 0, 'last_checkpoint_dir': None,
                    'last_best_checkpoint_dir': None, 'last_best_score': None}

    if os.path.exists(args.output_dir):
        save_file = os.path.join(args.output_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                texts = f.read().split('\n')
                last_saved = texts[0]
                last_saved = last_saved.strip()
                last_best_saved = texts[1].split('best: ')[-1].strip()
                last_best_score = json.loads(texts[2])

        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        if last_saved:
            folder_name = os.path.splitext(last_saved.split('/')[0])[0] # in the form of checkpoint-00001 or checkpoint-00001/pytorch_model.bin
            recover_args['last_checkpoint_dir'] = os.path.join(args.output_dir, folder_name)
            recover_args['epoch'] = int(folder_name.split('-')[1])
            recover_args['global_step'] = int(folder_name.split('-')[2])
            recover_args['last_best_checkpoint_dir'] = os.path.join(args.output_dir, last_best_saved)
            recover_args['last_best_score'] = last_best_score
            assert os.path.isfile(os.path.join(recover_args['last_checkpoint_dir'], WEIGHTS_NAME)), "Last_checkpoint detected, but file not found!"

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if recover_args['last_checkpoint_dir'] is not None: # recovery
        args.model_name_or_path = recover_args['last_checkpoint_dir']
        logger.info(" -> Recovering model from {}".format(recover_args['last_checkpoint_dir']))

    # tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base",
                                              cache_dir="/gallery_getty/jaewoo.ahn/multimodal-thread-sum/text_only_roberta/")

    # Prepare model
    model = RobertaForClusterIdentification.from_pretrained('roberta-base',
                                                         cache_dir="/gallery_getty/jaewoo.ahn/multimodal-thread-sum/text_only_roberta/")
    if recover_args['last_checkpoint_dir'] is not None or args.model_name_or_path != '': # recovery
        ### model_logging = model.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin')))
        ### logger.info(f"{model_logging}")
        model = RobertaForClusterIdentification.from_pretrained(args.model_name_or_path)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info('Total Parameters: {}'.format(total_params))

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # load eval dataset
    eval_dataset = RedditThreadDataset(args, tokenizer, 'val')

    # load tensorboard
    tb_log_dir = os.path.join(args.output_dir, 'train_logs')
    meters = TensorboardLogger(
        log_dir=tb_log_dir,
        delimiter="  ",
    )

    # training
    if args.do_train:
        train_dataset = RedditThreadDataset(args, tokenizer, 'train')
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, meters, recover_args)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    # test
    if args.do_test:
        test_dataset = RedditThreadDataset(args, tokenizer, 'test')
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))

        try:
            with open(os.path.join(args.output_dir, "last_checkpoint"), "r") as f:
                texts = f.read().split('\n')
                best_saved = texts[1].split('best: ')[-1].strip()
            checkpoints = [ckpt for ckpt in checkpoints if best_saved in ckpt]
        except:
            pass
        logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        test_log_json = []
        for checkpoint in checkpoints:
            epoch = checkpoint.split('-')[-2]
            global_step = checkpoint.split('-')[-1]
            ### model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model.bin')))
            model = RobertaForClusterIdentification.from_pretrained(checkpoint)
            model.to(args.device)
            ### from torch.utils.data import ConcatDataset
            ### concatenated_dataset = ConcatDataset([train_dataset, eval_dataset, test_dataset])
            ### test_scores = evaluate(args, model, concatenated_dataset, 'test', prefix=global_step)
            test_scores = evaluate(args, model, test_dataset, 'test', prefix=global_step)

            epoch_log = {'epoch': epoch, 'test_scores': test_scores}
            test_log_json.append(epoch_log)

            if args.local_rank in [-1, 0]:
                with open(args.output_dir + '/test_logs.json', 'w') as fp:
                    json.dump(test_log_json, fp)

    # close the tb logger
    meters.close()
    logger.info("Good Job Computer!")

def train(args, train_dataset, eval_dataset, model, meters, recover_args):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, sampler=train_sampler, batch_size=args.train_batch_size, pin_memory=True)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = []
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                     num_warmup_steps=args.warmup_steps,
                                     num_training_steps=t_total)

    if recover_args['global_step'] > 0 and os.path.isfile(os.path.join(recover_args['last_checkpoint_dir'], 'optimizer.pth')): # recovery
        last_checkpoint_dir = recover_args['last_checkpoint_dir']
        logger.info(
            "Load optimizer from {}".format(last_checkpoint_dir))
        optimizer_to_load = torch.load(
            os.path.join(last_checkpoint_dir, 'optimizer.pth'),
            map_location=torch.device("cpu"))
        optimizer.load_state_dict(optimizer_to_load.pop("optimizer"))
        scheduler.load_state_dict(optimizer_to_load.pop("scheduler"))

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = recover_args['global_step']
    start_epoch = recover_args['epoch'] + 1 if global_step > 0 else 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    best_scores = {
        'epoch': 0,
        'global_step': 0,
        'scores': {'f1': 0.0}
    }
    if recover_args['last_best_score'] is not None:
        best_scores = recover_args['last_best_score']

    # load loss function
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, int(args.num_train_epochs)):
        t_start = time.time()
        tbar = tqdm(train_dataloader, ncols=70)
        for step, batch in enumerate(tbar):
            tbar.set_description(f'Training loss = {logging_loss}')
            model.train()

            input_ids = batch[0].to(args.device, non_blocking=True)
            attention_mask = batch[1].to(args.device, non_blocking=True)
            labels = batch[2].to(args.device, non_blocking=True)

            if args.fp16:
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        mode='train',
                    )
                    loss = outputs[0]
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    mode='train',
                )
                loss = outputs[0]

            if args.n_gpu > 1: loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            logging_loss = round(loss.item(), 5)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # do gradient clipping
                if args.max_grad_norm > 0:
                   torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()  # Update learning rate schedule
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                model.zero_grad()
                global_step += 1

                # update tensorboard
                meters.update_metrics({'batch_metrics': {'loss': loss}})
                meters.update_params({'params': {'bert_lr': optimizer.param_groups[0]['lr']}})

                if args.logging_steps > 0 and (global_step + 1) % args.logging_steps == 0:
                    meters.get_logs(global_step+1)

        # Evaluation
        logger.info("Epoch: %d, global_step: %d" % (epoch, global_step))
        eval_scores = evaluate(args, model, eval_dataset, 'val', prefix=global_step)

        # Select f1 score as metric
        if eval_scores['f1'] > best_scores['scores']['f1']:
            best_scores['scores'] = eval_scores
            best_scores['epoch'] = epoch
            best_scores['global_step'] = global_step

        # save checkpoints
        if (args.local_rank in [-1, 0]) and (args.save_epoch>0 and epoch % args.save_epoch == 0) and (epoch > args.save_after_epoch):
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(epoch, global_step))
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            optimizer_to_save = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }

            save_num = 0
            while (save_num < 3):
                try:
                    logger.info("Saving model attempt: {}".format(save_num))
                    ### torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    torch.save(optimizer_to_save, os.path.join(output_dir, 'optimizer.pth'))
                    save_file = os.path.join(args.output_dir, 'last_checkpoint')
                    with open(save_file, 'w') as f:
                        f.write('checkpoint-{}-{}/pytorch_model.bin\n'.format(epoch, global_step))
                        f.write(f'best: checkpoint-{best_scores["epoch"]}-{best_scores["global_step"]}\n')
                        json.dump(best_scores, f)
                    break
                except:
                    save_num += 1
            logger.info("Saving model checkpoint {0} to {1}".format(epoch, output_dir))

        epoch_log = {'epoch': epoch, 'eval_scores': eval_scores, 'best_scores': best_scores['scores']}
        log_json.append(epoch_log)

        logger.info("PROGRESS: {}%".format(round(100*(epoch + 1) / args.num_train_epochs, 4)))

        if args.local_rank in [-1, 0]:
            with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                json.dump(log_json, fp)

            t_end = time.time()
            logger.info('Epoch: %d, Train Time: %.3f' % (epoch, t_end - t_start))

    return global_step, tr_loss / global_step

def evaluate(args, model, eval_dataset, mode, prefix=''):
    t_start = time.time()
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    test_dataloader = DataLoader(eval_dataset, num_workers=args.num_workers, sampler=test_sampler, batch_size=args.eval_batch_size, pin_memory=True)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    results_dict = defaultdict(list)
    for batch in tqdm(test_dataloader, ncols=70):
        model.eval()

        input_ids = batch[0].to(args.device, non_blocking=True)
        attention_mask = batch[1].to(args.device, non_blocking=True)
        labels = batch[2].to(args.device, non_blocking=True)
        sub_ids = batch[3]
        cluster_ids = batch[4]

        with torch.no_grad():
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                mode=mode,
                args=args,
            )
            probs = F.softmax(logits)[:,1]
            preds = torch.argmax(logits, dim=1)
            if args.local_rank != -1:
                raise NotImplementedError
                loss = gather_tensor(loss)
                preds = gather_tensor(preds)
                labels = gather_tensor(labels)
            results_dict['loss'].append(loss.cpu().detach().numpy())
            results_dict['probs'].append(probs.cpu().detach().numpy())
            results_dict['preds'].append(preds.cpu().detach().numpy())
            results_dict['labels'].append(labels.cpu().detach().numpy())
            results_dict['submission_ids'] += sub_ids
            results_dict['cluster_ids'] += cluster_ids

    for key, value in results_dict.items():
        if key in ['submission_ids', 'cluster_ids']:
            continue
        elif results_dict[key][0].shape == ():
            results_dict[key] = np.array(value)
        else:
            results_dict[key] = np.concatenate(value, axis=0)

    acc, r, p, f1 = compute_accuracy(results_dict['preds'], results_dict['labels'])

    total_scores = {
        'loss': round(np.mean(results_dict['loss']).item(), 4),
        'accuracy': round(acc, 4),
        'recall': round(r, 4),
        'precision': round(p, 4),
        'f1': round(f1, 4),
    }

    logger.info("Eval Results:")
    logger.info(f'Eval Score: {total_scores}')

    t_end = time.time()
    logger.info('Eval Time Cost: %.3f' % (t_end - t_start))

    with open(os.path.join(args.output_dir, 'pred_clusters.pkl'), 'wb') as fp:
        pred_clusters = {}
        for i,x in enumerate(results_dict['preds']):
            if x == 1:
                cluster_ids = results_dict['cluster_ids'][i]
                submission_id = results_dict['submission_ids'][i]
                prob = results_dict['probs'][i]
                pred_clusters[cluster_ids] = {'submission_id': submission_id, 'prob': prob}
            else:
                continue
        ### pred_cluster_ids = [results_dict['cluster_ids'][i] for i,x in enumerate(results_dict['preds']) if x == 1]
        pickle.dump(pred_clusters, fp)

    return total_scores

def compute_accuracy(preds, labels):
    is_right = [int(x[0]==x[1]) for x in zip(preds, labels)]
    accuracy = sum(is_right) / len(is_right)

    num_tp, num_fp, num_fn, num_tn = 0, 0, 0, 0
    for i,j in zip(preds, labels):
        if i==1 and j==1:
            num_tp += 1
        elif i==1 and j==0:
            num_fp += 1
        elif i==0 and j==1:
            num_fn += 1
        else:
            num_tn += 1
    if num_tp + num_fn == 0:
        recall = 0
    else:
        recall = num_tp / (num_tp + num_fn)
    if num_tp + num_fp == 0:
        precision = 0
    else:
        precision = num_tp / (num_tp + num_fp)
    if recall <= 0 and precision <= 0:
        f1 = 0
    else:
        f1 = 2 * recall * precision / (recall + precision)

    return accuracy, recall, precision, f1


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def gather_tensor(tensor):
    t = tensor.clone()
    gt = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gt, t)
    return torch.cat(gt, dim=0)

if __name__ == '__main__':
    main()

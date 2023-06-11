from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, T5Tokenizer
from tqdm import tqdm
import pytorch_lightning as pl
import torch
import numpy as np
from utils.utils import pad_sents, get_mask
import pdb

class OurDataset(Dataset):
    """Summarization dataset"""
    def __init__(self, args, mode):
        self.args = args
        # initial tokenizer and text
        if 't5' in self.args.model:
            self.tokenizer = T5Tokenizer.from_pretrained('/gallery_tate/keighley.overbay/thread-summarization/models/t5-base_cnn')
        else:
            self.tokenizer = BartTokenizer.from_pretrained('/gallery_tate/keighley.overbay/thread-summarization/models/bart-base_cnn')
        if mode == 'train':
            src_path = args.train_src_path
            tgt_path = args.train_tgt_path
        if mode == 'val':
            src_path = args.val_src_path
            tgt_path = args.val_tgt_path
        if mode == 'test':
            src_path = args.test_src_path
            tgt_path = args.test_tgt_path
        self.src = self.file_reader(src_path)
        self.tgt = self.file_reader(tgt_path)

        num_data = len(self.src)
        # get start & end index
        if args.split_id < args.num_splits - 1:
            start_idx = (num_data // args.num_splits) * args.split_id
            end_idx = start_idx + (num_data // args.num_splits)
        else:
            assert args.split_id == args.num_splits - 1
            start_idx = (num_data // args.num_splits) * args.split_id
            end_idx = num_data

        # pdb.set_trace()
        self.data_id = [item.split()[0] for item in self.tgt][start_idx:end_idx]
        self.src = [" ".join(item.split()[1:]) for item in self.src][start_idx:end_idx]
        self.tgt = [" ".join(item.split()[1:]) for item in self.tgt][start_idx:end_idx]

        if self.args.model == 'hierarchical_t5' or self.args.model =='mmhierarchical_t5':
            split_inputs = [item.split("|||") for item in self.src]
            turns = []
            src_tokens = []
            print('==================== Tokening {} set and generating turns ======================'.format(mode))

            # Tokenize here instead...allows for length match between input tokens and turns
            for split_input in split_inputs:
                turn = []
                src_token = []
                for i, input in enumerate(split_input):
                    input = self.tokenizer.encode(input, add_special_tokens=True)
                    turn.extend([i+1] * len(input))
                    src_token.extend(input)
                turns.append(turn)
                src_tokens.append(src_token)

            self.src_ids = src_tokens
            self.tgt_ids = self.tokenize(self.tgt)
            self.src_turns = turns

        else:
            # Just use tokenize func for non-hierarchical models
            print('==================== Tokening {} set ======================'.format(mode))
            self.src_ids = self.tokenize(self.src)
            self.tgt_ids = self.tokenize(self.tgt)
            self.src_turns = [[0 * len(i)] for i in self.src_ids]


    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src_ids[idx], self.tgt_ids[idx], self.src_turns[idx], self.data_id[idx]

    def tokenize(self, data):
        tokenized_text = [self.tokenizer.encode(i, add_special_tokens=True) for i in tqdm(data)]
        return tokenized_text

    def file_reader(self, file_path):
        file = open(file_path, 'r')
        lines = [item.strip('\n') for item in file.readlines()]
        return lines

    def collate_fn(self, data):
        if self.args.model == 'text_only_bart':
            # rebuild the raw text and truncate to max length
            max_input_len = self.args.max_input_len
            max_output_len = self.args.max_output_len
            raw_src = [pair[0] for pair in data]
            raw_tgt = [pair[1] for pair in data]
            raw_src = [i[:max_input_len-1] for i in raw_src]
            raw_tgt = [i[:max_output_len-1] for i in raw_tgt]
            src = []
            tgt = []
            # remove blank data
            for i in range(len(raw_src)):
                src.append(raw_src[i])
                tgt.append(raw_tgt[i])
            # make input mask
            mask = torch.tensor(get_mask(src, max_len=max_input_len))
            # make input ids
            src_ids = torch.tensor(pad_sents(src, 1, max_len=max_input_len)[0])
            # make output ids
            decoder_ids = [[0]+i for i in tgt]
            # make output labels
            label_ids = [[0]+i for i in tgt]
            # print(f"src_ids: {src_ids}")
            # print(f"label_ids: {label_ids}")
            # print(f"decoder_ids: {decoder_ids}")
            decoder_ids = torch.tensor(pad_sents(decoder_ids, 1, max_len=max_output_len)[0])
            # label_ids = torch.tensor(pad_sents(label_ids, -100, max_len=max_output_len)[0])
            label_ids = torch.tensor(pad_sents(tgt, -100, max_len=max_output_len)[0])

            # return src_ids, decoder_ids, mask, label_ids
            return src_ids, mask, label_ids, [x[-1] for x in data]

        elif self.args.model == 'multi_modal_bart':
            # rebuild the raw text and truncate to max length
            max_input_len = self.args.max_input_len
            max_output_len = self.args.max_output_len
            max_img_len = self.args.max_img_len
            raw_src = [pair[0] for pair in data]
            raw_tgt = [pair[1] for pair in data]
            data_id = [pair[3] for pair in data]
            raw_src = [i[:max_input_len-1] for i in raw_src]
            raw_tgt = [i[:max_output_len-1] for i in raw_tgt]
            data_id = [i.split('-')[0] for i in data_id]
            src = []
            tgt = []
            # ORIGINAL VERSION: FOR 1 x 2048 RESNEXT EMBEDS
            # img = np.zeros([len(raw_src), self.args.max_img_len, 2048])
            # NEW VERSION: FOR 1x768 VIT-B-16 EMBEDS
            img = np.zeros([len(raw_src), self.args.max_img_len, 768])
            img_len = []
            # remove blank data
            for i in range(len(raw_src)):
                src.append(raw_src[i])
                tgt.append(raw_tgt[i])
                # ORIGINAL VERSION: FOR 1 x 2048 RESNEXT EMBEDS
                # image_feature = np.load(self.args.image_feature_path + data_id[i]+ '.npy')[0]
                # NEW VERSION: FOR 197x768 VIT-B-16 EMBEDS
                image_feature = np.load(self.args.image_feature_path + data_id[i]+ '.npy')[:max_img_len]

                img[i][:image_feature.shape[0]] = image_feature
                # print(img[i])
                img_len.append(image_feature.shape[0])
            img = img[:,:max(img_len)]

            # make input mask
            mask = torch.tensor(get_mask(src, max_len=max_input_len))
            # make input ids
            src_ids = torch.tensor(pad_sents(src, 1, max_len=max_input_len)[0])
            # make output ids
            # decoder_ids = [[0]+i for i in tgt]
            # # make output labels
            # label_ids = [i+[0] for i in tgt]
            # decoder_ids = torch.tensor(pad_sents(decoder_ids, 1, max_len=max_output_len)[0])
            label_ids = torch.tensor(pad_sents(tgt, -100, max_len=max_output_len)[0])
            # return src_ids, decoder_ids, mask, label_ids, torch.tensor(img), img_len
            return src_ids, mask, label_ids, torch.tensor(img), img_len, [x[-1] for x in data]

        elif self.args.model == 'text_only_t5':
            # rebuild the raw text and truncate to max length
            max_input_len = self.args.max_input_len
            max_output_len = self.args.max_output_len
            raw_src = [pair[0] for pair in data]
            raw_tgt = [pair[1] for pair in data]
            raw_src = [i[:max_input_len-1] for i in raw_src]
            raw_tgt = [i[:max_output_len-1] for i in raw_tgt]
            src = []
            tgt = []
            # remove blank data
            for i in range(len(raw_src)):
                src.append(raw_src[i])
                tgt.append(raw_tgt[i])
            # make input mask
            mask = torch.tensor(get_mask(src, max_len=max_input_len))
            # make input ids
            src_ids = torch.tensor(pad_sents(src, 0, max_len=max_input_len)[0])
            # make output ids
            # decoder_ids = [[0]+i for i in tgt]
            # # make output labels
            # label_ids = [i+[0] for i in tgt]
            # decoder_ids = torch.tensor(pad_sents(decoder_ids, 0, max_len=max_output_len)[0])
            label_ids = torch.tensor(pad_sents(tgt, 0, max_len=max_output_len)[0])

            # return src_ids, decoder_ids, mask, label_ids
            return src_ids, mask, label_ids, [x[-1] for x in data]

        elif self.args.model == 'multi_modal_t5':
            # rebuild the raw text and truncate to max length
            max_input_len = self.args.max_input_len
            max_output_len = self.args.max_output_len
            max_img_len = self.args.max_img_len
            raw_src = [pair[0] for pair in data]
            raw_tgt = [pair[1] for pair in data]
            data_id = [pair[3] for pair in data]
            raw_src = [i[:max_input_len-1] for i in raw_src]
            raw_tgt = [i[:max_output_len-1] for i in raw_tgt]
            data_id = [i.split('-')[0] for i in data_id]
            src = []
            tgt = []
            # ORIGINAL VERSION: FOR 1 x 2048 RESNEXT EMBEDS
            # img = np.zeros([len(raw_src), self.args.max_img_len, 2048])
            # NEW VERSION: FOR 197x768 VIT-B-16 EMBEDS
            img = np.zeros([len(raw_src), self.args.max_img_len, 768])
            img_len = []
            # remove blank data
            for i in range(len(raw_src)):
                src.append(raw_src[i])
                tgt.append(raw_tgt[i])
                if self.args.vision_use_noise:
                    image_feature = np.load(self.args.image_feature_path + data_id[i] + '_noise.npy')[:max_img_len]
                else:
                    # image_feature = np.load(self.args.image_feature_path + data_id[i] + '.npy')[:max_img_len]
                    # NEW VERSION: FOR 197x768 VIT-B-16 EMBEDS
                    image_feature = np.load(self.args.image_feature_path + data_id[i]+ '.npy')[:max_img_len]
                # image_feature = np.load(self.args.image_feature_path + data_id[i]+ '.npy')[:max_img_len]
                img[i][:image_feature.shape[0]] = image_feature
                img_len.append(image_feature.shape[0])

            img = img[:,:max(img_len)]

            # make input mask
            mask = torch.tensor(get_mask(src, max_len=max_input_len))
            # make input ids
            src_ids = torch.tensor(pad_sents(src, 0, max_len=max_input_len)[0])
            # make output ids
            # decoder_ids = [[0]+i for i in tgt]
            # # make output labels
            # label_ids = [i+[0] for i in tgt]
            # decoder_ids = torch.tensor(pad_sents(decoder_ids, 0, max_len=max_output_len)[0])
            label_ids = torch.tensor(pad_sents(tgt, 0, max_len=max_output_len)[0])
            # return src_ids, decoder_ids, mask, label_ids, torch.tensor(img), img_len
            return src_ids, mask, label_ids, torch.tensor(img), img_len, [x[-1] for x in data]

        elif self.args.model == 'hierarchical_t5':
            # rebuild the raw text and truncate to max length
            max_input_len = self.args.max_input_len
            max_output_len = self.args.max_output_len
            raw_src = [pair[0] for pair in data]
            raw_tgt = [pair[1] for pair in data]
            raw_turns = [pair[2] for pair in data]
            # print(f"raw src {raw_src}")
            # print(f"raw ttgt {raw_tgt}")
            # print(f"raw turns {raw_turns}")
            raw_src = [i[:max_input_len-1] for i in raw_src]
            raw_tgt = [i[:max_output_len-1] for i in raw_tgt]
            raw_turns = [i[:max_input_len-1] for i in raw_turns]

            src = []
            tgt = []
            turns = []
            # turn_ids = []
            # remove blank data
            for i in range(len(raw_src)):
                src.append(raw_src[i])
                tgt.append(raw_tgt[i])
                turns.append(raw_turns[i])
                # turn_ids.append(raw_turns[i] + [0] * (max_input_len - len(raw_turns[i])))
            # make input mask
            mask = torch.tensor(get_mask(src, max_len=max_input_len))
            # make input ids
            src_ids = torch.tensor(pad_sents(src, 0, max_len=max_input_len)[0])
            # turn_ids = torch.tensor(turn_ids)
            # turn_ids.append(turn_input +)
            turn_ids = torch.tensor(pad_sents(turns, 0, max_len=max_input_len)[0])
            new_src_ids = torch.zeros((src_ids.size()[0],src_ids.size()[1] + turn_ids.size()[1]), dtype=torch.long)
            # print(f" src_id size {src_ids.size()}")
            # print(f" turn_id size {turn_ids.size()}")
            # print(f" newsrc_id size {new_src_ids.size()}")
            # print(f"src ids before: {src_ids}")
            for i, src_id in enumerate(src_ids):
                new_src_ids[i] = torch.cat([src_id, turn_ids[i]])
            # print(f"src ids after: {new_src_ids}")
            # new_src_ids = torch.tensor(new_src_ids)
            # src_ids = torch.cat([src_ids, turn_ids])
            mem_attn_mask = torch.ones((len(src), self.args.max_turn_length), dtype=torch.long)
            mask = torch.cat([mem_attn_mask, mask], dim=1)
            # make output ids
            # decoder_ids = [[0]+i for i in tgt]
            # # make output labels
            # label_ids = [i+[0] for i in tgt]
            # decoder_ids = torch.tensor(pad_sents(decoder_ids, 0, max_len=max_output_len)[0])
            label_ids = torch.tensor(pad_sents(tgt, 0, max_len=max_output_len)[0])

            # return src_ids, decoder_ids, mask, label_ids
            return new_src_ids, mask, label_ids

        elif self.args.model == 'mmhierarchical_t5':
            # rebuild the raw text and truncate to max length
            max_input_len = self.args.max_input_len
            max_output_len = self.args.max_output_len
            max_img_len = self.args.max_img_len
            raw_src = [pair[0] for pair in data]
            raw_tgt = [pair[1] for pair in data]
            raw_turns = [pair[2] for pair in data]
            data_id = [pair[3] for pair in data]

            img = np.zeros([len(raw_src), self.args.max_img_len, 768])
            img_len = []
            # print(f"raw src {raw_src}")
            # print(f"raw ttgt {raw_tgt}")
            # print(f"raw turns {raw_turns}")
            raw_src = [i[:max_input_len-1] for i in raw_src]
            raw_tgt = [i[:max_output_len-1] for i in raw_tgt]
            raw_turns = [i[:max_input_len-1] for i in raw_turns]

            src = []
            tgt = []
            turns = []
            # turn_ids = []
            # remove blank data
            for i in range(len(raw_src)):
                src.append(raw_src[i])
                tgt.append(raw_tgt[i])
                turns.append(raw_turns[i])
                if self.args.vision_use_noise:
                    image_feature = np.load(self.args.image_feature_path + data_id[i] + '_noise.npy')[:max_img_len]
                else:
                    image_feature = np.load(self.args.image_feature_path + data_id[i] + '.npy')[:max_img_len]

                img[i][:image_feature.shape[0]] = image_feature
                img_len.append(image_feature.shape[0])

            img = img[:,:max(img_len)]

            # make input mask
            mask = torch.tensor(get_mask(src, max_len=max_input_len))
            # make input ids
            src_ids = torch.tensor(pad_sents(src, 0, max_len=max_input_len)[0])
            # turn_ids = torch.tensor(turn_ids)
            # turn_ids.append(turn_input +)
            turn_ids = torch.tensor(pad_sents(turns, 0, max_len=max_input_len)[0])
            new_src_ids = torch.zeros((src_ids.size()[0],src_ids.size()[1] + turn_ids.size()[1]), dtype=torch.long)
            # print(f" src_id size {src_ids.size()}")
            # print(f" turn_id size {turn_ids.size()}")
            # print(f" newsrc_id size {new_src_ids.size()}")
            # print(f"src ids before: {src_ids}")
            for i, src_id in enumerate(src_ids):
                new_src_ids[i] = torch.cat([src_id, turn_ids[i]])
            # print(f"src ids after: {new_src_ids}")
            # new_src_ids = torch.tensor(new_src_ids)
            # src_ids = torch.cat([src_ids, turn_ids])
            mem_attn_mask = torch.ones((len(src), self.args.max_turn_length), dtype=torch.long)
            mask = torch.cat([mem_attn_mask, mask], dim=1)
            # make output ids
            # decoder_ids = [[0]+i for i in tgt]
            # # make output labels
            # label_ids = [i+[0] for i in tgt]
            # decoder_ids = torch.tensor(pad_sents(decoder_ids, 0, max_len=max_output_len)[0])
            label_ids = torch.tensor(pad_sents(tgt, 0, max_len=max_output_len)[0])

            # return src_ids, decoder_ids, mask, label_ids
            return new_src_ids, mask, label_ids, torch.tensor(img), img_len

        else:
            raise ValueError("Invalid model")

# Create a dataloading module as per the PyTorch Lightning Docs
class SummaryDataModule(pl.LightningDataModule):
  def __init__(self, args):
    super().__init__()
    train_set = OurDataset(args, 'train')
    val_set = OurDataset(args, 'val')
    test_set = OurDataset(args, 'test')
    self.train_loader = DataLoader(dataset=train_set, \
                                    batch_size=args.batch_size, \
                                    num_workers=3, \
                                    shuffle=True, \
                                    collate_fn=train_set.collate_fn)
    self.val_loader = DataLoader(dataset=val_set, \
                                    batch_size=args.batch_size, \
                                    num_workers=3, \
                                    shuffle=False, \
                                    collate_fn=val_set.collate_fn)
    self.test_loader = DataLoader(dataset=test_set, \
                                    batch_size=args.batch_size, \
                                    num_workers=3, \
                                    shuffle=False, \
                                    collate_fn=test_set.collate_fn)

  def train_dataloader(self):
    return self.train_loader

  def val_dataloader(self):
    return self.val_loader

  def test_dataloader(self):
    return self.test_loader

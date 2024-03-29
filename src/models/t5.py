from models.base_model import BaseModel
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration
from models.modeling_t5 import T5ForMultiModalGeneration
from models.modeling_ht5 import T5ForHierarchicalGeneration
from models.modeling_vght5 import T5ForMMHierarchicalGeneration
from datasets import load_metric
import pdb
from rouge import Rouge

class T5Origin(BaseModel):

    def __init__(self,args):
        self.args = args
        super(T5Origin, self).__init__(args)
        self.model = T5ForConditionalGeneration.from_pretrained('/gallery_tate/keighley.overbay/thread-summarization/models/t5-base_cnn', local_files_only=True)
        self.tokenizer = T5Tokenizer.from_pretrained('/gallery_tate/keighley.overbay/thread-summarization/models/t5-base_cnn', local_files_only=True)
        # self.rouge = load_metric('rouge', experiment_id=self.args.log_name)
        self.rouge = Rouge()

    # def forward(self, input_ids, attention_mask, decoder_input_ids, labels):
    def forward(self, input_ids, attention_mask, labels):

        # loss = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)[0]
        loss = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)[0]
        return loss

    def validation_step(self, batch, batch_idx):
        # batch
        # src_ids, decoder_ids, mask, label_ids = batch
        src_ids, mask, label_ids, data_ids = batch
        # get summary
        summary_ids = self.model.generate(input_ids=src_ids,
                                            attention_mask=mask,
                                            num_beams=self.args.n_beams,
                                            max_length=self.args.max_output_len,
                                            early_stopping=True,
                                            no_repeat_ngram_size=self.args.no_repeat_ngram_size)
        return [summary_ids, label_ids, data_ids]

    def validation_epoch_end(self, outputs):
        summary = []
        reference = []
        total_data_ids = []
        for item in outputs:
            summary_id = item[0]
            label_id = item[1]
            data_ids = item[2]
            one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
            one_reference = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in label_id]
            summary += one_summary
            reference += one_reference
            total_data_ids += data_ids
        avg_rouge1, avg_rouge2, avg_rougeL = self.calrouge(summary, reference, self.rouge)
        self.log('validation_Rouge1_one_epoch', avg_rouge1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_Rouge2_one_epoch', avg_rouge2, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_RougeL_one_epoch', avg_rougeL, on_epoch=True, prog_bar=True, sync_dist=True)

        self.save_txt(self.args.val_save_file+'_reference', reference)
        self.save_txt(self.args.val_save_file+'_summary', summary)

        self.save_txt(self.args.val_save_file+'_reference_with_ids', reference, total_data_ids)
        self.save_txt(self.args.val_save_file+'_summary_with_ids', summary, total_data_ids)

    def test_step(self, batch, batch_idx):
        # batch
        # src_ids, decoder_ids, mask, label_ids = batch
        src_ids, mask, label_ids, data_ids = batch
        # get summary
        summary_ids = self.model.generate(input_ids=src_ids,
                                            attention_mask=mask,
                                            num_beams=self.args.n_beams,
                                            max_length=self.args.max_output_len,
                                            early_stopping=True,
                                            no_repeat_ngram_size=self.args.no_repeat_ngram_size)
        return [summary_ids, label_ids, data_ids]

    def test_epoch_end(self, outputs):
        # rouge = load_metric('rouge', experiment_id=self.args.log_name)
        summary = []
        reference = []
        total_data_ids = []
        for item in outputs:
            summary_id = item[0]
            label_id = item[1]
            data_ids = item[2]
            one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
            one_reference = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in label_id]
            summary += one_summary
            reference += one_reference
            total_data_ids += data_ids
        avg_rouge1, avg_rouge2, avg_rougeL = self.calrouge(summary, reference, self.rouge)
        self.log('test_Rouge1_one_epoch', avg_rouge1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_Rouge2_one_epoch', avg_rouge2, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_RougeL_one_epoch', avg_rougeL, on_epoch=True, prog_bar=True, sync_dist=True)

        ### self.save_txt(self.args.test_save_file+f'_{self.args.split_id}_{self.args.num_splits}'+'_reference', reference)
        ### self.save_txt(self.args.test_save_file+f'_{self.args.split_id}_{self.args.num_splits}'+'_summary', summary)
        self.save_txt(self.args.test_save_file+'_reference', reference)
        self.save_txt(self.args.test_save_file+'_summary', summary)

        ### self.save_txt(self.args.test_save_file+f'_{self.args.split_id}_{self.args.num_splits}'+'_reference_with_ids', reference, total_data_ids)
        ### self.save_txt(self.args.test_save_file+f'_{self.args.split_id}_{self.args.num_splits}'+'_summary_with_ids', summary, total_data_ids)
        self.save_txt(self.args.test_save_file+'_reference_with_ids', reference, total_data_ids)
        self.save_txt(self.args.test_save_file+'_summary_with_ids', summary, total_data_ids)

    def calrouge(self, summary, reference, rouge):
        new_reference = []
        for r in reference:
            if r == '':
                print(f'Detected blank target summary')
                new_reference.append('{}')
            else:
                new_reference.append(r)
        # rouge.add_batch(predictions=summary, references=reference)
        # final_results = rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
        # R1_F1 = final_results["rouge1"].mid.fmeasure * 100
        # R2_F1 = final_results["rouge2"].mid.fmeasure * 100
        # RL_F1 = final_results["rougeL"].mid.fmeasure * 100
        # return R1_F1, R2_F1, RL_F1
        scores = rouge.get_scores(hyps=summary,refs=new_reference,avg=True)
        R1_F1 = scores['rouge-1']['f'] * 100
        R2_F1 = scores['rouge-2']['f'] * 100
        RL_F1 = scores['rouge-l']['f'] * 100
        return R1_F1, R2_F1, RL_F1

    def save_txt(self, file_name, list_data, data_ids=None):
        file = open(file_name, 'w')
        if data_ids is None:
            list_data = [item+'\n' for item in list_data]
        else:
            list_data = [f'{data_ids[i]} '+item+'\n' for i,item in enumerate(list_data)]
        file.writelines(list_data)
        file.close()


class T5MultiModal(BaseModel):

    def __init__(self, args):
        self.args = args
        super(T5MultiModal, self).__init__(args)
        self.model = T5ForMultiModalGeneration.from_pretrained('/gallery_tate/keighley.overbay/thread-summarization/models/t5-base_cnn/',
                                                                 fusion_layer=args.fusion_layer,
                                                                 use_img_trans=args.use_img_trans,
                                                                 use_forget_gate=args.use_forget_gate,
                                                                 cross_attn_type=args.cross_attn_type,
                                                                 dim_common=args.dim_common,
                                                                 image_encoder=args.image_encoder,
                                                                 n_attn_heads=args.n_attn_heads)
        self.tokenizer = T5Tokenizer.from_pretrained('/gallery_tate/keighley.overbay/thread-summarization/models/t5-base_cnn/')
        self.rouge = Rouge()

    # def forward(self, input_ids, attention_mask, decoder_input_ids, labels, image_features, image_len):
    def forward(self, input_ids, attention_mask, labels, image_features, image_len):
        loss = self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                        #   decoder_input_ids=decoder_input_ids,
                          labels=labels,
                          image_features=image_features,
                          image_len=image_len)[0]
        return loss

    def training_step(self, batch, batch_idx):
        # batch
        # src_ids, decoder_ids, mask, label_ids, image_features, image_len = batch
        src_ids, mask, label_ids, image_features, image_len, _ = batch
        # get loss
        # loss = self(input_ids=src_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids, image_features=image_features.float(), image_len=image_len)
        loss = self(input_ids=src_ids, attention_mask=mask, labels=label_ids, image_features=image_features.float(), image_len=image_len)
        # logs
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # batch
        # src_ids, decoder_ids, mask, label_ids, image_features, image_len = batch
        src_ids, mask, label_ids, image_features, image_len, data_ids = batch
        # get summary
        summary_ids = self.model.generate(input_ids=src_ids,
                                            attention_mask=mask,
                                            num_beams=self.args.n_beams,
                                            max_length=self.args.max_output_len,
                                            early_stopping=True,
                                            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                            image_features=image_features.float(),
                                            image_len=image_len)
        return [summary_ids, label_ids, data_ids]

    def validation_epoch_end(self, outputs):
        summary = []
        reference = []
        total_data_ids = []
        for item in outputs:
            summary_id = item[0]
            label_id = item[1]
            data_ids = item[2]
            one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
            one_reference = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in label_id]
            summary += one_summary
            reference += one_reference
            total_data_ids += data_ids
        avg_rouge1, avg_rouge2, avg_rougeL = self.calrouge(summary, reference, self.rouge)
        self.log('validation_Rouge1_one_epoch', avg_rouge1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_Rouge2_one_epoch', avg_rouge2, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_RougeL_one_epoch', avg_rougeL, on_epoch=True, prog_bar=True, sync_dist=True)

        self.save_txt(self.args.val_save_file+'_reference', reference)
        self.save_txt(self.args.val_save_file+'_summary', summary)

        self.save_txt(self.args.val_save_file+'_reference_with_ids', reference, total_data_ids)
        self.save_txt(self.args.val_save_file+'_summary_with_ids', summary, total_data_ids)

    def test_step(self, batch, batch_idx):
        # batch
        # src_ids, decoder_ids, mask, label_ids, image_features, image_len = batch
        src_ids, mask, label_ids, image_features, image_len, data_ids = batch
        # get summary
        summary_ids = self.model.generate(input_ids=src_ids,
                                            attention_mask=mask,
                                            num_beams=self.args.n_beams,
                                            max_length=self.args.max_output_len,
                                            early_stopping=True,
                                            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                            image_features=image_features.float(),
                                            image_len=image_len)
        return [summary_ids, label_ids, data_ids]

    def test_epoch_end(self, outputs):
        # rouge = load_metric('rouge', experiment_id=self.args.log_name)
        summary = []
        reference = []
        total_data_ids = []
        for item in outputs:
            summary_id = item[0]
            label_id = item[1]
            data_ids = item[2]
            one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
            one_reference = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in label_id]
            summary += one_summary
            reference += one_reference
            total_data_ids += data_ids
        avg_rouge1, avg_rouge2, avg_rougeL = self.calrouge(summary, reference, self.rouge)
        self.log('test_Rouge1_one_epoch', avg_rouge1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_Rouge2_one_epoch', avg_rouge2, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_RougeL_one_epoch', avg_rougeL, on_epoch=True, prog_bar=True, sync_dist=True)

        ### self.save_txt(self.args.test_save_file+f'_{self.args.split_id}_{self.args.num_splits}'+'_reference', reference)
        ### self.save_txt(self.args.test_save_file+f'_{self.args.split_id}_{self.args.num_splits}'+'_summary', summary)
        self.save_txt(self.args.test_save_file+'_reference', reference)
        self.save_txt(self.args.test_save_file+'_summary', summary)

        ### self.save_txt(self.args.test_save_file+f'_{self.args.split_id}_{self.args.num_splits}'+'_reference_with_ids', reference, total_data_ids)
        ### self.save_txt(self.args.test_save_file+f'_{self.args.split_id}_{self.args.num_splits}'+'_summary_with_ids', summary, total_data_ids)
        self.save_txt(self.args.test_save_file+'_reference_with_ids', reference, total_data_ids)
        self.save_txt(self.args.test_save_file+'_summary_with_ids', summary, total_data_ids)

    def calrouge(self, summary, reference, rouge):
        new_reference = []
        for r in reference:
            if r == '':
                print(f'Detected blank target summary')
                new_reference.append('{}')
            else:
                new_reference.append(r)
        # rouge.add_batch(predictions=summary, references=reference)
        # final_results = rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
        # R1_F1 = final_results["rouge1"].mid.fmeasure * 100
        # R2_F1 = final_results["rouge2"].mid.fmeasure * 100
        # RL_F1 = final_results["rougeL"].mid.fmeasure * 100
        # return R1_F1, R2_F1, RL_F1
        scores = rouge.get_scores(hyps=summary,refs=new_reference,avg=True)
        R1_F1 = scores['rouge-1']['f'] * 100
        R2_F1 = scores['rouge-2']['f'] * 100
        RL_F1 = scores['rouge-l']['f'] * 100
        return R1_F1, R2_F1, RL_F1

    def save_txt(self, file_name, list_data, data_ids=None):
        file = open(file_name, 'w')
        if data_ids is None:
            list_data = [item+'\n' for item in list_data]
        else:
            list_data = [f'{data_ids[i]} '+item+'\n' for i,item in enumerate(list_data)]
        file.writelines(list_data)
        file.close()

class T5Hierarchical(BaseModel):

    def __init__(self, args):
        self.args = args
        super(T5Hierarchical, self).__init__(args)
        self.model = T5ForHierarchicalGeneration.from_pretrained('../../models/t5-base_cnn',
                                                                max_input_length=args.max_input_len,
                                                                memory_length=args.max_turn_length,
                                                                local_files_only=True)
        self.tokenizer = T5Tokenizer.from_pretrained('../../models/t5-base_cnn')
        # self.rouge = load_metric('rouge', experiment_id=self.args.log_name)
        self.rouge = Rouge()

    def forward(self, input_ids, attention_mask, labels):

        # loss = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)[0]
        loss = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)[0]
        return loss

    def validation_step(self, batch, batch_idx):
        # batch
        # src_ids, decoder_ids, mask, label_ids = batch
        src_ids, mask, label_ids = batch
        # get summary
        # pdb.set_trace()
        summary_ids = self.model.generate(input_ids=src_ids,
                                            attention_mask=mask,
                                            num_beams=self.args.n_beams,
                                            max_length=self.args.max_output_len,
                                            early_stopping=True,
                                            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                            # max_input_length=self.args.max_input_len,
                                            # memory_length=self.args.max_turn_length
                                            )
        return [summary_ids, label_ids]

    def validation_epoch_end(self, outputs):
        summary = []
        reference = []
        for item in outputs:
            summary_id = item[0]
            label_id = item[1]
            one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
            one_reference = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in label_id]
            summary += one_summary
            reference += one_reference
        avg_rouge1, avg_rouge2, avg_rougeL = self.calrouge(summary, reference, self.rouge)
        self.log('validation_Rouge1_one_epoch', avg_rouge1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_Rouge2_one_epoch', avg_rouge2, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_RougeL_one_epoch', avg_rougeL, on_epoch=True, prog_bar=True, sync_dist=True)
        self.save_txt(self.args.val_save_file+'reference', reference)
        self.save_txt(self.args.val_save_file, summary)

    def test_step(self, batch, batch_idx):
        # batch
        # src_ids, decoder_ids, mask, label_ids = batch
        src_ids, mask, label_ids = batch
        # get summary
        summary_ids = self.model.generate(input_ids=src_ids,
                                            attention_mask=mask,
                                            num_beams=self.args.n_beams,
                                            max_length=self.args.max_output_len,
                                            early_stopping=True,
                                            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                            # max_input_length=self.args.max_input_len,
                                            # memory_length=self.args.max_turn_length
                                            )
        return [summary_ids, label_ids]

    def test_epoch_end(self, outputs):
        # rouge = load_metric('rouge', experiment_id=self.args.log_name)
        summary = []
        reference = []
        for item in outputs:
            summary_id = item[0]
            label_id = item[1]
            one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
            one_reference = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in label_id]
            summary += one_summary
            reference += one_reference
        avg_rouge1, avg_rouge2, avg_rougeL = self.calrouge(summary, reference, self.rouge)
        self.log('test_Rouge1_one_epoch', avg_rouge1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_Rouge2_one_epoch', avg_rouge2, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_RougeL_one_epoch', avg_rougeL, on_epoch=True, prog_bar=True, sync_dist=True)
        self.save_txt(self.args.test_save_file, summary)

    def calrouge(self, summary, reference, rouge):
        new_reference = []
        for r in reference:
            if r == '':
                print(f'Detected blank target summary')
                new_reference.append('{}')
            else:
                new_reference.append(r)
        # rouge.add_batch(predictions=summary, references=reference)
        # final_results = rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
        # R1_F1 = final_results["rouge1"].mid.fmeasure * 100
        # R2_F1 = final_results["rouge2"].mid.fmeasure * 100
        # RL_F1 = final_results["rougeL"].mid.fmeasure * 100
        # return R1_F1, R2_F1, RL_F1
        scores = rouge.get_scores(hyps=summary,refs=new_reference,avg=True)
        R1_F1 = scores['rouge-1']['f'] * 100
        R2_F1 = scores['rouge-2']['f'] * 100
        RL_F1 = scores['rouge-l']['f'] * 100
        return R1_F1, R2_F1, RL_F1

    def save_txt(self, file_name, list_data):
        file = open(file_name, 'w')
        list_data = [item+'\n' for item in list_data]
        file.writelines(list_data)
        file.close()


class T5MMHierarchical(BaseModel):

    def __init__(self, args):
        self.args = args
        super(T5MMHierarchical, self).__init__(args)
        self.model = T5ForMMHierarchicalGeneration.from_pretrained('../../models/t5-base_cnn',
                                                                max_input_length=args.max_input_len,
                                                                memory_length=args.max_turn_length,
                                                                fusion_layer=args.fusion_layer,
                                                                 use_img_trans=args.use_img_trans,
                                                                 use_forget_gate=args.use_forget_gate,
                                                                 cross_attn_type=args.cross_attn_type,
                                                                 dim_common=args.dim_common,
                                                                 n_attn_heads=args.n_attn_heads,
                                                                local_files_only=True)
        self.tokenizer = T5Tokenizer.from_pretrained('../../models/t5-base_cnn')
        # self.rouge = load_metric('rouge', experiment_id=self.args.log_name)
        self.rouge = Rouge()

    def forward(self, input_ids, attention_mask, labels,image_features, image_len):

        # loss = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)[0]
        loss = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,image_features=image_features,
                          image_len=image_len)[0]
        return loss

    def training_step(self, batch, batch_idx):
        # batch
        # src_ids, decoder_ids, mask, label_ids, image_features, image_len = batch
        src_ids, mask, label_ids, image_features, image_len = batch
        # get loss
        # loss = self(input_ids=src_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids, image_features=image_features.float(), image_len=image_len)
        loss = self(input_ids=src_ids, attention_mask=mask, labels=label_ids, image_features=image_features.float(), image_len=image_len)
        # logs
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # batch
        # src_ids, decoder_ids, mask, label_ids = batch
        src_ids, mask, label_ids, image_features, image_len = batch
        # get summary
        # pdb.set_trace()
        summary_ids = self.model.generate(input_ids=src_ids,
                                            attention_mask=mask,
                                            num_beams=self.args.n_beams,
                                            max_length=self.args.max_output_len,
                                            early_stopping=True,
                                            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                            image_features=image_features.float(),
                                            image_len=image_len#
                                            )
        return [summary_ids, label_ids]

    def validation_epoch_end(self, outputs):
        summary = []
        reference = []
        for item in outputs:
            summary_id = item[0]
            label_id = item[1]
            one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
            one_reference = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in label_id]
            summary += one_summary
            reference += one_reference
        avg_rouge1, avg_rouge2, avg_rougeL = self.calrouge(summary, reference, self.rouge)
        self.log('validation_Rouge1_one_epoch', avg_rouge1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_Rouge2_one_epoch', avg_rouge2, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_RougeL_one_epoch', avg_rougeL, on_epoch=True, prog_bar=True, sync_dist=True)
        self.save_txt(self.args.val_save_file+'reference', reference)
        self.save_txt(self.args.val_save_file, summary)

    def test_step(self, batch, batch_idx):
        # batch
        # src_ids, decoder_ids, mask, label_ids = batch
        src_ids, mask, label_ids, image_features, image_len = batch
        # get summary
        summary_ids = self.model.generate(input_ids=src_ids,
                                            attention_mask=mask,
                                            num_beams=self.args.n_beams,
                                            max_length=self.args.max_output_len,
                                            early_stopping=True,
                                            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                            image_features=image_features.float(),
                                            image_len=image_len# max_input_length=self.args.max_input_len,
                                            # memory_length=self.args.max_turn_length
                                            )
        return [summary_ids, label_ids]

    def test_epoch_end(self, outputs):
        # rouge = load_metric('rouge', experiment_id=self.args.log_name)
        summary = []
        reference = []
        for item in outputs:
            summary_id = item[0]
            label_id = item[1]
            one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
            one_reference = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in label_id]
            summary += one_summary
            reference += one_reference
        avg_rouge1, avg_rouge2, avg_rougeL = self.calrouge(summary, reference, self.rouge)
        self.log('test_Rouge1_one_epoch', avg_rouge1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_Rouge2_one_epoch', avg_rouge2, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_RougeL_one_epoch', avg_rougeL, on_epoch=True, prog_bar=True, sync_dist=True)
        self.save_txt(self.args.test_save_file, summary)

    def calrouge(self, summary, reference, rouge):
        new_reference = []
        for r in reference:
            if r == '':
                print(f'Detected blank target summary')
                new_reference.append('{}')
            else:
                new_reference.append(r)
        # rouge.add_batch(predictions=summary, references=reference)
        # final_results = rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
        # R1_F1 = final_results["rouge1"].mid.fmeasure * 100
        # R2_F1 = final_results["rouge2"].mid.fmeasure * 100
        # RL_F1 = final_results["rougeL"].mid.fmeasure * 100
        # return R1_F1, R2_F1, RL_F1
        scores = rouge.get_scores(hyps=summary,refs=new_reference,avg=True)
        R1_F1 = scores['rouge-1']['f'] * 100
        R2_F1 = scores['rouge-2']['f'] * 100
        RL_F1 = scores['rouge-l']['f'] * 100
        return R1_F1, R2_F1, RL_F1

    def save_txt(self, file_name, list_data):
        file = open(file_name, 'w')
        list_data = [item+'\n' for item in list_data]
        file.writelines(list_data)
        file.close()

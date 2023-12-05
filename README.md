# mRedditSum

Welcome! 👋🏻\
This is the official repository of our EMNLP 2023 paper: \
**[mRedditSum: A Multimodal Abstractive Summarization Dataset of Reddit Threads with Images.](https://openreview.net/forum?id=k3i6PKlKY8)**

![mredditsum illustration](assets/cover.png)

Please cite our work if you found the resources in this repository useful:

```bib
@inproceedings{overbay2023mredditsum,
    title={mRedditSum: A Multimodal Abstractive Summarization Dataset of Reddit Threads with Images.},
    author={Keighley Overbay and Jaewoo Ahn and Fatemeh Pesaran Zadeh and Joonsuk Park and Gunhee Kim},
    booktitle={EMNLP},
    year=2023
}
```

## Dataset

You can download our mRedditSum dataset directly by clicking this [link](https://drive.google.com/file/d/1WhOgsEWmLSnEG2-K8R2n_hogFLGPkI8I/view?usp=sharing).

The data comes preprocessed and ready for training.

## Model Training

To reproduce the results of our paper, you can run the models as follows.

To run the models, use run.py under the src file.

### Text-only models

Example of running regular text-only BART:

```bash
python src/run.py \
      -model=text_only_bart \
      -train_src_path=../data/train_processed_src.txt \
      -train_tgt_path=../data/train_processed_tgt.txt \
      -val_src_path=../data/val_processed_src.txt \
      -val_tgt_path=../data/val_processed_tgt.txt \
      -test_src_path=../data/test_processed_src.txt \
      -test_tgt_path=../data/test_processed_tgt.txt \
      -val_save_file=../results/val/bart_val.txt \
      -test_save_file=../results/test/bart_test.txt \
      -log_name=text_only_bart \
      -gpus='1' \
      -batch_size=4 \
      -learning_rate=3e-5 \
      -scheduler_lambda1=10 \
      -scheduler_lambda2=0.95 \
      -num_epochs=50 \
      -grad_accumulate=5 \
      -max_input_len=1024 \
      -max_output_len=256 \
      -n_beams=5 \
      -random_seed=0 \
      -do_train=True \
      -do_test=False \
      -limit_val_batches=1 \
      -val_check_interval=1 \
      -img_lr_factor=5 \
      -checkpoint=None
```

See the run.py file for more notes on input arguments.

### Our CMS (cluster-based multi-stage summarization) models

For running the CMS model, there are a few more steps including data preprocessing steps.

Example of running CMS-T5-ImgCap:

#### 1. first-stage summarization: preprocessing dataset on cluster summary labels & training model

```bash
# Iterate over the modes using a for loop
for mode in train val test
do
  echo $mode
  python scripts/convert_to_datapoints_first_stage.py \
      -f "../data/${mode}_edited_csums.json" \
      -s "../data/two_stage_summary_text_only_t5_imgcap/${mode}_edited_csums_firststage_processed.json" \
      -p clustergold -sep " " -imgcap

  python scripts/convert_to_txt.py \
      -f "../data/two_stage_summary_text_only_t5_imgcap/${mode}_edited_csums_firststage_processed.json"
done

# train first-stage summarization model
python src/run.py \
      -model=text_only_t5 \
      -train_src_path=../data/two_stage_summary_text_only_t5_imgcap/train_edited_csums_firststage_processed_src.txt \
      -train_tgt_path=../data/two_stage_summary_text_only_t5_imgcap/train_edited_csums_firststage_processed_tgt.txt \
      -val_src_path=../data/two_stage_summary_text_only_t5_imgcap/val_edited_csums_firststage_processed_src.txt \
      -val_tgt_path=../data/two_stage_summary_text_only_t5_imgcap/val_edited_csums_firststage_processed_tgt.txt \
      -test_src_path=../data/two_stage_summary_text_only_t5_imgcap/test_edited_csums_firststage_processed_src.txt \
      -test_tgt_path=../data/two_stage_summary_text_only_t5_imgcap/test_edited_csums_firststage_processed_tgt.txt \
      -image_feature_path=../data/image_features/vitb-32/ \
      -val_save_file=../results/val/text_only_t5_imgcap_val_firststage.txt \
      -test_save_file=../results/test/text_only_t5_imgcap_test_firststage.txt \
      -output_dir='../checkpoint' \
      -log_name=text_only_t5_imgcap_firststage \
      -gpus='1' \
      -batch_size=4 \
      -learning_rate=3e-05 \
      -scheduler_lambda1=10 \
      -scheduler_lambda2=0.95 \
      -num_epochs=50 \
      -grad_accumulate=5 \
      -max_input_len=1024 \
      -max_output_len=256 \
      -max_img_len=1 \
      -n_beams=5 \
      -random_seed=0 \
      -do_train=True \
      -do_test=True \
      -limit_val_batches=1 \
      -val_check_interval=1 \
      -img_lr_factor=5 \
      -checkpoint=None \
      -use_forget_gate \
      -cross_attn_type=0 \
      -use_img_trans \
```

#### 2. first-stage summarization (gen): preprocessing dataset for generating cluster summary & generating it based on the trained model

```bash
# Iterate over the modes using a for loop
for mode in train val test
do
  echo $mode
  python scripts/convert_to_datapoints_first_stage.py \
      -f "../data/${mode}_edited_csums.json" \
      -s "../data/two_stage_summary_text_only_t5_imgcap/${mode}_edited_csums_firststage_generated_processed.json" \
      -p clustergold -sep " " -gen -imgcap

  python scripts/convert_to_txt.py \
      -f "../data/two_stage_summary_text_only_t5_imgcap/${mode}_edited_csums_firststage_generated_processed.json"
done

for mode in train val test
do
  echo $mode
  # generate first-stage summary based on trained model
  python src/run.py \
      -model=text_only_t5 \
      -train_src_path=../data/two_stage_summary_text_only_t5_imgcap/train_edited_csums_firststage_generated_processed_src.txt \
      -train_tgt_path=../data/two_stage_summary_text_only_t5_imgcap/train_edited_csums_firststage_generated_processed_tgt.txt \
      -val_src_path=../data/two_stage_summary_text_only_t5_imgcap/val_edited_csums_firststage_generated_processed_src.txt \
      -val_tgt_path=../data/two_stage_summary_text_only_t5_imgcap/val_edited_csums_firststage_generated_processed_tgt.txt \
      -test_src_path=../data/two_stage_summary_text_only_t5_imgcap/${mode}_edited_csums_firststage_generated_processed_src.txt \
      -test_tgt_path=../data/two_stage_summary_text_only_t5_imgcap/${mode}_edited_csums_firststage_generated_processed_tgt.txt \
      -image_feature_path=../data/image_features/vitb-32/ \
      -val_save_file=../results/val/text_only_t5_imgcap_val_firststage_generated.txt \
      -test_save_file=../results/test/text_only_t5_imgcap_${mode}_firststage_generated.txt \
      -output_dir='../checkpoint' \
      -log_name=text_only_t5_imgcap_firststage \
      -gpus='1' \
      -batch_size=4 \
      -learning_rate=3e-05 \
      -scheduler_lambda1=10 \
      -scheduler_lambda2=0.95 \
      -num_epochs=10 \
      -grad_accumulate=5 \
      -max_input_len=1024 \
      -max_output_len=256 \
      -max_img_len=1 \
      -n_beams=5 \
      -random_seed=0 \
      -do_train=False \
      -do_test=True \
      -limit_val_batches=1 \
      -val_check_interval=1 \
      -img_lr_factor=5 \
      -checkpoint='../checkpoint/text_only_t5_imgcap_firststage/your/best/checkpoint.ckpt' \
      -use_forget_gate \
      -cross_attn_type=0 \
      -use_img_trans \
done
```

#### 3. second-stage summarization: preprocessing dataset for generating final summary with generated cluster summary & training model

```bash
# Iterate over the modes using a for loop
for mode in train val test
do
  echo $mode
  python scripts/convert_to_datapoints_second_stage.py \
      -f "../data/${mode}_edited_csums.json" \
      -s "../data/two_stage_summary_text_only_t5_imgcap/${mode}_edited_csums_generated_secondstage_processed.json" \
      -p clusterpred -sep " " -imgcap \
      -predfirst "../results/test/text_only_t5_imgcap_${mode}_firststage_generated.txt_summary_with_ids"
  python scripts/convert_to_txt.py \
      -f "../data/two_stage_summary_text_only_t5_imgcap/${mode}_edited_csums_generated_secondstage_processed.json"
done

# train second-stage summarization model
python src/run.py \
      -model=text_only_t5 \
      -train_src_path=../data/two_stage_summary_text_only_t5_imgcap/train_edited_csums_generated_secondstage_processed_src.txt \
      -train_tgt_path=../data/two_stage_summary_text_only_t5_imgcap/train_edited_csums_generated_secondstage_processed_tgt.txt \
      -val_src_path=../data/two_stage_summary_text_only_t5_imgcap/val_edited_csums_generated_secondstage_processed_src.txt \
      -val_tgt_path=../data/two_stage_summary_text_only_t5_imgcap/val_edited_csums_generated_secondstage_processed_tgt.txt \
      -test_src_path=../data/two_stage_summary_text_only_t5_imgcap/test_edited_csums_generated_secondstage_processed_src.txt \
      -test_tgt_path=../data/two_stage_summary_text_only_t5_imgcap/test_edited_csums_generated_secondstage_processed_tgt.txt \
      -image_feature_path=../data/image_features/vitb-32/ \
      -val_save_file=../results/val/text_only_t5_imgcap_seed0_val_generated_secondstage.txt \
      -test_save_file=../results/test/text_only_t5_imgcap_seed0_test_generated_secondstage.txt \
      -output_dir='../checkpoint' \
      -log_name=text_only_t5_imgcap_seed0_generated_secondstage \
      -gpus='1' \
      -batch_size=4 \
      -learning_rate=3e-5 \
      -scheduler_lambda1=10 \
      -scheduler_lambda2=0.95 \
      -num_epochs=50 \
      -grad_accumulate=5 \
      -max_input_len=1024 \
      -max_output_len=256 \
      -max_img_len=1 \
      -n_beams=5 \
      -random_seed=0 \
      -do_train=True \
      -do_test=True \
      -limit_val_batches=1 \
      -val_check_interval=1 \
      -img_lr_factor=5 \
      -checkpoint=None \
      -use_forget_gate \
      -use_img_trans
```

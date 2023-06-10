# Multimodal Summarization of Reddit Threads

This is the GitHub repository for the paper "mRedditSum: Multimodal Summarization of Reddit Threads".

To prepare the data, move your "mredditsum" data split into the "data" folder.

Then, run the following:

```
python scripts/convert_to_datapoints.py -f data/val.json -s data/val_processed.json -sep " "
```
```
python scripts/convert_to_txt.py -f data/val_processed.json
```

To get the resnext embeddings, move all downloaded images into the data/images folder and run the following:

```
python scripts/extract_features.py -f data/images
```

To run the models, use run.py under the src file.

Example of running training on VG-T5:
```
python src/run.py \
        -model=multi_modal_t5 \
        -train_src_path=../data/train_processed_src.txt \
        -train_tgt_path=../data/train_processed_tgt.txt \
        -val_src_path=../data/val_processed_src.txt \
        -val_tgt_path=../data/val_processed_tgt.txt \
        -test_src_path=../data/test_processed_src.txt \
        -test_tgt_path=../data/test_processed_tgt.txt \
        -image_feature_path=../data/image_features/resnext101/ \
        -val_save_file=../results/val/multi_t5_val.txt \
        -test_save_file=../results/test/multi_t5_test.txt \
        -log_name=multi_t5 \
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
        -do_test=False \
        -limit_val_batches=1 \
        -val_check_interval=1 \
        -img_lr_factor=5 \
        -checkpoint=None \
        -use_forget_gate \
        -cross_attn_type=0 \
        -use_img_trans
```

Example of running regular text-only BART:

```
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

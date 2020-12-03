python tools/preprocess_data.py \
       --input data/baike_process.json \
       --output-prefix my-gpt2 \
       --vocab tokenizer/vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file tokenizer/merges.txt \
       --append-eod

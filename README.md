# 11-747-Projects

Contributors: 
1. Daniel Huang, danielhu@andrew.cmu.edu 
2. Yi Shen (Eason), yishen@andrew.cmu.edu
3. Tianhao Wang (Danny), tianhao2@andrew.cmu.edu


The /code directory contains the source code from https://github.com/facebookresearch/SpanBERT. The only modification is on 
SpanBERT/code/run_squad.py, which incorporates the commit from pull request https://github.com/facebookresearch/SpanBERT/pull/60 that is necessary 
for us to run with the `apex` library successfully. 

We (almost) reproduced the F1 score 83.6 with SpanBERT-base model evaluated on SQuAD 2.0 dev sets. The detail steps can be found in our Colab notebook
`HW3_SpanBERT_reproduced_f1.ipynb`. In terms of runtime, we run in GPU + high RAM mode with my personal Colab Pro account. Notice that even with this setup, the 
recommended config of finetuning on SQuAD 2.0 by Facebook still cause GPU out-of-memory error. After experimentation, we decreased the train and eval batchsizes, both from 32 to 16, and was able to finish training on the downstream finetuning task in around 7.5 hours. 

The final result can be found in /code/SpanBERT/squad_output/eval_results.txt
```
HasAns_exact = 81.66329284750337
HasAns_f1 = 88.02528196756217
HasAns_total = 5928
NoAns_exact = 70.95037846930194
NoAns_f1 = 70.95037846930194
NoAns_total = 5945
batch_size = 16
best_exact = 80.63673881916955
best_exact_thresh = -5.0234375
best_f1 = 83.34820529512551
best_f1_thresh = -5.0234375
epoch = 3
exact = 76.29916617535585
f1 = 79.47560612344888
global_step = 25311
learning_rate = 2e-05
total = 11873

```

The best f1 score we achieved by using the pretrained SpanBERT-base and training on SQuAD 2.0 ourself is *83.35* and Facebook Research team reported *83.6* with their same fune-tuning model, as shown above. We attribute this slightly-worse performance delta to the smaller batch_size we use due to the constraint of our computational resources. 

Finally, we omitted to commit our downloaded pretrained model in this repo, as the model exceeds the 100mb size limit of a single file on GitHub. The original path should be `/code/SpanBERT/squad_output/pytorch_model.bin`, and one can be found it along with the model config (hyperparameters) in `https://huggingface.co/SpanBERT/spanbert-base-cased/tree/main`. 

# PNPT
The code for the paper *"PNPT: Prototypical Network with Prompt Template for Few-shot Relation Extraction"*.

### Environments
- ``python 3.8``
- ``PyTorch 1.12.0``
- ``transformers 4.24.0``

### Code
Put all data in the **data** folder, CP pretrained model in the **CP_model** folder (you can download CP model from https://github.com/thunlp/RE-Context-or-Names/tree/master/pretrain or [Google Drive](https://drive.google.com/drive/folders/1AwQLqlHJHPuB1aKJ8XPHu8nu237kgtWj?usp=sharing)), and then you can simply use script: *run_main.sh* for train, evaluation and test.

Set the corresponding parameter values in the script, and then run:
```
sh run_main.sh
```
Some explanations of the parameters in the script:
```
--model_name_or_path
	the path for the BERT-base-uncased
--use_cp
	select bert or cp as backend model
--num_valid_iters
	1000, the evaluation iteration
--load_ckpt
	the path of the trained model
--do_test
  run the test stage after train.
```

# SVM README
Data is stored here: '/data/tir/projects/tir6/strubell/data/wilds/data',
Results are logged:  '/data/tir/projects/tir6/strubell/jaredfer/projects/wild-time/results/icml'

Use `bash run_train.sh $DATASET $METHOD $SEED` or launch as an sbatch job with CLI params it calls  `main.py`.

Most important code is available in `wildtime/methods/base_trainer.py` with method specific code being overwritten in `wildtime/methods/METHOD_DIR/`.

Dataset options: {'fmow', 'arxiv', 'huffpost', 'yearbook'}
Method options: {'ft', 'ewc', ... } -- additional are listed in `wildtime/methods`

Default configs are listed in `wildtime/configs/eval_fix/configs_DATASET.py`


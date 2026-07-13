# DeepCoder + LCB Run 

## Download Dataset 
```
uv add gdown

uv run python examples/train/livecodebench/lcb_download.py --local_dir ~/data/lcb/download

uv run python examples/train/livecodebench/lcb_dataset.py --dataset_dir ~/data/lcb/download --local_dir ~/data/lcb/
```

## Note
* Use the generated JSONL files for training and evaluation. A single large JSON array can be too large for normal `datasets.load_dataset("json")` and may hit PyArrow int32 block-size limits.

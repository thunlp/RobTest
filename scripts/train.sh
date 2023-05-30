nohup python src/train.py --data_name sst2 --model_type base > sst2_base.log &
nohup python src/train.py --data_name agnews --model_type base > agnews_base.log &
nohup python src/train.py --data_name jigsaw --model_type base > jigsaw_base.log &

nohup python src/train.py --data_name sst2 --model_type large > sst2_base.log &
nohup python src/train.py --data_name agnews --model_type large > agnews_base.log &
nohup python src/train.py --data_name jigsaw --model_type large > jigsaw_base.log &

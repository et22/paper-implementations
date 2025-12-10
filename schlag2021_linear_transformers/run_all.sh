python train.py --attn_name "softmax" --update_rule "sum"
python train.py --attn_name "linear" --update_rule "sum"
python train.py --attn_name "dpfp" --update_rule "sum" --attn_arg 1
python train.py --attn_name "dpfp" --update_rule "sum" --attn_arg 2
python train.py --attn_name "dpfp" --update_rule "sum" --attn_arg 3
python plot_figures.py

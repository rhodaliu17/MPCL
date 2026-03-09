gpu='0'

load_path='../results/LA4_LAint4/checkpoints'
python test.py --gpu $gpu \
                     --data_dir='../Data/LA' \
                     --list_dir '../datalist/LA' \
                     --load_path $load_path
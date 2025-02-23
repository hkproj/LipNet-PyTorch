gpu = '0'
random_seed = 0
data_type = 'unseen'
video_path = 'lip/'
train_list = f'data/{data_type}_train.txt'
val_list = f'data/{data_type}_val.txt'
anno_path = 'GRID_align_txt'
vid_padding = 75
txt_padding = 200
batch_size = 24
base_lr = 2e-5
num_workers = 16
max_epoch = 10000
display = 10
test_step = 1000
save_prefix = f'weights/LipNet_{data_type}'
is_optimize = True
italip = False

# ItaLip specific options
clip = None
zero_infinity_loss = False

#weights = 'pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt'

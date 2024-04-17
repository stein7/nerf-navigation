import torch

device_gpu = torch.device('cuda:0')
sr_thresh = 0.0001

def gpu_alloc(gpu_num):
    global device_gpu

    if torch.cuda.is_available():
        device_gpu = torch.device('cuda:'+str(gpu_num))
    else:
        device_gpu = 'cpu'

def change_thresh(thresh_new):
    global sr_thresh

    sr_thresh = thresh_new

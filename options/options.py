import os
from collections import OrderedDict
from datetime import datetime
import json

from utils import util

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def parse(opt_path):
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)     #利用json.loads函数将最原始的Json文件进行解析，放在opt中

    opt['timestamp'] = get_timestamp()           #给opt字典增加一个键值
    scale = opt['scale']
    rgb_range = opt['rgb_range']

    # # export CUDA_VISIBLE_DEVICES
    # if torch.cuda.is_available():     #判断GPU是否可用
    #     gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    #     os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    #     print('===> Export CUDA_VISIBLE_DEVICES = [' + gpu_list + ']')
    # else:
    #     print('===> CPU mode is set (NOTE: GPU is recommended)')
    # 确定使用电脑中的哪一块GPU
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('===> Export CUDA_VISIBLE_DEVICES = [' + gpu_list + ']')

    # datasets,在此处为train何val datasets增加了三个键值信息  因为我们的训练数据集需要这些信息
    for phase, dataset in opt['datasets'].items():
        #phase = phase.split('_')[0]        #这句话在这里是一句废话
        dataset['phase'] = phase
        dataset['scale'] = scale
        dataset['rgb_range'] = rgb_range
        
    # for network initialize
    opt['networks']['scale'] = opt['scale']     #给networks子字典增加一个scale.
    network_opt = opt['networks']
    #输出文件夹的名字，采用了字符串格式化
    config_str = '%s_in%df%d_x%d'%(network_opt['which_model'].upper(), network_opt['in_channels'],
                                                        network_opt['num_features'], opt['scale'])

    exp_path = os.path.join(os.getcwd(), 'experiments', config_str)  #将当前路径，experimnets 和字符串合成一个路径

    if opt['is_train'] and opt['solver']['pretrain']:
        if 'pretrained_path' not in list(opt['solver'].keys()): raise ValueError("[Error] The 'pretrained_path' does not declarate in *.json")
        exp_path = os.path.dirname(os.path.dirname(opt['solver']['pretrained_path']))
        if opt['solver']['pretrain'] == 'finetune': exp_path += '_finetune'

    exp_path = os.path.relpath(exp_path)

    path_opt = OrderedDict()
    path_opt['exp_root'] = exp_path
    path_opt['epochs'] = os.path.join(exp_path, 'epochs')
    path_opt['visual'] = os.path.join(exp_path, 'visual')
    path_opt['records'] = os.path.join(exp_path, 'records')
    opt['path'] = path_opt

    if opt['is_train']:
        # create folders
        if opt['solver']['pretrain'] == 'resume':
            opt = dict_to_nonedict(opt)
        else:
            util.mkdir_and_rename(opt['path']['exp_root'])      # rename old experiments if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'exp_root'))
            save(opt)
            opt = dict_to_nonedict(opt)

        print("===> Experimental DIR: [%s]"%exp_path)

    return opt


def save(opt):
    dump_dir = opt['path']['exp_root']
    dump_path = os.path.join(dump_dir, 'options.json')
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)


class NoneDict(dict):
    def __missing__(self, key):
        return None

# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
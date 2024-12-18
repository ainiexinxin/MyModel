import argparse
from recbole.quick_start import run_recbole as run_recbole
from datetime import datetime
import common.utils.tool as tool

if __name__ == '__main__':
    import argparse
    from datetime import datetime

    # 1.set param
    parser = argparse.ArgumentParser()
    # set model
    parser.add_argument('--model', '-m', type=str, default='MyModel', help='name of models')
    # set datasets
    parser.add_argument('--dataset', '-d', type=str, default='steam', help='name of datasets')
    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--train_batch_size', type=int, default=512, help='train batch size')
    parser.add_argument('--augment_threshold', type=int, default=12, help='augment threshold')
    # get param
    args, _ = parser.parse_known_args()
    # config list
    config_file_list = ['./allconfig.yaml']
    parameter_dict = vars(args)

    # 先声明 RUNNING_FLAG 为全局变量
    global RUNNING_FLAG

    # 如果 RUNNING_FLAG 未定义，则初始化为 None
    if 'RUNNING_FLAG' not in globals():
        RUNNING_FLAG = None

    # 赋值 RUNNING_FLAG
    RUNNING_FLAG = f'RF{datetime.now().strftime("%Y%m%d%H%M%S")}' if RUNNING_FLAG is None else RUNNING_FLAG
    parameter_dict['running_flag'] = RUNNING_FLAG
    print(parameter_dict)

    # 2.call recbole_trm: config, dataset, model, trainer, training, evaluation
    if config_file_list:
        run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=parameter_dict)
    else:
        run_recbole(model=args.model, dataset=args.dataset, config_dict=parameter_dict)

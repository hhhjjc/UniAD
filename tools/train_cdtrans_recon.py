import argparse
import logging
import os
import pprint
import torch
import yaml
from datasets.data_builder import build_dataloader
from easydict import EasyDict
from models.model_helper import ModelHelper
from processor.processor_cdtrans_recon import do_train_cdtrans_recon, do_inference_cdtrans_recon
from utils.dist_helper import setup_distributed
from utils.misc_helper import (
    create_logger,
    get_current_time,
    load_state,
    set_random_seed,
    update_config,
)

parser = argparse.ArgumentParser(description="CDTrans-Recon Framework")
parser.add_argument("--config", default="./config.yaml")
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("--local_rank", default=None, help="local rank for dist")

def main():
    global args, config, key_metric, best_metric
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.port = config.get("port", None)
    rank, world_size = setup_distributed(port=config.port)
    config = update_config(config)

    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    config.evaluator.eval_dir = os.path.join(config.exp_path, config.evaluator.save_dir)
    if rank == 0:
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)

        current_time = get_current_time()
        logger = create_logger(
            "global_logger", config.log_path + "/dec_{}.log".format(current_time)
        )
        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))
    else:
        logger = None

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)

    # 创建模型
    model = ModelHelper(config.net)
    model.cuda()
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
    )

    # 定义冻结和活跃层
    layers = []
    for module in config.net:
        layers.append(module["name"])
    frozen_layers = config.get("frozen_layers", [])
    active_layers = list(set(layers) ^ set(frozen_layers))
    if rank == 0:
        logger.info("layers: {}".format(layers))
        logger.info("active layers: {}".format(active_layers))

    # 选择需要更新的参数
    parameters = [
        {"params": getattr(model.module, layer).parameters()} for layer in active_layers
    ]

    # 创建优化器
    from utils.optimizer_helper import get_optimizer
    optimizer = get_optimizer(parameters, config.trainer.optimizer)
    
    # 创建学习率调度器
    from utils.lr_helper import get_scheduler
    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)

    # 初始化最佳指标
    key_metric = config.evaluator["key_metric"]
    best_metric = 0
    last_epoch = 0

    # 加载模型（如果有）
    auto_resume = config.saver.get("auto_resume", True)
    resume_model = config.saver.get("resume_model", None)
    load_path = config.saver.get("load_path", None)

    if resume_model and not resume_model.startswith("/"):
        resume_model = os.path.join(config.exp_path, resume_model)
    lastest_model = os.path.join(config.save_path, "ckpt.pth.tar")
    if auto_resume and os.path.exists(lastest_model):
        resume_model = lastest_model
    if resume_model:
        best_metric, last_epoch = load_state(resume_model, model, optimizer=optimizer)
    elif load_path:
        if not load_path.startswith("/"):
            load_path = os.path.join(config.exp_path, load_path)
        load_state(load_path, model)

    # 创建数据加载器
    train_loader, val_loader = build_dataloader(config.dataset, distributed=True)

    # 创建用于CDTrans训练的中心损失
    from loss.center_loss import CenterLoss
    center_criterion = CenterLoss(num_classes=config.get("num_classes", 10), feat_dim=256)
    
    # 创建损失函数
    from utils.criterion_helper import build_criterion
    criterion = build_criterion(config.criterion)

    # 评估或训练
    if args.evaluate:
        do_inference_cdtrans_recon(config, model, val_loader, 0)
    else:
        do_train_cdtrans_recon(
            config,
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer,  # 使用同一个优化器
            lr_scheduler,
            criterion,
            0,  # num_query
            local_rank
        )

if __name__ == "__main__":
    main()
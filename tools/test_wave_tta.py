import argparse
import logging
import os
import pprint
import yaml
import torch
import torch.distributed as dist
from datasets.data_builder import build_dataloader
from easydict import EasyDict
from models.model_helper import ModelHelper
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.dist_helper import setup_distributed
from utils.eval_helper import dump, log_metrics, merge_together, performances
from utils.misc_helper import (
    create_logger,
    get_current_time,
    load_state,
    set_random_seed,
    update_config,
)
from utils.vis_helper import visualize_compound, visualize_single
from sam import SAM
from wave_tta import WaveAD_TTA, configure_model, collect_filter_params
from utils.optimizer_helper import get_optimizer

parser = argparse.ArgumentParser(description="WaveAD Test-Time Adaptation")
parser.add_argument("--config", default=None)
parser.add_argument("--load_path", default=None, 
                   help="Path to load trained model")
parser.add_argument("--local_rank", default=None, help="local rank for dist")

def main():
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    rank, world_size = setup_distributed(port=config.get("port", None))
    config = update_config(config)

    # 设置保存路径
    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    config.evaluator.eval_dir = os.path.join(config.exp_path, config.evaluator.save_dir + "_tta")
    
    if rank == 0:
        os.makedirs(config.evaluator.eval_dir, exist_ok=True)
        current_time = get_current_time()
        logger = create_logger(
            "global_logger", config.log_path + f"/wave_tta_{current_time}.log"
        )
        logger.info(f"config: {pprint.pformat(config)}")
    else:
        logger = None

    # 设置随机种子
    random_seed = config.get("random_seed", None)
    if random_seed:
        set_random_seed(random_seed)

    # 构建模型
    model = ModelHelper(config.net)
    model.cuda()
    
    # 加载预训练模型
    if args.load_path:
        if not args.load_path.startswith("/"):
            load_path = os.path.join(config.exp_path, args.load_path)
        else:
            load_path = args.load_path
            
        if rank == 0:
            logger.info(f"Loading model from {load_path}")
        load_state(load_path, model)
    
    # 配置模型用于TTA，只更新filter函数参数
    model = configure_model(model)
    
    # 收集filter参数用于优化
    filter_params, param_names = collect_filter_params(model)
    if rank == 0:
        logger.info(f"Parameters to be updated during TTA: {param_names}")
    
    # 使用SAM优化器 (Sharpness-Aware Minimization)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(filter_params, base_optimizer, lr=0.001, momentum=0.9)
    # optimizer = get_optimizer(filter_params, config.trainer.optimizer)
    # 应用TTA包装器
    tta_model = WaveAD_TTA(model, optimizer, steps=1, episodic=False)
    
    # DDP包装
    local_rank = int(os.environ["LOCAL_RANK"])
    tta_model = DDP(
        tta_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )
    
    # 准备数据加载器
    _, val_loader = build_dataloader(config.dataset, distributed=True)
    
    # 开始评估
    if rank == 0:
        logger.info("Starting TTA evaluation...")
    
    # TTA评估
    validate(val_loader, tta_model, config, logger, rank)
    
    # 清理
    dist.barrier()
    if rank == 0:
        logger.info("TTA evaluation completed.")

def validate(val_loader, model, config, logger, rank):
    model.eval()  # 虽然是eval模式，但TTA会在内部设置需要梯度的层为训练模式
    
    # 等待所有进程就绪
    dist.barrier()
    
    with torch.no_grad():  # TTA在必要的地方会启用梯度
        for i, input in enumerate(val_loader):
            # 前向传播，会包含TTA更新
            outputs = model(input)
            dump(config.evaluator.eval_dir, outputs)
            
            if (i + 1) % 10 == 0 and rank == 0:
                logger.info(f"Processed batch {i+1}/{len(val_loader)}")

    # 等待所有进程完成
    dist.barrier()
    
    # 最终结果汇总和评估（只在rank=0上进行）
    if rank == 0:
        logger.info("Gathering final results...")
        fileinfos, preds, masks = merge_together(config.evaluator.eval_dir)
        
        # 评估性能
        ret_metrics = performances(fileinfos, preds, masks, config.evaluator.metrics)
        log_metrics(ret_metrics, config.evaluator.metrics)
        
        # 可视化
        # if config.evaluator.get("vis_compound", None):
        #     visualize_compound(
        #         fileinfos,
        #         preds,
        #         masks,
        #         config.evaluator.vis_compound,
        #         config.dataset.image_reader,
        #     )
        # if config.evaluator.get("vis_single", None):
        #     visualize_single(
        #         fileinfos,
        #         preds,
        #         config.evaluator.vis_single,
        #         config.dataset.image_reader,
        #     )
            
        # 返回关键指标
        key_metric = config.evaluator["key_metric"]
        logger.info(f"TTA {key_metric}: {ret_metrics[key_metric]:.4f}")
        return ret_metrics

if __name__ == "__main__":
    main()
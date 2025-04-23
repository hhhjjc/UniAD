import argparse
import logging
import os
import pprint
import shutil
import time

import torch
import torch.distributed as dist
import torch.optim
import yaml
from datasets.data_builder import build_dataloader
from easydict import EasyDict
from models.model_helper import ModelHelper
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.criterion_helper import build_criterion
from utils.dist_helper import setup_distributed
from utils.eval_helper import dump, log_metrics, merge_together, performances
from utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    load_state,
    set_random_seed,
    update_config,
)
from utils.vis_helper import visualize_compound, visualize_single

parser = argparse.ArgumentParser(description="UniAD Framework with TTA")
parser.add_argument("--config", default="./config_tta.yaml")
parser.add_argument("--local_rank", default=None, help="local rank for dist")
parser.add_argument("--tta_steps", default=1, type=int, help="number of TTA steps")
parser.add_argument("--tta_lr", default=1e-5, type=float, help="learning rate for TTA")
parser.add_argument("--tta_threshold", default=0.5, type=float, help="reconstruction error threshold for TTA")
parser.add_argument("--checkpoint", required=True, help="path to checkpoint")


class TTAWrapper:
    def __init__(self, model, tta_lr=1e-5, tta_threshold=0.5):
        self.model = model
        self.tta_lr = tta_lr
        self.tta_threshold = tta_threshold
        
        # 识别需要更新的filter参数
        self.tta_params = []
        self.collect_filter_params()
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(self.tta_params, lr=self.tta_lr)
        
        # 保存原始参数以便回滚
        self.original_params = {}
        self.save_original_params()
    
    def collect_filter_params(self):
        """收集所有需要更新的filter参数"""
        # 检查模型是否包含Wave模块
        if hasattr(self.model.module, 'transformer'):
            transformer = self.model.module.transformer
            
            # 收集编码器中的filter参数
            if hasattr(transformer, 'encoder'):
                for layer in transformer.encoder.layers:
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'filter'):
                        self.tta_params.extend(layer.self_attn.filter.parameters())
            
            # 收集解码器中的filter参数
            if hasattr(transformer, 'decoder'):
                for layer in transformer.decoder.layers:
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'filter'):
                        self.tta_params.extend(layer.self_attn.filter.parameters())
                    if hasattr(layer, 'multihead_attn') and hasattr(layer.multihead_attn, 'filter'):
                        self.tta_params.extend(layer.multihead_attn.filter.parameters())
    
    def save_original_params(self):
        """保存原始参数以便回滚"""
        for param in self.tta_params:
            self.original_params[param] = param.data.clone()
    
    def restore_original_params(self):
        """恢复原始参数"""
        for param in self.tta_params:
            param.data = self.original_params[param].clone()
    
    def tta_step(self, input):
        """执行一步TTA更新"""
        # 前向传播
        with torch.cuda.amp.autocast():
            outputs = self.model(input)
        
        # 计算重建误差
        feature_rec = outputs["feature_rec"]
        feature_align = outputs["feature_align"]
        
        # 计算每个样本的重建误差
        batch_size = feature_rec.size(0)
        rec_errors = []
        for i in range(batch_size):
            error = torch.sqrt(torch.sum((feature_rec[i] - feature_align[i]) ** 2))
            rec_errors.append(error)
        rec_errors = torch.stack(rec_errors)
        
        # 选择性更新：只在重建误差小于阈值的样本上更新
        mask = rec_errors < self.tta_threshold
        
        if mask.sum() > 0:
            # 计算鲁棒的重建损失
            temperature = 0.1
            loss = torch.mean(
                torch.pow(feature_rec[mask] - feature_align[mask], 2) * 
                torch.exp(-temperature * torch.pow(feature_rec[mask] - feature_align[mask], 2))
            )
            
            # 添加正则化项防止过度适应
            reg_loss = 0.0
            for param in self.tta_params:
                reg_loss += torch.norm(param - self.original_params[param])
            
            total_loss = loss + 0.001 * reg_loss
            
            # 更新参数
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            return total_loss.item(), outputs, mask.sum().item()
        else:
            return 0.0, outputs, 0


def main():
    global args, config
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
            "global_logger", config.log_path + "/tta_{}.log".format(current_time)
        )
        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))

    random_seed = config.get("random_seed", None)
    if random_seed:
        set_random_seed(random_seed)

    # 创建模型
    model = ModelHelper(config.net)
    model.cuda()
    local_rank = int(os.environ["LOCAL_RANK"])
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    # 加载checkpoint
    load_state(args.checkpoint, model)

    # 创建数据加载器
    _, val_loader = build_dataloader(config.dataset, distributed=True)

    # 创建TTA包装器
    tta_wrapper = TTAWrapper(model, tta_lr=args.tta_lr, tta_threshold=args.tta_threshold)

    # 运行测试和TTA
    test_with_tta(val_loader, model, tta_wrapper, args.tta_steps)


def test_with_tta(val_loader, model, tta_wrapper, tta_steps):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)
    updates = AverageMeter(0)

    # 保持BatchNorm和Dropout在eval模式，但允许filter参数更新
    model.eval()
    
    # 设置模型为TTA模式（仅更新filter参数）
    for param in model.parameters():
        param.requires_grad = False
    for param in tta_wrapper.tta_params:
        param.requires_grad = True
    
    rank = dist.get_rank()
    logger = logging.getLogger("global_logger")
    criterion = build_criterion(config.criterion)
    end = time.time()

    if rank == 0:
        os.makedirs(config.evaluator.eval_dir, exist_ok=True)
    dist.barrier()
    
    # 保存当前性能用于回滚判断
    best_performance = float('inf')
    
    with torch.set_grad_enabled(True):
        for i, input in enumerate(val_loader):
            batch_performance_history = []
            
            # 对每个batch进行多步TTA
            for tta_step in range(tta_steps):
                loss_value, outputs, num_updates = tta_wrapper.tta_step(input)
                updates.update(num_updates / len(input["filename"]))
                
                # 记录每步性能
                with torch.no_grad():
                    loss = 0
                    for name, criterion_loss in criterion.items():
                        weight = criterion_loss.weight
                        loss += weight * criterion_loss(outputs)
                    batch_performance_history.append(loss.item())
                
                # 如果性能连续下降，回滚参数
                if tta_step > 0 and batch_performance_history[-1] > batch_performance_history[-2] * 1.1:
                    if rank == 0:
                        logger.info(f"Performance degraded at batch {i}, step {tta_step}. Rolling back...")
                    tta_wrapper.restore_original_params()
                    break
                
                if tta_step == tta_steps - 1:  # 最后一步，保存结果
                    dump(config.evaluator.eval_dir, outputs)
                    
                    # 记录最终损失
                    num = len(outputs["filename"])
                    losses.update(loss.item(), num)
            
            # 在每个batch结束后决定是否保存当前参数
            current_performance = batch_performance_history[-1] if batch_performance_history else float('inf')
            if current_performance < best_performance:
                best_performance = current_performance
                tta_wrapper.save_original_params()  # 更新"原始"参数为当前最好的参数
            
            # 测量时间
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % config.trainer.print_freq_step == 0 and rank == 0:
                logger.info(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Updates {updates.val:.2f} ({updates.avg:.2f})\t"
                    "Current Loss {:.5f}".format(
                        i + 1, len(val_loader), current_performance,
                        batch_time=batch_time, updates=updates
                    )
                )

    # 收集最终结果
    dist.barrier()
    total_num = torch.Tensor([losses.count]).cuda()
    loss_sum = torch.Tensor([losses.avg * losses.count]).cuda()
    dist.all_reduce(total_num, async_op=True)
    dist.all_reduce(loss_sum, async_op=True)
    final_loss = loss_sum.item() / total_num.item()

    ret_metrics = {}
    if rank == 0:
        logger.info("Gathering final results with TTA...")
        logger.info(" * Loss {:.5f}\ttotal_num={}".format(final_loss, total_num.item()))
        fileinfos, preds, masks = merge_together(config.evaluator.eval_dir)
        shutil.rmtree(config.evaluator.eval_dir)
        
        ret_metrics = performances(fileinfos, preds, masks, config.evaluator.metrics)
        log_metrics(ret_metrics, config.evaluator.metrics)
        
        if config.evaluator.get("vis_compound", None):
            visualize_compound(
                fileinfos,
                preds,
                masks,
                config.evaluator.vis_compound,
                config.dataset.image_reader,
            )
        if config.evaluator.get("vis_single", None):
            visualize_single(
                fileinfos,
                preds,
                config.evaluator.vis_single,
                config.dataset.image_reader,
            )
    
    return ret_metrics


if __name__ == "__main__":
    main()
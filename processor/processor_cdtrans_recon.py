import logging
import numpy as np
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, Class_accuracy_eval
from torch.cuda import amp
import torch.distributed as dist
import random
from tqdm import tqdm

def do_train_cdtrans_recon(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    """CDTrans-Recon训练函数"""
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info('开始CDTrans-Recon训练')
    
    if device:
        model.to(local_rank)
        if cfg.MODEL.DIST_TRAIN:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # 训练度量
    loss_meter = AverageMeter()
    rec_loss_meter = AverageMeter()  # 重建损失
    cons_loss_meter = AverageMeter()  # 一致性损失
    align_loss_meter = AverageMeter()  # 特征对齐损失
    
    # 评估器
    if cfg.MODEL.TASK_TYPE == 'classify_DA':
        evaluator = Class_accuracy_eval(dataset=cfg.DATASETS.NAMES, logger=logger)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    
    scaler = amp.GradScaler()
    best_model_mAP = 0
    min_mean_ent = 1e5
    
    # 创建类别信息映射
    class_to_idx = {}
    if hasattr(train_loader.dataset, 'classes'):
        for idx, cls_name in enumerate(train_loader.dataset.classes):
            class_to_idx[cls_name] = idx
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        rec_loss_meter.reset()
        cons_loss_meter.reset()
        align_loss_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        
        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
            if len(img) == 1:
                continue
                
            # 准备源域和目标域数据
            # 采用样本配对策略
            batch_size = img.size(0)
            if batch_size % 2 != 0:
                img = img[:-1]
                vid = vid[:-1]
                target_cam = target_cam[:-1]
                target_view = target_view[:-1]
                batch_size -= 1
            
            # 将批次分为源域和目标域
            half_size = batch_size // 2
            src_img = img[:half_size]
            tgt_img = img[half_size:]
            src_vid = vid[:half_size]
            tgt_vid = vid[half_size:]
            
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            src_img = src_img.to(device)
            tgt_img = tgt_img.to(device)
            src_vid = src_vid.to(device)
            tgt_vid = tgt_vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            
            with amp.autocast(enabled=True):
                # 源域到源域重建（自重建）
                src_input = {"image": src_img}
                src_features = model.module.backbone(src_input)["features"]
                src_feature_align = model.module.neck({"features": src_features})["feature_align"]
                src_output = model({"feature_align": src_feature_align, "clsname": ["source"] * half_size})
                
                # 目标域到目标域重建（自重建）
                tgt_input = {"image": tgt_img}
                tgt_features = model.module.backbone(tgt_input)["features"]
                tgt_feature_align = model.module.neck({"features": tgt_features})["feature_align"]
                tgt_output = model({"feature_align": tgt_feature_align, "clsname": ["target"] * half_size})
                
                # 源域到目标域重建（跨域重建）
                cross_output = model({
                    "feature_align": tgt_feature_align, 
                    "clsname": ["target"] * half_size
                }, prototype_idx=src_vid)
                
                # 计算损失
                # 1. 重建损失
                src_rec_loss = loss_fn(src_output)
                tgt_rec_loss = loss_fn(tgt_output)
                rec_loss = (src_rec_loss + tgt_rec_loss) / 2.0
                
                # 2. 一致性损失 - 确保跨域重建与自重建一致
                cons_loss = F.mse_loss(cross_output["feature_rec"], tgt_output["feature_rec"])
                
                # 3. 特征对齐损失
                # 计算源域和目标域特征的MMD距离
                src_feat = src_output["feature_align"].view(half_size, -1)
                tgt_feat = tgt_output["feature_align"].view(half_size, -1)
                align_loss = mmd_distance(src_feat, tgt_feat)
                
                # 总损失
                loss = rec_loss + 0.1 * cons_loss + 0.01 * align_loss
            
            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            
            # 更新度量
            loss_meter.update(loss.item(), half_size)
            rec_loss_meter.update(rec_loss.item(), half_size)
            cons_loss_meter.update(cons_loss.item(), half_size)
            align_loss_meter.update(align_loss.item(), half_size)
            
            # 日志记录
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] "
                    "Loss: {:.3f}, RecLoss: {:.3f}, ConsLoss: {:.3f}, AlignLoss: {:.3f}, "
                    "Base Lr: {:.2e}"
                    .format(
                        epoch, n_iter + 1, len(train_loader),
                        loss_meter.avg, rec_loss_meter.avg, 
                        cons_loss_meter.avg, align_loss_meter.avg,
                        scheduler._get_lr(epoch)[0]
                    )
                )
        
        # 每个epoch结束后的处理
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info(
            "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
            .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch)
        )
        
        # 保存检查点
        if epoch % checkpoint_period == 0:
            torch.save(
                model.state_dict(),
                os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch))
            )
        
        # 验证
        if epoch % eval_period == 0:
            validate_cdtrans_recon(cfg, model, val_loader, evaluator, epoch, logger)

    # 最终评估
    logger.info("训练完成，加载最佳模型进行评估")
    best_model_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        validate_cdtrans_recon(cfg, model, val_loader, evaluator, epochs, logger, final_eval=True)


def validate_cdtrans_recon(cfg, model, val_loader, evaluator, epoch, logger, final_eval=False):
    """CDTrans-Recon验证函数"""
    device = "cuda"
    model.eval()
    evaluator.reset()
    
    # 收集每个类别的特征，用于更新原型
    features_dict = {}
    
    with torch.no_grad():
        for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            
            # 特征提取
            input_dict = {"image": img}
            features = model.module.backbone(input_dict)["features"]
            feature_align = model.module.neck({"features": features})["feature_align"]
            
            # 根据任务类型进行不同处理
            if cfg.MODEL.TASK_TYPE == 'classify_DA':
                # 分类任务
                output = model({"feature_align": feature_align})
                evaluator.update((output["pred"], vid))
                
                # 收集类别特征
                for i, cls_id in enumerate(vid):
                    cls_id = cls_id.item()
                    if cls_id not in features_dict:
                        features_dict[cls_id] = []
                    # 提取全局特征
                    global_feat = feature_align[i].mean(dim=(1, 2))
                    features_dict[cls_id].append(global_feat)
            else:
                # 重建任务
                output = model({"feature_align": feature_align})
                evaluator.update((output["pred"], vid, camid))
        
        # 计算结果
        if cfg.MODEL.TASK_TYPE == 'classify_DA':
            accuracy, mean_ent = evaluator.compute()
            
            # 根据验证结果判断是否保存最佳模型
            is_best = False
            if not final_eval:
                if accuracy > logger.best_model_mAP:
                    logger.best_model_mAP = accuracy
                    logger.min_mean_ent = mean_ent
                    is_best = True
            
            logger.info(
                "验证结果 - Epoch: {}, Accuracy: {:.1%}, Mean Entropy: {:.1%}"
                .format(epoch, accuracy, mean_ent)
            )
            
            if is_best and not final_eval:
                torch.save(
                    model.state_dict(),
                    os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best_model.pth')
                )
        else:
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            
            # 根据验证结果判断是否保存最佳模型
            is_best = False
            if not final_eval:
                if mAP > logger.best_model_mAP:
                    logger.best_model_mAP = mAP
                    is_best = True
            
            logger.info(
                "验证结果 - Epoch: {}, mAP: {:.1%}"
                .format(epoch, mAP)
            )
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            
            if is_best and not final_eval:
                torch.save(
                    model.state_dict(),
                    os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best_model.pth')
                )
    
    # 更新模型中的类别原型（仅在最终评估时）
    if final_eval and hasattr(model.module, 'update_prototypes'):
        logger.info("更新类别原型")
        # 将特征列表转换为张量
        for cls_id, feat_list in features_dict.items():
            if len(feat_list) > 0:
                features_dict[cls_id] = torch.stack(feat_list)
        model.module.update_prototypes(features_dict)


def do_inference_cdtrans_recon(cfg,
                 model,
                 val_loader,
                 num_query):
    """CDTrans-Recon推理函数"""
    device = "cuda"
    logger = logging.getLogger("reid_baseline.test")
    logger.info("开始推理")
    
    # 选择评估器
    if cfg.MODEL.TASK_TYPE == 'classify_DA':
        evaluator = Class_accuracy_eval(dataset=cfg.DATASETS.NAMES, logger=logger)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    
    evaluator.reset()
    model.eval()
    
    # 自适应步骤：先收集少量无标签样本更新原型
    if cfg.get('ADAPTATION', False) and cfg.ADAPTATION.ENABLED:
        logger.info("执行测试时适应")
        adapt_samples = cfg.ADAPTATION.NUM_SAMPLES
        adapt_iters = cfg.ADAPTATION.ITERATIONS
        adapt_lr = cfg.ADAPTATION.LEARNING_RATE
        
        # 收集适应样本
        adapt_features = []
        adapt_imgs = []
        sample_count = 0
        
        for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
            if sample_count >= adapt_samples:
                break
            
            with torch.no_grad():
                img = img.to(device)
                # 只使用正常样本进行适应
                normal_indices = (vid == 0).nonzero(as_tuple=True)[0]
                if len(normal_indices) == 0:
                    continue
                
                img = img[normal_indices]
                adapt_imgs.append(img)
                
                # 特征提取
                input_dict = {"image": img}
                features = model.backbone(input_dict)["features"]
                feature_align = model.neck({"features": features})["feature_align"]
                adapt_features.append(feature_align)
                
                sample_count += len(normal_indices)
        
        if len(adapt_features) > 0:
            # 合并适应样本
            adapt_features = torch.cat(adapt_features, dim=0)
            adapt_imgs = torch.cat(adapt_imgs, dim=0)
            
            # 创建优化器，只更新原型和适应层
            adapt_params = []
            if hasattr(model, 'prototype_embed'):
                adapt_params.append(model.prototype_embed.parameters())
            if hasattr(model, 'domain_adapter'):
                adapt_params.append(model.domain_adapter.parameters())
            
            if len(adapt_params) > 0:
                adapt_optimizer = torch.optim.Adam(adapt_params, lr=adapt_lr)
                
                # 执行适应迭代
                for i in range(adapt_iters):
                    adapt_optimizer.zero_grad()
                    
                    # 重建损失作为自监督信号
                    output = model({"feature_align": adapt_features})
                    rec_loss = F.mse_loss(output["feature_rec"], adapt_features)
                    rec_loss.backward()
                    
                    adapt_optimizer.step()
                    logger.info(f"适应迭代 {i+1}/{adapt_iters}, 损失: {rec_loss.item():.4f}")
    
    # 执行常规评估
    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            
            # 特征提取与异常检测
            input_dict = {"image": img}
            features = model.backbone(input_dict)["features"]
            feature_align = model.neck({"features": features})["feature_align"]
            
            if cfg.MODEL.TASK_TYPE == 'classify_DA':
                output = model({"feature_align": feature_align})
                evaluator.update((output["pred"], vid))
            else:
                output = model({"feature_align": feature_align})
                evaluator.update((output["pred"], vid, camid))
    
    # 计算结果
    if cfg.MODEL.TASK_TYPE == 'classify_DA':
        accuracy, mean_ent = evaluator.compute()
        logger.info(f"测试结果 - Accuracy: {accuracy:.1%}, Mean Entropy: {mean_ent:.1%}")
        return accuracy
    else:
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logger.info(f"测试结果 - mAP: {mAP:.1%}")
        for r in [1, 5, 10]:
            logger.info(f"CMC curve, Rank-{r}:{cmc[r-1]:.1%}")
        return cmc[0], mAP


def mmd_distance(source_features, target_features):
    """计算两组特征之间的MMD距离"""
    # 计算内核矩阵
    def gaussian_kernel(x, y, sigma=1.0):
        dist = torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=2)
        return torch.exp(-dist / (2 * sigma ** 2))
    
    # 使用多个带宽
    sigmas = [1.0, 5.0, 10.0]
    
    xx_kernel = 0
    xy_kernel = 0
    yy_kernel = 0
    
    for sigma in sigmas:
        xx_kernel += gaussian_kernel(source_features, source_features, sigma)
        xy_kernel += gaussian_kernel(source_features, target_features, sigma)
        yy_kernel += gaussian_kernel(target_features, target_features, sigma)
    
    n = source_features.shape[0]
    m = target_features.shape[0]
    
    # 计算MMD平方距离
    mmd = xx_kernel.sum() / (n * n) + yy_kernel.sum() / (m * m) - 2 * xy_kernel.sum() / (n * m)
    return mmd

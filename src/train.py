import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from data.sejong_southbay import SejongSouthbayTupleLoader

from models.salsa import SALSA

from loss.loss import find_loss
from utils.misc_utils import tuple_collate_fn, read_yaml_config

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def print_nb_params(m):  # 定义一个函数，用于打印模型的可训练参数数量
    model_parameters = filter(
        lambda p: p.requires_grad, m.parameters()
    )  # 过滤出模型中所有需要梯度更新的参数
    params = sum([np.prod(p.size()) for p in model_parameters])  # 计算所有可训练参数的总数量
    print(f"Trainable parameters: {params/1e6:.3}M")  # 打印可训练参数的数量，以百万为单位
    del model_parameters, params  # 删除变量，释放内存


def print_model_size(model):  # 定义一个函数，用于打印模型的大小
    param_size = 0  # 初始化参数大小为 0
    for param in model.parameters():  # 遍历模型的所有参数
        param_size += param.nelement() * param.element_size()  # 计算每个参数的大小，并累加到参数大小中
    buffer_size = 0  # 初始化缓冲区大小为 0
    for buffer in model.buffers():  # 遍历模型的所有缓冲区
        buffer_size += (
            buffer.nelement() * buffer.element_size()
        )  # 计算每个缓冲区的大小，并累加到缓冲区大小中

    size_all_mb = (param_size + buffer_size) / 1024**2  # 计算模型的总大小，以 MB 为单位
    print("model size: {:.3f}MB".format(size_all_mb))  # 打印模型的大小


def main():

    # 打印当前使用的 CUDA 设备编号
    print("Current device:", torch.cuda.current_device())
    # 打印当前使用的 CUDA 设备名称
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # 读取训练配置文件
    config = read_yaml_config(
        os.path.join(os.path.dirname(__file__), "config/train.yaml")
    )
    # 初始化 TensorBoard 的 SummaryWriter，用于记录训练过程中的指标
    writer = SummaryWriter(config["writer_loc"])

    # 获取批量大小
    batch_size = config["batch_size"]
    # 数据增强变换（此处为 None）
    train_transform = None
    # 初始化数据集加载器
    dataset = SejongSouthbayTupleLoader(
        cached_queries=config["cached_queries"], pcl_transform=train_transform
    )
    # 获取设备信息（CPU 或 GPU）
    device = config["device"]

    # 初始化模型并将其移动到指定设备上
    model = SALSA(voxel_sz=0.5).to(device)

    # 检查是否存在已保存的模型权重文件
    checkpoint_path = os.path.join(
        os.path.dirname(__file__), "checkpoints/SALSA/Model/model_27.pth"
    )
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        # 如果存在，加载模型权重
        print(f"Loading model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
        # 读取已训练的轮次
        start_epoch = int(checkpoint_path.split("_")[-1].split(".")[0]) + 1
        print("continue train from epoch", start_epoch)

    # 设置模型为训练模式
    model.train()
    # 打印模型参数数量
    print_nb_params(model)
    # 打印模型大小
    print_model_size(model)
    # 获取最大训练轮次
    MAX_EPOCH = config["max_epoch"]

    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # 初始化批次计数器和子缓存计数器
    kk_batch = 0
    kk_subcache = 0
    # 进入训练轮次循环
    for e in range(start_epoch, MAX_EPOCH):
        # 初始化每轮次的损失列表
        EPOCH_LOSS = []
        # 记录当前时间
        time1 = time.time()
        # 更新数据集
        dataset.new_epoch()
        # 计算每轮次的步数
        steps_per_epoch = int(np.ceil(1000 / batch_size))
        # 初始化学习率调度器
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["max_lr"],
            epochs=dataset.nCacheSubset,
            steps_per_epoch=steps_per_epoch,
            anneal_strategy="cos",
            cycle_momentum=False,
        )
        # 获取当前学习率
        lr_list = [scheduler.get_last_lr()]
        # 更新学习率列表
        for ii in range(dataset.nCacheSubset):
            scheduler.step((ii + 1) * steps_per_epoch)
            lr_list.append(scheduler.get_last_lr())

        # 进入子缓存循环
        for current_subset in range(0, dataset.nCacheSubset):
            # 初始化子缓存损失列表
            CACHE_LOSS = []
            # 设置当前子缓存
            dataset.current_subset = current_subset
            # 更新子缓存数据
            dataset.update_subcache(model, outputdim=config["outdim"])
            # 如果子缓存中的三元组数量为零，则跳过该子缓存
            if len(dataset.triplets) == 0:
                continue
            # 设置模型为训练模式
            model.train()
            # 初始化数据加载器
            data_loader = torch.utils.data.DataLoader(
                dataset=dataset,
                shuffle=True,
                batch_size=batch_size,
                collate_fn=tuple_collate_fn,
                num_workers=16,
            )
            # 计算每个批次的学习率
            scheduler_lr = np.linspace(
                lr_list[current_subset], lr_list[current_subset + 1], len(data_loader)
            )

            # 进入批次循环
            for i, batch_data in enumerate(data_loader):
                # 清零模型和优化器的梯度
                model.zero_grad()
                optimizer.zero_grad()
                # 获取批次数据并移动到设备上
                # coord是0.5^3 的体素化坐标，并且减去了最小值进行归一化
                # xyz和feat完全相同，是data_loader返回的numpy数组
                coord, xyz, feat, batch_number, labels, point_pos_pairs = batch_data
                coord, xyz, feat, batch_number, labels = (
                    coord.to(device),
                    xyz.to(device),
                    feat.to(device),
                    batch_number.to(device),
                    labels.to(device),
                )
                # 前向传播计算局部特征和全局描述符
                local_features, global_descriptor = model(
                    coord, xyz, feat, batch_number
                )

                # 计算损失并反向传播
                loss = find_loss(local_features, global_descriptor, point_pos_pairs)
                loss.backward()
                optimizer.step()
                # 更新学习率
                for param_group in optimizer.param_groups:
                    last_lr = param_group["lr"]
                    param_group["lr"] = scheduler_lr[i][0]
                # 记录批次损失和学习率到 TensorBoard
                writer.add_scalar("Batch Loss", loss.item(), kk_batch)
                writer.add_scalar("Batch LR", last_lr, kk_batch)
                # 更新批次计数器
                kk_batch += 1
                # 更新子缓存损失列表
                CACHE_LOSS.append(loss.item())
                # 打印训练进度
                sys.stdout.write(
                    "\r"
                    + "Epoch "
                    + str(e + 1)
                    + " / "
                    + str(MAX_EPOCH)
                    + " Subset "
                    + str(current_subset + 1)
                    + " / "
                    + str(dataset.nCacheSubset)
                    + " Progress "
                    + str(i + 1)
                    + " / "
                    + str(len(data_loader))
                    + " Loss "
                    + str(format(loss.item(), ".2f"))
                    + " time "
                    + str(format(time.time() - time1, ".2f"))
                    + " seconds."
                )

            # 保存模型权重
            torch.save(
                model.state_dict(),
                os.path.join(
                    os.path.dirname(__file__), f"checkpoints/SALSA/Model/model_{e}.pth"
                ),
            )
            torch.save(model.state_dict(), checkpoint_path)  # 保存最新的模型权重
            # 删除不再需要的变量并清理内存
            del (
                coord,
                xyz,
                feat,
                batch_number,
                labels,
                local_features,
                global_descriptor,
                point_pos_pairs,
            )
            gc.collect()
            torch.cuda.empty_cache()
            # 计算子缓存的平均损失
            cache_loss_avg = sum(CACHE_LOSS) / len(CACHE_LOSS) * steps_per_epoch

            # 记录子缓存损失和学习率到 TensorBoard
            writer.add_scalar("Subcache Loss", cache_loss_avg, kk_subcache)
            writer.add_scalar("Subcache LR", last_lr, kk_subcache)
            # 更新子缓存计数器
            kk_subcache += 1

            # 更新轮次损失列表
            EPOCH_LOSS.append(cache_loss_avg)
            # 打印子缓存的平均损失
            print(" ")
            print("Avg. Subcache Loss", cache_loss_avg)
        # 保存模型权重
        torch.save(
            model.state_dict(),
            os.path.join(
                os.path.dirname(__file__), f"checkpoints/SALSA/Model/model_{e}.pth"
            ),
        )
        torch.save(model.state_dict(), checkpoint_path)  # 保存最新的模型权重
        # 计算轮次的平均损失
        epoch_loss_avg = sum(EPOCH_LOSS) / len(EPOCH_LOSS)
        # 打印轮次的平均损失
        print(" ")
        print("Avg. EPOCH Loss", epoch_loss_avg)
        # 记录轮次损失到 TensorBoard
        writer.add_scalar("Epoch Loss", epoch_loss_avg, e)


if __name__ == "__main__":
    seed = 1100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    gc.collect()
    main()

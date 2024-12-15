import functools  # 导入functools模块，提供高阶函数用于操作或返回其他函数
import warnings  # 导入warnings模块，用于发出警告信息
from collections import OrderedDict  # 从collections模块导入OrderedDict类，保持字典的插入顺序

import numpy as np  # 导入numpy库，并简写为np，用于科学计算
import spconv.pytorch as spconv  # 导入spconv.pytorch库，并简写为spconv，用于稀疏卷积操作
import torch  # 导入PyTorch库
import torch.nn as nn  # 从PyTorch库中导入神经网络模块，并简写为nn
from spconv.core import ConvAlgo  # 从spconv.core模块导入ConvAlgo类
from spconv.pytorch.modules import SparseModule  # 从spconv.pytorch.modules模块导入SparseModule类
from torch_scatter import scatter_mean  # 从torch_scatter库中导入scatter_mean函数，用于张量的散射平均

from .spherical_transformer import SphereFormer  # 从当前包中导入spherical_transformer模块中的SphereFormer类


class ResidualBlock(SparseModule):  # 定义ResidualBlock类，继承自SparseModule
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法
        if in_channels == out_channels:  # 如果输入通道数等于输出通道数
            self.i_branch = spconv.SparseSequential(  # 定义恒等分支
                nn.Identity()  # 恒等层，不做任何改变
            )
        else:  # 如果输入通道数不等于输出通道数
            self.i_branch = spconv.SparseSequential(  # 定义卷积分支
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)  # 1x1卷积层
            )
        self.conv_branch = spconv.SparseSequential(  # 定义卷积分支
            norm_fn(in_channels),  # 归一化层
            nn.ReLU(),  # ReLU激活函数
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),  # 3x3卷积层
            norm_fn(out_channels),  # 归一化层
            nn.ReLU(),  # ReLU激活函数
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)  # 3x3卷积层
        )

    def forward(self, input):  # 前向传播方法
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)  # 创建稀疏卷积张量
        output = self.conv_branch(input)  # 通过卷积分支
        output = output.replace_feature(output.features + self.i_branch(identity).features)  # 将卷积分支的输出与恒等分支的输出相加
        return output  # 返回输出


class VGGBlock(SparseModule):  # 定义VGGBlock类，继承自SparseModule
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法
        self.conv_layers = spconv.SparseSequential(  # 定义卷积层序列
            norm_fn(in_channels),  # 归一化层
            nn.ReLU(),  # ReLU激活函数
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)  # 3x3卷积层
        )

    def forward(self, input):  # 前向传播方法
        return self.conv_layers(input)  # 返回卷积层序列的输出


def get_downsample_info(xyz, batch, indice_pairs):  # 定义获取下采样信息的函数
    pair_in, pair_out = indice_pairs[0], indice_pairs[1]  # 获取输入和输出索引对
    valid_mask = (pair_in != -1)  # 创建有效掩码，过滤掉无效索引
    valid_pair_in, valid_pair_out = pair_in[valid_mask].long(), pair_out[valid_mask].long()  # 获取有效的输入和输出索引
    xyz_next = scatter_mean(xyz[valid_pair_in], index=valid_pair_out, dim=0)  # 计算下采样后的坐标
    batch_next = scatter_mean(batch.float()[valid_pair_in], index=valid_pair_out, dim=0)  # 计算下采样后的批次
    return xyz_next, batch_next  # 返回下采样后的坐标和批次


class UBlock(nn.Module):  # 定义UBlock类，继承自nn.Module
    def __init__(self, nPlanes,  # 初始化方法，定义UBlock的各个参数
        norm_fn,  # 归一化函数
        block_reps,  # 块重复次数
        block,  # 块类型
        window_size,  # 窗口大小
        window_size_sphere,  # 球形窗口大小
        quant_size,  # 量化大小
        quant_size_sphere,  # 球形量化大小
        head_dim=16,  # 头部维度
        window_size_scale=[2.0, 2.0],  # 窗口大小缩放比例
        rel_query=True,  # 相对查询
        rel_key=True,  # 相对键
        rel_value=True,  # 相对值
        drop_path=0.0,  # 丢弃路径
        indice_key_id=1,  # 索引键ID
        grad_checkpoint_layers=[],  # 梯度检查点层
        sphere_layers=[1,2,3,4,5],  # 球形层
        a=0.05*0.25,  # 参数a
    ):

        super().__init__()  # 调用父类的初始化方法
        self.nPlanes = nPlanes  # 设置nPlanes属性
        self.indice_key_id = indice_key_id  # 设置indice_key_id属性
        self.grad_checkpoint_layers = grad_checkpoint_layers  # 设置grad_checkpoint_layers属性
        self.sphere_layers = sphere_layers  # 设置sphere_layers属性

        # 创建块的有序字典
        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) for i in range(block_reps)}
        blocks = OrderedDict(blocks)  # 转换为有序字典
        self.blocks = spconv.SparseSequential(blocks)  # 创建稀疏序列

        if indice_key_id in sphere_layers:  # 如果当前索引键ID在球形层中
            self.window_size = window_size  # 设置窗口大小
            self.window_size_sphere = window_size_sphere  # 设置球形窗口大小
            num_heads = nPlanes[0] // head_dim  # 计算头部数量
            self.transformer_block = SphereFormer(  # 创建SphereFormer块
                nPlanes[0],
                num_heads,
                window_size,
                window_size_sphere,
                quant_size,
                quant_size_sphere,
                indice_key='sphereformer{}'.format(indice_key_id),
                rel_query=rel_query,
                rel_key=rel_key,
                rel_value=rel_value,
                drop_path=drop_path[0],
                a=a,
            )

        if len(nPlanes) > 1:  # 如果nPlanes的长度大于1
            self.conv = spconv.SparseSequential(  # 创建稀疏卷积序列
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id), algo=ConvAlgo.Native)
            )

            # 计算下一个窗口大小和量化大小
            window_size_scale_cubic, window_size_scale_sphere = window_size_scale
            window_size_next = np.array([
                window_size[0]*window_size_scale_cubic,
                window_size[1]*window_size_scale_cubic,
                window_size[2]*window_size_scale_cubic
            ])
            quant_size_next = np.array([
                quant_size[0]*window_size_scale_cubic,
                quant_size[1]*window_size_scale_cubic,
                quant_size[2]*window_size_scale_cubic
            ])
            window_size_sphere_next = np.array([
                window_size_sphere[0]*window_size_scale_sphere,
                window_size_sphere[1]*window_size_scale_sphere,
                window_size_sphere[2]
            ])
            quant_size_sphere_next = np.array([
                quant_size_sphere[0]*window_size_scale_sphere,
                quant_size_sphere[1]*window_size_scale_sphere,
                quant_size_sphere[2]
            ])
            self.u = UBlock(nPlanes[1:],  # 创建下一个UBlock
                norm_fn,
                block_reps,
                block,
                window_size_next,
                window_size_sphere_next,
                quant_size_next,
                quant_size_sphere_next,
                window_size_scale=window_size_scale,
                rel_query=rel_query,
                rel_key=rel_key,
                rel_value=rel_value,
                drop_path=drop_path[1:],
                indice_key_id=indice_key_id+1,
                grad_checkpoint_layers=grad_checkpoint_layers,
                sphere_layers=sphere_layers,
                a=a
            )

            self.deconv = spconv.SparseSequential(  # 创建稀疏反卷积序列
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id), algo=ConvAlgo.Native)
            )

            blocks_tail = {}  # 创建尾部块的字典
            for i in range(block_reps):  # 遍历块重复次数
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)  # 转换为有序字典
            self.blocks_tail = spconv.SparseSequential(blocks_tail)  # 创建稀疏序列

    def forward(self, inp, xyz, batch):  # 前向传播方法

        assert (inp.indices[:, 0] == batch).all()  # 断言输入的索引与批次一致

        output = self.blocks(inp)  # 通过残差块
        # transformer
        if self.indice_key_id in self.sphere_layers:  # 如果当前索引键ID在球形层中
            if self.indice_key_id in self.grad_checkpoint_layers:  # 如果当前索引键ID在梯度检查点层中
                def run(feats_, xyz_, batch_):  # 定义运行函数
                    return self.transformer_block(feats_, xyz_, batch_)
                transformer_features = torch.utils.checkpoint.checkpoint(run, output.features, xyz, batch)  # 使用梯度检查点
            else:
                transformer_features = self.transformer_block(output.features, xyz, batch)  # 直接运行transformer块
            output = output.replace_feature(transformer_features)  # 替换输出特征
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)  # 创建稀疏卷积张量

        if len(self.nPlanes) > 1:  # 如果nPlanes的长度大于1
            # downsample
            output_decoder = self.conv(output)  # 通过卷积层
            indice_pairs = output_decoder.indice_dict['spconv{}'.format(self.indice_key_id)].indice_pairs  # 获取索引对
            xyz_next, batch_next = get_downsample_info(xyz, batch, indice_pairs)  # 获取下采样信息

            output_decoder = self.u(output_decoder, xyz_next, batch_next.long())  # 递归调用UBlock

            # upsample
            output_decoder = self.deconv(output_decoder)  # 通过反卷积层
            output = output.replace_feature(torch.cat((identity.features, output_decoder.features), dim=1))  # 拼接特征
            output = self.blocks_tail(output)  # 通过尾部块

        return output  # 返回输出


class Semantic(nn.Module):  # 定义Semantic类，继承自nn.Module
    def __init__(self,  # 初始化方法，定义Semantic的各个参数
        input_c,  # 输入通道数
        m,  # 中间通道数
        classes,  # 类别数
        block_reps,  # 块重复次数
        block_residual,  # 是否使用残差块
        layers,  # 层数
        window_size,  # 窗口大小
        window_size_sphere,  # 球形窗口大小
        quant_size,  # 量化大小
        quant_size_sphere,  # 球形量化大小
        rel_query=True,  # 相对查询
        rel_key=True,  # 相对键
        rel_value=True,  # 相对值
        drop_path_rate=0.0,  # 丢弃路径率
        window_size_scale=2.0,  # 窗口大小缩放比例
        grad_checkpoint_layers=[],  # 梯度检查点层
        sphere_layers=[1,2,3,4,5],  # 球形层
        a=0.05*0.25,  # 参数a
    ):
        super().__init__()  # 调用父类的初始化方法

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)  # 定义归一化函数

        if block_residual:  # 如果使用残差块
            block = ResidualBlock  # 块类型为ResidualBlock
        else:  # 否则
            block = VGGBlock  # 块类型为VGGBlock

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 7)]  # 计算丢弃路径率

        #### backbone
        self.input_conv = spconv.SparseSequential(  # 定义输入卷积层
            spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )
        self.unet = UBlock(layers,  # 定义UBlock
            norm_fn,
            block_reps,
            block,
            window_size,
            window_size_sphere,
            quant_size,
            quant_size_sphere,
            window_size_scale=window_size_scale,
            rel_query=rel_query,
            rel_key=rel_key,
            rel_value=rel_value,
            drop_path=dpr,
            indice_key_id=1,
            grad_checkpoint_layers=grad_checkpoint_layers,
            sphere_layers=sphere_layers,
            a=a,
        )

        self.output_layer = spconv.SparseSequential(  # 定义输出层
            norm_fn(m),
            nn.ReLU()
        )

        #### semantic segmentation
        self.linear = nn.Linear(m, classes)  # 定义线性层，用于语义分割

        self.apply(self.set_bn_init)  # 应用批量归一化初始化

    @staticmethod
    def set_bn_init(m):  # 定义批量归一化初始化方法
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:  # 如果是批量归一化层
            m.weight.data.fill_(1.0)  # 初始化权重
            m.bias.data.fill_(0.0)  # 初始化偏置

    def forward(self, input, xyz, batch):  # 前向传播方法
        '''
        :param input_map: (N), int, cuda
        '''

        output = self.input_conv(input)  # 通过输入卷积层
        output = self.unet(output, xyz, batch)  # 通过UBlock
        output = self.output_layer(output)  # 通过输出层

        #### semantic segmentation
        semantic_scores = self.linear(output.features)  # 计算语义分割得分
        return semantic_scores  # 返回语义分割得分
        return output.features  # 返回输出特征


import torch
from torch import nn
from torch.nn import functional as F
import kornia


from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
)

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    1.膨胀卷积阶段： 当 expand_ratio 大于 1 时，输入通道数量会被膨胀。张量形状从[B,C,H,W]到[B,C'=expand_ratio*C,H,W]
    2.深度卷积阶段:输出张量形状仍为 [B,C',H,W]。
    3.se:启用此模块,通道会被“压缩”到较小的尺寸,经过全局平均池化操作后,再通过1*1 卷积进行通道缩减和扩展。输入输出张量形状仍为 [B,C',H,W]。
    4.输出阶段:使用1*1 卷积将输出通道从 expand_ratio \times input_filters 转换为 output_filters.
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()



class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    1.输入张量为 [B,3,H,W] 2.Stem阶段:使用卷积层将输入图像从3个通道转换为更高维度的特征图,具体的输出通道数根据EfficientNet版本调整。
    2.blocks阶段:不同阶段的输出尺寸逐渐缩小,通道数逐渐增加.
    3.Head阶段:将最后一个Block的输出通道从Block的 output_filters 转换为1280.
    4.分类阶段:全局平均池化将特征图变为[B,1280,1,1],再展平为[B,1280],全连接层最终分类，形状为[B,num_classes].
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None,escape=''):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self.escape=escape
        self._global_params = global_params
        self._blocks_args = blocks_args
        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        #使卷积层的输出特征图的空间尺寸（宽度和高度）与输入特征图的尺寸保持一致。这在设计深度神经网络时非常有用，尤其是在像 EfficientNet 这样的模型中，网络层次很深，确保每层的特征图尺寸不会在不必要的情况下过快缩减有助于保持特征分辨率。
        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum#新旧均值方差占的比例
        bn_eps = self._global_params.batch_norm_epsilon#很小的正数

        # Stem
        in_channels = 3  # rgb
        #round_filters它确保每一层的通道数根据模型配置（EfficientNet-B0、B1等）进行合理的缩放
        out_channels = round_filters(32, self._global_params)  # 根据EfficientNet的全局参数，动态调整卷积层的输出通道数，以确保模型的计算量和参数量与目标模型如EfficientNet-B0、B1等）相匹配。
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])#使用 nn.ModuleList 初始化一个空列表
        self.stage_map=[]
        stage_count=0
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )#通常用于根据全局参数调整过滤器数量。
            stage_count+=1
            self.stage_map+=['']*(block_args.num_repeat - 1)#语法：创建一个包含 (block_args.num_repeat - 1) 个空字符串的列表。然后，+= 操作符将这个列表附加到 self.stage_map 中。
            #这行代码在 stage_map 中为即将要添加的模块（如果有重复的模块）预留空位。如果当前模块会重复多个次数，只有第一次的阶段标识会被添加，其他重复的部分用空字符串占位。
            self.stage_map.append('b%s'%stage_count)#语法：'b%s' % stage_count 是字符串格式化的用法，将 stage_count 的值嵌入字符串中。然后，append 方法将这个新字符串添加到 self.stage_map 的末尾。
            #作用：这行代码向 stage_map 添加当前模块的标识，形式为 b{stage_count}。这样可以清楚地记录每个模块在模型构建过程中的位置。
            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            #将新创建的 MBConvBlock 实例添加到 _blocks 列表中。构建了一个新的卷积块。
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        # self._fc = KAN([out_channels,1024, self._global_params.num_classes])
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):#激活函数
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)


    def extract_features(self, inputs,layers):
        """ Returns output of the final convolution layer """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))#conv_stem这个卷积操作用于提取输入图像的初始特征。
        layers['b0']=x#layers: 一个空字典，用于保存各个阶段的特征输出，键为层的名字（例如 b0, b1 等），值为对应的特征张量。
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            stage=self.stage_map[idx]#根据当前块的索引 idx 从 stage_map 中获取当前块的对应阶段标识。
            if stage:#检查当前阶段是否有名称（即是否定义在 stage_map 中）。#（阶段指EfficientNet的不同 网络层级，不同层级阶段处理不同尺寸的特征图）
                layers[stage]=x#将当前阶段的特征输出 x 存入 layers 字典中，键为该阶段的标识符 stage。
                if stage==self.escape:
                    return None
        # Head
        x = self._bn1(self._conv_head(x))
        x=self._swish(x)
        return x

    def forward(self, x):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = x.size(0)
        # print("x",x.shape)
        layers={}
        x = self.extract_features(x,layers)
        if x is None:
            return layers
        
        layers['final']=x
        x = self._avg_pooling(x)
        
        x = x.view(bs, -1)#view() 方法将张量 x 重新整形为一个二维的张量，-1代表自动确定该维度的大小。
        x1 =x
        x = self._dropout(x)
        # print(x.shape)
        x = self._fc(x)
        # print(x1.shape)
        layers['logits']=x
        # print(layers["b3"].shape)
        # for key, value in layers.items():
        #     print(f"{key}: {value.shape}")
        # b0: torch.Size([32, 48, 112, 112])
        # b1: torch.Size([32, 24, 112, 112])
        # b2: torch.Size([32, 32, 56, 56])
        # b3: torch.Size([32, 56, 28, 28])
        # b4: torch.Size([32, 112, 14, 14])
        # b5: torch.Size([32, 160, 14, 14])
        # b6: torch.Size([32, 272, 7, 7])
        # b7: torch.Size([32, 448, 7, 7])
        # final: torch.Size([32, 1792, 7, 7])
        # logits: torch.Size([32, 1792])
        return layers
        # return x1
    

    @classmethod
    def from_name(cls, model_name, override_params=None,escape=''):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params,escape)

    @classmethod
    def from_pretrained(cls, model_name, advprop=False, num_classes=1000, in_channels=3,escape=''):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes},escape=escape)
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size = model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model
    
    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """ 
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

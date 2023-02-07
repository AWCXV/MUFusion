import torch
import torch.nn.functional as F

EPSILON = 1e-10


# addition fusion strategy
def AVGFusion(tensor1, tensor2):
    return (tensor1 + tensor2)/2

def MAXFusion(tensor1, tensor2):
    return torch.max(tensor1,tensor2);

# attention fusion strategy, average based on weight maps
def L1Fusion(tensor1, tensor2):
    # avg, max, nuclear
    f_spatial = spatial_fusion(tensor1, tensor2)
    tensor_f = f_spatial
    return tensor_f
    
def SCFusion(tensor1,tensor2):
    f_spatial = spatial_fusion(tensor1, tensor2);
    f_channel = channel_fusion(tensor1, tensor2);
    a = 0;
    print("a="+str(a));
    tensor_f = a*f_spatial + (1-a)*f_channel;
    return tensor_f;
    
# 基于通道注意力的融合
def channel_fusion(tensor1, tensor2):
    # 全局池化
    shape = tensor1.size()
    # 计算通道注意力 得到的是每个通道对应的一个值(一个表征)
    global_p1 = channel_attention(tensor1)
    global_p2 = channel_attention(tensor2)

    # EPSILON 加上去是防止分母为零吧, 做 softmax 操作，算出两个权重向量, 即每个通道的占比。
    global_p_w1 = global_p1 / (global_p1+global_p2+EPSILON)
    global_p_w2 = global_p2 / (global_p1+global_p2+EPSILON)

    #把每个通道都填满算出来的 全局池化数字 ，这样就方便进行运算了。
    global_p_w1 = global_p_w1.repeat(1,1,shape[2],shape[3])
    global_p_w2 = global_p_w2.repeat(1,1,shape[2],shape[3])

    tensorf = global_p_w1 * tensor1 + global_p_w2 * tensor2

    return tensorf    

# 通道注意力
def channel_attention(tensor, pooling_type = 'avg'):
    # 全局池化
    shape = tensor.size()
    #池化层的核为整个图片的大小，则直接取均值了,生成一个1*1的。
    global_p = F.avg_pool2d(tensor,kernel_size=shape[2:])
    return global_p

def spatial_fusion(tensor1, tensor2, spatial_type='sum'):
    shape = tensor1.size()
    # calculate spatial attention
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)
    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    
    #print(spatial_w1);
    #print(spatial_w2);
    

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2
    #print(tensor_f);

    return tensor_f


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    if spatial_type is 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type is 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial





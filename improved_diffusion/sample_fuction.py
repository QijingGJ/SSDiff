import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial
import torch.nn.functional as F


class AttentionHook:
    def __init__(self, unet):
        # 将所有局部变量（包括默认参数）更新为类的实例属性
        self.__dict__.update(locals())

        self.model = unet
        # 定义一个列表，包含要从注意力模块选择中排除的模块名称
        remove_modules = ['mid']  # 可能还有其他模块名称，这里只给出了一个示例

        # 遍历unet模型的所有模块，选择具有'attn2'属性的模块，但排除名称中包含remove_modules列表中任何字符串的模块
        self.att_modules = {n: mod.attn2 for n, mod in unet.named_modules() if hasattr(mod, 'attn2')
                            and not any(rm in n for rm in remove_modules)}

        # 初始化一个字典，用于存储每个注意力模块的查询（query），初始值为None
        self.queries = {mod: None for mod in self.att_modules}  # queries是前向过程中获得

        # 遍历所有注意力模块，为每个模块注册一个前向钩子
        for name, b in self.att_modules.items():
            # 初始化模块的前向钩子列表（虽然这里直接覆盖了，但通常用于存储多个钩子）
            b._forward_hooks = OrderedDict()

            # 定义一个内部函数作为前向钩子
            def hook(mod, input, output, name):
                # 提取输入的第一个元素，通常是隐藏状态
                hidden_states = input[0]
                # 使用模块的to_q方法将隐藏状态转换为查询
                query = mod.to_q(hidden_states)
                # 计算内部维度和头维度
                inner_dim = query.shape[-1]
                head_dim = inner_dim // mod.heads
                # 获取批量大小
                batch_size = hidden_states.shape[0]
                # 重新塑形查询并转置，以便后续处理
                query = query.view(1, -1, mod.heads, head_dim).squeeze().transpose(1, 0)
                # 将查询存储在self.queries字典中
                self.queries[name] = query

                # 使用partial函数为钩子函数提供额外的name参数，并注册到模块上

            b.register_forward_hook(partial(hook, name=name))

            # def processor(mod, hidden_states, hooker=None):
            #     pass
            #
            # b._forward = partial(processor, mod=b, hooker=self)
            # # 定义一个处理器函数，它可能会调用模块的原始processor方法（但这里未完全展示）
            # def processor(hidden_states, mod=None, hooker=None, **kwargs):
            #     # 注意：这里的mod.processor调用可能不存在于所有模块中，需要确保attn2模块有此方法
            #     return mod.processor(mod, hidden_states, **kwargs)
            #
            #     # 使用partial函数为处理器函数提供额外的mod和hooker参数，并替换模块的forward方法
            #
            # b.forward = partial(processor, mod=b, hooker=self)

    def set_image_embeddings(self, cine_myo):
        # keys是image_embeddings直接映射的
        image_embeddings = self.model.lstm_model(cine_myo)
        # 遍历所有注意力模块，为每个模块计算键（key）
        self.keys = {key: mod.to_k(image_embeddings) for key, mod in self.att_modules.items()}
        # 重新塑形并转置键，以便与查询匹配
        self.keys = {
            key: self.keys[key].view(1, -1, mod.heads, self.keys[key].shape[-1] // mod.heads).squeeze().transpose(1, 0)
            for key, mod in self.att_modules.items()}

    def compute(self, l2_norm=False, grad_norm=False):
        # 初始化一个空列表，用于存储所有注意力分数
        attention_scores_list = []

        # 遍历所有注意力模块，计算注意力分数
        for mod_name, mod in self.att_modules.items():
            # 获取键和查询
            # mod_name是名字：'input_blocks.7.1.transformer_blocks.0'
            # mod是网络结构 CrossAttention
            key = self.keys[mod_name]  # 参数[8,17,32]
            query = self.queries[mod_name]  # [8,1024,32]

            # 如果需要梯度归一化，则对键和查询进行归一化
            if grad_norm:
                key_norm = key.norm(dim=-1, keepdim=True)
                key = key / (1e-4 + key_norm / key_norm.detach())
                query_norm = query.norm(dim=-1, keepdim=True)
                query = query / (1e-4 + query_norm / query_norm.detach())
                # 如果需要L2归一化，则对键和查询进行L2归一化
            elif l2_norm:
                key_norm = key.norm(dim=-1, keepdim=True)
                key = key / (1e-4 + key_norm / 10)
                query_norm = query.norm(dim=-1, keepdim=True)
                query = query / (1e-4 + query_norm / 10)

                # 使用torch.baddbmm计算批量矩阵乘法，并加上一个标量倍数的矩阵（这里是mod.scale）
            # key.transpose(-1, -2):[8,32,17]
            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1],
                            dtype=query.dtype, device=query.device),
                query, key.transpose(-1, -2), beta=0, alpha=mod.scale)  # (8,1024,17)
            # 转置注意力分数，以便后续处理（根据具体需求可能需要此步骤）
            attention_scores = attention_scores.permute(0, 2, 1)  # (8,17,1024)

            # 如果注意力分数的形状是64x64，则应用平均池化层进行下采样
            sq = int(attention_scores.shape[-1] ** .5)
            if sq == 32:
                rescale = nn.AvgPool2d(kernel_size=2, stride=2)
                # 重新塑形为四维张量并应用平均池化，然后展平最后两个维度
                attention_scores = rescale(attention_scores.reshape(mod.heads, -1, sq, sq)).flatten(-2)

                # 将注意力分数添加到列表中
            if attention_scores.shape[-1] == 256:
                attention_scores_list.append(attention_scores)

            # 将所有注意力分数拼接在一起并返回
        return torch.cat(attention_scores_list)


def bce_loss(att_maps, gts, ls=0.2, eps=1e-3):
    gts_c = gts.clip(ls, 1 - ls)
    # ls is for label smoothing
    # test for each map !
    att_maps = att_maps.squeeze(1)
    # avoid pred problem
    att_maps = (att_maps + eps) / (1 + 2 * eps)
    att_maps_norm = att_maps - att_maps.min(dim=-1, keepdim=True).values
    att_maps_norm = att_maps_norm / att_maps_norm.max(dim=-1, keepdim=True).values
    att_maps_norm = (att_maps_norm + eps) / (1 + 2 * eps)
    # disable gradient on very small values
    disable_grad = ((att_maps - gts).abs() < ls).detach().requires_grad_(False).half()
    att_maps2 = att_maps * (1 - disable_grad) + disable_grad * gts_c.half()
    bce_losses = torch.stack([nn.BCELoss(reduction='none')(mp, gt) - nn.BCELoss(reduction='none')(gt, gt)
                              for mp, gt in zip(att_maps2, gts_c)]).mean(0)

    # disable gradient on vsmall values
    disable_grad_norm = ((att_maps_norm - gts).abs() < ls).detach().requires_grad_(False).half()
    att_maps_norm2 = att_maps_norm * (1 - disable_grad_norm) + disable_grad_norm * gts_c.half()

    norm_losses = torch.stack([nn.BCELoss(reduction='none')(mp, gt) - nn.BCELoss(reduction='none')(gt, gt)
                               for mp, gt in zip(att_maps_norm2, gts_c)])
    # norm_losses = (norm_losses*(1-disable_grad_norm)).mean(0)
    return bce_losses + norm_losses


def cal_loss(model, img, t, cine, y, n_attention_maps, segmentation_mask):
    attn_hook = AttentionHook(model)
    out = model(img, t, cine=cine, y=None)

    # print(out.size())
    # noise = torch.randn([1, 1, 128, 128]).cuda()
    # loss_out = (noise - out[0]).mean()
    # gradient = torch.autograd.grad(outputs=loss, inputs=img)
    # print(gradient[0].size())

    attn_hook.set_image_embeddings(cine_myo=cine)
    attention_scores = attn_hook.compute()  # (80,17,256)
    attention_maps = attention_scores.softmax(1)  # (80,17,256)-->(10,1,256)

    attention_maps_for_loss = attention_maps.view(10, 8, 17, 256).sum(dim=1).permute(1, 0, 2)  # (17,10,256)
    selected_maps = []
    for j, amlist in enumerate(attention_maps_for_loss):
        # amlist:(10, 256)
        scores = [mp.float().quantile(0.99) * (1 - mp.float().quantile(1 - 0.99)) for mp in amlist]  # 10
        sel_idx = torch.stack(scores).argsort()[-n_attention_maps:]  # 选前5个得分高的map
        selected_maps.append(amlist[sel_idx].mean(0))  # 求平均
    attention_maps_for_loss = torch.stack(selected_maps)  # (17,256)

    # 分割掩码引导
    if len(segmentation_mask.shape) == 2:
        segmentation_mask = segmentation_mask[None]  # (1,128,128)
    if len(segmentation_mask.shape) == 4:
        segmentation_mask = segmentation_mask[0]
    idWords_inMask = {-1: 'others', 0: 'myo', 1: 'inf'}
    mask_gt = torch.stack([F.interpolate((segmentation_mask == i).float()[None], 16, mode='area')[0, 0]  # (3,16,16)
                           for i, _ in idWords_inMask.items()]).half()

    attention_maps_for_loss = attention_maps_for_loss.view(17, 16, 16).unsqueeze(1).repeat(1, 3, 1, 1)
    # (17, 3, 16, 16) // (1, 3, 16, 16) --> (1, 3, 16, 16)

    import torchvision.utils as vutils
    xx = attention_maps_for_loss[8:9, :, :, :]
    yy = mask_gt.float().unsqueeze(0)
    vutils.save_image(xx, fp='./sample_c2l/attention_maps_for_loss.jpg')
    vutils.save_image(yy, fp='./sample_c2l/mask_gt.jpg')
    xxx
    loss_maps = bce_loss(attention_maps_for_loss, mask_gt.float().unsqueeze(0))

    segment_weights = torch.ones(len(idWords_inMask)).cuda()
    loss = (segment_weights[:, None, None] * loss_maps.squeeze()).mean()  # + loss_out / 100
    return loss

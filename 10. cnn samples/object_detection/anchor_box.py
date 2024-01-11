import torch
from d2l import torch as d2l

# 锚框定义：
# 图像的高度为h，宽度为w。
# 以像素为中心，：缩放比为s ∈ (0, 1]，宽高比为r > 0
# 锚框的宽度和高度分别是h * s * math.sqrt(r)和 h * s / math.sqrt(r)
#
# 交并比：IoU - |A ∩ B|/ |A ∪ B|
# intersection over union


def multi_box_prior(data, sizes, ratios):
    """以每个像素为中心生成不同 size 的锚框"""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5

    # 在y轴上缩放步长
    steps_h = 1.0 / in_height
    # 在x轴上缩放步长
    steps_w = 1.0 / in_width

    # 生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w

    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成“boxes_per_pixel”个高和宽，之后用于创建锚框的四角坐标(xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width

    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))

    # 除以2来获得半高和半宽
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    # 每个中心点都将有“boxes_per_pixel”个锚框
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
    d2l.plt.show()


def box_iou(boxes1, boxes2):
    """计算两个锚框的交并比"""
    # lambda 后面的参数为 : 后面表达式的参数，表达式可以类比为函数
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)

    # [:, None, :2] 切片操作中的
    # : 表示对所有元素的全选
    # None 表示在第二个维度插入一个新的维度
    # :2 表示在第三个维度上选择前两个元素
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    #  clamp(min=0) 将小于零的值截断为零
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)

    # inters 是一个张量，假设它的形状是 (n, m, 2)，其中 n 表示矩形框的数量，m 表示每个矩形框的交集矩形的宽度和高度
    # [:, :, 0] 切片操作中的冒号 : 表示对所有元素的全选，而 0 表示选择在第三个维度上的索引为0的元素
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas/union_areas


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框 - 不断最接近的"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]

    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)

    # 对于每个锚框，分配的真实边界框的张量
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)

    # 根据阈值决定是否分配真实边框
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """锚框偏移量的转换"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset


def multi_box_target(anchors, labels):
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]

    for i in range(batch_size):
        label = labels[i, :, :]
        # 获取与真实框接近的锚框
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)

        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)

        # 使用真实边界框来标记锚框的类别
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]

        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)

    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)

    return bbox_offset, bbox_mask, class_labels


# 非极大值抑制：
# 对于一个预测边界框B，目标检测模型会计算每个类别的预测概率。
# 假设最大的预测概率为p，则该概率所对应的类别B即为预测的类别
# 将p称为预测边界框B的置信度（confidence）
def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox


def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    B = torch.argsort(scores, dim=-1, descending=True)
    # 保留预测边界框的指标
    keep = []
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1:
            break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)


def multi_box_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)


if __name__ == "__main__":
    print("展示像素级锚框：")
    # load img
    img = d2l.plt.imread('./imgs/dog_and_cat.jpg')
    h, w = img.shape[:2]

    # bboxes
    X = torch.rand(size=(1, 3, h, w))
    Y = multi_box_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    boxes = Y.reshape(h, w, 5, 4)

    d2l.set_figsize()
    bbox_scale = torch.tensor((w, h, w, h))
    fig = d2l.plt.imshow(img)
    show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
                ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
                 's=0.75, r=0.5'])

    print("\n锚框标签 - 在训练数据中标注锚框：")
    ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                                 [1, 0.55, 0.2, 0.9, 0.88]])
    anchors = torch.tensor([[0, 0.1, 0.2, 0.3],
                            [0.15, 0.2, 0.4, 0.4],
                            [0.63, 0.05, 0.88, 0.98],
                            [0.66, 0.45, 0.8, 0.8],
                            [0.57, 0.3, 0.92, 0.9]])
    fig = d2l.plt.imshow(img)
    show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
    show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])

    print("\n使用非极大值抑制预测边界框：")
    anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92],
                            [0.08, 0.2, 0.56, 0.95],
                            [0.15, 0.3, 0.62, 0.91],
                            [0.55, 0.2, 0.9, 0.88]])
    offset_preds = torch.tensor([0] * anchors.numel())
    cls_probs = torch.tensor([[0] * 4, # 背景的预测概率
                              [0.9, 0.8, 0.7, 0.1], # 狗的预测概率
                              [0.1, 0.2, 0.3, 0.9]]) # 猫的预测概率
    fig = d2l.plt.imshow(img)
    show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
    output = multi_box_detection(cls_probs.unsqueeze(dim=0),
                                offset_preds.unsqueeze(dim=0),
                                anchors.unsqueeze(dim=0),
                                nms_threshold=0.5)
    fig = d2l.plt.imshow(img)
    for i in output[0].detach().numpy():
        if i[0] == -1:
            continue
        label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
        show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)






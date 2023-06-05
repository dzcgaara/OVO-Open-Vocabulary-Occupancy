import torch 
import torch.nn.functional as F

def get_dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1) # |X⋂Y|
    b = torch.sum(input * input, 1) + 0.001  # |X|
    c = torch.sum(target * target, 1) + 0.001  # |Y|
    d = (2 * a) / (b + c)
    return 1-d
    
def compute_mask_matching_loss(mask_gts, mask_pred):
    # mask_gts shape should be [bs, num_gt, 30, 18, 30]
    # mask_pred shape should be [bs, num_query, 30, 18, 30]
    # print("in mask matching loss")
    total_loss = 0
    for i in range(mask_gts.shape[0]):
        # step 1. we should get dice_similirty
        for j in range(mask_gts.shape[1]):
            tmp_gt = mask_gts[i][j].unsqueeze(0)
            
            if torch.sum(tmp_gt) < 1:
                break
            
            num_pred = mask_pred.shape[1]
            
            tmp_gts = tmp_gt.repeat(num_pred, 1, 1, 1)
            tmp_preds = mask_pred[i]
            
            # both gt and pred shape should be [num_query, 30, 18, 30]
            dice_loss = get_dice_loss(tmp_preds, tmp_gts)
            total_loss += torch.min(dice_loss)

    return total_loss

def focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs
    
    # alpha = 1 - torch.sum(targets) / (30*18*30)
    alpha = -1
    # print(alpha, torch.sum(targets), (30*18*30))
    print("num pos vox is ", torch.sum(targets))

    ce_loss = torch.nn.BCELoss()(prob, targets)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
        
    return loss.sum() / (30*18*30)
    # print(loss.shape)
    # exit()

    # return loss.mean(1).sum() / inputs.shape[1]

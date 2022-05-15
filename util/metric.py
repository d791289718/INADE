import numpy as np
from loguru import logger

class Evaluator(object):
    def __init__(self, num_class, log_dir, has_unknown):
        self.has_unknown = has_unknown
        self.num_class = num_class
        logger.add(log_dir)
        self.total_imgs = 0
        self.confusion_matrix = np.zeros((self.num_class,) * 2)  # 混淆矩阵,行代表ground truth类别,列代表preds的类别

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        # tmp = self._generate_matrix(gt_image, pre_image)
        self.confusion_matrix += self._generate_matrix(gt_image.astype(int).flatten(), pre_image.astype(int).flatten())
        self.total_imgs += 1

    def log(self, epoch):
        logger.info("Checkpoints_{}: mIoU is {}".format(epoch, self.mIoU()))
        logger.info("Checkpoints_{}: FWIoU is {}".format(epoch, self.Frequency_Weighted_IOU()))
        logger.info("Checkpoints_{}: PA is {}".format(epoch, self.pixel_accuracy()))
        logger.info("Checkpoints_{}: MPA is {}".format(epoch, self.mean_pixel_accuracy()))
        logger.info("Checkpoints_{}: CPA is {}".format(epoch, self.class_pixel_accuracy()))
        logger.info("Checkpoints_{}: class_IOU is {}".format(epoch, self.class_IOU()))

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.total_imgs = 0

    # 计算混淆矩阵
    def _generate_matrix(self, gt_image, pre_image):
        nc = self.num_class + 1 if self.has_unknown else self.num_class
        mask = (gt_image >= 0) & (gt_image < nc) # ground truth中所有正确(值在[0, classe_num])的像素label的mask
        label = nc * gt_image[mask].astype('int') + pre_image[mask] # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数（也就是混淆矩阵啦），返回值形状(n, n)
        count = np.bincount(label, minlength=nc**2)
        if self.has_unknown:
            confusion_matrix = count.reshape(nc, nc)[:-1,:-1]
        else:
            confusion_matrix = count.reshape(nc, nc)
        return confusion_matrix

    def pixel_accuracy(self):
        # PA 所有gt像素中预测对的
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def class_pixel_accuracy(self):
        # CPA 每一类gt像素中预测对的
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return acc

    def mean_pixel_accuracy(self):
        # MPA CPA算平均
        acc = self.class_pixel_accuracy()
        acc = np.nanmean(acc)
        return acc

    def class_IOU(self):
        # 每个类别IOU
        IoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        return IoU

    def mIoU(self):
        # 所有类别IOU算mean
        MIoU = self.class_IOU()
        MIoU = np.nanmean(MIoU)  # 跳过nan值求mean
        return MIoU

    def Frequency_Weighted_IOU(self):
        # 算mIoU的基础上以gt像素出现的频率作为weight
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = self.class_IOU()
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

if __name__ == "__main__":
    metric = Evaluator(3, './', False)
    gt = np.array([[0,1,0],[2,1,0],[2,2,2]])
    pre = np.array([[0,2,0],[2,1,0],[1,2,1]])
    metric.add_batch(gt, pre)
    print('!')

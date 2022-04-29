import numpy as np
from loguru import logger

class Evaluator(object):
    def __init__(self, num_class, log_dir, has_unknown):
        self.has_unknown = has_unknown
        self.num_class = num_class
        logger.add(log_dir)
        self.confusion_matrix = np.zeros((self.num_class,) * 2)  # 21*21的矩阵,行代表ground truth类别,列代表preds的类别

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        # tmp = self._generate_matrix(gt_image, pre_image)
        self.confusion_matrix += self._generate_matrix(gt_image.astype(int), pre_image.astype(int))

    def log(self, epoch):
        logger.info("Checkpoints_{}: mIoU is {}".format(epoch, self.mIoU()))
        logger.info("Checkpoints_{}: Acc is {}".format(epoch, self.pixel_accuracy()))
        logger.info("Checkpoints_{}: class mIoU is {}".format(epoch, self.class_IOU()))
        logger.info("Checkpoints_{}: class Acc is {}".format(epoch, self.pixel_accuracy_class()))

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    # 计算混淆矩阵
    def _generate_matrix(self, gt_image, pre_image):
        # ground truth中所有正确(值在[0, classe_num])的像素label的mask
        nc = self.num_class + 1 if self.has_unknown else self.num_class
        mask = (gt_image >= 0) & (gt_image < nc)
        label = nc * pre_image[mask].astype('int') + gt_image[mask]
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(label, minlength=nc**2)
        if self.has_unknown:
            confusion_matrix = count.reshape(nc, nc)[:-1,:-1]
        else:
            confusion_matrix = count.reshape(nc, nc)
        return confusion_matrix

    def pixel_accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def pixel_accuracy_class(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        acc = np.nanmean(acc)
        return acc

    def mIoU(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)  # 跳过0值求mean
        return MIoU

    def class_IOU(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    '''
    参数的传入:
        evaluator = Evaluate(4)           #只需传入类别数4
        evaluator.add_batch(target, preb) #target:[batch_size, 512, 512]    ,    preb:[batch_size, 512, 512]
        在add_batch中统计这个epoch中所有图片的预测结果和ground truth的对应情况, 累计成confusion矩阵(便于之后求mean)


    参数列表对应:
        gt_image: target  图片的真实标签            [batch_size, 512, 512]
        per_image: preb   网络生成的图片的预测标签   [batch_size, 512, 512]

    parameters:
        mask: ground truth中所有正确(值在[0, classe_num])的像素label的mask---为了保证ground truth中的标签值都在合理的范围[0, 20]
        label: 为了计算混淆矩阵, 混淆矩阵中一共有num_class*num_class个数, 所以label中的数值也是在0与num_class**2之间. [batch_size, 512, 512]
        cout(reshape): 记录了每个类别对应的像素个数,行代表真实类别,列代表预测的类别,count矩阵中(x, y)位置的元素代表该张图片中真实类别为x,被预测为y的像素个数
        np.bincount: https://blog.csdn.net/xlinsist/article/details/51346523
        confusion_matrix: 对角线上的值的和代表分类正确的像素点个数(preb与target一致),对角线之外的其他值的和代表所有分类错误的像素的个数
    '''

if __name__ == "__main__":
    metric = Evaluator(15)

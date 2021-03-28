from collections import defaultdict
import torch
import numpy as np
import random
import math
import copy
import cv2
import torchvision.transforms as T
from lib.sincurlimg import SincurlImage
from lib.floodfill import FloodFill


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img



class RandomScaleCrop(object):

    def __init__(self, probability=0.6):
        self.probability = probability

    def __call__(self, img):
        if torch.rand(1) > self.probability:
            return img
        (h, w) = img.shape[0:2]
        h = int(h)
        w = int(w)
        e_size = [h * 1.1, w * 1.1]
        e_size = (int(e_size[1]), int(e_size[0]))
        img = cv2.resize(img, e_size)
        s_x = random.randint(0, e_size[0] - w)
        s_y = random.randint(0, e_size[1] - h)

        img = img[s_y:s_y + h, s_x:s_x + w, :]

        return img

class RandomPaddingCrop(object):
    def __init__(self, probability = 1.0):
        self.probability = probability
        self._tnum = 0
        pass

    def __call__(self, img):
        if torch.rand(1) > self.probability:
            return img
        (h, w) = img.shape[0:2]
        h = int(h)
        w = int(w)
        nh = h + 20
        nw = w + 20
        nimg = np.zeros((nh, nw, img.shape[2])).astype(img.dtype)
        nimg[10:h+10,10:w+10,:] = img
        s_x = random.randint(0, nw - w)
        s_y = random.randint(0, nh - h)
        img = nimg[s_y:s_y + h, s_x:s_x + w, :]

        # cv2.imwrite('/Users/sytm/Documents/Codes/python/cudalearn/sincurlimg/result-%d.jpg' % self._tnum, img)
        # self._tnum = self._tnum + 1

        return img

class ValTransform(object):
    def __init__(self, cfg):
        self.aimsize = cfg.INPUTSIZE
        self.flip = cv2.flip
        self.transforms = T.Compose([T.ToTensor(),
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __call__(self, imgs, pids):
        imgs = [cv2.resize(img, self.aimsize) for img in imgs]
        res_imgs = []
        for i in range(len(imgs)):
            if torch.rand(1) < 0.5:
                res_imgs.append(self.transforms(self.flip(imgs[i], 1)))
            else:
                res_imgs.append(self.transforms(imgs[i]))
        pids = torch.tensor(pids, dtype=torch.int64)
        res_imgs = torch.stack(res_imgs, dim=0)
        return res_imgs, pids


class MultiTransform(object):

    def __init__(self, cfg):
        self.aimsize = cfg.INPUTSIZE
        self.flip = cv2.flip
        self.toTensor = T.ToTensor()
        self.normal = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.randomErasing = RandomErasing(probability=0.5, sh=0.4, mean=(0.4914, 0.4822, 0.4465))
        self.sinCurlImage = SincurlImage(probability = 1)
        # added on 2020.11.19
        self.floodfill = FloodFill()
        self.randomPaddingCrop = RandomPaddingCrop(probability = 0.5)
        self.make_zero = cfg.make_zero
        self.make_first = cfg.make_first
        self.make_sincurl_pic = cfg.make_sincurl_pic
        self.make_floodfill_pic = cfg.make_floodfill_pic
        self.flood_rand_img_nums = 2
        self.sincurl_pro = 0.8
        self.epochs = cfg.MAX_EPOCHS
        self.warm_epochs = cfg.WARMUP_EPOCHS


    def __call__(self, imgs, pids, epoch=0):
        '''

        :param imgs: PIL images in list
        :param pids: pid, every pid has four images
        :return: imgs * 3, with type of 3d Tensor
        '''

        imgs0 = [cv2.resize(img, self.aimsize) for img in imgs]

        if self.make_first:
            imgs1 = copy.deepcopy(imgs0)

        if self.make_sincurl_pic:
            imgs2 = copy.deepcopy(imgs0)

        if self.make_floodfill_pic:
            imgs3 = copy.deepcopy(imgs0)
            pid_dic = defaultdict(list)

            # deal imgs3 by gatter color from imgs2
            distinct_pids_set = set()
            for index, pid in enumerate(pids):
                pid_dic[pid].append(index)  # every pid has four index
                distinct_pids_set.add(pid)

            distinct_pids = list(distinct_pids_set)
            shuffle_pids = copy.deepcopy(distinct_pids)
            random.shuffle(shuffle_pids)

            for i in range(len(shuffle_pids)):
                pid = shuffle_pids[i]
                indexs = copy.deepcopy(pid_dic[pid])
                random.shuffle(indexs)
                for j in range(len(indexs)):
                    img_id1 = indexs[j]
                    # random_imgids = [pid_dic[shuffle_pids[self.getRandom(shuffle_pids,pid)]][j]  for _ in range(1)]
                    random_imgids = [pid_dic[shuffle_pids[self.getRandom(shuffle_pids, pid)]][0] for _ in range(1)]
                    imgs3[img_id1]=self.floodfill(imgs3[img_id1], imgs0, random_imgids)

        ans = []
        target = []
        if self.make_zero:
            imgs0 = self.deal_zero(imgs0)
            ans.extend(imgs0)
            target.extend(pids)

        if self.make_first:
            imgs1 = self.deal_first(imgs1)
            ans.extend(imgs1)
            target.extend(pids)

        if self.make_sincurl_pic:
            imgs2 = self.deal_second(imgs2)
            ans.extend(imgs2)
            target.extend(pids)

        if self.make_floodfill_pic:
            imgs3 = self.deal_third(imgs3)
            ans.extend(imgs3)
            target.extend(pids)

        target = torch.tensor(target, dtype=torch.int64)
        ans = torch.stack(ans, dim=0)
        return ans, target

    def getRandImgNums(self, epoch = 0):
        if epoch <= self.warm_epochs:
            return self.flood_rand_img_nums
        epoch = epoch - self.warm_epochs
        total_epochs = self.epochs - self.warm_epochs
        ans = int(math.ceil((total_epochs - epoch)/total_epochs * self.flood_rand_img_nums))
        return max(ans,2)

    def getSincurlProb(self, epoch = 0):
        if epoch <= self.warm_epochs:
            return self.sincurl_pro
        epoch = epoch - self.warm_epochs
        total_epochs = self.epochs - self.warm_epochs
        return self.sincurl_pro - epoch / total_epochs * (self.sincurl_pro/2)
    def getRandom(self,pids, pid):
        '''
        :param pids: all pids in list
        :param pid: the excluded pid
        :return:
        '''
        rv = random.randint(0,len(pids)-2)
        if pids[rv] == pid:
            return rv+1
        return rv

    def deal_zero(self, imgs):
        res_imgs = []
        # use the undealed images
        for i in range(len(imgs)):
            res_imgs.append(self.normal(self.toTensor(self.flipImg(imgs[i]))))
        return res_imgs

    def deal_first(self, imgs):
        res_imgs = []
        for i in range(len(imgs)):
            res_imgs.append(
                self.randomErasing(self.normal(self.toTensor(self.randomPaddingCrop(self.flipImg(imgs[i]))))))
        return res_imgs

    def deal_second(self, imgs):
        res_imgs = []
        for i in range(len(imgs)):
            res_imgs.append(
                self.randomErasing(self.normal(self.toTensor(self.randomPaddingCrop(self.sinCurlImage(self.flipImg(imgs[i])))))))
        return res_imgs

    def deal_third(self, imgs):
        res_imgs = []
        for i in range(len(imgs)):
            res_imgs.append(self.normal(self.toTensor(self.flipImg(imgs[i]))))
        return res_imgs

    def flipImg(self,img):
        if torch.rand(1) > 0.5:
            return self.flip(img, 1)
        return img
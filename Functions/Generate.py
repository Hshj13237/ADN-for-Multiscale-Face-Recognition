import cv2
import numpy as np
import random

class ImageGenerator:
    def __init__(self):
        self.bgRoot = '../Use_Background/'
        self.fgRoot = '../Use_Foreground/'
        self.maskfile = '../Use_Foreground/mask.npy'
        self.trainingSet = [[] for _ in range(10)]
        self.testSet = [[] for _ in range(10)]
        self.makeDataSet()


    # randomize to get the background
    def getBackground(self):
        img_index = random.randint(1,13)
        image_path = self.bgRoot + str(img_index) +'.tiff'
        image = cv2.imread(image_path)
        height, width,_ = image.shape
        x = random.randint(0, height-38)
        y = random.randint(0, width-38)
        background = image[x:x+38, y: y+38,:]
        return background

    def getForeground(self, t1, t2):
        fgpath = self.fgRoot + str(t1) + '/' + str(t2) + '.jpg'
        fg = cv2.imread(fgpath)
        return fg



    def getMask(self, t1, t2):
        mat = np.load(self.maskfile)
        mask = mat[t1][t2]
        mask = mask.reshape(112,92)
        return mask


    def generate(self, fg_size, fg, mask):
        '''
        :param fg: 112 x 92
        :param bg: 38 x 38
        :param mask: 112 x 92
        :return: 350p x 38x38
        '''

        dim = 38-fg_size
        result = np.zeros((dim-1, dim-1, 38*38))

        for i in range(1, dim):
            for j in range(1, dim):
                fg = cv2.resize(fg, dsize=(fg_size, fg_size))
                bg = self.getBackground()
                mask = cv2.resize(mask, dsize=(fg_size, fg_size))
                mask[mask>0]=1
                mask_reverse = mask - [1]
                mask_reverse[mask_reverse<0] = 1
                mask_reverse = cv2.resize(mask_reverse, dsize=(fg_size, fg_size))
                width, height, channels = fg.shape
                output = bg[:,:,0]
                fg_temp = fg[:, :, 0]
                # output[i:i + height, j:j + width] = cv2.bitwise_or(
                #     cv2.bitwise_and(output[i:i + height, j:j + width], mask_reverse),
                #     cv2.bitwise_and(fg, mask))

                output[i:i + height, j:j + width] = output[i:i + height, j:j + width] * mask_reverse+fg_temp*mask

                # cv2.imwrite(r'C:\Users\Administrator\Desktop\Program_diff_Scales\ds\\'+str(fg_size)+'_'+str(i)+'_'+str(j)+'.jpg', output)

                result[i-1][j-1] = output.reshape(1, -1)
        return result

    def makeDataSet(self):
        for i in range(10):
            for j in range(4):
                result = {}
                idx = [x+1 for x in range(4)]
                random.shuffle(idx)
                fg = self.getForeground(i+1, idx[j])
                mask = self.getMask(i, j)

                # size_24 = self.generate(24, fg, mask)
                # size_29 = self.generate(29, fg, mask)
                # size_34 = self.generate(34, fg, mask)
                # result['size_24'] = size_24
                # result['size_29'] = size_29
                # result['size_34'] = size_34

                # for t in range(24, 35):
                for t in range(24, 33):
                    if t % 2:
                        continue
                    temp_var = 'size_'+str(t)
                    temp = self.generate(t, fg, mask)
                    result[temp_var] = temp
                if j < 3:
                    self.trainingSet[i].append(result)
                else:
                    self.testSet[i].append(result)

    def generateDataSet(self):
        return self.trainingSet, self.testSet


# def preprocess(x):
#     x = x - np.mean(x)
#     max_val = np.max(x)
#     min_val = np.min(x)
#     if (max_val - min_val) != 0:
#         x = (x - min_val) / (max_val - min_val)
#     return x
#
#
# test = ImageGenerator()
# a, b = test.generateDataSet()
# np.save(r'C:\Users\Administrator\Desktop\Program_diff_Scales\ds\training_set.npy',a)
# np.save(r'C:\Users\Administrator\Desktop\Program_diff_Scales\ds\test_set.npy',b)
# image = a[0][0]['size_24'][0][0].reshape(38,38)
# dl = DataLoader.DataLoader(a, False)
#
# while True:
#     a = next(dl)
#     if len(a)> 0:
#         print(a)
# image = preprocess(image)
# cv2.imshow("test", image)
# cv2.waitKey()



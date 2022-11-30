import numpy as np
import random

class DataLoader:
    def __init__(self, dataTuple,shuflle_flag):
        self.dataTuple = dataTuple
        self.dataList = self.get_Info()
        self.seq = [x for x in range(len(self.dataList))]
        if shuflle_flag:
            random.shuffle(self.seq)
        self.cursor = 0

    def get_Info(self):
        result = []
        len1 = len(self.dataTuple)
        for i in range(len1):
            len2 = len(self.dataTuple[i])
            for j in range(len2):

                for key in self.dataTuple[i][j]:
                    height, width, channel = self.dataTuple[i][j][key].shape

                    for l in range(height):
                        for m in range(width):
                            resultTuple = {}
                            resultTuple['image'] = self.dataTuple[i][j][key][l][m]
                            size = int(key.split('_')[-1])
                            resultTuple['lable'] = [i, size, l*15+m, [l, m , l+size,m+size]] ###################
                            result.append(resultTuple)
        return result

    def __iter__(self):
        return self

    def __next__(self):
        if self.cursor < len(self.dataList):
            result = self.dataList[self.seq[self.cursor]]
            self.cursor += 1
            return result
        else:
            self.cursor = 0
            return []







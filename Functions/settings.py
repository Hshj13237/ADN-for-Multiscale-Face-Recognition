# coding: utf-8
import DN
import numpy as np
import cv2
import random
import os
import DataLoader
import Generate


def log(fe,ft,image_name, type, location, mv, mi, flag):
    if flag:  # training stage
        log_text = image_name + ' ' + str(type) + ' ' + str(location) + ' ' + str(mv) + ' ' + str(mi)
        print(log_text, file=ft)
    else:  # training stage
        log_text = image_name + ' ' + str(type) + ' ' + str(location) + ' ' + str(mv) + ' ' + str(mi)
        print(log_text, file=fe)


def openfile(logfile):
    try:
        fe = open(logfile + '/test_log.txt', "a")
    except IOError:
        raise ("open error")

    try:
        ft = open(logfile + '/train_log.txt', "a")
    except IOError:
        raise ("open error")
    try:
        fa = open(logfile + '/result.txt', "a")
    except IOError:
        raise ("open error")
    return fe, ft, fa


logfile = '../Log/'
saverFolder = '../Saver/'


input_dim = [38, 38]
z_neuron_num = [10, 11, 225]
y_neuron_num = 10000

y_top_k = 1

dataSet = Generate.ImageGenerator()
train_set, test_set = dataSet.generateDataSet()

dn = DN.DN(input_dim, y_neuron_num, y_top_k, z_neuron_num, logfile, True, True)
training_num = 20
dl_train = DataLoader.DataLoader(train_set, False)
dl_test = DataLoader.DataLoader(test_set, False)

for i in range(training_num):
    saver = saverFolder + 'N2000_r_' + str(i)+'/'
    os.mkdir(saver)
    logger = saver + 'log'
    os.mkdir(logger)
    fe,ft,fa = openfile(logger)
    count = 0
    while True:
        data = next(dl_train) # 获取训练样本和样本的label
        if len(data) == 0:
            break
        image = data['image']
        image = image.reshape(1,-1)
        image = image.reshape(38, 38)
        label = data['lable']
        type = label[0]
        size = label[1]-24
        location = label[2]
        where_loc = label[3]

        count += 1

        true_z = [type, size, location]
        true_z = np.array(true_z)
        mv, mi = dn.dn_learn(image, true_z, where_loc)
        if count%10000 ==0:
            print(count)

        # log(fe,ft,imgpath, classnum, location, mv, mi, True)

        dn.dn_learn(image, true_z, where_loc)
    error = np.zeros((1,3))

    total_num = 0
    location_error = 0
    size_error_a = 0
    size_error_r = 0
    while True:
        # error counter

        data = next(dl_test)  # 获取训练样本和样本的label
        if len(data) == 0:
            break
        image = data['image']
        image = image.reshape(1, -1)
        image = image.reshape(38, 38)
        label = data['lable']
        type = label[0]
        size = label[1]-24
        location = label[2]
        r = location // 15
        c = location % 15
        where_loc = label[3]

        true_z = [type, size, location]
        true_z = np.array(true_z)
        z_out, mv, mi = dn.dn_test(image)
        #                 print(z_output, true_z)
        error = error + (z_out != true_z)
        if z_out[0] != true_z[0]:
            size_error_a += abs(z_out[1]-true_z[1])
            size_error_r += abs(z_out[1] - true_z[1])/(true_z[1]+24)
            tr = z_out[2] // 15
            tc = z_out[2] % 15
            location_error += ((tr - r)**2+(tc - c)**2)**0.5

        total_num += 1

        # if any(z_output!= true_z):
            # print(imgpath,z_output)
    print(str(i) + " training, current performance: " + str(1.0 - error / total_num)+ '  '+ str(location_error) + ' ' + str(size_error_a)+ ' '+ str(size_error_r))
    print(
        str(i) + " training, current performance: " + str(1.0 - error / total_num) + '  ' + str(
            location_error) + ' ' + str(size_error_a) + ' ' + str(size_error_r), file= fa)
    dn.dn_save(saver)

    fe.close()
    ft.close()
    fa.close()




import numpy as np
import logging
import time


def preprocess(x):
    x = x - np.mean(x)
    max_val = np.max(x)
    min_val = np.min(x)
    if (max_val - min_val) != 0:
        x = (x - min_val) / (max_val - min_val)
    return x

class DN:
    def __init__(self, input_dim, y_neuron_num, y_top_k, z_neuron_num, log_file, attention_flag, synapse_flag):
        self.x_neuron_num = 1
        self.input_dim = input_dim
        input_dim = np.array([input_dim]).reshape(-1)  # reshape(-1)�������У��ĳ�һ��
        for item in input_dim:
            self.x_neuron_num = self.x_neuron_num * item
        self.y_neuron_num = y_neuron_num
        self.z_area_num = len(z_neuron_num)
        self.z_neuron_num = z_neuron_num
        self.y_top_k = y_top_k
        self.log_t, self.log_d = self.dn_log(log_file)  # set a log file
        self.attention_flag = attention_flag
        self.synapse_flag = synapse_flag
        self.dn_create()


    def dn_create(self):
        # some super parameters
        self.y_bottom_up_percent = 1 / 2
        self.y_top_down_percent = 1 / 2

        # responses
        self.x_response = np.zeros((1, self.x_neuron_num))

        self.y_bottom_up_response = np.zeros((1, self.y_neuron_num))
        self.y_top_down_response = np.zeros((self.z_area_num, self.y_neuron_num))
        self.y_pre_response = np.zeros((1, self.y_neuron_num))

        self.y_response = np.zeros((1, self.y_neuron_num))
        self.z_response = []
        for i in range(self.z_area_num):
            self.z_response.append(np.zeros((1, self.z_neuron_num[i])))

        # weights
        self.y_bottom_up_weight = np.random.random_sample((self.y_neuron_num, self.x_neuron_num))
        self.y_top_down_weight = []
        for i in range(self.z_area_num):
            self.y_top_down_weight.append(np.random.random_sample((self.y_neuron_num, self.z_neuron_num[i])))

        # attention mask
        self.bottom_up_mask = np.ones(self.y_bottom_up_weight.shape)

        # age and flag
        self.y_lsn_flag = np.zeros((1, self.y_neuron_num))
        self.y_firing_age = np.zeros((1, self.y_neuron_num))
        self.set_flag = np.zeros((1, self.y_neuron_num))

        # z weights
        self.z_bottom_up_weight = []
        self.z_firing_age = []
        for i in range(self.z_area_num):
            self.z_bottom_up_weight.append(np.zeros((self.y_neuron_num, self.z_neuron_num[i])))  # ????
            self.z_firing_age.append(np.zeros((1, self.z_neuron_num[i])))
        # synapse trimming
        self.synapse_coefficient = [0.8, 1.2]
        self.synapse_age = 20
        self.bottom_up_diff = np.zeros(self.y_bottom_up_weight.shape)
        self.bottom_up_factor = np.ones(self.y_bottom_up_weight.shape)

        # inhibit
        self.inhibit_age =np.zeros((1,self.y_neuron_num))

        # firing the initial neuron
        self.y_threshold = np.zeros((1, self.y_neuron_num))

        self.log_t.info('DN creat complete')

    def preprocess(self, x):
        x = x - np.mean(x)
        max_val = np.max(x)
        min_val = np.min(x)
        if (max_val - min_val) != 0:
            x = (x - min_val) / (max_val - min_val)
        return x

    def compute_response(self,input_vec, weight_vec, rec_field_mask, synapse_factor):
        neuron_num, input_dim = weight_vec.shape  # Y*X
        temp = np.tile(input_vec, (neuron_num, 1))
        temp = temp * synapse_factor * rec_field_mask
        temp = self.normalize(temp, rec_field_mask)
        weight_vec = weight_vec * synapse_factor * rec_field_mask
        weight_vec = self.normalize(weight_vec, rec_field_mask)
        # for i in range(neuron_num):
        #     result[0][i] = np.dot(temp[i].reshape(1, -1), weight_vec[i].reshape(-1, 1))[0, 0]
        result = np.sum(temp * weight_vec, axis=1)
        return result  # 1 x 25

    def get_learning_rate(self, firing_age):
        lr = 1.0 / (firing_age + 1.0)
        if lr < 1.0 / 50.0:
            lr = 1.0 / 50.0
        return lr

    def mean(self, input_vec):
        input_vec = input_vec.reshape(1, -1)
        _, lenth = input_vec.shape
        use_lenth = 0
        mean = 0
        for i in range(lenth):
            if input_vec[0][i] > 0:
                use_lenth += 1
                mean += input_vec[0][i]

        if use_lenth == 0:
            use_lenth = 1
        return mean / use_lenth

    def normalize(self, input, mask):
        _, input_dim = input.shape
        input_normed = input * mask
        norm = np.sqrt(np.sum(input_normed * input_normed, axis=1))
        norm[norm==0] = 1
        result = input_normed / np.tile(norm.reshape(-1, 1), (1, input_dim))
        return result

    def dn_learn(self, training_image, true_z, where_loc):
        self.x_response = training_image.reshape(1, -1)  # x_response��������ͼ������һ��
        for i in range(self.z_area_num):  # z_area_num: location and type��һ��2��
            self.z_response[i] = np.zeros(self.z_response[i].shape)
            self.z_response[i][0, true_z[i]] = 1
        self.x_response = preprocess(self.x_response)
        if self.attention_flag:
            whereID = true_z[-1]
        else:
            whereID = -1
        tempmask = self.getMask(where_loc)

        # compute response
        self.y_bottom_up_response = self.compute_response(self.x_response,
                                                          self.y_bottom_up_weight,
                                                          self.bottom_up_mask,
                                                          self.bottom_up_factor)

        for i in range(self.z_area_num):
            self.y_top_down_response[i] = self.compute_response(self.z_response[i],
                                                                self.y_top_down_weight[i],
                                                                np.ones(self.y_top_down_weight[i].shape),
                                                                np.ones(self.y_top_down_weight[i].shape))
        # top-down + bottom-up response
        self.y_pre_response = (self.y_bottom_up_percent * self.y_bottom_up_response +
                               self.y_top_down_percent * np.mean(self.y_top_down_response, axis=0).reshape(1,
                                                                                                           -1)) / (
                                      self.y_bottom_up_percent + self.y_top_down_percent)  # mean or with weight

        max_response, max_index = self.top_k_competition(True)

        self.log_d.info(str(true_z)+' ' + str(max_response) + ' ' + str(max_index))
        self.setMask(where_loc, max_index)
        self.hebbian_learning(tempmask)
        self.updateInhibit(max_index, max_response)
        return max_response, max_index


    def updateInhibit(self, max_index, max_response):
        # inhibit  Y response
        lr = self.get_learning_rate(self.y_firing_age[0][max_index]-1)
        self.y_threshold[0][max_index] = lr * max_response + (1-lr)*self.y_threshold[0][max_index]





    def synapse_maintainence(self, synapse_diff, synapse_factor):
        current_diff = synapse_diff
        mean_diff = np.mean(current_diff)
        lower_thresh = self.synapse_coefficient[0] * mean_diff
        upper_thresh = self.synapse_coefficient[1] * mean_diff

        synapse_factor[synapse_diff > upper_thresh] = 0
        synapse_factor[synapse_diff < lower_thresh] = 1

        synapse_factor[((synapse_diff <= upper_thresh) * (synapse_diff >= lower_thresh)) == 1] = ((
                                                                                                          synapse_diff[
                                                                                                              ((
                                                                                                                       synapse_diff <= upper_thresh) * (
                                                                                                                       synapse_diff >= lower_thresh) == 1)]
                                                                                                          - upper_thresh) / (
                                                                                                          lower_thresh - upper_thresh))
        return synapse_factor


    def top_k_competition(self, flag):

        if flag:
            self.y_response = np.zeros(self.y_response.shape)
            # response_output = np.zeros(self.y_response.shape)
            seq_high = np.argsort(-self.y_pre_response)
            # for i in range(self.y_top_k):
            #     response_output[0, seq_high[i]] = 1.0
            max_response = self.y_pre_response[0][seq_high[0][0]]
            if max_response > self.y_threshold[0][seq_high[0][0]]:
                self.y_response[0][seq_high[0][0]] = 1
                return max_response, seq_high[0][0]

            if self.set_flag[0][seq_high[0][0]] < 1:
                self.y_response[0][seq_high[0][0]] = 1
                return max_response, seq_high[0][0]
            else:
                _, lenth = seq_high.shape
                for i in range(lenth):
                    if self.set_flag[0][seq_high[0][i]] <1:
                        self.y_response[0][seq_high[0][i]] = 1
                        return max_response, seq_high[0][i]
            self.y_response[0][seq_high[0][0]] = 1
            return max_response, seq_high[0][0]
        else:
            self.y_response = np.zeros(self.y_response.shape)
            # response_output = np.zeros(self.y_response.shape)
            seq_high = np.argsort(-self.y_pre_response)
            # for i in range(self.y_top_k):
            #     response_output[0, seq_high[i]] = 1.0
            max_response = self.y_pre_response[0][seq_high[0][self.y_top_k - 1]]
            self.y_response[0][seq_high[0][self.y_top_k - 1]] = 1
            return max_response, seq_high[0][0]

    def getMask(self, where_loc):

        result = np.zeros((self.input_dim[0], self.input_dim[1]))
        result[where_loc[0]:where_loc[2],where_loc[1]:where_loc[3]] = 1
        return result.reshape(1, -1)

    def setMask(self, where_loc, i):
        if self.set_flag[0][i] == 0:

            result = np.zeros((self.input_dim[0], self.input_dim[1]))
            result[where_loc[0]:where_loc[2],where_loc[1]:where_loc[3]] = 1

            self.bottom_up_mask[i] = result.reshape(1, -1)
            self.set_flag[0][i] = 1

    def hebbian_learning(self, mask):
        for i in range(self.y_neuron_num):
            if self.y_response[0, i] == 1:  # firing neuron, currently set response to 1
                if self.y_lsn_flag[0, i] == 0:
                    self.y_lsn_flag[0, i] = 1
                    self.y_firing_age[0, i] = 0
                lr = self.get_learning_rate(self.y_firing_age[0, i])  # learning rate
                # self.y_bottom_up_weight[i] = normalize(self.y_bottom_up_weight[i], mask)

                self.y_bottom_up_weight[i] = (1 - lr) * self.y_bottom_up_weight[i] + lr * self.x_response# ����Ȩ��
                if self.synapse_flag:
                    self.bottom_up_diff[i] = (1 - lr) * self.bottom_up_diff[i] + \
                                             lr*(np.abs(self.y_bottom_up_weight[i] -
                                                       self.x_response))
                    if self.y_firing_age[0, i] > self.synapse_age:
                        self.bottom_up_factor[i] = self.synapse_maintainence(self.bottom_up_diff[i], self.bottom_up_factor[i])

                # top-down weight and synapse factor
                for j in range(self.z_area_num):
                    # self.z_response[j] = self.normalize(self.z_response[j], np.ones(self.z_response[j].shape))
                    self.y_top_down_weight[j][i] = (1 - lr) * self.y_top_down_weight[j][i] + lr * self.z_response[
                        j]
                self.y_firing_age[0, i] = self.y_firing_age[0, i] + 1



        # z neuron learning
        for area_idx in range(self.z_area_num):
            for i in range(self.z_neuron_num[area_idx]):
                if self.z_response[area_idx][0, i] == 1:
                    lr = self.get_learning_rate(self.z_firing_age[area_idx][0, i])
                    self.z_bottom_up_weight[area_idx][:, i] = (1 - lr) * self.z_bottom_up_weight[area_idx][:,
                                                                         i] + lr * self.y_response.reshape(-1)

                    self.z_firing_age[area_idx][0, i] = self.z_firing_age[area_idx][0, i] + 1

    def dn_save(self, save_folder):
        # y_bottom_up weight:ybu.npy
        ybu = save_folder + 'ybu.npy'
        np.save(ybu, self.y_bottom_up_weight)
        # y_top_down weight:ytd.npy
        # ytd = save_folder + 'ytd.npy'
        # np.save(ytd, self.y_top_down_weight)
        # y bottom up mask:bum.npy
        bum = save_folder + 'bum.npy'
        np.save(bum, self.bottom_up_mask)
        # y lsn flag: ylf.npy
        ylf = save_folder + 'ylf.npy'
        np.save(ylf, self.y_lsn_flag)
        # y set flag: ysf.npy
        ysf = save_folder + 'ysf.npy'
        np.save(ysf, self.set_flag)
        # y firing ege: yfa.npy
        yfa = save_folder + 'yfa.npy'
        np.save(yfa, self.y_firing_age)
        # z bottom up weight: zbu.npy
        # zbu = save_folder + 'zbu.npy'
        # np.save(zbu, self.z_bottom_up_weight)
        # # z firing age: zfa.npy
        # zfa = save_folder + 'zfa.npy'
        # np.save(zfa, self.z_firing_age)
        bus = save_folder + 'bus.npy'
        np.save(bus, self.bottom_up_factor)

        ####
        print(np.sum(self.set_flag))


    def dn_set(self, set_folder):
        # y_bottom_up weight:ybu.npy
        ybu = set_folder + 'ybu.npy'
        self.y_bottom_up_weight = np.load(ybu)
        # y_top_down weight:ytd.npy
        ytd = set_folder + 'ytd.npy'
        self.y_top_down_weight = np.load(ytd)
        # y bottom up mask:bum.npy
        bum = set_folder + 'bum.npy'
        self.bottom_up_mask = np.load(bum)
        # y lsn flag: ylf.npy
        ylf = set_folder + 'ylf.npy'
        self.y_lsn_flag = np.load(ylf)
        # y set flag: ysf.npy
        ysf = set_folder + 'ysf.npy'
        self.set_flag = np.load(ysf)
        # y firing ege: yfa.npy
        yfa = set_folder + 'yfa.npy'
        self.y_firing_age = np.load(yfa)

        # z bottom up weight: zbu.npy
        zbu = set_folder + 'zbu.npy'
        self.z_bottom_up_weight = np.load(zbu)
        # z firing age: zfa.npy
        zfa = set_folder + 'zfa.npy'
        self.z_firing_age = np.load(zfa)

    def dn_log(self, logfile):
        logger1 = logging.getLogger('a')
        logger2 = logging.getLogger('b')
        logger1.setLevel(logging.INFO)
        logger2.setLevel(logging.INFO)
        logfileTotal = logfile + 'Info.log'
        logfileDetail = logfile + 'Detail.log'
        ft = logging.FileHandler(logfileTotal, mode='w')
        fd = logging.FileHandler(logfileDetail, mode='w')
        logger1.addHandler(ft)
        logger2.addHandler(fd)
        return logger1, logger2

    def set_log(self, setinfo):
        self.log_t.info(setinfo)

    def dn_test(self, test_image):
        self.x_response = test_image.reshape(1, -1)
        self.y_bottom_up_response = self.compute_response(self.x_response,
                                                     self.y_bottom_up_weight,
                                                      self.bottom_up_mask, self.bottom_up_factor)
        self.y_pre_response = self.y_bottom_up_response * self.set_flag
        max, index = self.top_k_competition(False)

        z_output = []
        for i in range(self.z_area_num):
            self.z_response[i] = np.dot(self.y_response,self.z_bottom_up_weight[i])
            z_output_i = np.argmax(self.z_response[i])
            z_output.append(z_output_i)
        return np.array(z_output), max, index
# code is highly inspired by
# https://github.com/dandxy89/BellkorAlgorithm
from params import *
import numpy as np
import time as Time
import pandas as pd
import datetime
from tqdm import trange


###
# data - matrix
# row = [index, timestamp, user, item, rating]
# each user and itme should be reassign to numbers
# in range [0, size(users/items])


class TimeSVD:
    INDEX, TIMESTAMP, USER, ITEM, RATING = range(0, 5)

    def __init__(self, data, **params):
        self.data = data

        # array of users and items
        self.users = np.arange(len(np.unique(data[:, self.USER])))
        self.items = np.arange(len(np.unique(data[:, self.ITEM])))

        # hyper parameters
        self.learning_rates = LEARNING_RATES
        self.regs = REGULARIZATIONS
        self.latent_factors_size = params["latent_size"]

        # dataset-related parameters
        self.start_time = params["start_time"]  # should be timestamp
        self.end_time = params["end_time"]  # should be timestamp
        self.time_diff = self.end_time - self.start_time
        self.mean = np.mean(data[:, self.RATING])
        self.average_times = []
        for user in self.users:
            self.average_times.append(np.mean(self.data[np.where(self.data[:, self.USER] == user), self.TIMESTAMP]))
        # print(self.average_times)

        # list of all timestamps from start to end with step 1 day
        self.date_range = list(map(lambda x: int(Time.mktime(x.timetuple())),
                                   pd.date_range(
                                       start=datetime.datetime.fromtimestamp(self.start_time),
                                       end=datetime.datetime.fromtimestamp(self.end_time),
                                       freq="D",
                                   ).tolist(),
                                   )
                               )

        self.time_range = np.arange(len(self.date_range))
        self.time_map = dict(zip(self.date_range, np.arange(self.time_range.shape[0])))

        # initial parameters setup
        self.params = Parameters()
        # users parameters
        for user in self.users:
            # For each User add their Parameters
            self.params.b_u[user] = 0
            self.params.alpha_u[user] = 0
            self.params.c_u[user] = 1
            self.params.p[user] = np.random.rand(self.latent_factors_size) * 1e-2
            self.params.alpha_p[user] = np.random.rand(self.latent_factors_size) * 1e-2

        # time bases user parameters
        self.params.b_ut = np.zeros(
            shape=(self.users.shape[0], self.time_range.shape[0])
        )
        self.params.c_ut = self.params.b_ut.copy()

        # items Parameters
        for item in self.items:
            self.params.b_i[item] = 0
            self.params.q[item] = np.random.rand(self.latent_factors_size) * 1e-2

        # time bases params for items
        self.params.b_ibin = np.tile(np.arange(30), reps=(self.items.shape[0], 1)) / 100

    def prepare_data(self, data, index):
        # prepare data for prediction
        time = data[index, self.TIMESTAMP]
        bin = int((time - self.start_time) / (self.time_diff / 29))
        time_index = int(self.time_map[int(time)])
        # time_index = int(self.time_map[int(
        #     Time.mktime(datetime.datetime.fromtimestamp(time).date().timetuple())
        # )])
        return {
            'time': time,
            'bin': bin,
            'time_idx': time_index,
            'user': data[index, self.USER],
            'item': data[index, self.ITEM]
        }

    def predict_one(self, d):
        """
        :param d: data about 'situation': shoule have:
        d['user'] d['item'] d['time'] d['time_idx'] d['bin']
        :return: prediction
        """
        # print(d['user'])
        # print(len(self.average_times), len(self.users))
        delta_time = d['time'] - self.average_times[d['user']]
        sign = -1 if delta_time < 0 else 1
        delta_time = abs(delta_time) / self.time_diff
        dev = sign * delta_time ** 0.4

        p = self.params.p[d['user']] + self.params.alpha_p[d['user']] * dev

        # print(self.params.b_u[d['user']])
        return (
                       self.mean
                       # user biases part
                       + self.params.b_u[d['user']]
                       + self.params.alpha_u[d['user']] * dev
                       + self.params.b_ut[d['user'], d['time_idx']]
                       # items biases part
                       + self.params.b_i[d['item']] + self.params.b_ibin[d['item'], d['bin']]
                       # implicit feedback part
                       + self.params.c_u[d['user']] + self.params.c_ut[d['item'], d['time_idx']]
                       # latent factors part
                       + np.dot(self.params.q[d['item']], p)
               ), dev

    def predict(self, x):
        """
        :param x: matrix, where columns are [Index, TimePeriod, User, Item, BaseRating]
        :return:
        """
        pred_results = []

        for index in x[:, 0]:
            index_ = int(index)
            prediction, _ = self.predict_one(self.prepare_data(x, index_))
            pred_results.append(prediction)

        self.RMSE = 0
        for idx, user in enumerate(x):
            self.RMSE += (user[self.RATING] - pred_results[idx])**2
        self.RMSE /= len(pred_results)
        self.RMSE = np.sqrt(self.RMSE)

        return pred_results

    def train(self, epochs=20, stochastic_size=1000):
        cost_history = []
        for epoch in trange(epochs):
            start = Time.time()

            # stochaastic grad descent
            for random_index in np.random.choice(self.data[:, 0], stochastic_size):
                # predict
                prediction, dev = self.predict_one(self.prepare_data(self.data, random_index))

                # calculate error and cost
                cost, error = self.cost(self.data[random_index, self.RATING], prediction, random_index)
                cost_history.append(cost)

                self.gradient_descent(error=error, index=random_index, dev=dev)

            elapsed_time = float(Time.time() - start)
            print(f"Time elapsed during epoch {epoch}: {'%.3f' % elapsed_time}s")
        return cost_history

    def cost(self, rating, prediction, index):
        d = self.prepare_data(self.data, index)
        error = rating - prediction
        cost = (
                error ** 2
                + self.regs['b_u'] * self.params.b_u[d['user']] ** 2
                + self.regs['alpha_u'] * self.params.alpha_u[d['user']] ** 2
                + self.regs['b_ut'] * self.params.b_ut[d['user'], d['time_idx']] ** 2
                + self.regs['b_i'] * self.params.b_i[d['item']] ** 2
                + self.regs['b_ibin'] * self.params.b_ibin[d['item'], d['bin']] ** 2
                + self.regs['c_u'] * self.params.c_u[d['user']] ** 2
                + self.regs['c_ut'] * self.params.c_ut[d['user'], d['time_idx']] ** 2
                + self.regs['p'] * self.params.p[d['user']] @ self.params.p[d['item']]
                + self.regs['q'] * self.params.q[d['item']] @ self.params.q[d['item']]
                + self.regs['alpha_p'] * self.params.alpha_p[d['user']] @ self.params.alpha_p[d['user']]
        )
        return cost, error
        # from original code, but probably buggy
        # cost = (
        #         error**2
        #         + self.regularizations.b_u * self.parameters.b_u[d['user']]**2
        #         + self.regularizations.alpha_u * self.parameters.alpha_u[d['user']]**2
        #         + self.regularizations.b_ut * self.parameters.b_ut[d['user'], d['time_idx']]**2
        #         + self.regularizations.b_i * self.parameters.b_i[d['item']]**2
        #         + self.regularizations.b_ibin * self.parameters.b_ibin[d['item'], d['bin']]**2
        #         + self.regularizations.c_u * self.parameters.c_u[d['user']]**2
        #         + self.regularizations.c_ut * self.parameters.c_ut[d['user'], d['time_idx']]**2
        #         + self.regularizations.p * self.parameters.p[d['user']]**2
        #         + self.regularizations.q * self.parameters.q[d['item']]**2
        #         + self.regularizations.alpha_p * self.parameters.alpha_p[d['user']]**2
        # )

    def gradient_descent(self, error, index, dev):
        d = self.prepare_data(self.data, index)

        b_u = self.params.b_u[d['user']]
        alpha_u = self.params.alpha_u[d['user']]
        b_ut = self.params.b_ut[d['user'], d['time_idx']]
        b_i = self.params.b_i[d['item']]
        b_ibin = self.params.b_ibin[d['item'], d['bin']]
        c_u = self.params.c_u[d['user']]
        c_ut = self.params.c_ut[d['user'], d['time_idx']]
        p = self.params.p[d['user']].copy()
        q = self.params.q[d['item']].copy()
        alpha_p = self.params.alpha_p[d['user']].copy()

        self.params.b_u[d['user']] -= \
            self.learning_rates['b_u'] * (-2 * error + 2 * self.regs['b_u'] * b_u)

        self.params.alpha_u[d['user']] -= \
            self.learning_rates['alpha_u'] * (2 * error * (-dev) + 2 * self.regs['alpha_u'] * alpha_u)

        self.params.b_ut[d['user'], d['time_idx']] -= \
            self.learning_rates['b_ut'] * (-2 * error * self.regs['b_ut'] * b_ut)

        self.params.b_i[d['item']] -= \
            self.learning_rates['b_i'] * (2 * error * (-c_u - c_ut) + 2 * self.regs['b_i'] * b_i)

        self.params.b_ibin[d['item'], d['bin']] -= \
            self.learning_rates['b_ibin'] * (2 * error * (-c_u - c_ut) + 2 * self.regs['b_ibin'] * b_ibin)

        self.params.c_u[d['user']] -= \
            self.learning_rates['c_u'] * (2 * error * (-b_i - b_ibin) + 2 * self.regs['c_u'] * c_u)
        self.params.c_ut[d['user'], d['time_idx']] \
            -= self.learning_rates['c_ut'] * (2 * error * (-b_i - b_ibin) + 2 * self.regs['c_ut'] * c_ut)
        self.params.p[d['user']] -= \
            self.learning_rates['p'] * (2 * error * (-q) + 2 * self.regs['p'] * p)
        self.params.q[d['item']] -= \
            self.learning_rates['q'] * (2 * error * (-p - alpha_p * dev) + 2 * self.regs['q'] * q)
        self.params.alpha_p[d['user']] -= \
            self.learning_rates['alpha_p'] * (2 * error * (-q * dev) + self.regs['alpha_p'] * alpha_p)

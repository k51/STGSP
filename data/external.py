import os
import numpy as np
import h5py
from .utils import timestamp2array, timestamp2vec_origin, transtr, transtrlong, transtr24

def external_taxibj(datapath, fourty_eight, previous_meteorol):
    def f(tsx, tsy, timeenc):
        exd = ExtDat(datapath)
        tsx = np.asarray([exd.get_bjextarray(N, timeenc, fourty_eight=fourty_eight) for N in tsx])
        tsy = exd.get_bjextarray(tsy, timeenc, fourty_eight=fourty_eight, previous_meteorol=previous_meteorol)

        print('there are totally', exd.tot_holiday, 'holidays in constructed data')

        return tsx, tsy

    return f

def external_bikenyc():
    def f(tsx, tsy, timeenc):
        if timeenc == 'd':
            tsx = np.asarray([timestamp2vec_origin(N) for N in tsx])
            tsy = timestamp2vec_origin(tsy)
        elif timeenc == 'm' or timeenc == 'w':
            tsx = np.asarray([timestamp2array(N, timeenc) for N in tsx])
            tsy = timestamp2array(tsy, timeenc)
        else:
            raise ValueError('The value of timeenc does not exist')
        return tsx, tsy
    return f

class ExtDat:
    def __init__(self, datapath, dataset='TaxiBJ'):
        self.tot_holiday = 0

        self.holidayfname = os.path.join(datapath, dataset, 'BJ_Holiday.txt')
        f = open(self.holidayfname, 'r')
        holidays = f.readlines()
        self.holidays = set([h.strip() for h in holidays])
        '''
        timeslots: the predicted timeslots
        In real-world, we dont have the meteorol data in the predicted timeslot, instead, we use the meteoral at
         previous timeslots, i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)
        '''
        fname = os.path.join(datapath, dataset, 'BJ_Meteorology.h5')
        f = h5py.File(fname, 'r')
        timeslots = f['date'][()]
        wind_speed = f['WindSpeed'][()]
        weather = f['Weather'][()]
        temperature = f['Temperature'][()]
        f.close()

        self.M = dict()
        for i, slot in enumerate(timeslots):
            self.M[slot.decode()] = i
        ws = []  # WindSpeed
        wr = []  # Weather
        te = []  # Temperature
        for slot in timeslots:
            cur_id = self.M[slot.decode()]
            ws.append(wind_speed[cur_id])
            wr.append(weather[cur_id])
            te.append(temperature[cur_id])

        ws = np.asarray(ws)
        wr = np.asarray(wr)
        te = np.asarray(te)

        ws = 1. * (ws - ws.min()) / (ws.max() - ws.min())
        te = 1. * (te - te.min()) / (te.max() - te.min())

        print("meteor shape: ", ws.shape, wr.shape, te.shape)
        self.meteor_data = np.hstack([wr, ws[:, None], te[:, None]])

    def get_bjextarray(self, timestamp_list, timeenc, fourty_eight=False, previous_meteorol=False):
        if timeenc == 'd':
            vecs_timestamp = timestamp2vec_origin(timestamp_list)
        elif timeenc == 'm' or timeenc == 'w':
            vecs_timestamp = timestamp2array(timestamp_list, timeenc, fourty_eight)
        else:
            raise ValueError('The value of timeenc does not exist')
        bits_holiday = self.get_holidayarray(timestamp_list)
        vecs_meteorol = self.get_meteorolarray(timestamp_list, previous_meteorol, fourty_eight)

        return np.hstack([vecs_timestamp, bits_holiday, vecs_meteorol])

    def get_holidayarray(self, timeslots):
        h = [0 for _ in range(len(timeslots))]
        for i, slot in enumerate(timeslots):
            transformat = transtr(slot)
            if transformat in self.holidays:
                h[i] = 1
                self.tot_holiday += 1
        return np.vstack(h)

    def get_meteorolarray(self, timestamp_list, previous_meteorol, fourty_eight):
        if fourty_eight:
            return np.array(
                [
                    self.meteor_data[
                        self.M[transtrlong(ts-np.timedelta64(30, 'm') if previous_meteorol else ts)]
                    ] for ts in timestamp_list
                ]
            )
        else:
            return np.array(
                [
                    self.meteor_data[
                        self.M[transtr24(ts-np.timedelta64(60, 'm') if previous_meteorol else ts)]
                    ] for ts in timestamp_list
                ]
            )

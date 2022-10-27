import datetime
import pylab as plt
from pkulast.validation.net import SURFRAD, HiWATER, PKULSTNet

time = datetime.datetime.strptime("2019-08-11 14:00:00", "%Y-%m-%d %H:%M:%S")

class TestValClass:

    def test_validation(self):
        # SURFRAD
        s = SURFRAD()
        print(s.available_sites)
        mins, downward_r, upward_r = s.get_data('bon', time)
        y = downward_r - upward_r
        x = downward_r
        print(s.get_info('bon'))
        print(s.get_lst('bon', time))
        # plt.plot(acq_time, x)
        # plt.plot(acq_time, upward_r)
        # plt.plot(acq_time, y / x)
        # plt.plot(acq_time, y)
        # plt.show()

        # HiWATER
        h = HiWATER()
        print(h.available_sites)
        print(h.get_info('arz'))
        print(h.get_lst('arz', datetime.datetime.strptime("2018-08-11 14:00:00", "%Y-%m-%d %H:%M:%S"), 0.99))
        # PKULSTNet
        pku = PKULSTNet()
        print(pku.available_sites)
        print(pku.get_info('hbc'))
        print(pku.get_lst('hbc', time, 0.99))
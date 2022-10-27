from pkulast.retrieval.algorithm import SC, SW
from pkulast.retrieval.coefficient import sc_coefficient, sw_coefficient
from pkulast.remote.sensor import RSR
import numpy as np
import joblib


class TestAlgClass:

    # def test_sc(self):
    #     alg="JM09"
    #     sw = SW(alg)
    #     landsat_rsr = landsat_rsr = RSR("landsat_8_tirs")
    #     landsat_rsr.subset([1, ])
    #     result = joblib.load(r"E:\CodeRepo\代码\simpir_tutorials\sc_L8B9_simulated_dataset.pkl")
    #     models = sw_coefficient(
    #                     landsat_rsr,
    #                     alg,
    #                     dataset=result,
    #                     # store_dataset=True,
    #                 )
    #     # sw.set_params(landsat_rsr)
    #     sw.add_models(models)
    #     bt = np.array([[300, 310, 320], [300, 310, 320]])
    #     e =  np.array([[0.97, 0.96, 0.96], [0.89, 0.94, 0.93]])
    #     print(sw.compute(bt, e=e))


    def test_sw(self):
        alg="Wan14"
        sw = SW(alg)
        landsat_rsr = RSR("landsat_8_tirs")
        result = joblib.load(r"E:\CodeRepo\代码\simpir_tutorials\simulated_dataset.pkl")
        models = sw_coefficient(
                        landsat_rsr,
                        alg,
                        dataset=result,
                        # store_dataset=True,
                    )
        sw.set_params(landsat_rsr)
        sw.add_models(models)
        bt = np.array([[300, 304, 308], [303, 306, 309]]).T
        e =  np.array([[0.97, 0.96, 0.96], [0.89, 0.94, 0.93]]).T
        print(sw.compute(bt, e=e))
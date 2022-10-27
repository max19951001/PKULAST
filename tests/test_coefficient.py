from pkulast.retrieval.coefficient import sc_coefficient, sw_coefficient
import joblib
from pprint import pprint
from pkulast.remote.sensor import RSR
from pkulast.config import RSR_LIB
from pkulast.rtm.model import RTMSimul
from pkulast.retrieval.algorithm import SC, SW

selected_type = ("tropical", "mid-latitude", "polar")
dTs = list(range(-10, 25, 5))
cwv_intervals = [[0.0, 2.5], [2.0, 3.5], [3.0, 4.5], [4.0, 5.5], [5.0, 6.3], [0, 6.3]]

class TestCoeffClass:


    def test_sc(self):
        landsat_rsr = RSR("landsat_8_tirs") #landsat_7_tm landsat_7_tm
        landsat_rsr.subset([1, ])
        # md = RTMSimul(landsat_rsr, range(-10, 25, 5))
        # result = md.simulate(selected_tcwv=[0, 8])
        # joblib.dump(result, r"E:\CodeRepo\代码\simpir_tutorials\sc_L8B9_simulated_dataset.pkl")
        result = joblib.load(r"E:\CodeRepo\代码\simpir_tutorials\sc_L8B9_simulated_dataset.pkl")
        for alg in SC.algorithms:
            print("%s\n**********************************************************\n"%alg)
            if alg == "RTE":
                continue
            models = sc_coefficient(
                landsat_rsr,
                alg,
                cwv_intervals=cwv_intervals,
                dataset=result,
                # store_dataset=True,
            )
            # joblib.dump(models[0]["dataset"], r"E:\CodeRepo\代码\simpir_tutorials\sc_L5B6_simulated_dataset.pkl")
            for i in range(len(models)):
                print("******************************\n %s-%s\n"%(alg, models[i]["label"]))
                pprint("RMSE %.3f"%models[i]["RMSE"])
                pprint("R2 %.3f"%models[i]["R2"])
                for coef in models[i]["coeffs"]:
                    pprint(coef)
                print("******************************\n")
    def test_sw(self):
        landsat_rsr = RSR("landsat_8_tirs") #landsat_7_tm landsat_7_tm
        # md = RTMSimul(landsat_rsr, range(-10, 25, 5))
        # result = md.simulate(selected_tcwv=[0, 8])
        # joblib.dump(result, r"E:\CodeRepo\代码\simpir_tutorials\sc_L8B9_simulated_dataset.pkl")
        result = joblib.load(r"E:\CodeRepo\代码\simpir_tutorials\simulated_dataset.pkl")
        alg_list = ["RO14", ] #["Wan14", ] #SW.algorithms
        for alg in alg_list:
            print("%s\n**********************************************************\n"%alg)
            models = sw_coefficient(
                landsat_rsr,
                alg,
                cwv_intervals=cwv_intervals,
                dataset=result,
                # store_dataset=True,
            )
            # joblib.dump(models[0]["dataset"], r"E:\CodeRepo\代码\simpir_tutorials\sc_L5B6_simulated_dataset.pkl")
            for i in range(len(models)):
                print("******************************\n %s-%s\n"%(alg, models[i]["label"]))
                pprint("RMSE %.3f"%models[i]["RMSE"])
                pprint("R2 %.3f"%models[i]["R2"])
                for coef in models[i]["coeffs"]:
                    pprint(coef)
                print("******************************\n")
            
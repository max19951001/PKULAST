from datetime import datetime, timedelta
import profile
from pkulast.config import TEST_DATA_DIR
from pkulast.atmosphere.profile import ClearSkyTIGR946, NWPLibrary, StandardLibrary, TIGRLibrary, SeeborLibrary, Grib, NWPTile, NWPCube
from pkulast.atmosphere.profile import DOEHandler, GDASHandler, GDAS25Handler, JRA55Handler, CFSv2Handler
from pkulast.rtm.model import ModtranWrapper, run_modtran
from pkulast.remote.sensor import RelativeSpectralResponse
lat = 40
lon = 120
extent = (39, 41, 119, 121)
acq_time = datetime(2020, 10, 21, 8, 48, 0)

class TestProfileClass:
    def test_profile(self):
        # profiles = ClearSkyTIGR946()
        # print(profiles[0].get_effective_temperature())
        # s = StandardLibrary()
        # for k, p in s.iter_profiles():
        #     # print(k)
        #     print(p.plot())
        # lat = 45.59721498
        # lon = 89.34080970
        # acq_time = datetime(2020, 10, 25, 16, 30, 0) - timedelta(hours=8)
        nwp = NWPLibrary()
        p = nwp.extract(acq_time, lat, lon)
        p = StandardLibrary.patch_profile(p)
        p.plot()
        p.savefig(f'NCEP_TASI.png')
        # p.save('atm_corr.txt')
        # nwp = NWPLibrary("JRA55")
        # p = nwp.extract(acq_time, 40, 120)
        # s = StandardLibrary()
        # "127.0.0.1:7890"
        # 'GFS', 'ERA5', 'GDAS', 'ERA5'
        # for t in ['MERRA2', 'GDAS25', 'GFS', 'ERA5', 'GDAS', 'JRA55', 'CFSv2']:
        #     nwp = NWPLibrary(t, proxy='127.0.0.1:7890')
        #     print(t)

        #     p = nwp.extract(acq_time, 40, 120)
        #     # p.plot()
        #     p.save(f'{t}.txt')
        # p.savefig(f'{t}.png')

        rsr = RelativeSpectralResponse('TASI')
        # rsr.plot()
        atm = run_modtran([p, ], rsr, is_satellite=False, flight_altitude=2.472, out_file='tasi_atm.txt')
        print(atm)

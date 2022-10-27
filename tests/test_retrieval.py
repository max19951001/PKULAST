from pkulast.remote.sensor import RelativeSpectralResponse
from pkulast.atmosphere.profile import ClearSkyTIGR946
from pkulast.rtm.model import ModtranWrapper, run_modtran

class TestRetrievalClass:

    def test_modtran(self):
        rsr = RelativeSpectralResponse('MASI')
        profiles = ClearSkyTIGR946()
        mw = ModtranWrapper(rsr, False, 4)
        a = mw.simulation(profiles[:2], tbound=300, vza=0, mult=True, albedo=0.5, include_solar=False, delta_altitude=100, out_file='ir_exclude.out')
        b = mw.simulation(profiles[:2], tbound=300, vza=0, mult=True, albedo=0.5, include_solar=True, delta_altitude=4, out_file='ir_include.out')
        atm = run_modtran(profiles[:2], rsr, False, 3, 0, mult= False, include_solar=False, vza=0, out_file='result.out')
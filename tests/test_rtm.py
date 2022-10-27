from pkulast.remote.sensor import RelativeSpectralResponse
from datetime import datetime
from pkulast.atmosphere.profile import ClearSkyTIGR946
from pkulast.rtm.model import run_modtran

class TestRTMClass:

    def test_modtran(self):
       f = RelativeSpectralResponse('mersi2')
       f.subset((20, 21, 22, 23, 24, 25))
       profiles = ClearSkyTIGR946()
       run_modtran(profiles[:2], f, is_satellite=True, flight_altitude=100, ground_altitude=3)
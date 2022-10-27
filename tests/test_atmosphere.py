from datetime import datetime
from pkulast.config import TEST_DATA_DIR
from pkulast.atmosphere.profile import ClearSkyTIGR946, NWPLibrary, TIGRLibrary, SeeborLibrary, Grib, NWPTile, NWPCube

lat = 40
lon = 120
extent = (39, 41, 119, 121)
acq_time = datetime(2019, 8, 13, 8, 48, 48)
filename = TEST_DATA_DIR + 'gfs_4_20180812_1800_000.grb2'


class TestAtmosphereClass:
    def test_grib_tile(self):
        g = Grib(filename)
        g.subset('Temperature', extent=extent)
        profiles = g.extract(extent=extent, stride=1)
        lats, lons = g.latlons(extent=extent, stride=1)
        nt = NWPTile(profiles, lats, lons, g.acq_time, extent)
        p = nt.interp(40.4, 119.499)
        p.plot()

    def test_cube(self):
        cube = NWPCube(acq_time, extent, tile_count=2)
        p = cube.interp(40.4, 119.499, temporal_method='quadratic')

    def test_profile(self):
        p = NWPLibrary()
        profile = p.extract(acq_time, lat, lon)
        profile.plot()

        p = NWPLibrary('ERA5')
        profile = p.extract(acq_time, lat, lon)
        profile.plot()

        s = SeeborLibrary()
        profile = s[1]
        print(profile.get_card2c())
        profile.plot()

        t = TIGRLibrary()
        profile = t[1]
        print(profile.get_card2c())
        profile.plot()

        profiles = ClearSkyTIGR946()
        profiles[0].plot()
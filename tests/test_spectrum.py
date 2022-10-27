from pkulast.surface.spectrum import SpectralLibrary


class TestSpectrumClass:
    def test_spectrum(self):
        usgs = SpectralLibrary('USGS')
        usgs.plot(100)
        print('USGS', usgs.count)
        print(usgs.contains('veg'))
        print(usgs.get_name(111))
        print(usgs.get_signature(111))
        relab = SpectralLibrary('RELAB')
        relab.plot(100)
        print('RELAB', relab.count)
        aster = SpectralLibrary('ASTER')
        aster.plot(100)
        print('ASTER', aster.count)
        ecostress = SpectralLibrary('ECOSTRESS')
        ecostress.plot(100)
        print('ECOSTRESS', ecostress.count)

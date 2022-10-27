from pkulast.remote.sensor import RelativeSpectralResponse


class TestRemoteClass:

    def test_rsr(self):
        modis_rsr = RelativeSpectralResponse('modis_EOS-Terra', name="modis")
        modis_rsr.subset([20, 29, 31, 32])
        modis_rsr.plot()
        aster_rsr = RelativeSpectralResponse("aster")
        aster_rsr.subset([10, 11, 12, 13, 14])
        RelativeSpectralResponse.composite_plot(modis_rsr, aster_rsr, show_span=False, unit='nm')

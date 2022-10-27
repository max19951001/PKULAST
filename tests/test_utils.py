from pkulast.constants import EPSILON
from pkulast.utils.thermal import planck, planckian, inverse_planckian, convert_wave_unit


class TestUtilsClass:

	def test_planckian(self):
		assert (planckian(10, 300) - 9.924029710212947) < EPSILON

	def test_inverse_planckian(self):
		assert (inverse_planckian(10, [10, ])[0] - 300) < 1

	def test_planck(self):
		assert (planck(10e-6, 300) / 1e6 -  9.924029710212947) < EPSILON

	def test_convert_wv(self):
		assert convert_wave_unit(10) == 1E6

# -*- coding:utf-8 -*-
# Copyright (c) 2021-2022.

################################################################
# The contents of this file are subject to the GPLv3 License
# you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# https://www.gnu.org/licenses/gpl-3.0.en.html

# Software distributed under the License is distributed on an "AS IS"
# basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
# License for the specific language governing rights and limitations
# under the License.

# The Original Code is part of the PKULAST python package.

# Initial Dev of the Original Code is Jinshun Zhu, PhD Student,
# Institute of Remote Sensing and Geographic Information System,
# Peking Universiy Copyright (C) 2022
# All Rights Reserved.

# Contributor(s): Jinshun Zhu (created, refactored and updated original code).
###############################################################
"""pkulast.validation.net module. Provides: LST validation interface
"""

import os
import re
import glob
import shutil
import calendar
import pandas as pd
import numpy as np
from datetime import datetime
import urllib.request
from collections import defaultdict

from pkulast.config import *
from pkulast.constants import SIGMA, ZERO_TEMPERATURE


def day_of_year(_date):
	"""Convert a datatime to day of year 
	"""
	if not isinstance(_date, datetime):
		raise TypeError('Invalid type for parameter "date" - expecting datetime')
	return _date.timetuple().tm_yday

class Net(object):
	def __init__(self):
		self.available_sites = None
		self.config = None
		self.name = None
		self.desc = None

	def get_info(self, site):
		''' Get site information
		'''
		self._check_if_available(site)
		return self.config['SITE'][site]
	
	def get_loc(self, site):
		''' Get site location
		'''
		self._check_if_available(site)
		info = self.get_info(site)
		return info['latitude'], info['longitude']

	def get_count(self):
		''' Get site count
		'''
		return self.config['COUNT']

	def get_interval(self):
		''' Get site measurement interval
		'''
		return self.config['INTERVAL']

	def get_desc(self):
		''' Get site identifiers
		'''
		return self.desc

	def _get_full_name(self, site):
		return self.config['SITE'][site]['alias']

	def _check_if_available(self, site):
		if site not in self.available_sites:
			raise ValueError(f'Site `{site}` is not included in available_sites, please re-check spelling mistakes\n Available sites includes {self.available_sites}')
	
	def __str__(self):
		return f"Validation net {self.name}(self.config['INTERVAL' min interval]), including {self.get_count()} sites:\n{self.desc}"

class SURFRAD(Net):
	"""SURFRAD sites for land surface temperature validation.
	"""
	def __init__(self):
		self.available_sites = SURFRAD_SITES
		self.config = Configuration['Validation']['SURFRAD']
		self.name = SURFRAD_PROJECT_NAME
		self.desc = SURFRAD_DESC

	def get_interval(self, time=None):
		''' Get site measurement interval
		'''
		if time:
			return 3 if time < SURFRAD_UPDATED_DATE else 1
		else:
			raise ValueError('SURFRAD instruments renewed at 2009-01-01 00:00:00, please give a spceifc time for accurate measurement interval.')

	def get_data(self, site, date):
		""" Return data files
		"""
		self._check_if_available(site)
		yr = str(date.year)
		doy = "%03d"%day_of_year(date)
		filename = f'{site}{yr[2:]}{doy}.dat'
		filepath = SURFRAD_DIR + filename
		if not os.path.exists(filepath):
			url = f'{SURFRAD_URL}{self._get_full_name(site)}/{yr}/{filename}'
			with urllib.request.urlopen(url) as response, open(filepath, 'wb') as out_file:
				shutil.copyfileobj(response, out_file)

		with open(filepath, 'r') as file:
			next(file)
			line = file.readline()
			self.lat, self.lon, self.elev, self.version = re.findall(r'[-+]?[0-9]*\.?[0-9]+', line)
		data = np.genfromtxt(filepath, skip_header=2)
		hours = data[:, 4]
		print(filepath)
		mins = hours * 60 + data[:, 5]
		downward_r = data[:, 16]
		print(downward_r)
		upward_r = data[:, 22]
		print(upward_r)
		return mins, downward_r, upward_r


	def get_lst(self, site, date, emiss=None):
		''' Get LST
		'''
		self._check_if_available(site)
		if not emiss:
			emiss = self.get_info(site)['emissivity']
		yr = str(date.year)
		doy = "%03d"%day_of_year(date)
		filename = f'{site}{yr[2:]}{doy}.dat'
		filepath = SURFRAD_DIR + filename
		if not os.path.exists(filepath):
			url = f'{SURFRAD_URL}{self._get_full_name(site)}/{yr}/{filename}'
			with urllib.request.urlopen(url) as response, open(filepath, 'wb') as out_file:
				shutil.copyfileobj(response, out_file)

		with open(filepath, 'r') as file:
			next(file)
			line = file.readline()
			self.lat, self.lon, self.elev, self.version = re.findall(r'[-+]?[0-9]*\.?[0-9]+', line)
		data = np.genfromtxt(filepath, skip_header=2)
		acq_time = date.hour * 60 + date.minute
		hours = data[:, 4]
		mins = hours * 60 + data[:, 5]
		index = np.where( np.abs(mins - acq_time) <= 5 )		
		step_time = 3 if date < SURFRAD_UPDATED_DATE else 1
		downward_r = data[index, 16]
		upward_r = data[index, 22]

		gLST = np.power( (upward_r - (1 - emiss) * downward_r) / (emiss * SIGMA), 0.25)
		avgLST = np.average(gLST)
		stdLST = np.std(gLST)
		medianLST = np.median(gLST)
		return avgLST, stdLST, gLST

class PKULSTNet(Net):
	'''PKULSTNet sites for land surface temperature validation
	'''
	def __init__(self):
		self.available_sites = PKULSTNet_SITES
		self.available_time = defaultdict(list)
		self._extract_time()
		self.config = Configuration['Validation']['PKULSTNet']
		self.name = PKULSTNet_PROJECT_NAME
		self.desc = PKULSTNet_DESC

	def get_lst(self, site, date, emiss):
		''' Get LST
		'''
		self._check_if_available(site, date)
		year = date.year
		month = date.month
		_, month_days = calendar.monthrange(year, month)
		start_day = "%02d%02d"%(date.month, 1)
		end_day = "%02d%02d"%(date.month, month_days)
		filename = f"Total{self._get_full_name(site)}CR200Series_Data-{year}{start_day}-{end_day}.dat"
		filepath = PKULSTNet_DIR + filename
		if not os.path.exists(filepath):
			raise ValueError(f"File {filename} may be worngly named, please fix it right in:\n {PKULSTNet_DIR}")
		str2date = lambda x: datetime.strptime(x.decode("utf-8").strip('"'), '%Y-%m-%d %H:%M:%S')
		data = np.genfromtxt(filepath, delimiter=',', skip_header=4, converters = {0: str2date})
		l_index, r_index = self._nearest(data, date, 0, len(data)-1)
		gLST = list(map(lambda l:self._retrieval(l, emiss), data[l_index - PKULSTNet_INTERVAL:r_index + 1 + PKULSTNet_INTERVAL])) # 9 / 12分钟均值
		avgLST = np.average(gLST)
		stdLST = np.std(gLST)
		medianLST = np.median(gLST)
		return avgLST, stdLST, gLST

	def _retrieval(self, item, emiss):
		T_down = item[2] - ZERO_TEMPERATURE
		T_up   = item[3] - ZERO_TEMPERATURE
		return pow((SIGMA * T_up ** 4 - (1 - emiss) * SIGMA * T_down ** 4) / (emiss * SIGMA), 0.25)

	def _nearest(self, data, time, start, end):
		if end - start<= 1:
			return start, end
		mid = int((start + end) / 2)
		mid_time = data[mid][0]
		if mid_time > time:
			return self._nearest(data, time, start, mid)
		elif mid_time < time:
			return self._nearest(data, time, mid, end)
		else:
			return mid, mid

	def _extract_time(self):
		'''In view of the fact that PKULSTNet's data missing problem, we need extract time span of available data first. 
		'''
		for filename in glob.glob(PKULSTNet_DIR + "*.dat"):
			info = re.search(r"Total([\u4e00-\u9fa5]+)CR200Series_Data-(\d{4})(\d{2})(\d{2})-\d{2}(\d{2})", filename)
			if info:
				site, year, month, start_day, end_day = info.groups()
				self.available_time[site].append(year + month)

	def _check_if_available(self, site, time=None):
		if site not in self.available_sites:
			raise ValueError(f'Site `{site}` is not included in available_sites, please re-check spelling mistakes\n Available sites includes {self._available_sites}')
		if not time:
			return
		time_str = "%d%02d"%(time.year, time.month)
		site_aka = self._get_full_name(site)
		if time_str not in self.available_time[site_aka]:
			available_time = '\n'.join(self.available_time[site_aka])
			current_time = time.strftime("%Y-%m")
			raise ValueError(f"Ground site {site} has no data in {current_time}, please try another time\n Available time for {site} includes {available_time}")

class HiWATER(Net):
	'''HiWATER sites for land surface temperature validation
	'''
	def __init__(self):
		self.available_sites = HiWATER_SITES
		self.config = Configuration['Validation']['HiWATER']
		self.name = HiWATER_PROJECT_NAME
		self.desc = HiWATER_DESC

	def get_lst(self, site, date, emiss):
		''' Get LST
		'''
		self._check_if_available(site)
		year = str(date.year)
		# filename = f"{year}/{year}年黑河流域地表过程综合观测网{self._get_full_name(site)}AWS.xlsx"
		# filepath = HiWATER_DIR + filename
		# if not os.path.exists(filepath):
		# 	raise ValueError(f"File {filename} may be wrongly named, please fix it right in:\n {HiWATER_DIR}/{year}")
		# WS = pd.read_excel(filepath, usecols=[0, 27, 28], engine='openpyxl')
		filename = f"{year}/{year}AWS.h5"
		h5_name = f"{year}年黑河流域地表过程综合观测网{self._get_full_name(site)}AWS"
		filepath = HiWATER_DIR + filename
		if not os.path.exists(filepath):
			raise ValueError(f"File {filename} may be wrongly named, please fix it right in:\n {HiWATER_DIR}/{year}")
		WS = pd.read_hdf(filepath, h5_name)
		# data = np.array(WS, names=True)
		recode_times = WS['TIMESTAMP']
		data = np.c_[WS['ULR_Cor'], WS['DLR_Cor']]
		l_index, r_index = self._nearest(recode_times, date, 0, len(data)-1)
		gLST = list(map(lambda l:self._retrieval(l, emiss), data[l_index - HiWATER_INTERVAL:r_index + 1 + HiWATER_INTERVAL])) # 9 / 12分钟均值
		avgLST = np.average(gLST)
		stdLST = np.std(gLST)
		medianLST = np.median(gLST)
		return avgLST, stdLST, gLST

	def _retrieval(self, item, emiss):
		R_down = item[0]
		R_up   = item[1]
		return pow((R_up - (1 - emiss)* R_down) / (emiss * SIGMA), 0.25)

	def _nearest(self, recode_times, time, start, end):
		if end - start <= 1:
			return start, end
		mid = int((start + end) / 2)
		mid_time = recode_times[mid]
		if mid_time > time:
			return self._nearest(recode_times, time, start, mid)
		elif mid_time < time:
			return self._nearest(recode_times, time, mid, end)
		else:
			return mid, mid

		
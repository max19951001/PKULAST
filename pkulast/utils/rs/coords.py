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
''' Implementation of coordinate related utilities
'''

from math import floor, ceil
import numpy as np
import pyproj
import rasterio

# Spherical conversions
def lonlat2xyz(lon, lat):
	"""Convert lon lat to cartesian."""
	lat = np.deg2rad(lat)
	lon = np.deg2rad(lon)
	x = np.cos(lat) * np.cos(lon)
	y = np.cos(lat) * np.sin(lon)
	z = np.sin(lat)
	return x, y, z

def xyz2lonlat(x, y, z, asin=False):
	"""Convert cartesian to lon lat."""
	lon = np.rad2deg(np.arctan2(y, x))
	if asin:
		lat = np.rad2deg(np.arcsin(z))
	else:
		lat = np.rad2deg(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))
	return lon, lat

def angle2xyz(azi, zen):
	"""Convert azimuth and zenith to cartesian."""
	azi = np.deg2rad(azi)
	zen = np.deg2rad(zen)
	x = np.sin(zen) * np.sin(azi)
	y = np.sin(zen) * np.cos(azi)
	z = np.cos(zen)
	return x, y, z

def xyz2angle(x, y, z, acos=False):
	"""Convert cartesian to azimuth and zenith."""
	azi = np.rad2deg(np.arctan2(x, y))
	if acos:
		zen = np.rad2deg(np.arccos(z))
	else:
		zen = 90 - np.rad2deg(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))
	return azi, zen

def longlat2window(lons, lats, dataset):
	"""
	Args:
		dataset: Rasterio dataset
	Returns:
		rasterio.windows.Window
	source: https://gis.stackexchange.com/questions/298345/making-subset-of-sentinel-2-with-rasterio
	Example:
		with rasterio.open(file) as src:
			window = longlat2window((-99.2, -99.17), (19.40, 19.43), src)
			arr = src.read(1, window=window)
	"""
	p = pyproj.Proj(dataset.crs)
	t = dataset.transform
	xmin, ymin = p(lons[0], lats[0])
	xmax, ymax = p(lons[1], lats[1])
	col_min, row_min = ~t * (xmin, ymin)
	col_max, row_max = ~t * (xmax, ymax)
	return rasterio.windows.Window.from_slices(rows=(floor(row_max), ceil(row_min)),
							  cols=(floor(col_min), ceil(col_max)))

def longlat2rowcol(lon, lat, dataset):
	src = pyproj.Proj(dataset.crs) # Pass CRS of image from rasterio
	lonlat = pyproj.Proj('epsg:4326')
	east, north = pyproj.transform(lonlat, src, lon, lat)
	row, col = src.index(east, north) # spatial --> image coordinates
	return row, col

def rowcol2longlat(row, col, dataset):
	src = pyproj.Proj(dataset.crs)
	lonlat = pyproj.Proj('epsg:4326')
	east, north = dataset.xy(row,col) # image --> spatial coordinates
	lon, lat = pyproj.transform(src, lonlat, east, north)
	return lon, lat

def get_raster_extent(infile):
	with rasterio.open(infile) as src:
		rowcol2longlat(0, 0, src)
		return rowcol2longlat(src.width, src.height, src)

def proj_units_to_meters(proj_str):
	"""Convert projection units from kilometers to meters."""
	proj_parts = proj_str.split()
	new_parts = []
	for itm in proj_parts:
		key, val = itm.split('=')
		key = key.strip('+')
		if key in ['a', 'b', 'h']:
			val = float(val)
			if val < 6e6:
				val *= 1000.
				val = '%.3f' % val

		if key == 'units' and val == 'km':
			continue

		new_parts.append('+%s=%s' % (key, val))

	return ' '.join(new_parts)
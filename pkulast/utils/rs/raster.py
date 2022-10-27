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

import os
import sys
import math
import cmaps
import pyproj
import rasterio
import rasterio.plot
import itertools
import subprocess
import numpy as np
import pylab as plt
import cartopy.crs as ccrs
from osgeo import gdal, gdal_array, gdalconst, ogr, osr
from pyresample import create_area_def, get_area_def
from pyresample import geometry as geom
from pyresample import kd_tree as kdt
from pkulast.config import TEST_DATA_DIR
from pkulast.exceptions import *
from rios.imagewriter import DEFAULTDRIVERNAME, dfltDriverOptions
from rios.parallel.jobmanager import find_executable
import rasterio


def coord2area(name, proj, min_lat, max_lat, min_lon, max_lon, resolution):
    """ get area_def from coordinates

	name: area_def name
	proj: projection
	min_lat, max_lat, min_lon, max_lon
	resolution: m
	"""
    area_id = name
    description = name
    proj_id = f'{name}_{proj}_{resolution}km'
    res   = resolution * 1000

    left  = math.floor(min_lon)
    right = math.ceil(max_lon)
    up    = math.floor(min_lat)
    down  = math.ceil(max_lat)

    lat_0 = (up + down) / 2
    lon_0 = (right + left) / 2
    p = pyproj.Proj(proj=proj, lat_0=lat_0, lon_0=lon_0, ellps="WGS84")
    left_ex1, up_ex1 = p(left, up)
    right_ex1, up_ex2 = p(right, up)
    left_ex2, down_ex1 = p(left, down)
    right_ex2, down_ex2 = p(right, down)
    left_ex3, dummy = p(left, lat_0)
    right_ex3, dummy = p(right, lat_0)

    area_extent = (min(left_ex1, left_ex2, left_ex3),
          min(up_ex1, up_ex2),
          max(right_ex1, right_ex2, right_ex3),
          max(down_ex1, down_ex2))
    x_size = int(round((area_extent[2] - area_extent[0]) / res)) # width
    y_size = int(round((area_extent[3] - area_extent[1]) / res)) # height
    proj_dict = {'proj': proj, 'lat_0':lat_0, 'lon_0': lon_0, 'ellps': 'WGS84'}
    projection = "+" + \
     " +".join(("proj=" + proj + ",lat_0=" + str(lat_0) +
          ",lon_0=" + str(lon_0) + ",ellps=WGS84").split(","))
    areadef = get_area_def(area_id, description, proj_id, proj_dict, x_size, y_size, area_extent)
    return areadef

def utmcode(lat, lon):
    """ get utm zone code from lat, lon
	"""
    utm = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm) == 1:
        utm = '0' + utm
    if lat >= 0:
        epsg_code = '326' + utm
    else:
        epsg_code = '327' + utm
    return epsg_code

def get_area_extent(proj, lats, lons):
    """ get extent from coord
	"""
    left  = math.floor(np.min(lons))
    right = math.ceil(np.max(lons))
    up    = math.floor(np.min(lats))
    down  = math.ceil(np.max(lats))
    lat_0 = (up + down) / 2
    lon_0 = (right + left) / 2
    p = pyproj.Proj(proj=proj, lat_0=lat_0, lon_0=lon_0, ellps="WGS84")
    left_ex1, up_ex1 = p(left, up)
    right_ex1, up_ex2 = p(right, up)
    left_ex2, down_ex1 = p(left, down)
    right_ex2, down_ex2 = p(right, down)
    left_ex3, dummy = p(left, lat_0)
    right_ex3, dummy = p(right, lat_0)
    area_extent = (min(left_ex1, left_ex2, left_ex3),
          min(up_ex1, up_ex2),
          max(right_ex1, right_ex2, right_ex3),
          max(down_ex1, down_ex2))
    return area_extent

def calculate_area_def(lats, lons, pixel_size, proj="UTM", utmzone=None):
    """ get row/col number under specific resolution.
	"""
    swath_def = geom.SwathDefinition(lons=lons, lats=lats)
    area_extent = None
    epsg = '4326'
    proj_fullname = 'Geographic'
    proj_name = 'longlat'
    proj_dict = {}
    center = [int(lats.shape[1]/2)-1, int(lats.shape[0]/2)-1]
    center_latitude, center_longitude = float(lats[center[0]][center[1]]), float(lons[center[0]][center[1]])
    if proj == "UTM":
        if utmzone is None:
            epsg = utmcode(center_latitude, center_longitude)
        else:
            epsg = utmzone
        epsg_convert = pyproj.Proj("EPSG:{}".format(epsg))
        proj_name, proj_fullname = 'utm', 'Universal Transverse Mercator'
        proj_dict = {'proj': proj_name, 'zone': epsg[-2:], 'ellps': 'WGS84', 'datum': 'WGS84', 'units': 'm'}
        if epsg[2] == "7":
            proj_dict['south'] = 'True'
        llLon, llLat = epsg_convert(np.min(lons), np.min(lats), inverse=False)
        urLon, urLat = epsg_convert(np.max(lons), np.max(lats), inverse=False)
        area_extent = (llLon, llLat, urLon, urLat)
        pixel_size = pixel_size
    if proj == "GEO":
        epsg_convert = pyproj.Proj("+proj=aeqd +lat_0={} +lon_0={}".format(center_latitude, center_longitude))
        llLon, llLat = epsg_convert(np.min(lons), np.min(lats), inverse=False)
        urLon, urLat = epsg_convert(np.max(lons), np.max(lats), inverse=False)

        area_extent = (llLon, llLat, urLon, urLat)
        cols = int(round((area_extent[2] - area_extent[0]) / pixel_size))
        rows = int(round((area_extent[3] - area_extent[1]) / pixel_size))
        epsg, proj_name, proj_fullname = '4326', 'longlat', 'Geographic'
        llLon, llLat, urLon, urLat = np.min(lons), np.min(lats), np.max(lons), np.max(lats)
        area_extent = (llLon, llLat, urLon, urLat)
        proj_dict = {'proj': proj_name, 'datum': 'WGS84'}
        area_def = geom.AreaDefinition(epsg, proj_fullname, proj_name, proj_dict, cols, rows, area_extent)
        pixel_size = np.min([area_def.pixel_size_x, area_def.pixel_size_y])  # Square pixels
    cols = int(round(float((area_extent[2] - area_extent[0]) / pixel_size)))  # Calculate the output cols
    rows = int(round(float((area_extent[3] - area_extent[1]) / pixel_size)))  # Calculate the output rows
    area_def = geom.AreaDefinition(epsg, proj_fullname, proj_name, proj_dict, cols, rows, area_extent)
    return area_def

def swath2grid(array, lats, lons, pixel_size, filename, proj="UTM", utmzone=None, fill_value=np.nan):
    """ convert swath data to grid data.
	"""
    swath_def = geom.SwathDefinition(lons=lons, lats=lats)
    mid = [int(lats.shape[0]/2)-1, int(lats.shape[1]/2)-1]
    area_extent = None
    epsg = '4326'
    proj_fullname = 'Geographic'
    proj_name = 'longlat'
    proj_dict = {}
    midLat, midLon = float(lats[mid[0]][mid[1]]), float(lons[mid[0]][mid[1]])
    if proj == "UTM":
        if utmzone is None:
            epsg = utmcode(midLat, midLon)
        else:
            epsg = utmzone
        epsg_convert = pyproj.Proj("EPSG:{}".format(epsg))
        proj_name, proj_fullname = 'utm', 'Universal Transverse Mercator'
        proj_dict = {'proj': proj_name, 'zone': epsg[-2:], 'ellps': 'WGS84', 'datum': 'WGS84', 'units': 'm'}
        if epsg[2] == "7":
            proj_dict['south'] = 'True'
        llLon, llLat = epsg_convert(np.min(lons), np.min(lats), inverse=False)
        urLon, urLat = epsg_convert(np.max(lons), np.max(lats), inverse=False)
        area_extent = (llLon, llLat, urLon, urLat)
        pixel_size = pixel_size
    if proj == "GEO":
        epsg_convert = pyproj.Proj("+proj=aeqd +lat_0={} +lon_0={}".format(midLat, midLon))
        llLon, llLat = epsg_convert(np.min(lons), np.min(lats), inverse=False)
        urLon, urLat = epsg_convert(np.max(lons), np.max(lats), inverse=False)

        area_extent = (llLon, llLat, urLon, urLat)
        cols = int(round((area_extent[2] - area_extent[0]) / pixel_size))
        rows = int(round((area_extent[3] - area_extent[1]) / pixel_size))
        epsg, proj_name, proj_fullname = '4326', 'longlat', 'Geographic'
        llLon, llLat, urLon, urLat = np.min(lons), np.min(lats), np.max(lons), np.max(lats)
        area_extent = (llLon, llLat, urLon, urLat)
        proj_dict = {'proj': proj_name, 'datum': 'WGS84'}
        area_def = geom.AreaDefinition(epsg, proj_fullname, proj_name, proj_dict, cols, rows, area_extent)
        pixel_size = np.min([area_def.pixel_size_x, area_def.pixel_size_y])  # Square pixels
    cols = int(round(float((area_extent[2] - area_extent[0]) / pixel_size)))  # Calculate the output cols
    rows = int(round(float((area_extent[3] - area_extent[1]) / pixel_size)))  # Calculate the output rows
    area_def = geom.AreaDefinition(epsg, proj_fullname, proj_name, proj_dict, cols, rows, area_extent)
    index, outdex, indexArr, distArr = kdt.get_neighbour_info(swath_def, area_def, 50000, neighbours=1)

    array_geo = kdt.get_sample_from_neighbour_info('nn', area_def.shape, array, index, outdex, indexArr, fill_value=fill_value)

    geotransform = [area_def.area_extent[0], pixel_size, 0, area_def.area_extent[3], 0, -pixel_size]
    height, width = array_geo.shape
    driver = gdal.GetDriverByName('GTiff')
    dtype = gdal_array.NumericTypeCodeToGDALTypeCode(array_geo.dtype)
    d = driver.Create(filename, width, height, 1, dtype)
    d.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(epsg))
    d.SetProjection(srs.ExportToWkt())
    srs.ExportToWkt()
    band = d.GetRasterBand(1)
    band.WriteArray(array_geo)
    if fill_value:
        band.SetNoDataValue(fill_value)
    band.FlushCache()
    d, band = None, None

def spectral_subset(infile, outfile, selected_bands, nodata=0):
    with rasterio.Env():
        with rasterio.open(infile) as src:
            profile = src.profile
            count = len(selected_bands)
            profile.update(count=count, tiled=True, nodata=nodata)
            with rasterio.open(outfile, "w", **profile) as dst:
                for sindex, dindex in enumerate(selected_bands):
                    dst.write(src.read(dindex), sindex + 1)

def spatial_subset(infile, outfile, window, nodata=0):
    with rasterio.Env():
        with rasterio.open(infile) as src:
            profile = src.profile
            profile.update(tiled=True, nodata=nodata, width=window.width, height=window.height)
            with rasterio.open(outfile, "w", **profile) as dst:
                for sindex, dindex in enumerate(src.count):
                    dst.write(src.read(dindex + 1, window=window), sindex + 1)

def save_array(array, filename):
    """ write array to GeoTiff file
	"""
    transform = rasterio.transform.from_origin(472137, 5015782, 0.5, 0.5)
    out_dataset = rasterio.open(filename, 'w', driver='GTiff',
           height = array.shape[1], width = array.shape[2],
           count=array.shape[0], dtype=str(array.dtype),
           crs='+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs',
           transform=transform)
    for iband in range(array.shape[0]):
        out_dataset.write(array[iband], iband+1)
    out_dataset.close()

def get_tiles(ncols, nrows, tile_width=None, tile_height=None, window_size=128):
    """ return tiles
	"""
    if not tile_width:
        tile_width = window_size
    if not tile_height:
        tile_height = window_size
    offsets = itertools.product(range(0, ncols, tile_width), range(0, nrows, tile_height))
    for col_s, row_s in offsets:
        col_e = col_s + tile_width if col_s + tile_width < ncols else ncols
        row_e = row_s + tile_height if row_s + tile_height < nrows else nrows
        yield (col_s, row_s, col_e, row_e)

def get_evenly_tiles(ncols, nrows, tile_width=None, tile_height=None, window_size=128):
    """ divide an image into evenly sized, overlapping if needed.
	"""
    np.seterr(divide='ignore', invalid='ignore')

    if not tile_width:
        tile_width = window_size

    if not tile_height:
        tile_height = window_size

    wTile = tile_width
    hTile = tile_height

    if tile_width > ncols or tile_height > nrows:
        raise ValueError("tile dimensions cannot be larger than origin dimensions")

    # Number of tiles
    nTilesX = np.uint8(np.ceil(ncols / wTile))
    nTilesY = np.uint8(np.ceil(nrows / hTile))

    # Total remainders
    remainderX = nTilesX * wTile - ncols
    remainderY = nTilesY * hTile - nrows

    # Set up remainders per tile
    remaindersX = np.ones((nTilesX-1, 1)) * np.uint16(np.floor(remainderX / (nTilesX-1)))
    remaindersY = np.ones((nTilesY-1, 1)) * np.uint16(np.floor(remainderY / (nTilesY-1)))
    remaindersX[0:np.remainder(remainderX, np.uint16(nTilesX-1))] += 1
    remaindersY[0:np.remainder(remainderY, np.uint16(nTilesY-1))] += 1

    # Initialize array of tile boxes
    tiles = np.zeros((nTilesX * nTilesY, 4), np.uint16)

    k = 0
    x = 0
    for i in range(nTilesX):
        y = 0
        for j in range(nTilesY):
            tiles[k, :] = (x, y, hTile, wTile)
            k += 1
            if j < (nTilesY-1):
                y = y + hTile - remaindersY[j]
        if i < (nTilesX-1):
            x = x + wTile - remaindersX[i]

    return tiles

DEFAULT_ROWS = 500
DEFAULT_COLS = 500
DEFAULT_PIXSIZE = 10
DEFAULT_DTYPE = gdal.GDT_Byte
DEFAULT_XLEFT = 500000
DEFAULT_YTOP = 7000000
DEFAULT_EPSG = 28355

def create_test_file(filename, numRows=DEFAULT_ROWS, numCols=DEFAULT_COLS,
    dtype=DEFAULT_DTYPE, numBands=1, epsg=28355, xLeft=DEFAULT_XLEFT,
    yTop=DEFAULT_YTOP, xPix=DEFAULT_PIXSIZE, yPix=DEFAULT_PIXSIZE,
    driverName='HFA', creationOptions=['COMPRESS=YES']):
    """
    Create a simple test file, on a standard footprint. Has some fairly arbitrary
    default values for all the relevant characteristics, which can be
    over-ridden as required. 
    
    Returns the dataset object. 
    
    """
    # Unless otherwise specified, use HFA driver, because it has lots of capabilities
    # we can test, and it is always a part of GDAL.
    driver = gdal.GetDriverByName(driverName)

    ds = driver.Create(filename, numCols, numRows, numBands, dtype, creationOptions)
    if ds is None:
        raise ImageOpenException('Cannot create an image')

    geotransform = (xLeft, xPix, 0, yTop, 0, -yPix)
    ds.SetGeoTransform(geotransform)

    sr = osr.SpatialReference()
    sr.ImportFromEPSG(epsg)
    projWKT = sr.ExportToWkt()
    ds.SetProjection(projWKT)

    return ds

def gen_ramp_array(nRows=DEFAULT_ROWS, nCols=DEFAULT_COLS):
    """
    Generate a simple 2-d linear ramp. Returns a numpy array of the data
    """
    (x, y) = np.mgrid[:nRows, :nCols]
    ramp = ((x + y) * 100.0 / (nRows-1 + nCols-1)).astype(np.uint8)
    return ramp

def gen_ramp_image_file(filename, reverse=False, xLeft=DEFAULT_XLEFT, yTop=DEFAULT_YTOP):
    """
    Generate a test image of a simple 2-d linear ramp. 
    """
    ds = create_test_file(filename, xLeft=xLeft, yTop=yTop)
    ramp = gen_ramp_array()
    if reverse:
        # Flip left-to-right
        ramp = ramp[:, ::-1]

    band = ds.GetRasterBand(1)
    band.WriteArray(ramp)
    del ds

def gen_thematic_file(filename):
    """
    Generate a thematic file
    """
    ds = create_test_file(filename)

    band = ds.GetRasterBand(1)
    arr = np.zeros((DEFAULT_ROWS, DEFAULT_COLS))
    band.WriteArray(arr)

    band.SetMetadataItem('LAYER_TYPE', 'thematic')
    del ds

def gen_vector_square(filename, epsg=DEFAULT_EPSG):
    """
    Generate a square, which would lie inside the rasters generated by the
    routines above.
    
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.CreateDataSource(filename)
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(epsg)

    layer = ds.CreateLayer(filename, geom_type=ogr.wkbPolygon, srs=sr)

    squareSize = 20
    xmin = DEFAULT_XLEFT + 10.6 * DEFAULT_PIXSIZE
    xmax = xmin + squareSize * DEFAULT_PIXSIZE
    ymin = DEFAULT_YTOP - 30.6 * DEFAULT_PIXSIZE
    ymax = ymin + squareSize * DEFAULT_PIXSIZE

    corners = [
        [xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin], [xmin, ymax]
    ]
    cornersStrList = ["%s %s"%(x, y) for (x, y) in corners]
    cornersStr = ','.join(cornersStrList)
    squareWKT = "POLYGON((%s))" % cornersStr
    geom = ogr.Geometry(wkt=squareWKT)
    featureDefn = ogr.FeatureDefn()
    feature = ogr.Feature(featureDefn)
    feature.SetGeometry(geom)
    layer.CreateFeature(feature)

    del layer
    del ds

# for GDAL command line utilities
DEFAULTDRIVERNAME = 'GTiff'
CMDLINECREATIONOPTIONS = []
if DEFAULTDRIVERNAME in dfltDriverOptions:
    for opt in dfltDriverOptions[DEFAULTDRIVERNAME]:
        CMDLINECREATIONOPTIONS.append('-co')
        CMDLINECREATIONOPTIONS.append(opt)

def transform_bounding_box(bounding_box, base_epsg, new_epsg, edge_samples=11):
    """Transform input bounding box to output projection.

    This transform accounts for the fact that the reprojected square bounding
    box might be warped in the new coordinate system.  To account for this,
    the function samples points along the original bounding box edges and
    attempts to make the largest bounding box around any transformed point
    on the edge whether corners or warped edges.

    Parameters:
        bounding_box (list): a list of 4 coordinates in `base_epsg` coordinate
            system describing the bound in the order [xmin, ymin, xmax, ymax]
        base_epsg (int): the EPSG code of the input coordinate system
        new_epsg (int): the EPSG code of the desired output coordinate system
        edge_samples (int): the number of interpolated points along each
            bounding box edge to sample along. A value of 2 will sample just
            the corners while a value of 3 will also sample the corners and
            the midpoint.

    Returns:
        A list of the form [xmin, ymin, xmax, ymax] that describes the largest
        fitting bounding box around the original warped bounding box in
        `new_epsg` coordinate system.
    """
    base_ref = osr.SpatialReference()
    base_ref.ImportFromEPSG(base_epsg)

    new_ref = osr.SpatialReference()
    new_ref.ImportFromEPSG(new_epsg)

    transformer = osr.CoordinateTransformation(base_ref, new_ref)

    p_0 = np.array((bounding_box[0], bounding_box[3]))
    p_1 = np.array((bounding_box[0], bounding_box[1]))
    p_2 = np.array((bounding_box[2], bounding_box[1]))
    p_3 = np.array((bounding_box[2], bounding_box[3]))

    def _transform_point(point):
        trans_x, trans_y, _ = (transformer.TransformPoint(*point))
        return (trans_x, trans_y)

    # This list comprehension iterates over each edge of the bounding box,
    # divides each edge into `edge_samples` number of points, then reduces
    # that list to an appropriate `bounding_fn` given the edge.
    # For example the left edge needs to be the minimum x coordinate so
    # we generate `edge_samples` number of points between the upper left and
    # lower left point, transform them all to the new coordinate system
    # then get the minimum x coordinate "min(p[0] ...)" of the batch.
    transformed_bounding_box = [
        bounding_fn([
            _transform_point(p_a * v + p_b * (1 - v))
            for v in np.linspace(0, 1, edge_samples)
        ]) for p_a, p_b, bounding_fn in [(
            p_0, p_1, lambda p_list: min([p[0] for p in p_list])
        ), (p_1, p_2, lambda p_list: min([p[1] for p in p_list])
            ), (p_2, p_3, lambda p_list: max([p[0] for p in p_list])
                ), (p_3, p_0, lambda p_list: max([p[1] for p in p_list]))]
    ]
    return transformed_bounding_box


def layer_stack(outFile, stackFiles):
    gdal_merge = find_executable("gdal_merge.py")
    if gdal_merge is None:
        msg = "Unable to find gdal_merge.py command. Check installation of GDAL package. "
        raise ValueError(msg)
    subprocess.check_call(
        [sys.executable, gdal_merge, '-q', '-of', DEFAULTDRIVERNAME] +
        CMDLINECREATIONOPTIONS + ['-separate', '-o', outFile] + stackFiles)

def plot_raster(file, band=1, nodata=None, cmap=cmaps.MPL_gist_gray):
    name = os.path.basename(file)
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(10, 10), dpi=300)
    src = rasterio.open(file)
    data = src.read(band)
    if nodata:
        data[data == nodata] = np.nan
    im = rasterio.plot.show(data, transform=src.transform, ax=ax, origin='upper', cmap=cmap)
    # cl = ax.coastlines()
    cb = plt.colorbar(im.images[0])
    plt.title(f'{name} band{band}')
    plt.get_current_fig_manager().full_screen_toggle()
    plt.plot()
    plt.show()

def CronvertRaster2LatLong(InputRasterFile,OutputRasterFile):

    """
    Convert a raster to lat long WGS1984 EPSG:4326 coordinates for global plotting

    MDH

    """
    # import modules
    import rasterio
    from rasterio.warp import reproject, calculate_default_transform as cdt, Resampling

    # read the source raster
    with rasterio.open(InputRasterFile) as src:
        #get input coordinate system
        Input_CRS = src.crs
        # define the output coordinate system
        Output_CRS = {'init': "epsg:4326"}
        # set up the transform
        Affine, Width, Height = cdt(Input_CRS,Output_CRS,src.width,src.height,*src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': Output_CRS,
            'transform': Affine,
            'affine': Affine,
            'width': Width,
            'height': Height
        })

        with rasterio.open(OutputRasterFile, 'w', **kwargs) as dst:
            for i in range(1, src.count+1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.affine,
                    src_crs=src.crs,
                    dst_transform=Affine,
                    dst_crs=Output_CRS,
                    resampling=Resampling.bilinear) 

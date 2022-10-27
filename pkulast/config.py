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


from concurrent.futures.process import _MAX_WINDOWS_WORKERS
import os
import getpass
import logging
import glob
import yaml
import tempfile
from datetime import datetime
from os.path import expanduser
from appdirs import AppDirs
try:
    # python 3.3+
    from collections.abc import Mapping
except ImportError:
    # deprecated (above can't be done in 2.7)
    from collections import Mapping
import pkg_resources

try:
    from yaml import UnsafeLoader
except ImportError:
    from yaml import Loader as UnsafeLoader


LOG = logging.getLogger(__name__)


os.environ['WINEDEBUG']="-all" # avoid wine output error

if os.environ.get('JUPYTERHUB_USER') is None:
    os.environ['JUPYTERHUB_USER'] = getpass.getuser()


BUILTIN_CONFIG_FILE = pkg_resources.resource_filename(__name__,
                                                      os.path.join('etc', 'config.yaml'))



ALG_FILE = pkg_resources.resource_filename(__name__,
                                                      os.path.join('etc', 'algorithm.yaml'))

CONFIG_FILE = os.environ.get('PKULAST_CONFIG')

if CONFIG_FILE is not None and (not os.path.exists(CONFIG_FILE) or
                                not os.path.isfile(CONFIG_FILE)):
    raise IOError(
        str(CONFIG_FILE) + " pointed to by the environment " +
        "variable PKULAST_CONFIG is not a file or does not exist!")


LOG = logging.getLogger(__name__)

def recursive_dict_update(d, u):
    """Recursive dictionary update.

    Copied from:

        http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            r = recursive_dict_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


def get_config():
    """Get the configuration from file."""
    if CONFIG_FILE is not None:
        configfile = CONFIG_FILE
    else:
        configfile = BUILTIN_CONFIG_FILE

    config = {}
    with open(configfile, 'r', encoding='utf-8') as fp_:
        config = recursive_dict_update(config, yaml.load(fp_, Loader=UnsafeLoader))

    app_dirs = AppDirs('pkulast', 'pkutir')
    user_datadir = app_dirs.user_data_dir
    config['tmp_dir'] = expanduser(config.get('tmp_dir', user_datadir))
    return config


CFG = get_config()


# ALG details
with open(ALG_FILE, 'r', encoding='utf-8') as fp_:
    ALG_CFG = recursive_dict_update({}, yaml.load(fp_, Loader=UnsafeLoader))

SW_ALG = list(ALG_CFG['SW'].keys())
SC_ALG = list(ALG_CFG['SC'].keys())

# DEM dir
DEM_DIR = os.path.join(CFG['extra'], 'dem/GMTED2km.tif')


LOCAL_RSR_DIR = CFG.get('rsr_dir')

try:
    os.makedirs(LOCAL_RSR_DIR)
except OSError:
    if not os.path.isdir(LOCAL_RSR_DIR):
        raise

TB2RAD_DIR = CFG.get('tb2rad_dir', tempfile.gettempdir())


# aerosol types
AEROSOL_TYPES = ['antarctic_aerosol', 'continental_average_aerosol',
                 'continental_clean_aerosol', 'continental_polluted_aerosol',
                 'desert_aerosol', 'marine_clean_aerosol',
                 'marine_polluted_aerosol', 'marine_tropical_aerosol',
                 'rayleigh_only', 'rural_aerosol', 'urban_aerosol']

# relative spectral response

RSR_LIB = []
ENVI_RSR_LIB = []
HDF5_RSR_LIB = []
NWP_RSR_LIB = []
USER_RSR_LIB = []

get_rsr_name = lambda filename: os.path.basename(os.path.splitext(filename)[0])

ENVI_RSR_DIR = LOCAL_RSR_DIR + '/ENVI/'
HDF5_RSR_DIR = LOCAL_RSR_DIR + '/HDF5/'
NWP_RSR_DIR = LOCAL_RSR_DIR + '/NWP/'
USER_RSR_DIR = LOCAL_RSR_DIR + '/USER/'
# ENVI rsr library
for rsr in glob.glob(ENVI_RSR_DIR + '*.sli'):
    key = get_rsr_name(rsr)
    ENVI_RSR_LIB.append(key)

# HDF5 rsr library
for rsr in glob.glob(HDF5_RSR_DIR + '*.h5'):
    _, sensor, platform =get_rsr_name(rsr).split('_')
    HDF5_RSR_LIB.append(f'{sensor}_{platform}')

# NWP rsr library
with open(NWP_RSR_DIR + 'nwp.yaml', 'r', encoding="utf-8") as f:
    data = yaml.load(f.read(), Loader=yaml.FullLoader)
    NWP_RSR_CONFIG = dict(data)
NWP_RSR_LIB = list(NWP_RSR_CONFIG.keys())

# USER rsr library
for __rsr in glob.glob(USER_RSR_DIR + '*.flt'):
    key = get_rsr_name(__rsr)
    USER_RSR_LIB.append(key)

# overall rsr library
RSR_LIB = ENVI_RSR_LIB + HDF5_RSR_LIB + NWP_RSR_LIB + USER_RSR_LIB
RSR_LIB.sort(key=lambda x: x.lower().strip())


EXTENT_FACTOR = 5
MIN_TMP = 173
MAX_TMP = 373

# MIN_TMP = 100
# MAX_TMP = 1000
MAX_STEPS = int((MAX_TMP - MIN_TMP) / 0.1) + 1
DIFF_LEVEL = 2   # differential level

# atmosphere

PROFILE_DIR = CFG['extra'] + 'atmosphere/'
UCAR_USERNAME = CFG['ucar_username']
UCAR_PASSWORD = CFG['ucar_password']
UCAR_LOGIN_URL = 'https://rda.ucar.edu/cgi-bin/login'
UCAR_PARAMS = {'email' : UCAR_USERNAME, 'passwd' : UCAR_PASSWORD, 'action' : 'login'}

Third_Party_DIR = CFG['extra'] + 'third_party/'
PROFILE_CONFIG_FILE = PROFILE_DIR + 'profiles.yaml'

# with open(PROFILE_CONFIG_FILE, 'r', encoding='utf-8') as f:
#     data = yaml.load(f.read(), Loader=yaml.FullLoader)
#     NWPConfiguration = dict(data)['NWP_PROFILE']

def get_nwp_config():
    with open(PROFILE_CONFIG_FILE, 'r', encoding='utf-8') as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
        NWPConfiguration = dict(data)['NWP_PROFILE']
    return NWPConfiguration

# PROFILE_TYPES = list(NWPConfiguration.keys())

WGRIB_EXE = Third_Party_DIR + 'wgrib/wgrib.exe'

WGRIB2_EXE = Third_Party_DIR + 'grib2/wgrib2/wgrib2'

WGET_EXE = Third_Party_DIR + 'wget/wget.exe'

ClearSkyTIGR946_FILE = PROFILE_DIR + 'tigr/TIGR_atmProfilesSelection.txt'

###########################################################################################################
# 1. NWP Configuration

GFS_DIR = PROFILE_DIR + 'gfs/'
ERA5_DIR = PROFILE_DIR + 'era5/'
CRA40_DIR = PROFILE_DIR + 'cra40/'
GDAS_DIR = PROFILE_DIR + 'gdas/'
GDAS25_DIR = PROFILE_DIR + 'gdas25/'
DOE_DIR = PROFILE_DIR + 'doe/'
MERRA2_DIR = PROFILE_DIR + 'merra2/'
JRA55_DIR = PROFILE_DIR + 'jra55/'
CFSv2_DIR = PROFILE_DIR + 'cfsv2/'
# NCEI related

NCEI_GFSANL4_URL = "https://www.ncei.noaa.gov/data/global-forecast-system/access/grid-004-0.5-degree/analysis/"
NCEI_GFSANL4_URL_HIS = "https://www.ncei.noaa.gov/data/global-forecast-system/access/historical/analysis/"

# AIRS related
AIRS_HOST = "https://www.ncdc.noaa.gov"
AIRS_ORDER_URL = "https://www.ncdc.noaa.gov/has/HAS.FileSelect"
AIRS_GFSANL4_URL = "https://www.ncdc.noaa.gov/has/HAS.FileAppRouter?datasetname=GFSANL4&subqueryby=STATION&applname=&outdest=FILE"
AIRS_CHECK_STATUS_URL = 'https://www.ncdc.noaa.gov/has/{}'
AIRS_GFSANL4_STATIONS = ['00', '06', '12', '18']
AIRS_GFSANL4_BODY = {
    'satdisptype': 'N/A',
    'stations': '',
    'station_lst': '',
    'typeofdata': 'MODEL',
    'dtypelist': '',
    'begdatestring': '',
    'enddatestring': '',
    'begyear': '',
    'begmonth': '',
    'begday': '',
    'beghour': '',
    'begmin': '',
    'endyear': '',
    'endmonth': '',
    'endday': '',
    'endhour': '',
    'endmin': '',
    'outmed': 'FTP',
    'outpath': '',
    'pri': '500',
    'datasetname': 'GFSANL4',
    'directsub': 'Y',
    'emailadd': 'test@about.com',
    'outdest': 'FILE',
    'applname': '',
    'subqueryby': 'STATION',
    'tmeth': 'Awaiting-Data-Transfer',
  }
AIRS_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Host': 'www.ncdc.noaa.gov',
    'Origin': 'https://www.ncdc.noaa.gov',
    'Referer': AIRS_GFSANL4_URL,
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
  }

# CMA related
CMA_URL = "http://data.cma.cn/en"
CMA_HEADERS = {
    'Accept':
    'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Encoding':
    'gzip, deflate, br',
    'Accept-Language':
    'zh-CN,zh;q=0.9',
    'Cache-Control':
    'max-age=0',
    'Connection':
    'keep-alive',
    'Content-Type':
    'application/x-www-form-urlencoded',
    'Host':
    'data.cma.cn',
    'Origin':
    CMA_URL,
    'Referer':
    CMA_URL,
    'Sec-Fetch-Dest':
    'document',
    'Sec-Fetch-Mode':
    'navigate',
    'Sec-Fetch-Site':
    'same-origin',
    'Sec-Fetch-User':
    '?1',
    'Upgrade-Insecure-Requests':
    '1',
    'User-Agent':
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
}

###########################################################################################################
# 2. Seeborv5.1 Configuration
# http://cimss.ssec.wisc.edu/training_data/

# Record number:15,704
# Record length: 338
# Datatype: real*4

# RECORD FIELDS
# 1:101  temperature profile   [K]
# 102:202 mixing ratio profile  [kg/kg]
# 203:303 ozone profile            [ppmv]
# 304 latitude
# 305 longitude
# 306 surface pressure [hPa]
# 307 skin temperature [K]
# 308 wind speed (m/s)
#     - value used for finding seawater emissivity
#       (equal to 1 2 3 4 5 6 7 8 9 10 12.5 or 15)
#        For Land, this field is -999
# 309 tpw [cm]
# 310 ecosystem, igbp classification
# 311 elevation [m]
# 312 fraction land
# 313 year
# 314 month
# 315 day
# 316 hour
# 317 profile type
#     1 NOAA - 88b    2 TIGR - 3   3 Radiosondes    4 Ozonesondes   5 ECMWF
# 318:327 frequency (wavenumber) of emissivity at  10 BF emis hinge points
# 328:337 emissivity spectra
# 338 spare

# 101 levels
Seebor_PRESSURE_LEVEL = [
0.005,.016,.038,.077,.137,.224,.345,.506,.714,
.975,1.297,1.687,2.153,2.701,3.340,4.077,4.920,
5.878,6.957,8.165,9.512,11.004,12.649,14.456,16.432,
18.585,20.922,23.453,26.183,29.121,32.274,35.651,39.257,
43.100,47.188,51.528,56.126,60.989,66.125,71.540,77.240,
83.231,89.520,96.114,103.017,110.237,117.777,125.646,133.846,
142.385,151.266,160.496,170.078,180.018,190.320,200.989,212.028,
223.441,235.234,247.408,259.969,272.919,286.262,300.000,314.137,
328.675,343.618,358.966,374.724,390.893,407.474,424.470,441.882,
459.712,477.961,496.630,515.720,535.232,555.167,575.525,596.306,
617.511,639.140,661.192,683.667,706.565,729.886,753.628,777.790,
802.371,827.371,852.788,878.620,904.866,931.524,958.591,986.067,
1013.948,1042.232,1070.917,1100.000
]

Seebor_FILENAME = PROFILE_DIR + "seebor/SeeBorV5.1_Training_data_Emis10inf2004.bin"

Seebor_TOTALCOUNT = 15704

Seebor_NLEVEL = 101

# NOTES:
# Number of NOAA88 profiles:
# 6137
# Number of TIGR-3 profiles:
# 1387
# Number of Radiosonde profiles:
# 570
# Number of Ozonesonde profiles:
# 1595
# Number of ECMWF profiles:
# 6015
# Total number profiles:
# 15704
# •
# emissivity spectra is derived from UW-Madison Global Gridded IR Emissivity Dataset
# (http://cimss.ssec.wisc.edu/iremis/)
# •
# saturation criteria for clear sky profile selection is RH 99 %.

###########################################################################################################
#3. Thermodynamic Initial Guess Retrieval(TIGR) dataset
#
"""
The format is as follows :
	iatm,ilon,ilat,idate
	(T(nl),nl=1,nlevel)
	Ts,Ps
	(RO_h2o(nl),nl=1,nlevel)
	(RO_o3(nl),nl=1,nlevel)
With :
	• ilat=latitude*100 (integer)
	• ilon=longitude*100 (integer)
	• idate=date of the profile (yymmdd) when dd is not defined ? dd = 00
	• T=temperature (°K)
	• RO_h2o=h2o mass mixing ratio (g/g)
	• RO_o3=ozone mass mixing ratio (g/g)
	• Ps=Surface pressure (hPa)
	• Ts=Surface Temperature (°K)
NB :
	• The surface temperature is set equal to the temperature of the "nlevel_th" level (43rd
	in the current case)
	• The surface pressure is equal to 1013.00 hPa
"""
# Atmoshphere section
TROPICAL = [1, 872]
MID_LAT = [873, 1614]
POLAR = [1615, 2311]

# Atmoshphere subsection
MID_LAT1 = [873, 1260]
MID_LAT2 = [1261, 1614]
POLAR1 = [1615, 1718]
POLAR2 = [1719, 2311]

# Pressure/Temperature/Water vapour/Ozone

TIGR_TOTALCOUNT = 2311

# Version 1.2
NLEVEL = 43
PRESSURE_N_LEVELS = [2.6e-3,8.9e-3,2.4e-2,
  0.5E-01,0.8999997E-01,0.17E+00,0.3E+00,0.55E+00,
  0.1E+01,0.15E+01,0.223E+01,0.333E+01,0.498E+01,
  0.743E+01,0.1111E+02,0.1660001E+02,0.2478999E+02,0.3703999E+02,
  0.4573E+02,0.5646001E+02,0.6971001E+02,0.8607001E+02,0.10627E+03,
  0.1312E+03,0.16199E+03,0.2E+03,0.22265E+03,0.24787E+03,
  0.27595E+03,0.3072E+03,0.34199E+03,0.38073E+03,0.4238501E+03,
  0.4718601E+03,0.525E+03,0.5848E+03,0.65104E+03,0.72478E+03,
  0.8E+03,0.8486899E+03,0.9003301E+03,0.9551201E+03,0.1013E+04]

TIGR_FILENAME = PROFILE_DIR + "tigr/atm4atigr2000_v1.2_43lev.dsf"

FIRST_LINE = 0
SECOND_LINE = 1
THIRD_LINE = 9
FOURTH_LINE = 10
FIFTH_LINE = 18
TOTAL_LINE = 26



# Version 1.1

# NLEVEL = 40
# PRESSURE_N_LEVELS = [0.5E-01,0.8999997E-01,0.17E+00,0.3E+00,0.55E+00,
#      0.1E+01,0.15E+01,0.223E+01,0.333E+01,0.498E+01,
#      0.743E+01,0.1111E+02,0.1660001E+02,0.2478999E+02,0.3703999E+02,
#      0.4573E+02,0.5646001E+02,0.6971001E+02,0.8607001E+02,0.10627E+03,
#      0.1312E+03,0.16199E+03,0.2E+03,0.22265E+03,0.24787E+03,
#      0.27595E+03,0.3072E+03,0.34199E+03,0.38073E+03,0.4238501E+03,
#      0.4718601E+03,0.525E+03,0.5848E+03,0.65104E+03,0.72478E+03,
#      0.8E+03,0.8486899E+03,0.9003301E+03,0.9551201E+03,0.1013E+04]

# TIGR_FILENAME = BASEDIR + "/tigr/atm4atigr2000_v1.1.dsf"


# FIRST_LINE = 0
# SECOND_LINE = 1
# THIRD_LINE = 8
# FOURTH_LINE = 9
# FIFTH_LINE = 16
# TOTAL_LINE = 23


TYPE = ["Pressure", "Temperature", "Water vapour", "Ozone"]

STANDARD_ATMOSPHERES_DIR = PROFILE_DIR + 'standard/'


ICRCCM_STANDARD_ATMOSPHERES_DIR = STANDARD_ATMOSPHERES_DIR + 'ICRCCM/'
ICRCCM_STANDARD_ATMOSPHERES = {
    'subarctic summer': 'sas.atm',
    'subarctic winter': 'saw.atm',
    'midlatitude summer': 'mls.atm',
    'midlatitude winter': 'mlw.atm',
    'tropical': 'tro.atm',
    'us-standard 1976': 'std.atm'
}
ENVI_STANDARD_ATMOSPHERES_DIR = STANDARD_ATMOSPHERES_DIR + 'ICRCCM_ENVI/'
ENVI_STANDARD_ATMOSPHERES = {
    'subarctic summer': 'SubArc_Summer.dat',
    'subarctic winter': 'SubArc_Winter.dat',
    'midlatitude summer': 'MidLat_Summer.dat',
    'midlatitude winter': 'MidLat_Winter.dat',
    'tropical': 'Tropical.dat',
    'us-standard 1976': 'US_Std_1976.dat'
}

# radiative transfer model

# RTTOV13
RTM_RTTOV13_DIR = CFG['extra'] + 'rtm/RTTOV13/'

# MODTRAN
RTM_MODTRAN_DIR = CFG['extra'] + 'rtm/MODTRAN/'
TP_DIR = RTM_MODTRAN_DIR + 'tape/'

# for multi-users
TP_DIR = os.path.join(TP_DIR, os.getenv('JUPYTERHUB_USER'))

if not os.path.exists(TP_DIR):
    try:
        original_umask = os.umask(0)
        os.makedirs(TP_DIR, 0o777)
    finally:
        os.umask(original_umask)
UWARD_ID = 'uward'
DWARD_ID = 'dward'
TRANS_ID = 'trans'
SIMUL_ID = 'simul'
UWARD_DIR = os.path.join(TP_DIR, UWARD_ID)
DWARD_DIR = os.path.join(TP_DIR, DWARD_ID)
TRANS_DIR = os.path.join(TP_DIR, TRANS_ID)
SIMUL_DIR = os.path.join(TP_DIR, SIMUL_ID)
SCALAR = 1E7

INTERPOLATION_METHODS = ['nearest', 'linear']

# test_data
TEST_DATA_DIR = CFG['extra'] + 'test_data/'

# emissivity

SPEC_ALB_FILE = RTM_MODTRAN_DIR + 'DATA/spec_alb.dat'
SPEC_ALB_BK_FILE = RTM_MODTRAN_DIR + 'DATA/spec_alb_bk.dat'

ASTER_HEADER = 26
ASTER_FOOTER = 1

PRN_HEADER = 22
PRN_FOOTER = 1

SPECTRAL_LIBRARY = [
 'ASTER',
 'USGS',
 'ECOSTRESS',
 'RELAB'
]

SURFACE_DIR = CFG['extra'] + 'surface/'
SPECTRA_DIR = SURFACE_DIR + 'spectral_library/'
USGS_FILE = SPECTRA_DIR + 'usgs.db'
ASTER_FILE = SPECTRA_DIR + 'aster.db'
ECOSTRESS_FILE = SPECTRA_DIR + 'ecostress.db'
RELAB_FILE = SPECTRA_DIR + 'relab.db'
RELAB_ARCHIVE = SPECTRA_DIR + "RelabDatabase2021Jun30.zip"

################################################################
# ECOSTRESS Spectral Library Header Information
# The numbering donates which line the metadata values can be found
# 1. Name
# 2. Type
# 3. Class
# 4. Genus
# 5. species
# 6. Sample No.
# 7. Owner
# 8. Wavelength Range
# 9. Origin
# 10. Collection Date
# 11. Description
# 12. Measurement
# 13. First Column
# 14. Second Column
# 15. X Units
# 16. Y Units
# 17. First X Value
# 18. Last X Value
# 19. Number of X Values
# 20. Additional Information: if value is TRUE this cateogory will be set to the ancillary text file name
# 21. Empty Space
# 22. List of Spectra Start on this line
################################################################
EMIS_LIB_70 = SURFACE_DIR + 'selected_emissivity/'
EMIS_LIB_290 = SURFACE_DIR + 'selected_emissivity_290/'
DEFAULT_EMIS_LIB = EMIS_LIB_70

# utilities
UTILS_DIR = CFG['extra'] + 'utils/'
LPDAAC_DatasetNameList = UTILS_DIR + 'LPDAAC_DatasetNameList.csv'
LPDAAC_NearRealTime = 14
CHINA_REGION = UTILS_DIR + 'region/China.geojson'


# validation

#######################################################
# Site Configuration
#######################################################
VALIDATION_DIR = CFG['extra'] + 'validation/'

validation_sites_yaml =  VALIDATION_DIR + 'validation_sites.yml'

def validation_config():
    with open(validation_sites_yaml, 'r', encoding="utf-8") as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
        configuration = dict(data)
    return configuration


Configuration = validation_config()
#######################################################
# Validation Net List
#######################################################
# 1. SURFRAD
#######################################################
SURFRAD_DESC= '''
	"bon" is the station identifier for Bondville, Illinois
	"fpk" is the station identifier for Fort Peck, Montana
	"gwn" is the station identifier for Goodwin Creek, Mississippi
	"tbl" is the station identifier for Table Mountain, Colorado
	"dra" is the station identifier for Desert Rock, Nevada 
	"psu" is the station identifier for Penn State, Pennsylvania 
	"sxf" is the station identifier for Sioux Falls, South Dakota
	"tbl" is the station identifier for Boulder_CO
				'''
SURFRAD_DIR = VALIDATION_DIR + 'SURFRAD/'
SURFRAD_URL = 'ftp://aftp.cmdl.noaa.gov/data/radiation/surfrad/'
SURFRAD_PROJECT_NAME = "(NOAA's) SURFace RADiation budget network"
SURFRAD_SITES = list(Configuration['Validation']['SURFRAD']['SITE'].keys())
SURFRAD_UPDATED_DATE = datetime.strptime("2009-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
#######################################################
# 2. PKULSTNet
#######################################################
PKULSTNet_DESC= '''
	"hbc" is the station identifier for 承德站
	"hnw" is the station identifier for 海南站
	"hnh" is the station identifier for 鹤壁站
	"xzl" is the station identifier for 林芝站
	"imb" is the station identifier for 内蒙古站
	"lnp" is the station identifier for 盘锦站
	"xjf" is the station identifier for 新疆站
	"cqb" is the station identifier for 重庆站
				'''
PKULSTNet_DIR = VALIDATION_DIR + 'PKULSTNet/'
PKULSTNet_PROJECT_NAME = "北京大学地表温度验证网"
PKULSTNet_SITES = list(Configuration['Validation']['PKULSTNet']['SITE'].keys())
PKULSTNet_INTERVAL = 1
#######################################################
# 3. HiWATER
#######################################################
HiWATER_DESC = '''
	"zyz" is the station identifier for 张掖湿地站
	"ykz" is the station identifier for 垭口站
	"jyl" is the station identifier for 景阳岭站
	"hhl" is the station identifier for 混合林站
	"hmz" is the station identifier for 荒漠站
	"hzz" is the station identifier for 花寨子荒漠站
	"hhz" is the station identifier for 黑河遥感站
	"dsl" is the station identifier for 大沙龙站
	"sdq" is the station identifier for 四道桥超级站
	"dmz" is the station identifier for 大满超级站
	"arz" is the station identifier for 阿柔超级站
				'''
HiWATER_DIR = VALIDATION_DIR + 'HiWATER/'
HiWATER_PROJECT_NAME = "黑河流域地表过程综合观测网"
HiWATER_SITES = list(Configuration['Validation']['HiWATER']['SITE'].keys())
HiWATER_INTERVAL = 0
#######################################################
# Validation nets and sites
#######################################################
AVAILABLE_NETS = list(Configuration['Validation'].keys())
AVAILAVBLE_SITES = SURFRAD_SITES + PKULSTNet_SITES + HiWATER_SITES

# parallel

NUM_WORKERS = 12
BLOCK_SIZE = 128


# MODTRAN convolve resolution
Modtran_Sampling_Resolution = 15 # cm^-1

# MODTRAN parallel directory
Modtran_Parallel_Dir = RTM_MODTRAN_DIR + 'parallel/'
CHUNK_SIZE = 6
MAX_WORKERS = 12 # _MAX_WINDOWS_WORKERS
# 3 12

# debug&logging

def debug_on():
    """Turn debugging logging on."""
    logging_on(logging.DEBUG)


_is_logging_on = False


def logging_on(level=logging.WARNING):
    """Turn logging on."""
    global _is_logging_on

    if not _is_logging_on:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("[%(levelname)s: %(asctime)s :"
                                               " %(name)s] %(message)s",
                                               '%Y-%m-%d %H:%M:%S'))
        console.setLevel(level)
        logging.getLogger('').addHandler(console)
        _is_logging_on = True

    log = logging.getLogger('')
    log.setLevel(level)
    for h in log.handlers:
        h.setLevel(level)


class NullHandler(logging.Handler):
    """empty handler."""

    def emit(self, record):
        """Record a message."""


def logging_off():
    """turn logging off."""
    logging.getLogger('').handlers = [NullHandler()]


def get_logger(name):
    """Return logger with null handle."""
    log = logging.getLogger(name)
    if not log.handlers:
        log.addHandler(NullHandler())
    return log

# if __name__ == '__main__':
#    config = get_config()
#    print(config)
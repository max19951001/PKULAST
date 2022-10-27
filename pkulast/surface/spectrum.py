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
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams["font.size"] = 14

from pkulast.surface.spectral import AsterDatabase, USGSDatabase, EcostressDatabase, RelabDatabase
from pkulast.config import RELAB_ARCHIVE, RELAB_FILE, SPECTRA_DIR, ASTER_FILE, USGS_FILE, ECOSTRESS_FILE, SPEC_ALB_FILE, SPEC_ALB_BK_FILE


class Spectra(object):
    ''' Spectra
	'''
    def __init__(self, x, y, number, name, unit='um'):
        self.x = x
        self.y = y
        self.number = number
        self.name = name
        self.unit = unit

    def __rsub__(self, scalar):
        self.x = scalar - self.x
        self.y = scalar - self.y

    def __sub__(self, scalar):
        self.x -= scalar
        self.y -= scalar

    def __add__(self, scalar):
        self.x += scalar
        self.y += scalar

    def plot(self):
        plt.figure(figsize=[12, 10])
        plt.plot(self.x, self.y, color='r', label=self.name)
        plt.title(f'Spectra {self.number}: {self.name}')
        plt.xlabel(f'wavelength({self.unit})')
        plt.ylabel('Spectra')
        plt.legend()
        plt.show()


class SpectralAlbedo(object):
    """ spectral albedo
	"""
    def __init__(self, spec_alb=SPEC_ALB_FILE):
        self.spec_alb = spec_alb
        self.spectrum = []
        self._load_spec_alb()

    @property
    def count(self):
        return len(self.spectrum)

    def print_summary(self):
        for spec in self.spectrum:
            print("|".join((spec.number, spec.name)))


    def restore(self):
        shutil.copyfile(SPEC_ALB_BK_FILE, SPEC_ALB_FILE)
        self.clear()
        self._load_spec_alb()

    def save(self, filename=SPEC_ALB_FILE):
        with open(filename, 'w') as f:
            f.write('! Table of Contents\n')
            toc = ''
            content = ''
            for id, spec in enumerate(self.spectrum):
                toc += f'! {id:>6d}  "{spec.name:>10s}"\n'
                content += f' {id}  {spec.name}\n'
                content += '\n'.join([
                 f'{_x:>8.5f}     {_y:>8.5f}'
                 for _x, _y in zip(spec.x, spec.y)
                ])
                content += '\n!\n'
            f.write(toc)
            f.write(content)

    def clear(self):
        self.spectrum = []

    def del_spec_alb(self, id):
        self.spectrum.remove(id)

    def pop_spec_alb(self, index=-1):
        if index == -1:
            self.spectrum.pop()
        else:
            self.spectrum.pop(index)

    def add_spec_alb(self, x, y, name):
        if not np.array_equal(sorted(x), x):
            x = np.array(list(reversed(x)), dtype=np.float64)
            y = np.array(list(reversed(y)), dtype=np.float64)
        x, indices = np.unique(x, return_index=True)
        y = y[indices]
        with open(self.spec_alb, 'a') as f:
            f.write('!\n')
            f.write(f' {self.count+1} 	{name}\n')
            for wv, alb in zip(x, y):
                f.write(f'{wv:>8.5f} 	{alb:>8.5f}\n')
            self.spectrum.append(Spectra(x, y, self.count + 1, name))
        return -self.count

    def _load_spec_alb(self):
        if os.path.exists(self.spec_alb):
            with open(self.spec_alb, 'r') as f:
                lines = f.readlines()
                sig = False
                wvs = []
                albs = []
                sample_number = 1
                sample_name = ''
                for line in lines:
                    if line.strip() == '':
                        continue
                    if line.strip()[0] == '!':
                        continue
                    else:
                        if '.' not in line:
                            if not sig:
                                sig = True
                            else:
                                self.spectrum.append(
                                 Spectra(np.array(wvs), np.array(albs),
                                   sample_number, sample_name))
                                wvs = []
                                albs = []
                                sample_number += 1
                                # print(line)
                            header = line.split('!')[0].strip().split()
                            # sample_number = header[0]
                            sample_name = '_'.join(header[1:])
                        else:
                            wv, alb = line.strip().split()
                            wvs.append(float(wv))
                            albs.append(float(alb))
                if sig:
                    self.spectrum.append(
                        Spectra(np.array(wvs), np.array(albs),
                        sample_number, sample_name))
        else:
            with open(self.spec_alb, 'a'):
                os.utime(self.spec_alb, None)

    def __getitem__(self, index):
        if abs(index) >= self.count:
            raise ValueError(f'index{index} out of range {self.count}')
        return self.spectrum[index]

    def __len__(self):
        return self.count

    def __iter__(self):
        self.current_index = -1
        return self

    def __next__(self):
        self.current_index += 1
        if self.current_index >= self.count:
            raise StopIteration
        return self.spectrum[self.current_index]


class SpectralLibrary(object):
    """ spectral library.
	"""
    def __init__(self, SPEC_LIB='ASTER'):
        self.spec_lib = SPEC_LIB
        self._load_database()

    def _load_database(self):
        self.add_args = ''
        if self.spec_lib == 'ASTER':
            if not os.path.exists(ASTER_FILE):
                AsterDatabase.create(ASTER_FILE, SPECTRA_DIR + 'ASTER')
            self.database = AsterDatabase(ASTER_FILE)
        elif self.spec_lib == 'ECOSTRESS':
            if not os.path.exists(ECOSTRESS_FILE):
                EcostressDatabase.create(ECOSTRESS_FILE,
                 SPECTRA_DIR + 'ECOSTRESS')
            self.database = EcostressDatabase(ECOSTRESS_FILE)
        elif self.spec_lib == 'USGS':
            if not os.path.exists(USGS_FILE):
                USGSDatabase.create(USGS_FILE, SPECTRA_DIR + 'USGS/ASCIIdata')
            self.database = USGSDatabase(USGS_FILE)
            self.add_args = 'AND LibName="splib07b"'
        elif self.spec_lib == 'RELAB':
            if not os.path.exists(RELAB_FILE):
                RelabDatabase.create(RELAB_FILE, RELAB_ARCHIVE)
            self.database = RelabDatabase(RELAB_FILE)
            # self.add_args = 'AND LibName="splib07b"'
        else:
            self.database = None

    @property
    def count(self):
        sql = 'SELECT COUNT() FROM Samples'
        ret = self.query(sql)
        return ret[0][0]

    def contains(self, name, limit=1000):
        sql = f'SELECT SampleID, Description FROM Samples WHERE Description LIKE "%{name}%" {self.add_args} limit {limit}'
        return self.query(sql)

    def get_type(self):
        if self.spec_lib in  ['ASTER', 'ECOSTRESS']:
            sql = "SELECT DISTINCT LOWER(Type) FROM Samples"
            results = self.query(sql)
            return [item[0] for item in results]
        else:
            raise NotImplementedError(f"Not NotImplemented for spectral library {self.spec_lib}")

    def get_class(self, type_='manmade'):
        if self.spec_lib in  ['ASTER', 'ECOSTRESS']:
            sql = f'SELECT DISTINCT LOWER(Class) FROM Samples WHERE Type LIKE "%{type_}%" '
            results = self.query(sql)
            return [item[0] for item in results]
        else:
            raise NotImplementedError(f"Not NotImplemented for spectral library {self.spec_lib}")

    def filter_type(self, type_, limit=1000):
        if self.spec_lib in  ['ASTER', 'ECOSTRESS']:
            sql = f'SELECT SampleID, Name, LOWER(Type), LOWER(Class), SubClass, Description FROM Samples WHERE LOWER(Type) LIKE "%{type_.lower()}%" {self.add_args} limit {limit}'
            return self.query(sql)
        else:
            raise NotImplementedError(f"Not NotImplemented for spectral library {self.spec_lib}")

    def filter_class(self, class_, limit=1000):
        if self.spec_lib in  ['ASTER', 'ECOSTRESS']:
            sql = f'SELECT SampleID, Name, LOWER(Type), LOWER(Class), SubClass, Description FROM Samples WHERE LOWER(Class) LIKE "%{class_.lower()}%" {self.add_args} limit {limit}'
            return self.query(sql)
        else:
            raise NotImplementedError(f"Not NotImplemented for spectral library {self.spec_lib}")

    def read_file(self, filename):
        return self.database.read_file(filename)

    def query(self, sql):
        cursor = self.database.query(sql)
        return cursor.fetchall()

    def close(self):
        self.database.close()

    def print_query(self, sql):
        self.database.print_query(sql)

    def get_description(self, id):
        return self.database.get_description(id)

    def get_sample(self, id):
        return self.database.get_sample(id)

    def get_signature(self, id):
        return self.database.get_signature(id)

    def get_name(self, id):
        return self.database.get_name(id)

    def get_spectrum(self, id):
        return self.database.get_spectrum(id)

    def plot(self, id):
        x, y = self.get_spectrum(id)
        plt.plot(x, y)
        plt.xlabel('wavelength($\mu m$)')
        plt.ylabel('reflectance(%)')
        plt.title(self.get_name(id))
        plt.show()

    def create_envi_spectral_library(self, spectrumIDs, bandInfo):
        self.database.create_envi_spectral_library(spectrumIDs, bandInfo)

    def __getitem__(self, id):
        if id < 1 or id > self.count:
            raise ValueError(f'id {id} not in range 1 to {self.count}')
        return self.get_spectrum(id)

    def __len__(self):
        return self.count

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        self._index += 1
        if self._index > self.count:
            raise StopIteration
        return self.get_spectrum(self._index)


# USGS
#     'CREATE TABLE Samples (SampleID INTEGER PRIMARY KEY, LibName TEXT, Record INTEGER, '
#     'Description TEXT, Spectrometer TEXT, Purity TEXT, MeasurementType TEXT, Chapter TEXT, FileName TEXT, '
#     'AssumedWLSpmeterDataID INTEGER, '
#     'NumValues INTEGER, MinValue FLOAT, MaxValue FLOAT, ValuesArray BLOB)',

# 	  'CREATE TABLE SpectrometerData (SpectrometerDataID INTEGER PRIMARY KEY, LibName TEXT, '
#     'Record INTEGER, MeasurementType TEXT, Unit TEXT, Name TEXT, Description TEXT, FileName TEXT, '
#     'NumValues INTEGER, MinValue FLOAT, MaxValue FLOAT, ValuesArray BLOB)'

# ASTER
#     'CREATE TABLE Samples (SampleID INTEGER PRIMARY KEY, Name TEXT, Type TEXT, Class TEXT, SubClass TEXT, '
#     'ParticleSize TEXT, SampleNum TEXT, Owner TEXT, Origin TEXT, Phase TEXT, Description TEXT)',
#
#     'CREATE TABLE Spectra (SpectrumID INTEGER PRIMARY KEY, SampleID INTEGER, SensorCalibrationID INTEGER, '
#     'Instrument TEXT, Environment TEXT, Measurement TEXT, '
#     'XUnit TEXT, YUnit TEXT, MinWavelength FLOAT, MaxWavelength FLOAT, '
#     'NumValues INTEGER, XData BLOB, YData BLOB)',

# RELAB

# 'CREATE TABLE Samples (SampleID INTEGER PRIMARY KEY, RelabSampleID TEXT, Name TEXT, PI TEXT, SI TEXT, SI2, TEXT,'
# 'Source TEXT, GeneralType1 TEXT, GeneralType2 TEXT, Type1 TEXT, Type2 TEXT, SubType TEXT, '
# 'Modified TEXT, MinSize TEXT, MaxSize TEXT, Particulate TEXT, SampleNum TEXT, Texture TEXT,'
# 'Origin TEXT, Location TEXT, Chem TEXT, Description TEXT)',

# 'CREATE TABLE Spectra (SpectrumID INTEGER PRIMARY KEY, RelabSpectrumID TEXT, SampleID INTEGER, RelabSampleID TEXT, Date TEXT, '
# 'ReleaseDate TEXT, SpecCode TEXT, Start INTEGER, Stop INTEGER, Resolution FLOAT, SourceAngle TEXT,'
# 'DetectAngle TEXT, AzimuthAngle TEXT, PhaseAngle TEXT, Temperature TEXT, Atmosphere TEXT, PhotoFile TEXT'
# 'Spinning TEXT, Aperture TEXT, Scrambler TEXT, Depolarizer TEXT, User TEXT, NASA_PI_Sponsor TEXT,'
# 'ResearchType TEXT, Reference TEXT, MinWavelength FLOAT, MaxWavelength FLOAT, '
# 'NumValues INTEGER, XData BLOB, YData BLOB)',

# 'CREATE TABLE Spectrometer (SpecID INTEGER PRIMARY KEY, SpecCode TEXT, Spectrometer TEXT, MeasurementMode TEXT, Affiliation)',

if __name__ == '__main__':
    db = SpectralLibrary('RELAB')
    x, y = db.get_spectrum(1)
    print(db.get_sample(121))
    plt.plot(x, y)
    plt.show()
    db.close()

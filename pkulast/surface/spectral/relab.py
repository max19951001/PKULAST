#!/usr/bin/python
# -*- coding: utf-8 -*-
# Filename: relab.py
# Author: Peter Zhu
from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import pandas as pd
from os import path, makedirs
import pickle as pkl
import itertools
import logging
from pkulast.utils.collections import frombytes, tobytes, open_file, readline, find_file_path
from pkulast.surface.spectral.aster import AsterDatabase
from pkulast.config import SPECTRA_DIR

# home = path.expanduser("~")


logger = logging.getLogger(__name__)

table_schemas = [
    'CREATE TABLE Samples (SampleID INTEGER PRIMARY KEY, RelabSampleID TEXT, Name TEXT, PI TEXT, SI TEXT, SI2, TEXT,'
    'Source TEXT, GeneralType1 TEXT, GeneralType2 TEXT, Type1 TEXT, Type2 TEXT, SubType TEXT, '
    'Modified TEXT, MinSize TEXT, MaxSize TEXT, Particulate TEXT, SampleNum TEXT, Texture TEXT,'
    'Origin TEXT, Location TEXT, Chem TEXT, Description TEXT)',
    'CREATE TABLE Spectra (SpectrumID INTEGER PRIMARY KEY, RelabSpectrumID TEXT, SampleID INTEGER, RelabSampleID TEXT, Date TEXT, '
    'ReleaseDate TEXT, SpecCode TEXT, Resolution FLOAT, SourceAngle TEXT,'
    'DetectAngle TEXT, AzimuthAngle TEXT, PhaseAngle TEXT, Temperature TEXT, Atmosphere TEXT, PhotoFile TEXT,'
    'Spinning TEXT, Aperture TEXT, Scrambler TEXT, Depolarizer TEXT, User TEXT, NASA_PI_Sponsor TEXT,'
    'ResearchType TEXT, Reference TEXT, MinWavelength FLOAT, MaxWavelength FLOAT, '
    'NumValues INTEGER, XData BLOB, YData BLOB)',
    'CREATE TABLE Spectrometer (SpecID INTEGER PRIMARY KEY, SpecCode TEXT, Spectrometer TEXT, MeasurementMode TEXT, Affiliation)',
    # 'CREATE TABLE Chem (SpecCode TEXT PRIMARY KEY'
    # Rec #	Sample ID	Mineral	%SiO2	%TiO2	%Al2O3	%Cr2O3
    # %Fe2O3	%FeO	%MnO	%MgO	%CaO	%Na2O	%K2O
    # %P2O5	%LOI	Cu(ppm)	Ni(ppm)	Co(ppm)	Zn(ppm)	V(ppm)	Source	Text
]


# These files contained malformed signature data and will be ignored.
bad_files = [
]

arraytypecode = chr(ord('f'))


class RelabDatabase(AsterDatabase):
    # Download Relab spectra library at http://www.planetary.brown.edu/relabdata/
    schemas = table_schemas

    # def _add_sample(self, sampleid, name, pi, si, si2, source, generaltype1,
    #                 generaltype2, type1, type2, subtype, modified, minSize,
    #                 maxSize, particulate, samplenum, Texture, origin, location,
    #                 chem, description):
    #     sql = '''INSERT INTO Samples (RelabSampleID, Name, PI, SI, SI2, Source, GeneralType1, GeneralType2 , Type1, Type2, SubType,
    #     Modified, MinSize, MaxSize, Particulate, SampleNum, Texture,
    #     Origin, Location, Chem, Description)
    #                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
    #     self.cursor.execute(
    #         sql, (sampleid, name, pi, si, si2, source, generaltype1, generaltype2, type1,
    #     type2, subtype, modified,
    #                 minSize, maxSize, particulate, samplenum, Texture,
    #                 origin, location, chem, description))
    #     rowId = self.cursor.lastrowid
    #     self.db.commit()
    #     return rowId

    def _add_signature(self, RelabSpectrumID, SampleID, RelabSampleID, Date,
                       ReleaseDate, SpecCode, Resolution, SourceAngle,
                       DetectAngle, AzimuthAngle, PhaseAngle, Temperature,
                       Atmosphere, PhotoFile, Spinning, Aperture, Scrambler,
                       Depolarizer, User, NASA_PI_Sponsor, ResearchType,
                       Reference, MinWavelength, MaxWavelength, NumValues,
                       xData, yData):
        import sqlite3
        import array
        sql = '''INSERT INTO Spectra (RelabSpectrumID, SampleID, RelabSampleID, Date,
                ReleaseDate, SpecCode, Resolution, SourceAngle, DetectAngle,
                AzimuthAngle, PhaseAngle, Temperature, Atmosphere, PhotoFile, Spinning,
                Aperture, Scrambler, Depolarizer, User, NASA_PI_Sponsor, ResearchType,
                Reference, MinWavelength, MaxWavelength, NumValues, XData, YData) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        xBlob = sqlite3.Binary(tobytes(array.array(arraytypecode, xData)))
        yBlob = sqlite3.Binary(tobytes(array.array(arraytypecode, yData)))
        self.cursor.execute(
            sql, (RelabSpectrumID, SampleID, RelabSampleID, Date, ReleaseDate,
                  SpecCode, Resolution, SourceAngle, DetectAngle, AzimuthAngle,
                  PhaseAngle, Temperature, Atmosphere, PhotoFile, Spinning,
                  Aperture, Scrambler, Depolarizer, User, NASA_PI_Sponsor,
                  ResearchType, Reference, MinWavelength, MaxWavelength,
                  NumValues, xBlob, yBlob))
        rowId = self.cursor.lastrowid
        self.db.commit()
        return rowId

    @classmethod
    def create(cls, filename, relab_data_dir=None):
        '''Creates an RELAB relational database by parsing RELAB data files.

        Arguments:

            `filename` (str):

                Name of the new sqlite database file to create.

            `relab_data_dir` (str):

                Path to the directory containing RELAB library data files. If
                this argument is not provided, no data will be imported.

        Returns:

            An :class:`~spectral.database.RelabDatabase` object.

        Example::

            >>> RelabDatabase.create("relab_lib.db", "/Relad_dir")

        This is a class method (it does not require instantiating an
        RelabDatabase object) that creates a new database by parsing all of the
        files in the RELAB library data directory.  Normally, this should only
        need to be called once.  Subsequently, a corresponding database object
        can be created by instantiating a new RelabDatabase object with the
        path the database file as its argument.  For example::

            >>> from spectral.database.relab import RelabDatabase
            >>> db = RelabDatabase("relab_lib.db")
        '''
        import os
        if os.path.isfile(filename):
            raise Exception('Error: Specified file already exists.')
        db = cls()
        db._connect(filename)
        for schema in cls.schemas:
            db.cursor.execute(schema)
        if relab_data_dir:
            db._import_files(relab_data_dir)
        return db

    def __init__(self, sqlite_filename=None):
        '''Creates a database object to interface an existing database.

        Arguments:

            `sqlite_filename` (str):

                Name of the database file.  If this argument is not provided,
                an interface to a database file will not be established.

        Returns:

            An :class:`~AsterDatabase` connected to the database.
        '''
        if sqlite_filename:
            self._connect(find_file_path(sqlite_filename))
        else:
            self.db = None
            self.cursor = None

    def _import_files(self, data_dir):
        '''Read each file in the Relab library.'''
        from glob import glob
        import zipfile
        import xlrd as excel
        import os
        if not os.path.exists(data_dir):
            raise Exception('Error: Invalid file name specified.')

        zip = zipfile.ZipFile(data_dir)

        def extract_zipfile(filename):
            zip.extract(f'catalogues/{filename}.xls',
                        SPECTRA_DIR + 'Relab')
            out = os.path.join(SPECTRA_DIR + 'Relab/',
                          f'catalogues/{filename}.xls').replace(
                              "/", "\\")
            # indexid = "SampleID"
            loadexcel = excel.open_workbook(out,
                                            on_demand=True)
            # sheet = loadexcel.sheet_by_name(loadexcel.sheet_names()[0])
            f = pd.read_excel(
                out,
                loadexcel.sheet_names()[0],
                header=0,
                # index_col=indexid,
                na_values=["   "],
            )
            return f
        samplecat = extract_zipfile('Sample_Catalogue')
        samplecat.rename(columns={'SampleID': 'RelabSampleID'},
                                inplace=True)
        samplecat.insert(0, 'SampleID', np.arange(len(samplecat)))

        # samplecat.index.name = "SampleID"
        samplecat.rename(columns={'Text': 'Description'}, inplace=True)
        samplecat.rename(columns={'SampleName': 'Name'}, inplace=True)
        samplecat.to_sql('Samples', self.db, if_exists='replace')
        spectracat = extract_zipfile('Spectra_Catalogue')
        spectracat.rename(columns={'SpectrumID': 'RelabSpectrumID'},
                         inplace=True)
        spectracat.rename(columns={'SampleID': 'RelabSampleID'}, inplace=True)
        # spectracat.insert(0, 'SpectrumID', np.arange(len(spectracat)))
        spectracat = pd.merge(samplecat, spectracat, on='RelabSampleID')
        spectracat.index.name = "SpectrumID"
        for index, row in spectracat.iterrows():
            f = row['RelabSampleID'].lower().split("-")
            txt_path = '/'.join(
                ["data", f[1], f[0], row['RelabSpectrumID'].lower() + '.txt'])
            try:
                fs = zip.open(txt_path, 'r')

                data = np.loadtxt( fs,
                            skiprows=2,
                            delimiter='\t',
                            comments='\r')
                xData, yData = data[:, 0], data[:, 1]
                NumValues = len(xData)
                self._add_signature(
                    row['RelabSpectrumID'], row['SampleID'], row['RelabSampleID'],
                    str(row['Date']), str(row['ReleaseDate']), row['SpecCode'],
                    row['Resolution'], row['SourceAngle'], row['DetectAngle'],
                    row['AzimuthAngle'], row['PhaseAngle'], row['Temperature'],
                    row['Atmosphere'], row['PhotoFile'], row['Spinning'],
                    row['Aperture'], row['Scrambler'], row['Depolarizer'],
                    row['User'], row['NASA_PI_Sponsor'], row['ResearchType'],
                    row['Reference'], row['Start'], row['Stop'], NumValues, xData,
                    yData)
                logger.info('Importing Relab spectrum {}: {} {}'.format(
                    index + 1, row['RelabSampleID'], row['RelabSpectrumID']))
            except Exception as e:
                logger.error('Failed to parse Relab spectrum: {}: {}'.format(
                    row['RelabSampleID'], row['RelabSpectrumID']))
                continue

if __name__ == '__main__':
    #relab = Relab()
    #relab.locate("RS-CMP-012", False)
    #relab.show_spectra()
    RelabDatabase.create("relab_lib.db")
    pass
# END
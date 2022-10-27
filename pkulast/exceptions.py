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

"""
All exceptions used within PKULAST. 

"""

class PKULASTException(Exception):
    pass

class PKULASTFileNotFoundException(PKULASTException):
    "Failed to find a file"

class FileOpenException(PKULASTException):
    "Failed to open an input or output file"

class ImageOpenException(FileOpenException):
    "Image wasn't able to be opened by GDAL"

class ParameterException(PKULASTException):
    "Incorrect parameters passed to function"

class GDALLayerNumberException(PKULASTException):
    "A GDAL layer number was given, but was out of range"

class ResampleNeededException(PKULASTException):
    "Images do not match - resample needs to be turned on"

class OutsideImageBoundsException(PKULASTException):
    "Requested Block is not available"

class GdalWarpNotFoundException(PKULASTException):
    "Unable to find gdalwarp"

class GdalWarpException(PKULASTException):
    "Exception while running gdalwarp"

class ThematicException(PKULASTException):
    "File unable to be set to thematic"

class ProcessCancelledException(PKULASTException):
    "Process was cancelled by user"

class KeysMismatch(PKULASTException):
    "Keys do not match expected"

class MismatchedListLengthsException(PKULASTException):
    "Two lists had different lengths, when they were supposed to be the same length"

class AttributeTableColumnException(PKULASTException):
    "Unable to find specified column"

class AttributeTableTypeException(PKULASTException):
    "Type does not match that expected"

class ArrayShapeException(PKULASTException):
    "Exception in shape of an array"

class TypeConversionException(PKULASTException):
    "Unknown type conversion"

class VectorAttributeException(PKULASTException):
    "Unable to find specified index in vector file"

class VectorGeometryTypeException(PKULASTException):
    "Unexpected Geometry type"

class VectorProjectionException(PKULASTException):
    "Vector projection does not match raster projection"

class VectorRasterizationException(PKULASTException):
    "Rasterisation of Vector dataset failed"

class VectorLayerException(PKULASTException):
    "Unable to find the specified layer"

class WrongControlsObject(PKULASTException):
    "The wrong type of control object has been passed to apply"

class RatBlockLengthException(PKULASTException):
    "Exception with RAT block length, in ratapplier"

class RatMismatchException(PKULASTException):
    "Inconsistent RATs on inputs to ratapplier"

class IntersectionException(PKULASTException):
    "Images don't have a common area"

class JobMgrException(PKULASTException):
    "Exceptions from Jobmanager class"

class ColorTableGenerationException(PKULASTException):
    "Exception generating a color table"

class PermissionException(PKULASTException):
    "Exception due to permissions on temp files"

class AlgorithmNotFoundException(PKULASTException):
    """Exception due to AlgorithmNotFound."""
    pass


def assert_required_keywords_provided(keywords, **kwargs):
    """
    This method checks if all the required keyword arguments to complete a computation
        are provided in **kwargs
    Args:
        keywords ([list[str]], optional): Required keywords.
    Raises:
        KeywordArgumentError: custom exception
    """
    for keyword in keywords:
        if keyword not in kwargs or kwargs[keyword] is None:
            message = (
                f"Keyword argument {keyword} must be provided for this computation "
            )
            raise ValueError(message)
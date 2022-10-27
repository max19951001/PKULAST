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

def ensure_dir(filename):
	""" Check if the dir of f exists, otherwise create it.
	"""
	directory = os.path.dirname(filename)
	if directory and not os.path.isdir(directory):
		os.mkdirs(directory)

def check_filename_exist(filename):
	if not os.path.exists(filename) or not os.path.isfile(filename):
		errmsg = ('File does not exist! Filename = ' + str(filename))
		raise IOError(errmsg)
	else:
		return True

def check_file_status(filepath, filesize):
    sys.stdout.write('\r')
    sys.stdout.flush()
    size = int(os.stat(filepath).st_size)
    percent_complete = (size/filesize)*100
    sys.stdout.write('%s %.2f %s' % (os.path.basename(filepath), percent_complete, '% Completed'))
    sys.stdout.flush()
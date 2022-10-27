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

""" algorithms utilities"""
import re
import numpy as np
import concurrent.futures
from multiprocessing import freeze_support


def parallel_exec(func, lst, call_back):
    max_workers = 12
    orders = []
    freeze_support()
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers) as executor:
        futures = [executor.submit(func, item) for item in lst]
        for future in futures:
            future.add_done_callback(call_back)
        for future in concurrent.futures.as_completed(futures):
            order = future.result()
            orders.append(order)
    return order


def get_operant(key, kwargs):
    """ get operant from kwargs
    """
    if str(key) in kwargs.keys():
        return kwargs[key]
    else:
        return np.array(key, dtype=np.float32)


def parse_formula(f_str, kwargs):
    op_rank = {'*': 2, '/': 2, '+': 1, '-': 1}  # define op rank
    stack = []
    postfix = []
    for s in re.findall(r'[A-Za-z0-9]+|[()]|[-/\+\*]', f_str):
        if s in '+-*/':
            while stack and stack[-1] in '+-*/' and op_rank.get(
                    stack[-1]) >= op_rank.get(s):  # priority
                postfix.append(stack.pop())
            stack.append(s)
        elif s == "(":
            stack.append(s)
        elif s == ")":
            while stack:
                top = stack.pop()
                if top == "(":
                    break
                postfix.append(top)
        else:  # operand
            postfix.append(s)
    while stack:
        postfix.append(stack.pop())
    stack = []
    for p in postfix:
        if p in '+-*/':  # operator
            str_2 = stack.pop()  # op 2
            str_1 = stack.pop()  # op 1
            if p == '+':
                result = get_operant(str_1, kwargs) + get_operant(
                    str_2, kwargs)
            elif p == '-':
                result = get_operant(str_1, kwargs) - get_operant(
                    str_2, kwargs)
            elif p == '*':
                result = get_operant(str_1, kwargs) * get_operant(
                    str_2, kwargs)
            else:
                result = get_operant(str_1, kwargs) / get_operant(
                    str_2, kwargs)
            stack.append(result.reshape(-1, 1))
        else:
            stack.append(get_operant(p, kwargs))
    return stack.pop()
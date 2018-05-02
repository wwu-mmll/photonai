# -*- coding: utf-8; -*-
#
# Licensed to Crate (https://crate.io) under one or more contributor
# license agreements.  See the NOTICE file distributed with this work for
# additional information regarding copyright ownership.  Crate licenses
# this file to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may
# obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations
# under the License.
#
# However, if you have executed another commercial license agreement
# with Crate these terms will supersede the license and you may use the
# software solely pursuant to the terms of the relevant commercial agreement.

"""Crate Sphix Theme for ReadTheDocs"""

import os

VERSION = (0, 5, 57)

__version__ = ".".join(str(v) for v in VERSION)
__version_full__ = __version__

def current_dir():
    return os.path.abspath(os.path.dirname(__file__))

def get_html_theme_path():
    """Return list of HTML theme paths."""
    return [current_dir()]

def get_html_static_path():
    """Return list of HTML static paths."""
    current_dir = current_dir()
    return [
        os.path.join(current_dir, 'crate', 'static'),
    ]

def get_html_template_path():
    """Return list of HTML template paths."""
    current_dir = current_dir()
    return [
        os.path.join(current_dir, 'crate'),
    ]

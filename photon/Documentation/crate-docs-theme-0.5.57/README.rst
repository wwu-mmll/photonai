CrateDB Docs Theme
==================

A Sphinx_ theme for `the CrateDB documentation`_.

Prerequisites
=============

You will need Python 2.7.

Installation
============

The CrateDB docs theme is available as a pip_ package.

To install, run::

    $ pip install crate-docs-theme

To update, run::

    $ pip install -U crate-docs-theme

Configuration
=============

The CrateDB `reference site`_ is composed of multiple documentation bases, seemlessly interlinked via the CrateDB docs theme navigation. The CrateDB docs theme includes one configuration module per project that can be listed in the navigation.

In your Sphinx ``conf.py``, import all of these modules like so::

   from crate.theme.rtd.conf import *

Or, if you only want to use a specific module, do this::

   from crate.theme.rtd.conf.crate_server import *

Contributing
============

This project is primarily maintained by Crate.io_, but we welcome community
contributions!

See the `developer docs`_ and the `contribution docs`_ for more information.

Help
====

Looking for more help?

- Check `StackOverflow`_ for common problems
- Chat with us on `Slack`_
- Get `paid support`_

.. _reference site: https://crate.io/docs/reference/
.. _contribution docs: CONTRIBUTING.rst
.. _Crate.io: https://crate.io
.. _developer docs: DEVELOP.rst
.. _paid support: https://crate.io/pricing/
.. _pip: https://pypi.python.org/pypi/pip
.. _Slack: https://crate.io/docs/support/slackin/
.. _Sphinx: http://www.sphinx-doc.org/en/stable/
.. _StackOverflow: https://stackoverflow.com/tags/crate
.. _the CrateDB documentation: https://crate.io/docs/reference/

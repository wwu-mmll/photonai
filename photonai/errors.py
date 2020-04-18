#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
__coverage__ = 0.98

from photonai.photonlogger.logger import logger


class PhotoaiError(Exception):
    pass


def raise_PhotoaiError(msg: str) -> None:
    """
    Photonai standard error, and logging.error of
    the same msg.

    Parameters
    ----------
    msg: The formatted or unformatted sting sent to error and log.

    Returns
    -------
    Traps into an error that is probably not recoverable.
    """
    logger.error(msg)
    raise PhotoaiError(msg)


class PhotoaiNotImplementedError(Exception):
    pass

def raise_PhotoaiNotImplementedError(msg: str):
    """
    Photonai Raise a NotImplemented error, and loogging.error of
    the same msg.

    Parameters
    ----------
    msg: The formatted or unformatted sting sent to error and log.

    Returns
    -------
    Traps into an error that is probably not recoverable.
    """
    logger.error(msg)
    raise PhotoaiNotImplementedError(msg)
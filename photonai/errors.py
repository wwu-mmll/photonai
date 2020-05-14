#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
__coverage__ = 0.98

from photonai.photonlogger.logger import logger


class PhotonaiError(Exception):
    pass


def raise_PhotonaiError(msg: str) -> None:
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
    raise PhotonaiError(msg)


class PhotoaiNotImplementedError(Exception):
    pass

def raise_PhotonaiNotImplementedError(msg: str):
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
    raise PhotonaiNotImplementedError(msg)

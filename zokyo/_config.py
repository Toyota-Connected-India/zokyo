# -*- coding: utf-8 -*-

import importlib


class _Config:
    """Singleton for storing package info lazily. We load lazily since it's
    expensive to load some of these modules and if we try to load them all,
    importing zokyo can take several seconds to import
    """

    def _see_if_available(self, package_name):
        attrname = "has_%s" % package_name.replace("-", "_")
        if hasattr(self, attrname):
            return getattr(self, attrname)  # True or False

        # First time:
        res = False
        try:
            importlib.import_module(package_name)
            res = True
        except (ImportError, ModuleNotFoundError):
            pass

        setattr(self, attrname, res)
        return res


_conf = _Config()

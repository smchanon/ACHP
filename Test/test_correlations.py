# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:44:45 2024

@author: SMCANANA
"""

import pytest
from ACHP.calculations.Correlations import *
from ACHP.models.Fluid import Fluid, FluidApparatusProps
from ACHP.models.HeatExchangers import HEXType

def pytest_generate_tests(metafunc):
    if metafunc.function.__name__ in metafunc.cls.params.keys():
        funcarglist = metafunc.cls.params[metafunc.function.__name__]
        argnames = sorted(funcarglist[0])
        metafunc.parametrize(
            argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
        )


@pytest.fixture
def fluidR744():
    return Fluid("R744", "HEOS")

@pytest.fixture
def fluidMEG50():
    return Fluid("MEG", "IncompressibleBackend", massFraction=0.50)

class TestGetTempDensityPhaseFromPandH:
    params = {"test_getPhaseFromPandH": [dict(fluid=Fluid("MEG", "IncompressibleBackend", massFraction=0.50), pressure=700000, enthalpy=125, hexType=HEXType.PLATE, expected='Subcooled'),#backend has 'incomp'
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=10000000.0, enthalpy=319892.71816103684, hexType=HEXType.PLATE, expected='Supercritical'),#p > pCrit && t > tcrit
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=10000000.0, enthalpy=274856.22998397355, hexType=HEXType.PLATE, expected='Supercritical'),#p > pCrit && t = tcrit
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=10000000.0, enthalpy=195530.26737918807, hexType=HEXType.PLATE, expected='Supercrit_liq'),#p > pCrit && t < tcrit
                                        # bug calculating at p=pCrit
                                        # dict(fluid=Fluid("R744", "HEOS"), pressure=7377300.0, enthalpy=426260.21486963786, hexType=HEXType.PLATE, expected='Supercritical'),#p = pCrit && t > tcrit
                                        # dict(fluid=Fluid("R744", "HEOS"), pressure=7377300.0, enthalpy=329138.0227391727, hexType=HEXType.PLATE, expected='Supercritical'),#p = pCrit && t = tcrit
                                        # dict(fluid=Fluid("R744", "HEOS"), pressure=7377300.0, enthalpy=196834.08085025015, hexType=HEXType.PLATE, expected='Supercrit_liq'),#p = pCrit && t < tcrit
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=450000, hexType=HEXType.PLATE, expected='Superheated'),#h > hsatV
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=432872.2974300893, hexType=HEXType.PLATE, expected='TwoPhase'),#h = hsatV
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=100000, hexType=HEXType.PLATE, expected='TwoPhase'),#h < hsatV && h > hsatL
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=94191.1371077584, hexType=HEXType.PLATE, expected='TwoPhase'),#h = hsatL
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=90000, hexType=HEXType.PLATE, expected='Subcooled')],#h < hsatL
            "test_getTempDensityPhaseFromPandHGetsTemp": [dict(fluid=Fluid("MEG", "IncompressibleBackend", massFraction=0.50), pressure=700000, enthalpy=22784.029047293378, tBubble=235, tDew=225, rhosatL=None, rhosatV=None, expected=299.8508),#backend has 'incomp'
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=10000000.0, enthalpy=319892.71816103684, tBubble=223.78094125730473, tDew=225.123456, rhosatL=None, rhosatV=None, expected=314.3),#p > pCrit && t > tcrit
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=10000000.0, enthalpy=274856.22998397355, tBubble=223.78094125730473, tDew=225.123456, rhosatL=None, rhosatV=None, expected=304.1282),#p > pCrit && t = tcrit
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=10000000.0, enthalpy=195530.26737918807, tBubble=223.78094125730473, tDew=225.123456, rhosatL=None, rhosatV=None, expected=273.15),#p > pCrit && t < tcrit
                                        # bug calculating at p=pCrit
                                        # dict(fluid=Fluid("R744", "HEOS"), pressure=7377300.0, enthalpy=426260.21486963786, tBubble=235, tDew=225, rhosatL=None, rhosatV=None, expected=314.3),#p = pCrit && t > tcrit
                                        # dict(fluid=Fluid("R744", "HEOS"), pressure=7377300.0, enthalpy=329138.0227391727, tBubble=235, tDew=225, rhosatL=None, rhosatV=None, expected=304.1282),#p = pCrit && t = tcrit
                                        # dict(fluid=Fluid("R744", "HEOS"), pressure=7377300.0, enthalpy=196834.08085025015, tBubble=235, tDew=225, rhosatL=None, rhosatV=None, expected=273.15),#p = pCrit && t < tcrit
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=450000, tBubble=223.78094125730473, tDew=225.123456, rhosatL=None, rhosatV=None, expected=242.1709),#h > hsatV
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=432872.2974300893, tBubble=223.78094125730473, tDew=225.123456, rhosatL=None, rhosatV=None, expected=225.1218),#h = hsatV
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=100000, tBubble=223.78094125730473, tDew=225.123456, rhosatL=None, rhosatV=None, expected=223.8039),#hsatL < h < hsatV
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=94191.1371077584, tBubble=223.78094125730473, tDew=225.123456, rhosatL=None, rhosatV=None, expected=223.7809),#h = hsatL
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=90000, tBubble=223.78094125730473, tDew=225.123456, rhosatL=None, rhosatV=None, expected=221.6530)],#h < hsatL
            "test_getTempDensityPhaseFromPandHGetsDensity": [dict(fluid=Fluid("MEG", "IncompressibleBackend", massFraction=0.50), pressure=700000, enthalpy=22784.029047293378, tBubble=235, tDew=225, rhosatL=None, rhosatV=None, expected=1061.2631),#backend has 'incomp'
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=10000000.0, enthalpy=319892.71816103684, tBubble=223.78094125730473, tDew=225.123456, rhosatL=None, rhosatV=None, expected=603.0474),#p > pCrit && t > tcrit
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=10000000.0, enthalpy=274856.22998397355, tBubble=223.78094125730473, tDew=225.123456, rhosatL=None, rhosatV=None, expected=761.2431),#p > pCrit && t = tcrit
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=10000000.0, enthalpy=195530.26737918807, tBubble=223.78094125730473, tDew=225.123456, rhosatL=None, rhosatV=None, expected=974.0506),#p > pCrit && t < tcrit
                                        # bug calculating at p=pCrit
                                        # dict(fluid=Fluid("R744", "HEOS"), pressure=7377300.0, enthalpy=426260.21486963786, tBubble=235, tDew=225, rhosatL=None, rhosatV=None, expected='Supercritical'),#p = pCrit && t > tcrit
                                        # dict(fluid=Fluid("R744", "HEOS"), pressure=7377300.0, enthalpy=329138.0227391727, tBubble=235, tDew=225, rhosatL=None, rhosatV=None, expected='Supercritical'),#p = pCrit && t = tcrit
                                        # dict(fluid=Fluid("R744", "HEOS"), pressure=7377300.0, enthalpy=196834.08085025015, tBubble=235, tDew=225, rhosatL=None, rhosatV=None, expected='Supercrit_liq'),#p = pCrit && t < tcrit
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=450000, tBubble=223.78094125730473, tDew=225.123456, rhosatL=None, rhosatV=None, expected=16.5154),#h > hsatV
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=432872.2974300893, tBubble=223.78094125730473, tDew=225.123456, rhosatL=None, rhosatV=None, expected=19.3762),#h = hsatV
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=100000, tBubble=223.78094125730473, tDew=225.123456, rhosatL=None, rhosatV=None, expected=575.3125),#h < hsatV && h > hsatL
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=94191.1371077584, tBubble=223.78094125730473, tDew=225.123456, rhosatL=None, rhosatV=None, expected=1152.2187),#h = hsatL
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=90000, tBubble=223.78094125730473, tDew=225.123456, rhosatL=None, rhosatV=None, expected=1160.2212)],#h < hsatL
            "test_getTempFromPandH": [dict(fluid=Fluid("MEG", "IncompressibleBackend", massFraction=0.50), pressure=700000, enthalpy=22784.029047293378, hexType=HEXType.PLATE, expected=299.8508),#backend has 'incomp'
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=10000000.0, enthalpy=319892.71816103684, hexType=HEXType.PLATE, expected=314.3),#p > pCrit && t > tcrit
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=10000000.0, enthalpy=274856.22998397355, hexType=HEXType.PLATE, expected=304.1282),#p > pCrit && t = tcrit
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=10000000.0, enthalpy=195530.26737918807, hexType=HEXType.PLATE, expected=273.15),#p > pCrit && t < tcrit
                                        # bug calculating at p=pCrit
                                        # dict(fluid=Fluid("R744", "HEOS"), pressure=7377300.0, enthalpy=426260.21486963786, hexType=HEXType.PLATE, expected=314.3),#p = pCrit && t > tcrit
                                        # dict(fluid=Fluid("R744", "HEOS"), pressure=7377300.0, enthalpy=329138.0227391727, hexType=HEXType.PLATE, expected=304.1282),#p = pCrit && t = tcrit
                                        # dict(fluid=Fluid("R744", "HEOS"), pressure=7377300.0, enthalpy=196834.08085025015, hexType=HEXType.PLATE, expected=273.15),#p = pCrit && t < tcrit
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=450000, hexType=HEXType.PLATE, expected=242.1709),#h > hsatV
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=432872.2974300893, hexType=HEXType.PLATE, expected=225.1235),#h = hsatV!
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=100000, hexType=HEXType.PLATE, expected=223.804),#h < hsatV && h > hsatL
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=94191.1371077584, hexType=HEXType.PLATE, expected=223.7809),#h = hsatL
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=90000, hexType=HEXType.PLATE, expected=221.6530)],#h < hsatL
            "test_getDensityFromPandH": [dict(fluid=Fluid("MEG", "IncompressibleBackend", massFraction=0.50), pressure=700000, enthalpy=22784.029047293378, hexType=HEXType.PLATE, expected=1061.2631),#backend has 'incomp'
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=10000000.0, enthalpy=319892.71816103684, hexType=HEXType.PLATE, expected=603.0474),#p > pCrit && t > tcrit
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=10000000.0, enthalpy=274856.22998397355, hexType=HEXType.PLATE, expected=761.2431),#p > pCrit && t = tcrit
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=10000000.0, enthalpy=195530.26737918807, hexType=HEXType.PLATE, expected=974.0506),#p > pCrit && t < tcrit
                                        # bug calculating at p=pCrit
                                        # dict(fluid=Fluid("R744", "HEOS"), pressure=7377300.0, enthalpy=426260.21486963786, hexType=HEXType.PLATE, expected='Supercritical'),#p = pCrit && t > tcrit
                                        # dict(fluid=Fluid("R744", "HEOS"), pressure=7377300.0, enthalpy=329138.0227391727, hexType=HEXType.PLATE, expected='Supercritical'),#p = pCrit && t = tcrit
                                        # dict(fluid=Fluid("R744", "HEOS"), pressure=7377300.0, enthalpy=196834.08085025015, hexType=HEXType.PLATE, expected='Supercrit_liq'),#p = pCrit && t < tcrit
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=450000, hexType=HEXType.PLATE, expected=16.5154),#h > hsatV
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=432872.2974300893, hexType=HEXType.PLATE, expected=19.3534),#h = hsatV
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=100000, hexType=HEXType.PLATE, expected=574.9674),#h < hsatV && h > hsatL
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=94191.1371077584, hexType=HEXType.PLATE, expected=1152.2187),#h = hsatL
                                        dict(fluid=Fluid("R744", "HEOS"), pressure=700000, enthalpy=90000, hexType=HEXType.PLATE, expected=1160.2212)],#h < hsatL
               }

    def test_getPhaseFromPandH(self, fluid, pressure, enthalpy, hexType, expected):
        fluid.fluidApparatiProps[HEXType.PLATE] = FluidApparatusProps()
        fluid.fluidApparatiProps[HEXType.PLATE].enthalpySatLiquid = 94191.1371077584
        fluid.fluidApparatiProps[HEXType.PLATE].enthalpySatVapor = 432872.2974300893
        actual = getPhaseFromPandH(fluid, pressure, enthalpy, hexType)
        assert(actual == expected)

    def test_getTempFromPandH(self, fluid, pressure, enthalpy, hexType, expected):
        fluid.fluidApparatiProps[HEXType.PLATE] = FluidApparatusProps()
        fluid.fluidApparatiProps[HEXType.PLATE].enthalpySatLiquid = 94191.1371077584
        fluid.fluidApparatiProps[HEXType.PLATE].enthalpySatVapor = 432872.2974300893
        fluid.fluidApparatiProps[HEXType.PLATE].tempDew=225.123456
        fluid.fluidApparatiProps[HEXType.PLATE].tempBubble=223.78094125730473
        actual = getTempFromPandH(fluid, pressure, enthalpy, hexType)
        assert(round(actual, 4) == expected)

    def test_getDensityFromPandH(self, fluid, pressure, enthalpy, hexType, expected):
        fluid.fluidApparatiProps[HEXType.PLATE] = FluidApparatusProps()
        fluid.fluidApparatiProps[HEXType.PLATE].enthalpySatLiquid = 94191.1371077584
        fluid.fluidApparatiProps[HEXType.PLATE].enthalpySatVapor = 432872.2974300893
        fluid.fluidApparatiProps[HEXType.PLATE].tempDew=225.123456
        fluid.fluidApparatiProps[HEXType.PLATE].tempBubble=223.78094125730473
        fluid.fluidApparatiProps[HEXType.PLATE].densitySatVapor=19.3533705476017
        fluid.fluidApparatiProps[HEXType.PLATE].densitySatLiquid=1152.2186710321578
        actual = getDensityFromPandH(fluid, pressure, enthalpy, hexType)
        assert(round(actual, 4) == expected)

    def test_getTempDensityPhaseFromPandHGetsTemp(self, fluid, pressure, enthalpy, tBubble, tDew, rhosatL, rhosatV, expected):
        actual = getTempDensityPhaseFromPandH(fluid, pressure, enthalpy, tBubble, tDew, rhosatL, rhosatV)
        assert(round(actual[0], 4) == expected)

    def test_getTempDensityPhaseFromPandHGetsDensity(self, fluid, pressure, enthalpy, tBubble, tDew, rhosatL, rhosatV, expected):
        actual = getTempDensityPhaseFromPandH(fluid, pressure, enthalpy, tBubble, tDew, rhosatL, rhosatV)
        assert(round(actual[1], 4) == expected)

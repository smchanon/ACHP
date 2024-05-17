# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:40:11 2024

@author: SMCANANA
"""

import pytest
from json import load
from ACHP.wrappers.JsonParser import *
from ACHP.models.Fluid import Fluid

def pytest_generate_tests(metafunc):
    if metafunc.function.__name__ in metafunc.cls.params.keys():
        funcarglist = metafunc.cls.params[metafunc.function.__name__]
        argnames = sorted(funcarglist[0])
        metafunc.parametrize(
            argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
        )

class TestJsonParser:
    params = {"test_parseFluid": [dict(filepath="Test/test_files/HEX_input_H.json", temp='hot', backEnd='HEOS', name='R744'),
                                  dict(filepath="Test/test_files/HEX_input_H.json", temp='cold', backEnd='IncompressibleBackend', name='MEG')],
              "test_parsePressure": [dict(filepath="Test/test_files/HEX_input_H.json", temp='hot', expected=[100000, 500000, 6000000]),
                                     dict(filepath="Test/test_files/HEX_input_H.json", temp='cold', expected=[350000, 100000])],
              "test_parseEnthalpy": [dict(filepath="Test/test_files/HEX_input_H.json", temp='hot', expected=[233.150, 300], pressures=[100000]),
                                   dict(filepath="Test/test_files/HEX_input_H.json", temp='cold', expected=[233150, 300000], pressures=[100000]),
                                   dict(filepath="Test/test_files/HEX_input_T.json", temp='hot', expected=[505853.79244995594, 504578.7314804369, 515270.02433395694, 527420.3658120929], pressures=[100000]),
                                   dict(filepath="Test/test_files/HEX_input_T.json", temp='cold', expected=[485588.39065255976, 509991.2480036709], pressures=[100000]),
                                   dict(filepath="Test/test_files/HEX_input_T.json", temp='hot', expected=[503503.1406317193, 502201.04707660875, 513102.87973879155, 525454.5219250775, 505853.79244995594, 504578.7314804369, 515270.02433395694, 527420.3658120929], pressures=[350000, 100000])],
              "test_parseMassFlow": [dict(filepath="Test/test_files/HEX_input_H.json", temp='hot', expected=[0.29694444444444446, 0.5555555555555556]),
                                      dict(filepath="Test/test_files/HEX_input_H.json", temp='cold', expected=[106900, 2000])]}

    def test_parseFluid(self, filepath, temp, backEnd, name):
        file = open(filepath)
        jsonData = load(file)
        file.close()
        fluid = parseFluid(jsonData['inputs'][0][temp])
        assert(isinstance(fluid, Fluid))
        assert(fluid.backEnd == backEnd)
        assert(fluid.name == name)

    def test_parsePressure(self, filepath, temp, expected):
        file = open(filepath)
        jsonData = load(file)
        file.close()
        pressure = parsePressure(jsonData['inputs'][0][temp]['pressureIn'])
        assert(pressure == expected)

    def test_parseEnthalpy(self, filepath, temp, pressures, expected):
        file = open(filepath)
        jsonData = load(file)
        file.close()
        fluid = Fluid('R744', 'HEOS')
        enthalpy = parseEnthalpy(jsonData['inputs'][0][temp]['h_or_T'], fluid, pressures)
        assert(enthalpy == expected)

    def test_parseMassFlow(self, filepath, temp, expected):
        file = open(filepath)
        jsonData = load(file)
        file.close()
        massFlow = parseMassFlow(jsonData['inputs'][0][temp]['massFlowIn'])
        assert(massFlow == expected)

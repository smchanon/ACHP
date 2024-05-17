# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:39:53 2024

@author: SMCANANA
"""
import pytest
from os import remove
from ACHP.wrappers.FileOperations import *
from ACHP.models.Fluid import Fluid
from json import load

def pytest_generate_tests(metafunc):
    if metafunc.function.__name__ in metafunc.cls.params.keys():
        funcarglist = metafunc.cls.params[metafunc.function.__name__]
        argnames = sorted(funcarglist[0])
        metafunc.parametrize(
            argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
        )

@pytest.fixture
def test_folder_name():
    return 'Test/test_files/'

@pytest.fixture
def csv_input(test_folder_name):
    with open(f"{test_folder_name}test_csv_input.json") as json_file:
        json_data = load(json_file)
    return json_data

@pytest.fixture
def json_input(test_folder_name):
    with open(f"{test_folder_name}test_json_input.json") as json_file:
        json_data = load(json_file)
    return json_data

@pytest.fixture
def csv_text(test_folder_name):
    with open(f"{test_folder_name}test_csv_expected.csv") as csv_file:
        csv_data = csv_file.read()
    return csv_data

@pytest.fixture
def json_pretty(test_folder_name):
    with open(f"{test_folder_name}test_json_expected.json") as json_file:
        json_data = json_file.read()
    return json_data

class TestFileOperations:
    params = {"test_getValuesFromInputFile": [dict(filename="HEX_input_H.json",
                                                      testName='test_test',
                                                      hexParams={'fluidHot': Fluid('R744', 'HEOS'),
                                                                 'fluidCold': Fluid('MEG', 'IncompressibleBackend', massFraction=0.50),
                                                                 'massFlowsCold': [106900, 2000],
                                                                 'massFlowsHot': [0.29694444444444446, 0.5555555555555556],
                                                                 'pressuresInCold': [350000, 100000],
                                                                 'pressuresInHot': [100000, 500000, 6000000],
                                                                 'enthalpiesInCold': [233150, 300000],
                                                                 'enthalpiesInHot': [233.150, 300],
                                                                 'innerTubeInnerDiameter':0.59,
                                                                 'innerTubeOuterDiameter':0.61,
                                                                 'outerTubeInnerDiameter':0.65,
                                                                 'length':5.43,
                                                                 'conductivity': 0.045,
                                                                 'hexType': 'Coaxial-HEX'},
                                                      values={'x-axis': [30, 50, 100, 150, 200, 250, 300, 350, 400, 450],
                                                              'y-axis': [10, 20, 30, 40, 50, 60],
                                                              'to_be_extracted': ['tempHotOut', 'tempHotIn']
                                                              }),
                                                 dict(filename="HEX_input_T.json",
                                                        testName='test_test2',
                                                        hexParams={'fluidHot': Fluid('R744', 'HEOS'),
                                                                   'fluidCold': Fluid('Water', 'HEOS'),
                                                                   'massFlowsCold': [106900],
                                                                   'massFlowsHot': [0.29694444444444446],
                                                                   'pressuresInCold': [350000, 100000],
                                                                   'pressuresInHot': [150000, 10000000, 8500000],
                                                                   'enthalpiesInCold': [3898.551121088151, 125422.40897047192, 3644.974626711335, 125194.32743019443],
                                                                   'enthalpiesInHot': [505387.74786185793, 504107.41465534386, 514839.83354385645, 527029.6300193968, 256379.85604709422, 252143.40658392495, 293617.5449981324, 384070.4045627197, 260981.4575081616, 256149.76685110378, 319399.6127334093, 425878.19097209204],
                                                                   'centerlineDistanceShort': 0.59,
                                                                   'centerlineDistanceLong': 0.61,
                                                                   'numPlates': 5,
                                                                   'thickness': 0.045,
                                                                   'moreChannels': 'hot',
                                                                   'volumeChannelSingle': 5.43,
                                                                   'conductivity': 0.65,
                                                                   'hexType': 'Plate-HEX'},
                                                        values={'x-axis': [20, 40, 60, 80, 100],
                                                                'y-axis': [15, 30, 45, 60],
                                                                'to_be_extracted': ['heatTransferred', 'tempColdOut']
                                                                })],
               "test_createCsvTable": [dict(value='tempHotOut', sorting=('pressure', 'massFlow'), reverse=(True, False), expected='Test/test_files/test_csv_expected_tf.csv'),
                                       dict(value='tempHotOut', sorting=('pressure', 'massFlow'), reverse=(True, True), expected='Test/test_files/test_csv_expected_tt.csv'),
                                       dict(value='tempHotOut', sorting=('pressure', 'massFlow'), reverse=(False, True), expected='Test/test_files/test_csv_expected_ft.csv'),
                                       dict(value='tempHotOut', sorting=('pressure', 'massFlow'), reverse=(False, False), expected='Test/test_files/test_csv_expected_ff.csv'),
                                       dict(value='resistanceTotal', sorting=('pressure', 'massFlow'), reverse=(True, False), expected='Test/test_files/test_csv_expected_r.csv'),
                                       #TODO: create output with more than 2 values to compare
                                       # dict(value='tempHotOut', sorting=('tempColdIn', 'massFlow'), reverse=(True, False), expected='Test/test_files/test_csv_expected_tm.csv'),
                                       # dict(value='tempHotOut', sorting=('pressure', 'tempColdIn'), reverse=(True, False), expected='Test/test_files/test_csv_expected_pt.csv')
                                      ]}

    def test_getValuesFromInputFile(self, test_folder_name, filename, testName, hexParams, values):
        actual = getValuesFromInputFile(f"{test_folder_name}{filename}")
        assert(actual[0] == testName)
        assert(actual[1] == hexParams)
        assert(actual[2] == values)

    def test_createJsonFile(self, mocker, test_folder_name, json_input, json_pretty):
        mocker.patch("ACHP.wrappers.FileOperations.output_folder", return_value=test_folder_name)
        createJsonFile(json_input, "test_name")
        with open(f"{test_folder_name}test_name.json") as file:
            actual = file.read()
            assert(actual == json_pretty)
            assert(actual != json_input)
        remove(f"{test_folder_name}test_name.json")

    def test_createCsvTable(self, mocker, test_folder_name, csv_input, value, sorting, reverse, expected):
        mocker.patch("ACHP.wrappers.FileOperations.output_folder", return_value=test_folder_name)
        createCsvTable(csv_input, "test_name", value, sorting, reverse=reverse)
        with open(f"{test_folder_name}test_name_{value}.csv") as file:
            text = file.read()
        with open(expected) as csv_file:
            csv_text = csv_file.read()
        assert(text == csv_text)
        remove(f"{test_folder_name}test_name_{value}.csv")

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:44:47 2024

@author: SMCANANA
"""
from operator import itemgetter
from json import dumps, load
from ACHP.wrappers.JsonParser import parseFluid, parseEnthalpy, parseMassFlow, parsePressure
from ACHP.calculations.Conversions import TemperatureConversions, PressureConversions, MassFlowConversions

def output_folder():
    return "C:/Users/smcanana/Documents/Spyder Projects/ACHP/Outputs/"

def createCsvTable(data, testName, value, sorting, reverse=(False, False)):
    """
    Creates a 2D table in csv format (but .txt extension) using 2 sorting parameters and one value
    to put in the table. Can only sort lowest to highest at the moment

    Parameters
    ----------
    data : string
        data in json format.
    filename : str
        name of file.
    testName : str
        name of test in json file.
    value : str
        name of value in test to be put in table.
    sorting : tuple of str
        2 parameters to sort the list by. The first parameter is the y-axis, and the second is the
        x-axis.
    reverse : tuple of bool, optional
        2 parameters for whether the list should be sorted in reverse order. The first parameter is
        for the y-axis, and the second is for the x-axis. The default is (False, False).

    Returns
    -------
    None.

    """
    jsonDataSorted = data[testName]
    for key, reverse_sort in zip(reversed(sorting), reversed(reverse)):
        jsonDataSorted = sorted(jsonDataSorted, key=itemgetter(key), reverse=reverse_sort)
    secondParamList = sorted(list(set(map(lambda x: x[sorting[1]], jsonDataSorted))), reverse=reverse[1])
    firstParam = 0
    i = 0
    with open(f"{output_folder()}{testName}_{value}.csv", "w") as file:
        file.write("sep=,")
        for point in jsonDataSorted:
            if firstParam == point[sorting[0]] and ((not reverse[1] and point[sorting[1]] > secondParamList[i]) or (reverse[1] and point[sorting[1]] < secondParamList[i])):
                file.write(f",{point[value]}")
            else:
                file.write(f"\n{point[sorting[0]]},{point[value]}")
                firstParam = point[sorting[0]]
            i = secondParamList.index(point[sorting[1]])
        file.write(f"\n,{','.join(str(x) for x in secondParamList)}")

def createJsonFile(jsonData, testName):
    tc = TemperatureConversions()
    mfc = MassFlowConversions()
    pc = PressureConversions()
    print(f"{output_folder()}{testName}.json")
    with open(f"{output_folder()}{testName}.json", "w") as file:
        for singlePoint in jsonData[testName]:
            for pointData in singlePoint.keys():
                if "pressure" in pointData:
                    singlePoint[pointData] = pc.convertPressure(pc.Unit.PA, pc.Unit.BAR, singlePoint[pointData])
                if "massFlow" in pointData:
                    singlePoint[pointData] = mfc.convertMassFlow(mfc.Unit.KGS, mfc.Unit.KGH, singlePoint[pointData])
                if "temp" in pointData:
                    singlePoint[pointData] = tc.convertTemperature(tc.Unit.K, tc.Unit.C, singlePoint[pointData])
        file.write(dumps(jsonData, indent=4))

def getValuesFromInputFile(filepath):
    with open(filepath) as file:
        jsonData = load(file)
    for jsonInput in jsonData["inputs"]:
        HEXParams = {}
        for sideTemp in ["hot", "cold"]:
            HEXParams[f"fluid{sideTemp.capitalize()}"] = parseFluid(jsonInput[sideTemp])
            HEXParams[f"pressuresIn{sideTemp.capitalize()}"] = parsePressure(jsonInput[sideTemp]["pressureIn"])
            HEXParams[f"enthalpiesIn{sideTemp.capitalize()}"] = parseEnthalpy(jsonInput[sideTemp]["h_or_T"],
                                                                                 HEXParams[f"fluid{sideTemp.capitalize()}"],
                                                                                 HEXParams[f"pressuresIn{sideTemp.capitalize()}"])
            HEXParams[f"massFlows{sideTemp.capitalize()}"] = parseMassFlow(jsonInput[sideTemp]["massFlowIn"])
        HEXParams['hexType'] = jsonInput["HEX"]["type"]
        for key, value in jsonInput["HEX"]["HEX_values"][HEXParams['hexType']].items():
            if value is not None:
                HEXParams[key] = value
    return jsonInput["testName"], HEXParams, jsonInput["values"]
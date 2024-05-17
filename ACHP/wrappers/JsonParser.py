# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:10:16 2024

@author: SMCANANA
"""
from ACHP.models.Fluid import Fluid, ThermoProps
from ACHP.calculations.Conversions import TemperatureConversions, MassFlowConversions, PressureConversions

def parseFluid(jsonDict):
    fluidName = jsonDict["fluidName"]
    if 'MEG' in fluidName: #assuming form of 'MEG::50%'
        fraction = float(fluidName.split('::')[1].replace('%', ''))/100
        return Fluid("MEG", "IncompressibleBackend", massFraction=fraction)
    else:
        return Fluid(fluidName, "HEOS")

def parsePressure(jsonDict):
    if jsonDict["unit"] in ["Pa", "Pascal", "Pascals", "PA"]:
        return jsonDict["values"]
    else:
        pc = PressureConversions()
        return [pc.convertPressure(getattr(pc.Unit, jsonDict["unit"]), pc.Unit.PA, pressure) for pressure in jsonDict["values"]]

def parseEnthalpy(jsonDict, fluid: Fluid=None, pressures=None):
    if jsonDict["enthalpyIn"]: #TODO: make sure units are correct
        return jsonDict["values"]
    else:
        enthalpyDict = []
        for pressure in pressures:
            if jsonDict["unit"] != "K":
                tc = TemperatureConversions()
                enthalpyDict.extend([fluid.calculateEnthalpy(ThermoProps.PT, pressure,
                                    tc.convertTemperature(getattr(tc.Unit, jsonDict["unit"]), tc.Unit.K,
                                    tempToConvert)) for tempToConvert in jsonDict["values"]])
            else:
                enthalpyDict.extend([fluid.calculateEnthalpy(ThermoProps.PT, pressure, temperature) for temperature in jsonDict["values"]])
        return enthalpyDict

def parseMassFlow(jsonDict):
    if jsonDict["unit"] in ["KGS", "kg/s", "kgs", "kgs per second"]:
        return jsonDict["values"]
    else:
        mfc = MassFlowConversions()
        return [mfc.convertMassFlow(getattr(mfc.Unit, jsonDict["unit"]), mfc.Unit.KGS, massFlow) for massFlow in jsonDict["values"]]

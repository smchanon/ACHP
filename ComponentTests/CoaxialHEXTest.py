# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 07:25:51 2024

@author: smcanana
"""
import numpy as np
import pylab
import matplotlib.pyplot as plt
import logging
from itertools import product
from ACHP.models.Fluid import Fluid, ThermoProps
from ACHP.models.HeatExchangers import CoaxialHEX, BrazedPlateHEX
from ACHP.wrappers.CoolPropWrapper import PropsSIWrapper
from ACHP.wrappers.FileOperations import createCsvTable, createJsonFile, getValuesFromInputFile
from ACHP.calculations.Conversions import TemperatureConversions, PressureConversions, MassFlowConversions

def ACHPCoax():
    TT=[]
    QQ=[]
    Q1=[]
    #cold
    refrigerant = Fluid("R290", "HEOS")
    #hot/glycol
    coolant = Fluid("Water", "HEOS")
    propsSI = PropsSIWrapper()
    for Tdew_evap in np.linspace(270,290.4):
        Tdew_cond=317.73
#        Tdew_evap=285.42
        pdew_cond = propsSI.calculatePressureFromTandQ(refrigerant, Tdew_cond, 1.0)
        h = propsSI.calculateEnthalpyFromTandP(refrigerant, Tdew_cond-7, pdew_cond)
        params = {
                'fluidCold': refrigerant,
                'fluidHot': coolant,
                'massFlowCold': 0.040,
                'massFlowHot': 0.38,
                'pressureInCold': propsSI.calculatePressureFromTandQ(refrigerant, Tdew_evap, 1.0),
                'pressureInHot': 300000,
                'enthalpyInCold': h,
                'enthalpyInHot': propsSI.calculateEnthalpyFromTandP(coolant, 290.52, 300000),
                #'tempInCold': 290.52, this is enthalpyInCold right now
                'innerTubeInnerDiameter': 0.0278,      #inner tube, Internal Diameter (ID)
                'innerTubeOuterDiameter': 0.03415,     #inner tube, Outer Diameter (OD)
                'outerTubeInnerDiameter': 0.045,       #outer tube (annulus), Internal Diameter (ID)
                'length': 50,
                'conductivity' : 237, #[W/m-K]
                }
        coaxialHX = CoaxialHEX(**params)
        coaxialHX.calculate()

        TT.append(Tdew_evap)
        QQ.append(coaxialHX.fluidProps["Cold"].heatTransferCoeffEffective["TwoPhase"])
        Q1.append(coaxialHX.fluidProps["Cold"].heatTransferCoeffEffective["Superheated"])
        print (coaxialHX.heatTransferred)
    pylab.plot(TT,QQ)
    pylab.plot(TT,Q1)
    pylab.show()

def DuessCoax():
    #inputs needed
    # testName
    # cold fluid name
    # cold fluid pressure(s) in
    # cold fluid enthalpy/ies or temperature(s) in
    # cold fluid mass flow(s) in
    # hot fluid name
    # hot fluid pressure(s) in
    # hot fluid enthalpy/ies or temperature(s) in
    # hot fluid mass flow(s) in
    testName = "test"
    coldFluid = Fluid("MEG", "IncompressibleBackend", massFraction=0.50)
    pressuresInCold = [160000] #Pa
    tempInCold = 28.9
    enthalpyInCold = coldFluid.calculateEnthalpy(ThermoProps.PT, pressuresInCold[0], tempInCold + 273.15)
    tempOutCold = 45.15
    massFlowsInCold = [1800] #kg/h
    hotFluid = Fluid("R744", "HEOS")
    pressureInHot = 12400000 #Pa
    tempsInHot = [151]
    tempOutHot = 40
    massFlowsInHot = [192] #kg/h
    uberhitzungRef = 10.0
    # testName = "mass flow cold vs pressure in cold_evap"
    # coldFluid = Fluid("R744", "HEOS")
    # pressuresInCold = range(15, 60, 5) #bar
    # # enthalpyInCold = 255000
    # massFlowsInCold = [30, 50, 100, 150, 200, 250, 300, 350, 400, 450] #kg/h
    # hotFluid = Fluid("MEG", "IncompressibleBackend", massFraction=0.50)
    # pressureInHot = 100000 #Pa
    # tempsInHot = [34.03]
    # massFlowsInHot = [1069]
    # uberhitzungRef = 10.0
    # massFlows = [100, 150, 200, 250, 300, 350, 400, 450]
    # pressures = range(15, 60, 5)
    # coldFluid = Fluid("MEG", "IncompressibleBackend", massFraction=0.50)
    # pressureInCold = 100000 #Pa
    # tempsInCold = [27.629373002977218,24.576487403950637,21.616354956084365,18.762298053789834,16.033583814207475,13.453680663720263,11.04831152421923,8.8421258788523,
    # 27.751328292678693,24.762042328163204,21.87492970188947,19.109728789215126,16.493508860343695,14.058482873742605,11.838358748296912,9.862068387382749,
    # 27.886989201786093,24.971214076547426,22.170164433619846,19.511915917318504,17.033973503538334,14.779671267642186,12.790786669258807,11.095936396152524,
    # 28.0360866278765,25.20399027947599,22.503651455239662,19.97475477596231,17.66923854404297,15.644123484634633,13.945581197783156,13.902421095715226,
    # 28.199880559268024,25.463346195729798,22.882523366127714,20.514160128199364,18.430511914833744,16.70181457065047,16.556218778361597,16.5561976840944,
    # 28.38097815062156,25.755361692577765,23.32085955225199,21.160565394077423,19.373111843971515,19.016878117411068,19.01685734536511,19.016856205328793,
    # 28.583809047697684,26.09088649628643,23.84494036004736,21.97032774273498,21.317803231134064,21.317784271786138,21.317783003703937,21.317782738821734,
    # 28.815856353400363,26.49002979588454,24.50654442579139,23.482298514090417,23.48229731702196,23.48229563185845,23.48229563185845,23.48229563185845,
    # 29.090759893131008,26.994438039210138,25.52691786210073,25.526916265101477,25.526914988442115,25.526914988442115,25.526914988442115,25.526914988442115]
    # massFlowsInCold = [1069]
    # hotFluid = Fluid("Water", "HEOS")
    # pressureInHot = 100000 #Pa
    # tempsInHot = [30]
    # massFlowsInHot = [1800]
    values_wanted = ['tempColdIn', 'tempHotIn', 'tempHotOut', 'tempColdOut', 'resistanceTotal']
    jsonData = {testName: []}
    tc = TemperatureConversions()
    mfc = MassFlowConversions()
    pc = PressureConversions()
    # for i, (tempInCold, massFlowInHot, massFlowInCold) in enumerate(product(tempsInCold, massFlowsInHot, massFlowsInCold)):
    for pressureInCold, massFlowInHot, massFlowInCold in product(pressuresInCold, massFlowsInHot, massFlowsInCold):
        pColdIn = {}

        for tempInHot in tempsInHot:
            tempInHot = tc.convertTemperature(tc.Unit.C, tc.Unit.K, tempInHot)
            params = {
                    'fluidCold': coldFluid,
                    'fluidHot': hotFluid,
                    'massFlowCold': mfc.convertMassFlow(mfc.Unit.KGH, mfc.Unit.KGS, massFlowInCold),
                    'massFlowHot': mfc.convertMassFlow(mfc.Unit.KGH, mfc.Unit.KGS, massFlowInHot),
                    'pressureInCold': pressureInCold,
                    'pressureInHot': pressureInHot,
                    'enthalpyInCold': enthalpyInCold,
                    # 'enthalpyInCold': coldFluid.calculateEnthalpy(ThermoProps.PT, pressureInCold, tc.convertTemperature(tc.Unit.C, tc.Unit.K, tempInCold)),
                    'enthalpyInHot': hotFluid.calculateEnthalpy(ThermoProps.PT, pressureInHot, tempInHot),
                    'innerTubeInnerDiameter': 0.011,      #inner tube, Internal Diameter (ID)
                    'innerTubeOuterDiameter': 0.012,     #inner tube, Outer Diameter (OD)
                    'outerTubeInnerDiameter': 0.023,       #outer tube (annulus), Internal Diameter (ID)
                    'length': 5.519,
                    'conductivity' : 15, #[W/m-K]
                    }
            coaxialHX = CoaxialHEX(**params)
            coaxialHX.tempHotOutReal = tempOutHot + 273.15
            coaxialHX.tempColdOutReal = tempOutCold + 273.15
            coaxialHX.calculate()
            pColdIn["pressureColdIn"] = pressureInCold
            pColdIn["tempColdIn"] = coldFluid.calculateTemperature(ThermoProps.HP, enthalpyInCold, pressureInCold)
            pColdIn["tempColdOut"] = coaxialHX.fluidProps['Cold'].tempOut
            # pColdIn["pressure"] = pressures[i//len(massFlows)]
            # pColdIn["massFlow"] = massFlows[i%len(massFlows)]
            pColdIn["massFlowColdIn"] = massFlowInCold/3600
            pColdIn["resistanceTotal"] = coaxialHX.resistanceTotal
            pColdIn["tempHotIn"] = coaxialHX.fluidProps['Hot'].tempIn
            pColdIn["tempHotOut"] = coaxialHX.fluidProps['Hot'].tempOut
            pColdIn["massFlowHotIn"] = massFlowInHot/3600
            # pColdIn["passes"] = coaxialHX.fluidProps['Cold'].tempOut - tempInCold >= uberhitzungRef
            jsonData[testName].append(pColdIn)
        # fig, ax = plt.subplots()
        # ax.plot(tempHotIn[f"pCold={pressureInCold}, mfHot={massFlowInHot}, mfCold={massFlowInCold}"],
        #         tempColdOut[f"pCold={pressureInCold}, mfHot={massFlowInHot}, mfCold={massFlowInCold}"],
        #         label="temp cold out")
        # ax.plot(tempHotIn[f"pCold={pressureInCold}, mfHot={massFlowInHot}, mfCold={massFlowInCold}"],
        #         tempHotOut[f"pCold={pressureInCold}, mfHot={massFlowInHot}, mfCold={massFlowInCold}"],
        #         label="temp hot out")
        # ax.set_xlabel("Temperature in, hot side (°C)")
        # ax.legend()
    createJsonFile(jsonData, testName)
    for value_wanted in values_wanted:
        createCsvTable(jsonData, testName, value_wanted,
                   sorting=("pressureColdIn", "massFlowColdIn"))
    # plt.show()

def heatExchangerCalcs(filepath):
    # tempOutHot = 40
    # tempOutCold = 45.15
    testName, hexParams, values = getValuesFromInputFile(filepath)
    hexType = hexParams.pop('hexType')
    hexSingleParams = {k: hexParams[k] for k in hexParams.keys() if all([True if y not in k else False for y in ["pressures", "enthalpies", "massFlows"]])}
    outputs = {testName: []}
    allVals = [x for x in values['to_be_extracted']]
    allVals.extend([values['x-axis'], values['y-axis']])
    for pressureInCold, enthalpyInCold, massFlowCold, pressureInHot, enthalpyInHot, massFlowHot in product(
            hexParams['pressuresInCold'], hexParams['enthalpiesInCold'], hexParams['massFlowsCold'],
            hexParams['pressuresInHot'], hexParams['enthalpiesInHot'], hexParams['massFlowsHot']):
        singleOutput = {}
        hexSingleParams = {**hexSingleParams, **{"pressureInCold": pressureInCold, "enthalpyInCold": enthalpyInCold,
                            "massFlowCold": massFlowCold, "pressureInHot": pressureInHot,
                            "enthalpyInHot": enthalpyInHot, "massFlowHot": massFlowHot}}
        if hexType == "Plate-HEX":
            heatExchanger = BrazedPlateHEX(**hexSingleParams)
        else:
            heatExchanger = CoaxialHEX(**hexSingleParams)
        # heatExchanger.tempHotOutReal = tempOutHot + 273.15
        # heatExchanger.tempColdOutReal = tempOutCold + 273.15
        heatExchanger.calculate()
        for value in allVals:
            for temp in ['Cold', 'Hot']:
                if temp in value and 'massFlow' not in value:
                    propertyString = value.replace(temp, '')
                    singleOutput[value] = getattr(heatExchanger.fluidProps[temp], propertyString)
                    break
            else:
                singleOutput[value] = getattr(heatExchanger, value)
        outputs[testName].append(singleOutput)
    createJsonFile(outputs, testName)
    for value_wanted in values['to_be_extracted']:
        createCsvTable(outputs, testName, value_wanted,
                   sorting=(values['x-axis'],  values['y-axis']), reverse=(True, False))



if __name__=='__main__':
    logging.basicConfig(filename="ACHPlog.log", level=logging.DEBUG, encoding='utf-8',
                        format='%(asctime)s - %(name)s.%(methodname)s - %(levelname)s - %(message)s')
    # ACHPCoax()
    # DuessCoax()
    heatExchangerCalcs("C:/Users/smcanana/Documents/Spyder Projects/ACHP/Inputs/coax_double_length.json")
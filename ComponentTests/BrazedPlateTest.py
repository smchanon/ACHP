# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:38:27 2024

@author: smcanana
"""
import numpy as np
import pylab
import logging
from ACHP.models.Fluid import Fluid, ThermoProps
from ACHP.models.HeatExchangers import BrazedPlateHEX
from ACHP.wrappers.CoolPropWrapper import PropsSIWrapper

def WyattPHEHX():
    #cold
    refrigerant = Fluid("R134a", "HEOS")
    #hot
    coolant = Fluid("Water", "HEOS")
    propsSI = PropsSIWrapper()
    tDew = propsSI.calculateTemperatureFromPandQ(refrigerant,962833,1.0)
    params = {
        'fluidCold': refrigerant,
        'massFlowCold': 0.073,
        'pressureInCold': 962833,
        'enthalpyInCold': propsSI.calculateEnthalpyFromTandQ(refrigerant,tDew,0.0), #[J/kg-K]
        # 'xInC': 0.0,
        'fluidHot': coolant,
        'massFlowHot': 100.017,
        'pressureInHot': propsSI.calculatePressureFromTandQ(coolant,115.5+273.15,1),
        'enthalpyInHot': propsSI.calculateEnthalpyFromTandQ(coolant,115.5+273.15,1), #[J/kg-K]d
        #Geometric parameters
        'centerlineDistanceShort': 0.119,
        'centerlineDistanceLong': 0.526, #Center-to-center distance between ports
        'numPlates': 110,
        'amplitude': 0.00102, #[m]
        'thickness': 0.0003, #[m]
        'wavelength': 0.0066, #[m]
        'inclinationAngle': np.pi/3,#[rad]
        'conductivity': 15.0, #[W/m-K]
        'surfaceRoughness': 1.0, #[microns] Surface roughness
        'moreChannels': 'Hot', #Which stream gets the extra channel, 'Hot' or 'Cold'
        'htpColdTuning': 1,
        'htpHotTuning': 1,
        'dpHotTuning': 1,
        'dpColdTuning': 1
    }
    plateHEX = BrazedPlateHEX(**params)
    plateHEX.calculate()

def SWEPVariedmdot():
    #cold
    refrigerant = Fluid("R290", "HEOS")
    #hot
    coolant = Fluid("Water", "HEOS")
    propsSI = PropsSIWrapper()

    temperatureIn = 8+273.15
    for massFlowHot in [0.4176,0.5013,0.6267,0.8357,1.254,2.508]:
        params = {
            'fluidCold': refrigerant,
            'massFlowCold': 0.03312,
            'pressureInCold': propsSI.calculatePressureFromTandQ(refrigerant,temperatureIn,1.0),
            'enthalpyInCold': propsSI.calculateEnthalpyFromTandQ(refrigerant,temperatureIn,0.15), #[J/kg-K]
            'fluidHot': coolant,
            'massFlowHot': massFlowHot,
            'pressureInHot': 200000,
            'enthalpyInHot': propsSI.calculateEnthalpyFromTandP(coolant,15+273.15,200000), #[J/kg-K]
            'centerlineDistanceShort': 0.101,
            'centerlineDistanceLong': 0.455, #Center-to-center distance between ports
            'numPlates': 46,
            'amplitude': 0.00102, #[m]
            'thickness': 0.0003, #[m]
            'wavelength': 0.00626, #[m]
            'inclinationAngle': 65/180*np.pi,#[rad]
            'conductivity': 15.0, #[W/m-K]
            'surfaceRoughness': 1.0, #[microns] Surface roughness
            'moreChannels': 'Hot', #Which stream gets the extra channel, 'Hot' or 'Cold'
            'htpColdTuning': 1,
            'htpHotTuning': 1,
            'dpHotTuning': 1,
            'dpColdTuning': 1
        }
        plateHEX = BrazedPlateHEX(**params)
        plateHEX.calculate()
        print(plateHEX.heatTransferred,',',plateHEX.fluidProps["Hot"].heatTransferCoeffEffective["Subcooled"],',',-plateHEX.fluidProps["Hot"].pressureDrop/1000)
        print(plateHEX.outputList())

def samplePHEHX():
    #cold
    refrigerant = Fluid("R290", "HEOS")
    #hot
    coolant = Fluid("Water", "HEOS")

    massFlow, QQ, Q1 = [], [], []
    temperatureIn = 8+273.15
    propsSI = PropsSIWrapper()
#    for temperatureIn in np.linspace(275,287,101):##
    for massFlowHot in [0.4176,0.5013,0.6267,0.8357,1.254,2.508]:
        params = {
            "fluidHot": coolant,
            "fluidCold": refrigerant,
            "massFlowHot": massFlowHot,
            "massFlowCold": 0.03312,
            "pressureInHot": 200000,
            "pressureInCold": propsSI.calculatePressureFromTandQ(refrigerant, temperatureIn, 1.0),
            "enthalpyInHot": propsSI.calculateEnthalpyFromTandP(coolant, 15+273.15, 200000),
            "enthalpyInCold": propsSI.calculateEnthalpyFromTandQ(refrigerant, temperatureIn, 0.15),
            "centerlineDistanceShort": 0.101,
            "centerlineDistanceLong": 0.455,
            "numPlates": 46,
            "amplitude": 0.00102,
            "thickness": 0.0003,
            "wavelength": 0.00626,
            "inclinationAngle": 65/180*np.pi,
            "conductivity": 15.0,
            "moreChannels": "Hot"
            }
        plateHEX = BrazedPlateHEX(**params)
        plateHEX.calculate()
        massFlow.append(massFlowHot)
        QQ.append(plateHEX.fluidProps["Cold"].heatTransferCoeffEffective["TwoPhase"])#plateHEX.Q/plateHEX.qMax)
        Q1.append(plateHEX.qFlux)#w_2phase_c)#plateHEX.Q/plateHEX.qMax)
#        print(plateHEX.heatTransferred/plateHEX.qMax,plateHEX.heatTransferred)
        print(plateHEX.heatTransferred,',',plateHEX.fluidProps["Hot"].heatTransferCoeffEffective["Subcooled"],',',-plateHEX.fluidProps["Hot"].pressureDrop/1000)
        print(plateHEX.outputList())
    print(massFlow)
    print(QQ)
    print(Q1)
    pylab.plot(massFlow,QQ)
    pylab.show()
    
def BDW16DW():
    refrigerant = Fluid("R744", "HEOS")
    coolant = Fluid("MEG", "IncompressibleBackend", massFraction=0.50)
    T_R_cond_in = 65 + 273.15
    T_W_supply_in = 20 + 273.15
    p_R_in = 7000000 # 7500000 Pa
    p_W_in = 100000 #Pa
    mf_W_condenser = 1200/(60*60) #kg/h / (min/h*sec/min)
    mf_R_cycle = 750/(60*60) #kg/h / (min/h*sec/min)
    #what mf_W for this T_R_cond_out
    T_R_cond_out = 25 + 273.15
    #what T_R_cond_out for this mf_W
    mf_W = 2000/(60*60) #kg/h / (min/h*sec/min)
    params = {
        'fluidHot': refrigerant,
        'fluidCold': coolant,
        "massFlowHot": mf_R_cycle,
        "massFlowCold": mf_W_condenser,
        "pressureInHot": p_R_in,
        "pressureInCold": p_W_in,
        "enthalpyInHot": refrigerant.calculateEnthalpy(ThermoProps.PT, p_R_in, T_R_cond_in),
        "enthalpyInCold": coolant.calculateEnthalpy(ThermoProps.PT, p_W_in, T_W_supply_in),
        "centerlineDistanceShort": 0.072, # m
        "centerlineDistanceLong": 0.329, # m
        "numPlates": 12,
        "volumeChannelSingle": 0.000061, # m^3
        "thickness": 0.0003, # guess, taken from PHEHX SamplePHEHX
        "conductivity": 15.37546, # 316 stainless steel
        "moreChannels": "Cold" #default for SWEP BHPs
        }
    plateHEX = BrazedPlateHEX(**params)
    plateHEX.calculate()
    print(plateHEX.fluidProps['Hot'].tempOut - 273.15)
    print(plateHEX.fluidProps['Cold'].tempOut - 273.15)

if __name__=='__main__':
    logging.basicConfig(filename="ACHPlog.log", level=logging.DEBUG, encoding='utf-8',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    BDW16DW()
    # samplePHEHX()
    #WyattPHEHX() #has two-phase to two-phase heat transfer, which is not yet implemented
    # SWEPVariedmdot()

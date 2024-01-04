# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 07:27:43 2023

@author: SMCANANA
"""
# from __future__ import division, print_function, absolute_import
from ACHP.wrappers.coolPropWrapper import AbstractStateWrapper, PropsSIWrapper
from ACHP.OilPropLib import Solubility_Ref_in_Liq, rho_oil

class CompressorClass():
    """
    Compressor Model based on 10-coefficient Model from `ANSI/AHRI standard 540 
    <http://www.ahrinet.org/App_Content/ahri/files/standards%20pdfs/ANSI%20standards%20pdfs/ANSI-ARI-540-2004%20latest.pdf>`_

    Required Parameters:

    ===========         ==========  ======================================================================
    Variable            Units       Description
    ===========         ==========  ======================================================================
    massFlowCoeffs      Ibm/hr      A numpy-like list of compressor map coefficients for mass flow
    powerCoeffs         Watts       A numpy-like list of compressor map coefficients for electrical power
    refrigerant         N/A         A string representing the refrigerant
    oil                 N/A         A string representing the lubricant oil
    tempInR             K           Refrigerant inlet temperature
    pressureInR         Pa          Refrigerant suction pressure (absolute)
    pressureOutR        Pa          Refrigerant discharge pressure (absolute)
    ambientPowerLoss    --          Fraction of electrical power lost as heat to ambient
    vDotRatio           --          Displacement Scale factor
    volumeOilSump       m^3         Total volume of oil sump inside the compressor shell
    shellPressure       N/A         A string defining the shell pressure of the compressor
    backEnd             N/A         A string defining the back end calculation method
    ===========         ==========  ======================================================================

    All variables are of double-type unless otherwise specified

    """

    def __init__(self, massFlowCoeffs: list, powerCoeffs : list, refrigerant: str, oil: str, 
                 tempInR: float, pressureInR: float, pressureOutR: float, ambientPowerLoss: float,
                 vDotRatio: float, volumeOilSump: float, shellPressure: str, backEnd='HEOS'):
        #inputs
        self.massFlowCoeffs = massFlowCoeffs
        self.powerCoeffs = powerCoeffs
        self.refrigerant = refrigerant
        self.oil = oil
        self.tempInR = tempInR
        self.pressureInR = pressureInR
        self.pressureOutR = pressureOutR
        self.ambientPowerLoss = ambientPowerLoss
        self.vDotRatio = vDotRatio
        self.volumeOilSump = volumeOilSump
        self.shellPressure = shellPressure
        self.backEnd = backEnd

# params={
#     'massFlowCoeffs':massFlowCoeffs,
#     'powerCoeffs':elecPowerCoeffs,
#     'refrigerant':Cycle.Ref,                                                              #refrigerant
#     'oil':Cycle.Oil, #Compressor lubricant oil
#     'volumeOilSump':0, #Volume of oil in the sump
#     'shellPressure':Cycle.shell_pressure, #Compressor shell pressure
#     'ambientPowerLoss':0.15, #Fraction of electrical power lost as heat to ambient            #shell heat loss
#     'vDotRatio': 1.0, #Displacement Scale factor                               #up- or downsize compressor (1=original)
#     'Verbosity': 0, # How verbose should the debugging statements be [0 to 10]

#   }
# Cycle.Compressor = Compressor(**params)

        #outputs
        self.power = 0
        self.massFlowR = 0
        self.tempOutR = 0
        self.enthalpyInR = 0
        self.enthalpyOutR = 0
        self.overallIsentropicEfficiency = 0
        self.vDotPumped = 0
        self.ambientHeatLoss = 0
        self.refrigerantChangeOilSump = 0

    def outputList(self):
        """
            Return a list of parameters for this component for further output

            It is a list of tuples, and each tuple is formed of items with indices:
                [0] Description of value

                [1] Units of value

                [2] The value itself
        """
        outputList = []
        for i, massFlowCoeff in enumerate(self.massFlowCoeffs):
            outputList.append(('M{i:d}','-',massFlowCoeff))
        for i, powerCoeff in enumerate(self.powerCoeffs):
            outputList.append(('P{i:d}','-',powerCoeff))
        outputList.append(('Heat Loss Fraction','-',self.ambientPowerLoss))
        outputList.append(('Displacement Scale Factor','-',self.vDotRatio))
        outputList.append(('Power','W',self.power))
        outputList.append(('Mass Flow Rate','kg/s',self.massFlowR))
        outputList.append(('Inlet Temperature','K',self.tempInR))
        outputList.append(('Outlet Temperature','K',self.tempOutR))
        outputList.append(('Inlet Enthalpy','J/kg',self.enthalpyInR))
        outputList.append(('Outlet Enthalpy','J/kg',self.enthalpyOutR))
        outputList.append(('Overall Isentropic Efficiency','-',self.overallIsentropicEfficiency))
        outputList.append(('Pumped Flow Rate','m^3/s',self.vDotPumped))
        outputList.append(('Ambient Heat Loss','W',self.ambientHeatLoss))
        outputList.append(('Refrigerant change in oil sump','kg',self.refrigerantChangeOilSump))
        return outputList

    def calculate(self):
        """
        Calculates everything :(

        Returns
        -------
        None.

        """
        abstractState = AbstractStateWrapper(self.backEnd, self.refrigerant)
        # abstractState = CP.AbstractState(self.backEnd, self.refrigerant)

        self.tempSatSuperheatK = abstractState.calculateTempFromPandQ(self.pressureInR, 1.0)
        self.tempSatDewK = abstractState.calculateTempFromPandQ(self.pressureOutR, 1.0)
        self.dTSuperheatK = self.tempInR-self.tempSatSuperheatK
        tempSatSuperheatF = self.convertKToF(self.tempSatSuperheatK)
        tempSatDewF = self.convertKToF(self.tempSatDewK)

        powerMap = self.applyCoeffs(tempSatDewF, tempSatSuperheatF, self.powerCoeffs)
        massFlowMap = self.applyCoeffs(tempSatDewF, tempSatSuperheatF, self.massFlowCoeffs)
        powerMap = self.scaleParameter(powerMap, self.vDotRatio)
        massFlowMap = self.scaleParameter(self.convertLbmphToKgps(massFlowMap), self.vDotRatio)

        temp1Actual = self.tempSatSuperheatK + self.dTSuperheatK
        temp1Map = self.tempSatSuperheatK + 20*5/9

        specificVolumeMap = abstractState.calculateSpecificVolumeFromPandT(self.pressureInR, temp1Map)
        entropyMap = abstractState.calculateEntropyFromPandT(self.pressureInR, temp1Map)
        enthalpyMap = abstractState.calculateEnthalpyFromPandT(self.pressureInR, temp1Map)
        
        specificVolumeActual = abstractState.calculateSpecificVolumeFromPandT(self.pressureInR, temp1Actual)
        self.entropyInR = abstractState.calculateEntropyFromPandT(self.pressureInR, temp1Actual)
        self.enthalpyInR = abstractState.calculateEnthalpyFromPandT(self.pressureInR, temp1Actual)
        self.massFlowR = (1 + 0.75*(specificVolumeMap/specificVolumeActual - 1))*massFlowMap

        h2sMap = abstractState.calculateEnthalpyFromPandS(self.pressureOutR, entropyMap)
        h2sActual = abstractState.calculateEnthalpyFromPandS(self.pressureOutR, self.entropyInR)

        #Shaft power based on 20F superheat calculation from fit overall isentropic efficiency
        self.power = powerMap*(self.massFlowR/massFlowMap)*(h2sActual - self.enthalpyInR)/\
            (h2sMap - enthalpyMap)

        self.enthalpyOutR = self.power*(1 - self.ambientPowerLoss)/self.massFlowR + self.enthalpyInR
        self.overallIsentropicEfficiency=self.massFlowR*(h2sActual-self.enthalpyInR)/(self.power)

        self.tempOutR = abstractState.calculateTempFromPandH(self.pressureOutR, self.enthalpyOutR)
        self.entropyOutR = abstractState.calculateEntropyFromPandH(self.pressureOutR, self.enthalpyOutR)

        self.cycleEnergyIn=self.power*(1-self.ambientPowerLoss)
        self.vDotPumped= self.massFlowR*specificVolumeActual
        self.ambientHeatLoss=-self.ambientPowerLoss*self.power

        # Estimate refrigerant dissolved in the oil sump
        tempAvg = (temp1Actual + self.tempOutR)/2
        if self.shellPressure == 'high-pressure':
            pShell = self.pressureOutR
        elif self.shellPressure == 'low-pressure':
            pShell = self.pressureInR

        self.xRef,error = Solubility_Ref_in_Liq(self.refrigerant,self.oil,tempAvg,pShell/1000)

        rhoMassOil = rho_oil(self.oil,tempAvg-273.15)
        self.massOil = self.volumeOilSump*rhoMassOil

        # Amount of refrigerant dissolved in the oil sump
        self.refrigerantChangeOilSump = self.massOil*self.xRef/(1-self.xRef)

    def convertKToF(self, temperature):
        """
        Converts temperatures from Kelvin to Fahrenheit

        Parameters
        ----------
        temperature : float
            temperature in Kelvin

        Returns
        -------
        float
            temperature in Fahrenheit

        """
        return temperature*9/5 - 459.67

    def applyCoeffs(self, tempSatDewF, tempSatSuperheatF, coefficients):
        """
        Applies a list of 10 coefficients from a 10-coefficient ARI compressor map to the
        saturation temperatures of the refrigerant. Used to describe power in W or
        mass flow in lbm/hr

        Parameters
        ----------
        tempSatDewF : float
            Dew saturation temperature
        tempSatSuperheatF : float
            Superheat saturation temperature
        coefficients : list
            list of 10 coefficients

        Returns
        -------
        float
            map-based version of what the coefficients are describing

        """
        return coefficients[0] + coefficients[1]*tempSatSuperheatF + \
            coefficients[2]*tempSatDewF + coefficients[3]*tempSatSuperheatF**2 + \
            coefficients[4]*tempSatSuperheatF*tempSatDewF + \
            coefficients[5]*tempSatDewF**2 + coefficients[6]*tempSatSuperheatF**3 + \
            coefficients[7]*tempSatDewF*tempSatSuperheatF**2 + \
            coefficients[8]*tempSatDewF**2*tempSatSuperheatF + \
            coefficients[9]*tempSatDewF**3

    def convertLbmphToKgps(self, lbmph):
        """
        Converts lbm/hr to kg/s

        Parameters
        ----------
        lbmph : float
            mass flow in lbm/hr

        Returns
        -------
        float
            mass flow in kg/s

        """
        return lbmph * 0.000125998

    def scaleParameter(self, parameterMap, ratio):
        """
        Scales a parameter using a given ratio

        Parameters
        ----------
        parameterMap : float
            parameter to be scaled
        ratio : float
            scaling ratio

        Returns
        -------
        float
            Scaled parameter

        """
        return parameterMap*ratio

if __name__=='__main__':
    #Abstract State
    REFRIGERANT = 'R134a'
    BACKEND = 'HEOS' #choose between: 'HEOS','TTSE&HEOS','BICUBIC&HEOS','REFPROP','SRK','PR'
    propsSI = PropsSIWrapper(REFRIGERANT)
    for i in range(1):
        kwds={
              'massFlowCoeffs':[217.3163128,5.094492028,-0.593170311,4.38E-02,-2.14E-02,1.04E-02,7.90E-05,-5.73E-05,1.79E-04,-8.08E-05],
              'powerCoeffs':[-561.3615705,-15.62601841,46.92506685,-0.217949552,0.435062616,-0.442400826,2.25E-04,2.37E-03,-3.32E-03,2.50E-03],
              'backEnd': BACKEND,
              'refrigerant': REFRIGERANT,
              'tempInR':280,
              'pressureInR':propsSI.calculatePressureFromTandQ(279,1),
              'pressureOutR':propsSI.calculatePressureFromTandQ(315,1),
              'ambientPowerLoss':0.15, #Fraction of electrical power lost as heat to ambient
              'vDotRatio': 1.0, #Displacement Scale factor
              'shellPressure': 'low-pressure',
              'oil': 'POE32',
              'volumeOilSump': 0.0,
              }
        Comp=CompressorClass(**kwds)
        Comp.calculate()
        print ('Power:', Comp.power,'W')
        print ('Flow rate:',Comp.vDotPumped,'m^3/s')
        print ('Heat loss rate:', Comp.ambientHeatLoss, 'W')
        print ('Refrigerant dissolved in oil sump:', Comp.refrigerantChangeOilSump,'kg')

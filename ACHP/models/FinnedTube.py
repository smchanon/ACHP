# -*- coding: utf-8 -*-
"""
Possibly only air-side calculations
Created on Fri Dec 22 10:41:29 2023

@author: SMCANANA
"""
from enum import StrEnum
import numpy as np
from ACHP.wrappers.CoolPropWrapper import HumidAirPropertiesWrapper

class Tubes():
    """
    Tube settings for fins and tubes heat exchanger
    """

    def __init__(self, numPerBank: int, numBanks: int, numCircuits: int,
                 length: float, innerDiam: float, outerDiam: float,
                 distFlow: float, distOrtho: float, thermalConductivityWall: float):
        assert 0.1 <= numPerBank <= 100, "Number of tubes per bank must be between 0.1 and 100"
        assert 1 <= numBanks <= 50, "Number of banks must be between 1 and 50"
        assert 1 <= numCircuits <= 50, "Number of circuits must be between 1 and 50"
        assert 0.001 <= length <= 10, "Length of tube must be between 0.001 and 10"
        assert 0.0001 <= innerDiam <= 1, "Inner diameter of tube must be between 0.0001 and 1"
        assert 0.0001 <= outerDiam <= 1, "Outer diameter of tube must be between 0.0001 and 1"
        assert 0.0001 <= distFlow <= 1, "Distance between center of tubes in flow direction must \
            be between 0.0001 and 1"
        assert 0.0001 <= distOrtho <= 1, "Distance between center of tubes orthogonal to flow \
            direction must be between 0.0001 and 1"
        assert 0.01 <= thermalConductivityWall <= 10000, "Thermal conductivity must be between \
            0.01 and 10000"
        self.numPerBank = numPerBank #number of tubes per bank/row
        self.numBanks = numBanks #number of banks
        self.numCircuits = numCircuits #number of circuits
        self.length = length #total tube length
        self.innerDiam = innerDiam #tube inner diameter
        self.outerDiam = outerDiam #tube outer diameter
        self.distFlow = distFlow #distance between center of tubes in flow direction
        self.distOrtho = distOrtho #distance between center of tubes orthogonal to flow direction
        self.thermalConductivityWall = thermalConductivityWall #wall thermal conductivity

    def __str__(self):
        tubeString = ''
        tubeVars = vars(self)
        for var in tubeVars:
            tubeString +=f'{var}: {tubeVars[var]}\n'
        return tubeString

    def calculateHeight(self):
        """

        Returns
        -------
        float
            Height of heat exchanger based on the distance between the center
            of the tubes orthogonal to flow, in m.

        """
        return self.distOrtho * (self.numPerBank + 1)

    def calculateDuctArea(self):
        """

        Returns
        -------
        float
            area of duct neglecting any additional fin height

        """
        return self.calculateHeight() * self.length

    def calculateDuctCrossSectionalArea(self, finOuterDiam, finThickness,
                                        numFinsInTubeSheet):
        """
        Calculates the cross-sectional area of the duct formed by the fin/tube
        heat exchanger

        Parameters
        ----------
        finOuterDiam : float
            outer diameter of the fin
        finThickness : float
            thickness of the fin
        numFinsInTubeSheet : float
            number of fins in the tube sheet

        Returns
        -------
        float
            cross-sectional area of duct

        """
        tubeSpaceHeight = self.calculateHeight() - finOuterDiam*self.numPerBank
        tubeSpaceLength = self.numPerBank*finOuterDiam*self.length
        return self.calculateDuctArea() - finThickness*numFinsInTubeSheet*\
            tubeSpaceHeight - tubeSpaceLength

    def calculateTubeOuterArea(self):
        """
        Calculates the outer surface area of the tube

        Returns
        -------
        float
            outer surface area of tube

        """
        return self.numPerBank*self.numBanks*np.pi*self.outerDiam*self.length

    def calculateEffectiveRadiusRatio(self):
        """
        Calculates circular fin radius divided by the outer radius of the tube, also known as
        the effective radius ratio of the tubes

        Returns
        -------
        float
            effective radius ratio

        """
        lateralRadialDistance = self.outerDiam/2
        diagonalRadialDistance = np.sqrt(self.distFlow**2 + (self.distOrtho**2)/4)/2
        orthogonalRadialDistance = self.distOrtho/2
        return 1.27*orthogonalRadialDistance/lateralRadialDistance*\
            np.sqrt(diagonalRadialDistance/orthogonalRadialDistance - 0.3)

class Air():
    """
    "air" settings for fins and tubes heat exchanger.
    """

    def __init__(self, vDotHA: float, tempDryBulb: float, pressure: float, relativeHumidity: float,
                 fanPower: float, correction=1.0):
        assert 0.001 <= vDotHA <= 10, "Volumetric flow must be between 0.001 and 10"
        assert (-80+273.15) <= tempDryBulb <= (200+273.15), "Dry bulb temperature must be between\
             193.15 and 473.15"
        assert 10.0 <= pressure <= 10000000, "Pressure must be between 10 and 10000000"
        assert 0.0 <= relativeHumidity <= 1.0, "Relative humidity must be between 0 and 1"
        assert 0.0 <= fanPower <= 4000, "Fan power must be between 0 and 4000"
        self.vDotHA = vDotHA
        self.tempDryBulb = tempDryBulb
        self.pressure = pressure
        self.relativeHumidity = relativeHumidity
        self.fanPower = fanPower
        self.correction = correction
        #TODO: get rid of humid air props and make it for all coolants
        humidAirProps = HumidAirPropertiesWrapper()
        self.humidityRatio = humidAirProps.calculateHumidityRatio(self.tempDryBulb, self.pressure,
                self.relativeHumidity)
        self.enthalpyDryAir = humidAirProps.calculateDryAirEnthalpy(self.tempDryBulb, self.pressure,
                self.humidityRatio)
        self.volumeDryAir = humidAirProps.calculateDryAirVolume(self.tempDryBulb, self.pressure,
                self.humidityRatio)
        self.viscosity = humidAirProps.calculateViscosity(self.tempDryBulb, self.pressure,
                self.humidityRatio)
        self.thermalConductivity = humidAirProps.calculateThermalConductivity(self.tempDryBulb,
                self.pressure, self.humidityRatio)
        self.heatCapacityDryAir = self.calculateHeatCapacityDryAir() #[J/kg_ha/K]
        self.heatCapacityHumidAir = self.heatCapacityDryAir/(1 + self.humidityRatio) #[J/kg_ha/K]
        self.prandtlNumber = self.calculatePrandtlNumber()
        self.rhoHumidAir, self.rhoDryAir = self.calculateRho()
        self.massFlowHumidAir, self.massFlowDryAir = self.calculateMassFlowRate()

    def __str__(self):
        airString =  f'volume flow of humid air: {self.vDotHA}\n'
        airString += f'dry bulb temperature: {self.tempDryBulb}\n'
        airString += f'pressure: {self.pressure}\n'
        airString += f'relative humidity: {self.relativeHumidity}\n'
        airString += f'fan power: {self.fanPower}\n'
        return airString

    def calculateRho(self):
        """
        Calculates rho of the current fluid

        Returns
        -------
        tuple
            rho of humid air, rho of dry air

        """
        rhoHumidAir = 1/self.volumeDryAir*(1 + self.humidityRatio)
        rhoDryAir = 1/self.volumeDryAir
        return rhoHumidAir, rhoDryAir

    def calculateMassFlowRate(self):
        """
        Calculates the mass flow rate of the current fluid

        Returns
        -------
        tuple
            mass flow of humid air, mass flow of dry air

        """
        massFlowHumidAir = self.vDotHA*self.rhoHumidAir
        massFlowDryAir = self.vDotHA*self.rhoDryAir
        return massFlowHumidAir, massFlowDryAir

    def calculateHeatCapacityDryAir(self):
        """
        Calculates the heat capacity of dry air

        Returns
        -------
        float
            heat capacity of dry air

        """
        deltaT = 0.0001 #[K]
        return (HumidAirPropertiesWrapper().calculateDryAirEnthalpy(
            self.tempDryBulb + deltaT, self.pressure, self.humidityRatio)\
             - self.enthalpyDryAir)/deltaT

    def calculatePrandtlNumber(self):
        """
        Calculates the Prandtl number of the current fluid

        Returns
        -------
        float
            Prandtl number of the current fluid

        """
        return self.heatCapacityHumidAir*self.viscosity/self.thermalConductivity

    def calculateReynoldsNumber(self, flowSpeed, characteristicLength):
        """
        Calculates the Reynolds number of the current fluid

        Parameters
        ----------
        flowSpeed : float
            fluid flow speed
        characteristicLength : float
            Reynolds number characteristic length

        Returns
        -------
        float
            Reynolds number of the current fluid

        """
        return self.rhoHumidAir*flowSpeed*characteristicLength/self.viscosity

class FinType(StrEnum):
    """
    Fin type enum
    """
    HERRINGBONE = 'HerringboneFins'
    WAVYLOUVERED = 'WavyLouveredFins'
    PLAIN = 'PlainFins'

class Fins():
    """
    Fin settings for fins and tubes heat exchanger

    Required inputs:
        finsPerInch             number of fins per inch
        amplitude               amplitude of fin
        period                  period of fin
        thickness               thickness of fin
        thermalConductivity     thermal conductivity of fin material
    """

    def __init__(self, finsPerInch: float, thickness: float, thermalConductivity: float):
        assert 0.1 <= finsPerInch <= 100, "Fins per inch must be between 0.1 and 100"
        assert 0.00001 <= thickness <= 0.01, "Thickness must be between 0.00001 and 0.01"
        assert 0.01 <= thermalConductivity <= 10000, "Thermal conductivity must be between \
            0.01 and 10000"
        self.finsPerInch = finsPerInch
        self.thickness = thickness
        self.thermalConductivity = thermalConductivity
        self.finsPerMeter = self.calculateFinsPerMeter()
        self.finPitch = self.calculateFinPitch()
        self.finType = ''

    def __str__(self):
        finString = ''
        finVars = vars(self)
        for var in finVars:
            finString +=f'{var}: {finVars[var]}\n'
        return finString

    def calculateFinsPerMeter(self):
        """
        Calculates the number of fins per meter given the number of fins per inch

        Returns
        -------
        float
            number of fins per meter

        """
        return self.finsPerInch/0.0254

    def calculateFinPitch(self):
        """
        Calculates the fin pitch (meters per fin)

        Returns
        -------
        float
            fins per meter^-1

        """
        return 1/self.finsPerMeter

    def finSpacing(self):
        """
        Calculates the fin spacing based on fins pitch and fin thickness

        Returns
        -------
        float
            spacing between fins

        """
        return self.finPitch  - self.thickness

    def calculateOutsideDiameter(self, tubeOuterDiameter):
        """
        Calculates the outer diameter of the tubes. Somehow changes based on fin
        geometry?

        Parameters
        ----------
        tubeOuterDiameter : float
            outer diameter of tubes

        Returns
        -------
        float
            corrected outer diameter of tubes

        """
        return tubeOuterDiameter

    def calculateAreaIncrease(self):
        """
        Calculates the increase in area created by different fin geometries

        Returns
        -------
        float
            fin geometry area correction

        """
        return 1.0

    def calculateColburnJFactor(self, reynoldsNumberAir, tubeOuterDiameter,
                tubeNumBanks, distFlow, distOrtho, totalArea, tubeOuterArea,
                crossSectionalArea):
        """
        Calculates the Chilton and Colburn j factor depending on the type of fin

        Parameters
        ----------
        reynoldsNumberAir : float
            Reynolds number of air/coolant fluid
        tubeOuterDiameter : float
            diameter of the outer surface of tube
        tubeNumBanks : float
            number of banks of tubes
        distFlow : float
            distance between tubes in the direction of flow
        distOrtho : float
            distance between tubes orthogonal to the direction of flow
        totalArea : float
            total surface area of heat exchanger
        tubeOuterArea : float
            outer surface area of tube
        crossSectionalArea : float
            cross-sectional area of the duct

        Returns
        -------
        float
            Chilton and Colburn j factor

        """

    def calculateAirSideFrictionFactor(self, reynoldsNumber, totalArea, tubeOuterArea,
                tubeOuterDiameter, tubeNumBanks, tubeDistOrtho, tubeDistFlow):
        """
        Calculates the air side friction factor depending on the type of fin

        Parameters
        ----------
        reynoldsNumber : float
            Reynolds number of air/coolant fluid
        totalArea : float
            total surface area of heat exchanger
        tubeOuterArea : float
            outer surface area of tube
        tubeOuterDiameter : float
            diameter of the outer surface of tube
        tubeNumBanks : float
            number of banks of tubes
        tubeDistOrtho : float
            distance between tubes orthogonal to the direction of flow
        tubeDistFlow : float
            distance between tubes in the direction of flow

        Returns
        -------
        float
            air-side friction factor

        """

class TexturedFins(Fins):
    """
    Textured fin umbrella type

    Required inputs:
        finsPerInch             number of fins per inch
        amplitude               amplitude of fin
        period                  period of fin
        thickness               thickness of fin
        thermalConductivity     thermal conductivity of fin material
    """

    def __init__(self, finsPerInch, thickness, thermalConductivity,
                 amplitude: float, period: float):
        super().__init__(finsPerInch, thickness, thermalConductivity)
        assert 0.00001 <= amplitude <= 0.01, "Amplitude must be between 0.00001 and 0.01"
        assert 0.00001 <= period <= 0.01, "Period must be between 0.00001 and 0.01"
        self.amplitude = amplitude
        self.period = period

class WavyLouveredFins(TexturedFins):
    """
    Fin type: wavy louvered

    Correlations from:
    Chi-Chuan Wang and Yu-Min Tsai and Ding-Chong Lu, 1998, "Comprehensive
    Study of Convex-Louver and Wavy Fin-and-Tube Heat Exchangers", Journal
    of Thermophysics and Heat Transfer

    || -    xf    ->
    ^              ==                          ==
    |           ==  |  ==                  ==
    Pd       ==     |     ==            ==
    |     ==        |        ==     ==
    =  ==           s             ==
                    |
                    |
                    |
                   ==                        ==
                ==     ==                 ==
             ==           ==           ==
          ==                 ==     ==
       ==                        ==

     t: thickness of fin plate
     Pf: fin pitch (centerline-to-centerline distance between fins)
     Pd: indentation for waviness (not including fin thickness)
     s: fin spacing (free space between fins) = Pf-t



                 |--       Pl      -|
                ___                 |
              /     \               |
       =     |       |              |
       |      \ ___ /               |
       |                            |
       |                           ___
       |                         /     \      |
      Pt                        |       |     D
       |                         \ ___ /      |
       |
       |        ___
       |      /     \
       =     |       |
              \ ___ /

    Required inputs:
        finsPerInch             number of fins per inch
        amplitude               amplitude of fin
        period                  period of fin
        thickness               thickness of fin
        thermalConductivity     thermal conductivity of fin material
    """

    def __init__(self, finsPerInch, thickness, thermalConductivity, amplitude,
                 period):
        super().__init__(finsPerInch, thickness, thermalConductivity,
                         amplitude, period)
        self.finType = FinType.WAVYLOUVERED


    def calculateAreaIncrease(self):
        return np.sqrt(self.period**2 + self.amplitude**2)/self.period

    def calculateColburnJFactor(self, reynoldsNumberAir, tubeOuterDiameter,
                tubeNumBanks, distFlow, distOrtho, totalArea, tubeOuterArea,
                crossSectionalArea):
        return 16.06*pow(reynoldsNumberAir,-1.02*(self.finPitch/tubeOuterDiameter) - 0.256)*\
            pow(totalArea/tubeOuterArea,-0.601)*pow(tubeNumBanks,-0.069)*\
            pow(self.finPitch/tubeOuterDiameter,0.84)

    def calculateAirSideFrictionFactor(self, reynoldsNumber, totalArea, tubeOuterArea,
                tubeOuterDiameter, tubeNumBanks, tubeDistOrtho, tubeDistFlow):
        if reynoldsNumber < 1e3:
            valA = 0.264
            valB = 0.105
            valC = 0.708
            valD = 225.0
            valE = -0.637
            valF = 0.263
            valG = -0.317
        else:
            valA = 0.768
            valB = 0.0494
            valC = 0.142
            valD = 1180.0
            valE = 0
            valF = 0.0195
            valG = -0.121
        return valA*(valB + valC*np.exp(-reynoldsNumber/valD))*pow(reynoldsNumber, valE)*\
            pow(totalArea/tubeOuterArea, valF)*pow(self.finPitch/tubeOuterDiameter, valG)

class HerringboneFins(TexturedFins):
    """
    Fin type: herringbone

    Source:
    Empirical correlations for heat transfer and flow friction characteristics of herringbone wavy
    fin-and-tube heat exchangers
    Chi-Chuan Wang, Young-Ming Hwang, Yur-Tsai Lin
    International Journal of Refrigeration, 25, 2002, 637-680

    Required inputs:
        finsPerInch             number of fins per inch
        amplitude               amplitude of fin
        period                  period of fin
        thickness               thickness of fin
        thermalConductivity     thermal conductivity of fin material
    """

    def __init__(self, finsPerInch, thickness, thermalConductivity, amplitude,
                 period):
        super().__init__(finsPerInch, thickness, thermalConductivity,
                         amplitude, period)
        self.finType = FinType.HERRINGBONE

    def calculateAreaIncrease(self):
        #TODO: !!used wavy louvered fins definition - there seems to be a bug in the paper !!
        return np.sqrt(self.period**2 + self.amplitude**2)/self.period

    def calculateOutsideDiameter(self, tubeOuterDiameter):
        return tubeOuterDiameter + 2*self.thickness

    def calculateColburnJFactor(self, reynoldsNumberAir, tubeOuterDiameter,
                tubeNumBanks, distFlow, distOrtho, totalArea, tubeOuterArea,
                crossSectionalArea):
        tanTheta = self.amplitude/self.period
        beta = (np.pi*tubeOuterDiameter**2)/(4.0*distOrtho*distFlow)
        oneMinusBeta = 1.0 - beta
        hydraulicDiameter = 2.0*self.finSpacing()*oneMinusBeta/\
            (oneMinusBeta*self.calculateAreaIncrease() + 2*self.finSpacing()*beta/tubeOuterDiameter)
        if reynoldsNumberAir < 1e3:
            jFactor1 = 0.0045 - 0.491*pow(reynoldsNumberAir,-0.0316 - \
                0.0171*np.log(tubeNumBanks*tanTheta))*\
                pow(distFlow/distOrtho,-0.109*np.log(tubeNumBanks*tanTheta))*\
                pow(tubeOuterDiameter/hydraulicDiameter,0.542 + 0.0471*tubeNumBanks)*\
                pow(self.finSpacing()/tubeOuterDiameter,0.984)*\
                pow(self.finSpacing()/distOrtho,-0.349)
            jFactor2 = -2.72 + 6.84*tanTheta
            jFactor3 = 2.66*tanTheta
            jFactor = 0.882*pow(reynoldsNumberAir,jFactor1)*\
                pow(tubeOuterDiameter/hydraulicDiameter,jFactor2)*\
                pow(self.finSpacing()/distOrtho,jFactor3)*\
                pow(self.finSpacing()/tubeOuterDiameter,-1.58)*pow(tanTheta,-0.2)
        else:
            jFactor1 = -0.0545 - 0.0538*tanTheta - 0.302*pow(tubeNumBanks,-0.24)*\
                pow(self.finSpacing()/distFlow,-1.3)*pow(distFlow/distOrtho,0.379)*\
                pow(distFlow/hydraulicDiameter,-1.35)*pow(tanTheta,-0.256)
            jFactor2 = -1.29*pow(distFlow/distOrtho,1.77 - 9.43*tanTheta)*\
                pow(tubeOuterDiameter/hydraulicDiameter,0.229-1.43*tanTheta)*\
                pow(tubeNumBanks,-0.166-1.08*tanTheta)*\
                pow(self.finSpacing()/distOrtho,-0.174*np.log(0.5*tubeNumBanks))
            jFactor = 0.0646*pow(reynoldsNumberAir,jFactor1)*\
                pow(tubeOuterDiameter/hydraulicDiameter,jFactor2)*\
                pow(self.finSpacing()/distOrtho,-1.03)*pow(distFlow/tubeOuterDiameter,0.432)*\
                pow(tanTheta,-0.692)*pow(tubeNumBanks,-0.737)
        return jFactor

    def calculateAirSideFrictionFactor(self, reynoldsNumber, totalArea, tubeOuterArea,
                tubeOuterDiameter, tubeNumBanks, tubeDistOrtho, tubeDistFlow):
        tanTheta = self.amplitude/self.period
        beta = (np.pi*tubeOuterDiameter**2)/(4.0*tubeDistOrtho*tubeDistFlow)
        oneMinusBeta = 1.0 - beta
        finSpacing = self.finSpacing()
        hydraulicDiameter=2.0*self.finSpacing()*oneMinusBeta/\
            (oneMinusBeta*self.calculateAreaIncrease() + 2*self.finSpacing()*beta/tubeOuterDiameter)
        if reynoldsNumber<1000.0:
            frictionFactor1 = -0.574 - 0.137*pow(np.log(reynoldsNumber) - 5.26,0.245)*\
                pow(tubeDistOrtho/tubeOuterDiameter,-0.765)*\
                pow(tubeOuterDiameter/hydraulicDiameter,-0.243)*\
                pow(finSpacing/hydraulicDiameter,-0.474)*pow(tanTheta,-0.217)*\
                pow(tubeNumBanks,0.035)
            frictionFactor2 = -3.05*tanTheta
            frictionFactor3 = -0.192*tubeNumBanks
            frictionFactor4 = -0.646*tanTheta
            frictionFactor=4.37*pow(reynoldsNumber,frictionFactor1)*\
                pow(finSpacing/hydraulicDiameter,frictionFactor2)*\
                pow(tubeDistFlow/tubeDistOrtho,frictionFactor3)*\
                pow(tubeOuterDiameter/hydraulicDiameter,0.2054)*pow(tubeNumBanks,frictionFactor4)
        else:
            frictionFactor1=-0.141*pow(finSpacing/tubeDistFlow,0.0512)*pow(tanTheta,-0.472)*\
                pow(tubeDistFlow/tubeDistOrtho,0.35)*\
                pow(tubeDistOrtho/hydraulicDiameter,0.449*tanTheta)*\
                pow(tubeNumBanks,-0.049+0.237*tanTheta)
            frictionFactor2=-0.562*pow(np.log(reynoldsNumber),-0.0923)*pow(tubeNumBanks,0.013)
            frictionFactor3=0.302*pow(reynoldsNumber,0.03)*\
                pow(tubeDistOrtho/tubeOuterDiameter,0.026)
            frictionFactor4=-0.306+3.63*tanTheta
            frictionFactor=0.228*pow(reynoldsNumber,frictionFactor1)*\
                pow(tanTheta,frictionFactor2)*pow(finSpacing/tubeDistFlow,frictionFactor3)*\
                pow(tubeDistFlow/tubeOuterDiameter,frictionFactor4)*\
                pow(tubeOuterDiameter/hydraulicDiameter,0.383)*\
                pow(tubeDistFlow/tubeDistOrtho,-0.247)
        return frictionFactor

class PlainFins(Fins):
    """
    Fin type: plain

    Source:
    Heat transfer and friction characteristics of plain fin-and-tube heat exchangers,
    part II: Correlation
    Chi-Chuan Wang, Kuan-Yu Chi, Chun-Jung Chang

    Required inputs:
        finsPerInch             number of fins per inch
        thickness               thickness of fin
        thermalConductivity     thermal conductivity of fin material
    """

    def __init__(self, finsPerInch: float, thickness: float,
                 thermalConductivity: float):
        super().__init__(finsPerInch, thickness, thermalConductivity)
        self.finType = FinType.PLAIN

    def calculateOutsideDiameter(self, tubeOuterDiameter):
        return tubeOuterDiameter + 2*self.thickness

    def calculateColburnJFactor(self, reynoldsNumberAir, tubeOuterDiameter,
                tubeNumBanks, distFlow, distOrtho, totalArea, tubeOuterArea,
                crossSectionalArea):
        hydraulicDiameter = 4*crossSectionalArea*(tubeNumBanks+1)*distFlow/totalArea
        if tubeNumBanks == 1:
            jFactor1 = 1.9 - 0.23*np.log(reynoldsNumberAir)
            jFactor2 = -0.236+0.126*np.log(reynoldsNumberAir)
            jFactor = 0.108*pow(reynoldsNumberAir,-0.29)*pow(distOrtho/distFlow, jFactor1)*\
                pow(self.finPitch/tubeOuterDiameter,-1.084)*\
                pow(self.finPitch/hydraulicDiameter,-0.786)*\
                pow(self.finPitch/distOrtho,jFactor2)
        else:
            jFactor1 = -0.361 - 0.042*tubeNumBanks/np.log(reynoldsNumberAir) + \
                0.158*np.log(tubeNumBanks*(self.finPitch/tubeOuterDiameter)**0.41)
            jFactor2 = -1.224 - 0.076*pow(distFlow/hydraulicDiameter,1.42)/\
                np.log(reynoldsNumberAir)
            jFactor3 = -0.083 + 0.058*tubeNumBanks/np.log(reynoldsNumberAir)
            jFactor4 = -5.735 + 1.21*np.log(reynoldsNumberAir/tubeNumBanks)
            jFactor = 0.086*pow(reynoldsNumberAir,jFactor1)*\
                pow(tubeNumBanks, jFactor2)*pow(self.finPitch/tubeOuterDiameter,jFactor3)*\
                pow(self.finPitch/hydraulicDiameter,jFactor4)*pow(self.finPitch/distOrtho,-0.93)
        return jFactor

    def calculateAirSideFrictionFactor(self, reynoldsNumber, totalArea, tubeOuterArea,
                tubeOuterDiameter, tubeNumBanks, tubeDistOrtho, tubeDistFlow):
        frictionFactor1 = -0.764 + 0.739*tubeDistOrtho/tubeDistFlow + \
            0.177*self.finPitch/tubeOuterDiameter - 0.00758/tubeNumBanks
        frictionFactor2 = -15.689 + 64.021/np.log(reynoldsNumber)
        frictionFactor3 = 1.696 - 15.695/np.log(reynoldsNumber)
        frictionFactor = 0.0267*pow(reynoldsNumber,frictionFactor1)*\
            pow(tubeDistOrtho/tubeDistFlow,frictionFactor2)*\
            pow(self.finPitch/tubeOuterDiameter,frictionFactor3)
        return frictionFactor

class FinnedTube():
    """
    Fins and tubes heat exchanger settings
    """
    def __init__(self, tubes: Tubes, fins: Fins, air: Air):
        self.tubes = tubes
        self.fins = fins
        self.air = air
        self.numFinsInTubeSheet = self.calculateNumFinsInTubeSheet()
        self.totalArea = self.calculateTotalArea()
        self.ductCrossSectionalArea = self.tubes.calculateDuctCrossSectionalArea(
            self.fins.calculateOutsideDiameter(self.tubes.outerDiam),
            self.fins.thickness, self.numFinsInTubeSheet)
        self.airSideMeanHeatTransfer = self.calculateAirSideMeanHeatTransfer()
        self.airSideFrictionFactor = self.calculateAirSideFrictionFactor()
        self.airSidePressureDrop = self.calculateAirSidePressureDrop()

    def __str__(self):
        return "Tubes:\n" + str(self.tubes) + "Fins:\n" + str(self.fins) + "Air:\n" + str(self.air)

    def calculateNumFinsInTubeSheet(self):
        """
        Calculates the number of fins in a tube sheet

        Returns
        -------
        float
            number of fins in tube sheet

        """
        return self.tubes.length * self.fins.finsPerMeter

    def calculateWettedArea1Fin(self):
        """
        Calculates the wetted area of one fin, assuming that the fin extends
        1/2 pt in front/after last tube in bundle

        Returns
        -------
        float
            wetted area of single fin

        """
        #TODO: plain fin uses D_c and not D_o like the others. Why?
        totalArea = self.tubes.calculateHeight()*self.tubes.distFlow*\
            (self.tubes.numBanks+1)*self.fins.calculateAreaIncrease()
        tubeArea = (self.tubes.numPerBank*self.tubes.numBanks*np.pi*self.tubes.outerDiam**2)/4
        return 2.0 * (totalArea - tubeArea)

    def calculateWettedAreaTotal(self):
        """
        Calculates the total wetted area of the heat exchanger

        Returns
        -------
        float
            total wetted area

        """
        return self.numFinsInTubeSheet*self.calculateWettedArea1Fin()

    def calculateTotalArea(self):
        """
        Calculates the total air side surface area of the heat exchanger

        Returns
        -------
        float
            total air side surface area

        """
        #TODO: plain fin uses D_c and not D_o like the others. Why?
        return self.calculateWettedAreaTotal() + self.tubes.numPerBank*self.tubes.numBanks*np.pi*\
            self.tubes.outerDiam*(self.tubes.length - self.numFinsInTubeSheet*self.fins.thickness)

    def calculateUMax(self):
        """
        Calculates the maximum velocity of air in the heat exchanger

        Returns
        -------
        float
            maximum air velocity

        """
        return self.air.massFlowHumidAir/(self.air.rhoHumidAir*self.ductCrossSectionalArea)

    def calculateReynoldsNumberAir(self):
        """
        Wrapper for calculation of the Reynolds number for air

        Returns
        -------
        float
            air side Reynolds number

        """
        return self.air.calculateReynoldsNumber(self.calculateUMax(),
            self.fins.calculateOutsideDiameter(self.tubes.outerDiam))

    def calculateColburnJFactor(self):
        """
        Calculates the Chilton and Colburn J-factor from the Chilton and Colburn
        J-factor analogy

        Returns
        -------
        float
            Chilton and Colburn j factor

        """
        return self.fins.calculateColburnJFactor(self.calculateReynoldsNumberAir(),
                self.fins.calculateOutsideDiameter(self.tubes.outerDiam), self.tubes.numBanks,
                self.tubes.distFlow, self.tubes.distOrtho, self.totalArea,
                self.tubes.calculateTubeOuterArea(), self.ductCrossSectionalArea)

    def calculateAirSideMeanHeatTransfer(self):
        """
        Calculate h_a, the mean heat transfer for the air side

        Returns
        -------
        float
            air side mean heat transfer

        """
        return self.calculateColburnJFactor()*self.air.rhoHumidAir*self.calculateUMax()*\
                self.air.heatCapacityHumidAir/pow(self.air.prandtlNumber, 2.0/3.0)

    def calculateMFactor(self):
        """
        Calculates m, a non-dimensional group in the fin efficiency calculation

        Returns
        -------
        float
            m, fin surface efficiency parameter

        """
        return np.sqrt(2*self.airSideMeanHeatTransfer*self.air.correction/\
            (self.fins.thermalConductivity*self.fins.thickness))

    def calculateSurfaceEfficiencyParam(self):
        """
        Calculates phi, a parameter used in the fin surface efficiency calculation

        Returns
        -------
        float
            phi, fin surface efficiency parameter

        """
        #TODO: check calculation
        effectiveRadiusRatio = self.tubes.calculateEffectiveRadiusRatio()
        lateralRadialDistance = self.tubes.outerDiam/2
        return (effectiveRadiusRatio - 1)*(1 + (0.3 + pow(\
                self.calculateMFactor()*(effectiveRadiusRatio*lateralRadialDistance - \
                lateralRadialDistance)/2.5,1.5 - effectiveRadiusRatio/12.0)*(\
                0.26*pow(effectiveRadiusRatio,0.3)-0.3)) * np.log(effectiveRadiusRatio))

    def calculateFinEfficiency(self):
        """
        Calculates eta_f, the surface efficiency of the heat exchanger fins

        Returns
        -------
        float
            fin surface efficiency

        """
        mrPhi = self.calculateMFactor()*self.tubes.outerDiam/2*\
            self.calculateSurfaceEfficiencyParam()
        return np.tanh(mrPhi)/mrPhi*np.cos(0.1*mrPhi)

    def calculateOverallSurfaceEfficiency(self):
        """
        Calculates eta_o, the overall surface efficiency of the heat exchanger

        Returns
        -------
        float
            overall surface efficiency

        """
        return 1 - self.calculateWettedAreaTotal()/\
            self.totalArea*(1 - self.calculateFinEfficiency())

    def calculateAirMassFlux(self):
        """
        Calculates G_c, the air side mass flux

        Returns
        -------
        float
            air side mass flux

        """
        return self.air.massFlowHumidAir/self.ductCrossSectionalArea

    def calculateAirSidePressureDrop(self):
        """
        Calculates the pressure drop on the air side of the heat exchanger

        Returns
        -------
        float
            air side pressure drop

        """
        return self.totalArea/self.ductCrossSectionalArea/self.air.rhoHumidAir*\
            self.calculateAirMassFlux()**2/2.0*self.calculateAirSideFrictionFactor()

    def calculateAirSideFrictionFactor(self):
        """
        Calculates C_f, the friction factor for the air side, used to calculate
        the air side pressure drop

        Returns
        -------
        float
            friction factor

        """
        return self.fins.calculateAirSideFrictionFactor(self.calculateReynoldsNumberAir(),
                self.totalArea, self.tubes.calculateTubeOuterArea(),
                self.fins.calculateOutsideDiameter(self.tubes.outerDiam),
                self.tubes.numBanks, self.tubes.distOrtho, self.tubes.distFlow)

if __name__=='__main__':
    tf = HerringboneFins(finsPerInch=14.5, thickness=0.00011, thermalConductivity=237,
                            amplitude=0.001, period=0.001)

    tubesTest = Tubes(numPerBank=32, numBanks=3, numCircuits=5, length=0.452, innerDiam=0.0089154,
                  outerDiam=0.009525, distFlow=0.0254, distOrtho=0.0219964,
                  thermalConductivityWall=237)

    finsWL = WavyLouveredFins(finsPerInch=14.5, thickness=0.00011, thermalConductivity=237,
                            amplitude=0.001, period=0.001)
    finsH = HerringboneFins(finsPerInch=14.5, thickness=0.00011, thermalConductivity=237,
                            amplitude=0.001, period=0.001)
    finsP = PlainFins(finsPerInch=14.5, thickness=0.00011, thermalConductivity=237)

    airTest = Air(vDotHA=0.5663, tempDryBulb=299.8, pressure=101325,
              relativeHumidity=0.51, fanPower=438)

    finnedTubeWL = FinnedTube(fins=finsWL, tubes=tubesTest, air=airTest)
    finnedTubeH = FinnedTube(fins=finsH, tubes=tubesTest, air=airTest)
    finnedTubeP = FinnedTube(fins=finsP, tubes=tubesTest, air=airTest)

    print(finnedTubeWL)
    print("Wavy-Louvered fins:","eta_a is:" + \
            str(finnedTubeWL.calculateOverallSurfaceEfficiency()) +
            ", dP_a is:" + str(finnedTubeWL.calculateAirSidePressureDrop()) +" Pa")
    print("Herringbone Fins fins:","eta_a is:" + \
            str(finnedTubeH.calculateOverallSurfaceEfficiency()) +
            ", dP_a is:" + str(finnedTubeH.calculateAirSidePressureDrop()) +" Pa")
    print("Plain Fins fins:","eta_a is:" + str(finnedTubeP.calculateOverallSurfaceEfficiency()) +
            ", dP_a is:" + str(finnedTubeP.calculateAirSidePressureDrop()) +" Pa")
    print("a graph for the fin correlations can be found here: " + r"\Documentation\Web\MPLPlots")

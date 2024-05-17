# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 10:39:20 2023

@author: SMCANANA
"""
import logging
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from ACHP.models.FinnedTube import FinnedTube
from ACHP.models.Fluid import Fluid, ThermoProps, FluidApparatusProps
from ACHP.DryWetSegment import DWSVals, DryWetSegment
from ACHP.calculations.Correlations import f_h_1phase_Tube, ShahEvaporation_Average, lmPressureGradientAvg,\
    calculateAccelerationalPressureDrop, twoPhaseDensity, KandlikarEvaporation_average

class Evaporator():
    """
    Class for evaporator-specific calculations, hopefully
    """
    def __init__(self, finnedTube: FinnedTube, refrigerant: Fluid, massFlowR: float,
                 pressureSatR: float, enthalpyInR: float=None, qualityInR: float=None,
                 meanHeatTransferTuningFactor=1.0, heatTransferCoeffRTuning=1.0,
                 pressureDropRTuning=1.0):
        assert 0.000001 <= massFlowR <= 10.0, "Refrigerant mass flow must be between 0.000001 and 10.0"
        assert 0.001 <= pressureSatR <= 100000000, "Refrigerant saturated pressure must be between 0.001 and 100000000"
        self.type = "Evaporator"
        self.logger = logging.getLogger(self.type)
        self.refrigerant = refrigerant
        self.refrigerant.fluidApparatiProps[self.type] = FluidApparatusProps(enthalpyIn=enthalpyInR,
                                                                             pressureIn=pressureSatR)
        self.finnedTube = finnedTube
        self.massFlowR = massFlowR
        self.pressureSatR = pressureSatR
        self.fluidProps = self.refrigerant.fluidApparatiProps[self.type]

        self.meanHeatTransferTuningFactor = meanHeatTransferTuningFactor
        self.heatTransferCoeffRTuning = heatTransferCoeffRTuning
        self.pressureDropRTuning = pressureDropRTuning

        #standalone refrigerant values
        self.tempSatLiquidR = self.refrigerant.calculateTemperature(ThermoProps.PQ, self.pressureSatR, 0.0)
        self.enthalpySatLiquidR = self.refrigerant.calculateEnthalpy(ThermoProps.PQ, self.pressureSatR, 0.0)
        self.entropySatLiquidR = self.refrigerant.calculateEntropy(ThermoProps.QT, 0.0, self.tempSatLiquidR)
        self.tempSatVaporR = self.refrigerant.calculateTemperature(ThermoProps.PQ, self.pressureSatR, 1.0)
        self.enthalpySatVaporR = self.refrigerant.calculateEnthalpy(ThermoProps.PQ, self.pressureSatR, 1.0)
        self.entropySatVaporR = self.refrigerant.calculateEntropy(ThermoProps.QT, 1.0, self.tempSatVaporR)
        self.tempSatMean = (self.tempSatLiquidR + self.tempSatVaporR)/2 #TODO: used only once, is it necessary here??
        self.latentHeat = self.enthalpySatVaporR - self.enthalpySatLiquidR
        self.refrigerant.fluidApparatiProps[self.type].set("tempDew", self.tempSatVaporR)
        self.refrigerant.fluidApparatiProps[self.type].set("tempBubble", self.tempSatLiquidR)

        #refrigerant values for evaporator
        self.qualityInR = (enthalpyInR - self.enthalpySatLiquidR)/\
            (self.enthalpySatVaporR - self.enthalpySatLiquidR) if qualityInR is None else qualityInR
        assert 0.0 <= self.qualityInR <= 1.0, 'Refrigerant quality must be between 0.0 and 1.0'
        self.enthalpyInR = self.qualityInR*self.enthalpySatVaporR + \
            (1 - self.qualityInR)*self.enthalpySatLiquidR if enthalpyInR is None else enthalpyInR
        assert -100000 <= self.enthalpyInR <= 10000000, 'Refrigerant enthalpy in must be between -100000 and 10000000'
        self.entropyInR = self.qualityInR*self.entropySatVaporR + (1 - self.qualityInR)*self.entropySatLiquidR
        self.tempInR = self.qualityInR*self.tempSatVaporR + (1 - self.qualityInR)*self.tempSatLiquidR

        #tubes values
        self.effectiveCircuitLength = self.finnedTube.tubes.length*\
            self.finnedTube.tubes.numPerBank*self.finnedTube.tubes.numBanks/\
            self.finnedTube.tubes.numCircuits
        self.areaWettedR = self.finnedTube.tubes.numCircuits*np.pi*\
            self.finnedTube.tubes.innerDiam*self.effectiveCircuitLength
        self.volumeMeanR = self.finnedTube.tubes.numCircuits*self.effectiveCircuitLength*np.pi*\
            self.finnedTube.tubes.innerDiam**2/4.0
        self.massFluxMeanR = self.massFlowR/(self.finnedTube.tubes.numCircuits*np.pi*\
                                             self.finnedTube.tubes.innerDiam**2/4.0)
        self.wallThermalResistance = np.log(self.finnedTube.tubes.outerDiam/self.finnedTube.tubes.innerDiam)/\
            (2*np.pi*self.finnedTube.tubes.thermalConductivityWall*self.effectiveCircuitLength*\
             self.finnedTube.tubes.numCircuits)
        self.qualityOutTwoPhase = 1.0 #TODO: why?

        #two-phase outputs
        self.lengthFractionTwoPhase: float
        self.heatTransferTwoPhase: float
        self.heatTransferTwoPhaseSensible: float

        #superheat inputs?
        #need to change f_h_1phase_Tube to accept Fluid instead of abstractState
        self.darcyFrictionFactorSuperheat, self.heatTransferCoeffSuperheatR, \
            self.reynoldsNumberSuperheatR = f_h_1phase_Tube(
            self.massFlowR/self.finnedTube.tubes.numCircuits, self.finnedTube.tubes.innerDiam,
            self.tempSatVaporR+3, self.pressureSatR, self.refrigerant, "Single")

        #superheat outputs
        self.lengthFractionSuperheat: float
        self.heatTransferSuperheat: float
        self.heatTransferSuperheatSensible: float
        self.dryFractionSuperheat: float
        self.tempOutSuperheatAir: float
        self.tempOutR: float
        self.chargeSuperheat: float
        self.pressureDropSuperheatR: float

        self.heatTransfer: float
        self.charge: float
        self.capacity: float
        self.enthalpyOutR: float
        self.entropyOutR: float
        self.tempChangeSuperheat: float
        self.sensibleHeatRatio: float
        self.tempOutAir: float
        self.pressureDropR: float
        self.heatTransferCoeffMeanR: float
        self.heatTransferConductanceR: float
        self.heatTransferConductanceAir: float
        self.heatTransferConductanceWall: float
        self.temperatureOfBends: float
        self.heatTransferCoeffTwoPhase: float
        self.dryFractionTwoPhase: float
        self.tempOutTwoPhaseAir: float
        self.chargeTwoPhase: float
        self.pressureDropTwoPhaseR: float

    def outputList(self):
        """
        list of parameters for evaporator to be output into a file.
        Need to figure out what's usable and necessary

        Returns
        -------
        None.

        """
        pass

    def airSideCalcs(self):
        """
        Calculates everything under fins and tubes that hasn't been calculated yet.
        Mainly air side calculations and geometries

        Returns
        -------
        None.

        """
        self.finnedTube.calculateOverallSurfaceEfficiency()
        self.finnedTube.calculateAirsidePressureDrop()

    def calculate(self):
        """
        Calculates everything to do with the evaporator...

        Returns
        -------
        None.

        """
        if self.twoPhaseForward(self.qualityOutTwoPhase) < 0:
            existsSuperheat = False
            #TODO: add whatever else needs to be added
        else:
            existsSuperheat = True
            self.lengthFractionTwoPhase = brentq(self.twoPhaseForward,0.00000000001,0.9999999999)
            self.superheatForward(1 - self.lengthFractionTwoPhase)

        self.heatTransfer = self.heatTransferSuperheat + self.heatTransferTwoPhase
        self.charge = self.chargeSuperheat + self.chargeTwoPhase
        if self.verbosity > 4:
            print(self.heatTransfer, "Evaporator.Q")
        self.capacity = self.heatTransfer - self.finnedTube.air.fanPower

        #Sensible heat ratio [-]
        self.sensibleHeatRatio = (self.heatTransferTwoPhaseSensible +\
                                  self.heatTransferSuperheatSensible)/self.heatTransfer
        #Average air outlet temperature (area fraction weighted average) [K]
        self.tempOutAir = self.lengthFractionSuperheat*self.tempOutSuperheatAir +\
            self.lengthFractionTwoPhase*self.tempOutTwoPhaseAir
        self.pressureDropR = (self.pressureDropSuperheatR + self.pressureDropTwoPhaseR)*self.pressureDropRTuning

        #Outlet enthalpy obtained from energy balance
        self.enthalpyOutR = self.enthalpyInR+self.heatTransfer/self.massFlowR

        #Outlet entropy
        if existsSuperheat:
            self.entropyOutR = self.refrigerant.calculateEntropy(ThermoProps.PT, self.pressureSatR,
                                                                            self.tempOutR)
        else:
            qualityOutR = (self.enthalpyOutR - self.enthalpySatLiquidR)/\
                (self.enthalpySatVaporR - self.enthalpySatLiquidR)
            entropySatLiquidR = self.refrigerant.calculateEntropy(ThermoProps.QT, 0.0, self.tempSatLiquidR)
            entropySatVaporR = self.refrigerant.calculateEntropy(ThermoProps.QT, 1.0, self.tempSatVaporR)
            self.entropyOutR = entropySatVaporR*qualityOutR + (1 - qualityOutR)*entropySatLiquidR

        #Outlet superheat and temperature (in case of two phase)
        if existsSuperheat:
            self.tempChangeSuperheat = self.tempOutR-self.tempSatVaporR
        else:
            heatCapacitySuperheat = self.refrigerant.calculateHeatCapacity(ThermoProps.QT, 1.0,
                                                                           self.tempSatVaporR)
            #Effective superheat
            self.tempChangeSuperheat = (self.enthalpyOutR - self.enthalpySatVaporR)/\
                heatCapacitySuperheat
            self.tempOutR = self.refrigerant.calculateTemperature(ThermoProps.PQ,
                self.pressureSatR + self.pressureDropR, qualityOutR)
        self.heatTransferCoeffMeanR = self.lengthFractionTwoPhase*self.heatTransferCoeffTwoPhase + \
            self.lengthFractionSuperheat*self.heatTransferCoeffSuperheatR
        self.heatTransferConductanceR = self.heatTransferCoeffMeanR*self.areaWettedR
        self.heatTransferConductanceAir = (self.finnedTube.airSideMeanHeatTransfer*\
                self.meanHeatTransferTuningFactor)*self.finnedTube.totalArea*\
                self.finnedTube.calculateOverallSurfaceEfficiency()
        self.heatTransferConductanceWall = 1/self.wallThermalResistance

        #Build a vector of temperatures at each point where there is a phase transition
        #along the averaged circuit
        if existsSuperheat:
            #Insert the shoulder point
            tempVector = [self.tempInR,self.tempSatVaporR,self.tempOutR]
            xValues = [0,self.lengthFractionTwoPhase,1]
        else:
            tempVector = [self.tempInR,qualityOutR*self.tempSatVaporR + \
                          (1 - qualityOutR)*self.tempSatLiquidR]
            xValues = [0,1]

        #Determine each bend temperature by interpolation
        #------------------------------------------------
        #Number of bends (including inlet and outlet of coil)
        numBends = 1 + self.effectiveCircuitLength/self.finnedTube.tubes.length
        #x-position of each point
        xVector = np.linspace(0,1,int(numBends))

        self.temperatureOfBends = interp1d(xValues,tempVector)(xVector)


    def twoPhaseForward(self, lengthFractionTwoPhase):
        """
        Calculates the two-phase portion of the evaporator


        Parameters
        ----------
        lengthFractionTwoPhase : float
            fractional length of the two-phase portion to be calculated over

        Raises
        ------
        ValueError
            Heat transfer target in evaporator should not be negative

        Returns
        -------
        float
            difference between calculated heat transfer and the heat transfer target

        """
        dryWetSegment = self.initializeDryWetSegment(lengthFractionTwoPhase, self.tempSatMean,
                                                     1.0e15, True)
        heatTransferTarget = self.massFlowR*(self.qualityOutTwoPhase - self.qualityInR)*self.latentHeat
        if heatTransferTarget < 0:
            raise ValueError('Q_target in Evaporator must be positive')

        dryWetSegment.h_r = self.calculateHeatTransferCoeffR(True,heatTransferTarget,dryWetSegment)

        #Run the DryWetSegment to carry out the heat and mass transfer analysis
        DryWetSegment(dryWetSegment)

        self.heatTransferTwoPhase = dryWetSegment.Q
        self.heatTransferTwoPhaseSensible = dryWetSegment.Q_sensible
        self.heatTransferCoeffTwoPhase = dryWetSegment.h_r
        self.dryFractionTwoPhase = dryWetSegment.f_dry
        self.tempOutTwoPhaseAir = dryWetSegment.Tout_a

        rhoMean = twoPhaseDensity(self.refrigerant, self.qualityInR, self.qualityOutTwoPhase, self.type)
        self.chargeTwoPhase = rhoMean*self.lengthFractionTwoPhase*self.volumeMeanR
        self.pressureDropTwoPhaseR = self.calculatePressureDropR()
        if self.verbosity > 7:
            print (self.lengthFractionTwoPhase,dryWetSegment.Q,heatTransferTarget,self.qualityInR,
                   "lengthFractionTwoPhase,dryWetSegment.Q,heatTransferTarget,self.qualityInR")
        return dryWetSegment.Q - heatTransferTarget

    def superheatForward(self, lengthFractionSuperheat):
        """
        Calculates the superheat portion of the evaporator

        Parameters
        ----------
        lengthFractionSuperheat : float
            fractional length of the superheat portion to be calculated over

        Returns
        -------
        None.

        """
        heatCapacityR = self.refrigerant.calculateHeatCapacity(ThermoProps.PT, self.pressureSatR,
                self.tempSatVaporR + 2.5)
        dryWetSegment = self.initializeDryWetSegment(lengthFractionSuperheat, self.tempSatVaporR,
                heatCapacityR)
        dryWetSegment.h_r = self.calculateHeatTransferCoeffR()

        DryWetSegment(dryWetSegment)

        #Set values
        self.heatTransferSuperheat = dryWetSegment.Q
        self.heatTransferSuperheatSensible = dryWetSegment.Q_sensible
        self.dryFractionSuperheat = dryWetSegment.f_dry
        self.tempOutSuperheatAir = dryWetSegment.Tout_a
        self.tempOutR = dryWetSegment.Tout_r

        rhoSuperheat = self.refrigerant.calculateDensity(ThermoProps.PT, self.pressureSatR,
                    (dryWetSegment.Tout_r + self.tempSatVaporR)/2.0)
        self.chargeSuperheat = self.lengthFractionSuperheat*self.volumeMeanR*rhoSuperheat
        self.pressureDropSuperheatR = self.calculatePressureDropR(rhoSuperheat, False)

        if self.verbosity > 7:
            print(self.lengthFractionSuperheat,dryWetSegment.Q,
                  "lengthFractionSuperheat,dryWetSegment.Q")

    def initializeDryWetSegment(self, quality, tempInR, heatCapacityR, isTwoPhase=False):
        """
        Initializes the DryWetSegment class to be used in calculations

        Parameters
        ----------
        quality : float
            Vapor quality of refrigerant.
        tempInR : float
            Input temperature of refrigerant.
        heatCapacityR : float
            heat capacity of refrigerant.
        isTwoPhase : boolean, optional
            Is calculation in two-phase section? The default is False.

        Returns
        -------
        dryWetSegment : DryWegSegment class
            Class for DryWetSegment calculations.

        """
        dryWetSegment = DWSVals()
        dryWetSegment.Fins = self.finnedTube
        dryWetSegment.FinsType = self.finnedTube.fins.finType
        dryWetSegment.A_a = self.finnedTube.totalArea*quality
        dryWetSegment.cp_da = self.finnedTube.air.heatCapacityDryAir
        dryWetSegment.eta_a = self.finnedTube.calculateOverallSurfaceEfficiency()
        dryWetSegment.h_a = self.finnedTube.airSideMeanHeatTransfer*\
            self.meanHeatTransferTuningFactor
        dryWetSegment.mdot_da = self.finnedTube.air.massFlowDryAir*quality
        dryWetSegment.pin_a = self.finnedTube.air.pressure
        dryWetSegment.Tdew_r = self.tempSatVaporR
        dryWetSegment.Tbubble_r = self.tempSatLiquidR

        dryWetSegment.Tin_a = self.finnedTube.air.tempDryBulb
        dryWetSegment.RHin_a = self.finnedTube.air.relativeHumidity

        dryWetSegment.Tin_r = tempInR
        dryWetSegment.A_r = self.areaWettedR*quality
        dryWetSegment.Rw = self.wallThermalResistance/quality
        dryWetSegment.cp_r = heatCapacityR
        dryWetSegment.pin_r = self.pressureSatR
        dryWetSegment.mdot_r = self.massFlowR
        dryWetSegment.IsTwoPhase = isTwoPhase
        return dryWetSegment

    def calculateHeatTransferCoeffR(self, isTwoPhase=False, heatTransferTarget=None,
                                    dryWetSegment=None):
        """
        Calculates the heat transfer coefficient of the refrigerant.

        Parameters
        ----------
        isTwoPhase : boolean, optional
            Is calculation in two-phase section? The default is False.
        heatTransferTarget : float, optional
            Heat transfer target. Only used in two-phase section. The default is None.
        dryWetSegment : DryWetSegment class, optional
            Class for dryWetSegment calculations. Only used in two-phase section.
            The default is None.

        Raises
        ------
        TypeError
            In the two-phase section, heatTransferTarget and dryWetSegment must be present.

        Returns
        -------
        heatTransferCoeffR : float
            Heat transfer coefficient for refrigerant.

        """
        if isTwoPhase:
            if not heatTransferTarget:
                raise TypeError('In the two-phase section, heatTransferTarget must be defined.')
            if not dryWetSegment:
                raise TypeError('In the two-phase section, dryWetSegment must be defined.')
            try:
                #TODO: change these to accept Fluid instead of abstractState
                if self.refrigerant.name in 'CarbonDioxide':
                    heatTransferCoeffR = KandlikarEvaporation_average(self.qualityInR,
                        self.qualityOutTwoPhase, self.refrigerant, self.massFluxMeanR,
                        self.finnedTube.tubes.innerDiam, self.pressureSatR,
                        heatTransferTarget/dryWetSegment.A_r,self.tempSatLiquidR,
                        self.tempSatVaporR)
                else:
                    heatTransferCoeffR = ShahEvaporation_Average(self.qualityInR,
                        self.qualityOutTwoPhase, self.refrigerant, self.massFluxMeanR,
                        self.finnedTube.tubes.innerDiam, self.pressureSatR,
                        heatTransferTarget/dryWetSegment.A_r, self.tempSatLiquidR,
                        self.tempSatVaporR)
            except:
                heatTransferCoeffR = ShahEvaporation_Average(self.qualityInR,
                        self.qualityOutTwoPhase, self.refrigerant, self.massFluxMeanR,
                        self.finnedTube.tubes.innerDiam,
                        heatTransferTarget/dryWetSegment.A_r,self.tempSatLiquidR,
                        self.tempSatVaporR)
            heatTransferCoeffR = heatTransferCoeffR*self.heatTransferCoeffRTuning
        else:
            #Use a guess value of 6K superheat to calculate the properties
            heatTransferCoeffR = self.heatTransferCoeffSuperheatR
        return heatTransferCoeffR

    def calculatePressureDropR(self, density=None, isTwoPhase=True):
        """
        Calculates the pressure drop of the refrigerant in evaporator

        Parameters
        ----------
        density : float, optional
            Density of the refrigerant. The default is None.
        isTwoPhase : boolean, optional
            Is calculation in two-phase section? The default is True.

        Raises
        ------
        TypeError
            In the superheat section, density must be present.

        Returns
        -------
        pressureDropR : float
            Pressure drop of the refrigerant.

        """
        if isTwoPhase:
            #TODO: change these to accept Fluid instead of abstractState
            frictionalPressureDrop = lmPressureGradientAvg(self.qualityInR,
                        self.qualityOutTwoPhase, self.refrigerant, self.massFluxMeanR,
                        self.finnedTube.tubes.innerDiam, self.tempSatLiquidR,
                        self.tempSatVaporR)*self.effectiveCircuitLength*self.lengthFractionTwoPhase
            try:
                if self.refrigerant.name in 'CarbonDioxide':
                    accelPressureDropTwoPhaseR = calculateAccelerationalPressureDrop(self.qualityInR,
                        self.qualityOutTwoPhase, self.refrigerant, self.massFluxMeanR,
                        self.tempSatLiquidR, self.tempSatVaporR,D=self.finnedTube.tubes.innerDiam,
                        slipModel='Premoli')*self.effectiveCircuitLength*self.lengthFractionTwoPhase
                else:
                    accelPressureDropTwoPhaseR = calculateAccelerationalPressureDrop(self.qualityInR,
                        self.qualityOutTwoPhase, self.refrigerant, self.massFluxMeanR,
                        self.tempSatLiquidR, self.tempSatVaporR, slipModel='Zivi')*\
                        self.effectiveCircuitLength*self.lengthFractionTwoPhase
            except:
                accelPressureDropTwoPhaseR = calculateAccelerationalPressureDrop(self.qualityInR,
                        self.qualityOutTwoPhase, self.refrigerant, self.massFluxMeanR,
                        self.tempSatLiquidR, self.tempSatVaporR, slipModel='Zivi')*\
                        self.effectiveCircuitLength*self.lengthFractionTwoPhase
            pressureDropR = frictionalPressureDrop + accelPressureDropTwoPhaseR
        else:
            if not density:
                raise TypeError('In the superheat section, density must be defined.')
            specificVolumeR = 1/density
            #Pressure gradient using Darcy friction factor
            pressureGradientR = -self.darcyFrictionFactorSuperheat*specificVolumeR*\
                self.massFluxMeanR**2/(2*self.finnedTube.tubes.innerDiam)  #Pressure gradient
            pressureDropR = pressureGradientR*self.effectiveCircuitLength*\
                self.lengthFractionSuperheat
        return pressureDropR

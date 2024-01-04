# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 10:39:20 2023

@author: SMCANANA
"""
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from ACHP.models.FinsAndTubes import FinsAndTubes
from ACHP.wrappers.coolPropWrapper import AbstractStateWrapper
from ACHP.DryWetSegment import DWSVals, DryWetSegment
from ACHP.Correlations import f_h_1phase_Tube, ShahEvaporation_Average, LMPressureGradientAvg,\
    AccelPressureDrop, TwoPhaseDensity, KandlikarEvaporation_average

class EvaporatorClass():
    """
    Class for evaporator-specific calculations, hopefully
    """
    def __init__(self, finsAndTubes: FinsAndTubes, refrigerant: str, backEnd: str,
                 massFlowR: float, pressureSatR: float, enthalpyInR: float=None,
                 qualityInR: float=None, verbosity=0, meanHeatTransferTuningFactor=1.0,
                 hTpTuning=1.0, dPTuning=1.0):
        assert 0.000001 <= massFlowR <= 10.0, "Refrigerant mass flow must be between 0.000001 and 10.0"
        assert 0.001 <= pressureSatR <= 100000000, "Refrigerant saturated pressure must be between 0.001 and 100000000"
        self.finsAndTubes = finsAndTubes
        self.abstractState = AbstractStateWrapper(backEnd, refrigerant)
        self.massFlowR = massFlowR
        self.pressureSatR = pressureSatR
        self.verbosity = verbosity

        #TODO: these tuning factors don't need to be here
        self.meanHeatTransferTuningFactor = meanHeatTransferTuningFactor
        self.hTpTuning = hTpTuning
        self.dPTuning = dPTuning

        #standalone refrigerant values
        self.tempSatLiquidR = self.abstractState.calculateTempFromPandQ(self.pressureSatR, 0.0)
        self.enthalpySatLiquidR = self.abstractState.calculateEnthalpyFromPandQ(self.pressureSatR, 0.0)
        self.entropySatLiquidR = self.abstractState.calculateEntropyFromQandT(0.0, self.tempSatLiquidR)
        self.tempSatVaporR = self.abstractState.calculateTempFromPandQ(self.pressureSatR, 1.0)
        self.enthalpySatVaporR = self.abstractState.calculateEnthalpyFromPandQ(self.pressureSatR, 1.0)
        self.entropySatVaporR = self.abstractState.calculateEntropyFromQandT(1.0, self.tempSatVaporR)
        self.tempSatMean = (self.tempSatLiquidR + self.tempSatVaporR)/2 #TODO: used only once, is it necessary here??
        self.latentHeat = self.enthalpySatVaporR - self.enthalpySatLiquidR

        #refrigerant values for evaporator
        self.qualityInR = qualityInR or (enthalpyInR - self.enthalpySatLiquidR)/\
            (self.enthalpySatVaporR - self.enthalpySatLiquidR)
        assert 0.0 <= qualityInR <= 1.0, 'Refrigerant quality must be between 0.0 and 1.0'
        self.enthalpyInR = enthalpyInR or self.qualityInR*self.enthalpySatVaporR + \
            (1 - self.qualityInR)*self.enthalpySatLiquidR
        assert -100000 <= enthalpyInR <= 10000000, 'Refrigerant enthalpy in must be between -100000 and 10000000'
        self.entropyInR = self.qualityInR*self.entropySatVaporR + (1 - self.qualityInR)*self.entropySatLiquidR
        self.tempInR = self.qualityInR*self.tempSatVaporR + (1 - self.qualityInR)*self.tempSatLiquidR

        #evaporator values
        self.effectiveCircuitLength = self.finsAndTubes.tubes.length*\
            self.finsAndTubes.tubes.numPerBank*self.finsAndTubes.tubes.numBanks/\
            self.finsAndTubes.tubes.numCircuits
        self.areaWettedR = self.finsAndTubes.tubes.numCircuits*np.pi*\
            self.finsAndTubes.tubes.innerDiam*self.effectiveCircuitLength
        self.volumeMeanR = self.finsAndTubes.tubes.numCircuits*self.effectiveCircuitLength*np.pi*\
            self.finsAndTubes.tubes.innerDiam**2/4.0
        self.massFluxMeanR = self.massFlowR/(self.finsAndTubes.tubes.numCircuits*np.pi*\
                                             self.finsAndTubes.tubes.innerDiam**2/4.0)
        self.wallThermalResistance = np.log(self.finsAndTubes.tubes.outerDiam/self.finsAndTubes.tubes.innerDiam)/\
            (2*np.pi*self.finsAndTubes.tubes.thermalConductivityWall*self.effectiveCircuitLength*\
             self.finsAndTubes.tubes.numCircuits)
        #TODO example h_in uses pressure at temperature 282K and quality of 1, and an actual quality of 0.15 why?
        self.qualityOutTwoPhase = 1.0 #TODO: why?

        #two-phase outputs
        self.lengthFractionTwoPhase: float
        self.heatTransferTwoPhase: float
        self.heatTransferTwoPhaseSensible: float

        #superheat inputs?
        self.darcyFrictionFactorSuperheat, self.heatTransferCoeffSuperheatR, \
            self.reynoldsNumberSuperheatR = f_h_1phase_Tube(
            self.massFlowR/self.finsAndTubes.tubes.numCircuits, self.finsAndTubes.tubes.innerDiam,
            self.tempSatVaporR+3, self.pressureSatR, self.abstractState, "Single")

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
        self.finsAndTubes.calculateOverallSurfaceEfficiency()
        self.finsAndTubes.calculateAirsidePressureDrop()

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
        self.capacity = self.heatTransfer - self.finsAndTubes.air.fanPower

        #Sensible heat ratio [-]
        self.sensibleHeatRatio = (self.heatTransferTwoPhaseSensible +\
                                  self.heatTransferSuperheatSensible)/self.heatTransfer
        #Average air outlet temperature (area fraction weighted average) [K]
        self.tempOutAir = self.lengthFractionSuperheat*self.tempOutSuperheatAir +\
            self.lengthFractionTwoPhase*self.tempOutTwoPhaseAir
        self.pressureDropR = (self.pressureDropSuperheatR + self.pressureDropTwoPhaseR)*self.dPTuning

        #Outlet enthalpy obtained from energy balance
        self.enthalpyOutR = self.enthalpyInR+self.heatTransfer/self.massFlowR

        #Outlet entropy
        if existsSuperheat:
            self.entropyOutR = self.abstractState.calculateEntropyFromPandT(self.pressureSatR,
                                                                            self.tempOutR)
        else:
            qualityOutR = (self.enthalpyOutR - self.enthalpySatLiquidR)/\
                (self.enthalpySatVaporR - self.enthalpySatLiquidR)
            entropySatLiquidR = self.abstractState.calculateEntropyFromQandT(0.0, self.tempSatLiquidR)
            entropySatVaporR = self.abstractState.calculateEntropyFromQandT(1.0, self.tempSatVaporR)
            self.entropyOutR = entropySatVaporR*qualityOutR + (1 - qualityOutR)*entropySatLiquidR

        #Outlet superheat and temperature (in case of two phase)
        if existsSuperheat:
            self.tempChangeSuperheat = self.tempOutR-self.tempSatVaporR
        else:
            heatCapacitySuperheat = self.abstractState.calculateHeatCapacityFromQandT(1.0, self.tempSatVaporR)
            #Effective superheat
            self.tempChangeSuperheat = (self.enthalpyOutR - self.enthalpySatVaporR)/\
                heatCapacitySuperheat
            self.tempOutR = self.abstractState.calculateTempFromPandQ(
                self.pressureSatR + self.pressureDropR, qualityOutR)
        self.heatTransferCoeffMeanR = self.lengthFractionTwoPhase*self.heatTransferCoeffTwoPhase + \
            self.lengthFractionSuperheat*self.heatTransferCoeffSuperheatR
        self.heatTransferConductanceR = self.heatTransferCoeffMeanR*self.areaWettedR
        self.heatTransferConductanceAir = (self.finsAndTubes.airSideMeanHeatTransfer*\
                self.meanHeatTransferTuningFactor)*self.finsAndTubes.totalArea*\
                self.finsAndTubes.calculateOverallSurfaceEfficiency()
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
        numBends = 1 + self.effectiveCircuitLength/self.finsAndTubes.tubes.length
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

        rhoMean = TwoPhaseDensity(self.abstractState, self.qualityInR, self.qualityOutTwoPhase,
                    self.tempSatVaporR,self.tempSatLiquidR,slipModel='Zivi')
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
        heatCapacityR = self.abstractState.calculateHeatCapacityFromPandT(self.pressureSatR,
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

        rhoSuperheat = self.abstractState.calculateDensityFromPandT(self.pressureSatR,
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
        dryWetSegment.Fins = self.finsAndTubes
        dryWetSegment.FinsType = self.finsAndTubes.fins.finType
        dryWetSegment.A_a = self.finsAndTubes.totalArea*quality
        dryWetSegment.cp_da = self.finsAndTubes.air.heatCapacityDryAir
        dryWetSegment.eta_a = self.finsAndTubes.calculateOverallSurfaceEfficiency()
        dryWetSegment.h_a = self.finsAndTubes.airSideMeanHeatTransfer*\
            self.meanHeatTransferTuningFactor
        dryWetSegment.mdot_da = self.finsAndTubes.air.massFlowDryAir*quality
        dryWetSegment.pin_a = self.finsAndTubes.air.pressure
        dryWetSegment.Tdew_r = self.tempSatVaporR
        dryWetSegment.Tbubble_r = self.tempSatLiquidR

        dryWetSegment.Tin_a = self.finsAndTubes.air.tempDryBulb
        dryWetSegment.RHin_a = self.finsAndTubes.air.relativeHumidity

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
                if self.abstractState.fluid in 'CarbonDioxide':
                    heatTransferCoeffR = KandlikarEvaporation_average(self.qualityInR,
                        self.qualityOutTwoPhase, self.abstractState, self.massFluxMeanR,
                        self.finsAndTubes.tubes.innerDiam, self.pressureSatR,
                        heatTransferTarget/dryWetSegment.A_r,self.tempSatLiquidR,
                        self.tempSatVaporR)
                else:
                    heatTransferCoeffR = ShahEvaporation_Average(self.qualityInR,
                        self.qualityOutTwoPhase, self.abstractState, self.massFluxMeanR,
                        self.finsAndTubes.tubes.innerDiam, self.pressureSatR,
                        heatTransferTarget/dryWetSegment.A_r, self.tempSatLiquidR,
                        self.tempSatVaporR)
            except:
                heatTransferCoeffR = ShahEvaporation_Average(self.qualityInR,
                        self.qualityOutTwoPhase, self.abstractState, self.massFluxMeanR,
                        self.finsAndTubes.tubes.innerDiam, self.pressureSatR,
                        heatTransferTarget/dryWetSegment.A_r,self.tempSatLiquidR,
                        self.tempSatVaporR)
            heatTransferCoeffR = heatTransferCoeffR*self.hTpTuning
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
            frictionalPressureDrop = LMPressureGradientAvg(self.qualityInR,
                        self.qualityOutTwoPhase, self.abstractState, self.massFluxMeanR,
                        self.finsAndTubes.tubes.innerDiam, self.tempSatLiquidR,
                        self.tempSatVaporR)*self.effectiveCircuitLength*self.lengthFractionTwoPhase
            try:
                if self.abstractState.fluid in 'CarbonDioxide':
                    accelPressureDropTwoPhaseR = AccelPressureDrop(self.qualityInR,
                        self.qualityOutTwoPhase, self.abstractState, self.massFluxMeanR,
                        self.tempSatLiquidR, self.tempSatVaporR,D=self.finsAndTubes.tubes.innerDiam,
                        slipModel='Premoli')*self.effectiveCircuitLength*self.lengthFractionTwoPhase
                else:
                    accelPressureDropTwoPhaseR = AccelPressureDrop(self.qualityInR,
                        self.qualityOutTwoPhase, self.abstractState, self.massFluxMeanR,
                        self.tempSatLiquidR, self.tempSatVaporR, slipModel='Zivi')*\
                        self.effectiveCircuitLength*self.lengthFractionTwoPhase
            except:
                accelPressureDropTwoPhaseR = AccelPressureDrop(self.qualityInR,
                        self.qualityOutTwoPhase, self.abstractState, self.massFluxMeanR,
                        self.tempSatLiquidR, self.tempSatVaporR, slipModel='Zivi')*\
                        self.effectiveCircuitLength*self.lengthFractionTwoPhase
            pressureDropR = frictionalPressureDrop + accelPressureDropTwoPhaseR
        else:
            if not density:
                raise TypeError('In the superheat section, density must be defined.')
            specificVolumeR = 1/density
            #Pressure gradient using Darcy friction factor
            pressureGradientR = -self.darcyFrictionFactorSuperheat*specificVolumeR*\
                self.massFluxMeanR**2/(2*self.finsAndTubes.tubes.innerDiam)  #Pressure gradient
            pressureDropR = pressureGradientR*self.effectiveCircuitLength*\
                self.lengthFractionSuperheat
        return pressureDropR

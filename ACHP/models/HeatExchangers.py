# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 08:16:00 2024

@author: smcanana
"""
from enum import StrEnum
from itertools import product
import logging
from scipy.optimize import brentq
import numpy as np
from ACHP.models.Fluid import Fluid, ThermoProps, FluidApparatusProps
from ACHP.models.Correlations import getTempDensityPhaseFromPandH, getPhaseFromPandH, Cooper_PoolBoiling, twoPhaseDensity,\
    LMPressureGradientAvg, calculateAccelerationalPressureDrop, LongoCondensation, Petterson_supercritical, f_h_1phase_Tube,\
    f_h_1phase_Annulus, KandlikarEvaporation_average

class HEXType(StrEnum):
    """
    Heat Exchanger type enum for all available plate types
    """
    PLATE = "Plate-HX"
    COAXIAL = "Coaxial-HX"

class HeatExchanger():
    r"""
    There are a number of possibilities:

        Each fluid can:
        a) Not change phase
        b) Evaporate
        c) Condense

        Possibility matrix

                                      Hot stream
         Cold Stream  || Subcooled ||  Two-Phase || Superheated || Supercritical || Supercrit_liq ||
                      ------------------------------------------------------------------------------
         Subcooled    ||           ||            ||             ||               ||               ||
                      ------------------------------------------------------------------------------
         Two-Phase    ||           ||            ||             ||               ||               ||
                      ------------------------------------------------------------------------------
         Superheated  ||           ||            ||             ||               ||               ||
                      ------------------------------------------------------------------------------

    Hot stream goes to the left in the matrix, cold stream goes down.  If
    hot stream comes in subcooled, there are only three combinations that
    might exist.

    Based on inlet states can figure out what states are possible.
    """
    def __init__(self, fluidHot: Fluid, fluidCold: Fluid, massFlowHot: float, massFlowCold: float,
                 pressureInHot: float, pressureInCold: float, enthalpyInHot: float, enthalpyInCold: float,
                 conductivity: float, effectiveLength: float, surfaceRoughness: float=1.0,
                 htpColdTuning: float=1.0, htpHotTuning: float=1.0, hrHotTuning: float=1.0,
                 dpHotTuning: float=1.0, dpColdTuning: float=1.0):
        self.logger = logging.getLogger(str(self.type))

        #fluid properties on entry into heat exchanger
        self.fluidHot = fluidHot
        self.massFlowHot = massFlowHot
        self.fluidCold = fluidCold
        self.massFlowCold = massFlowCold

        self.fluidHot.fluidApparatiProps[self.type] = FluidApparatusProps(
            pressureIn=pressureInHot, enthalpyIn=enthalpyInHot)
        self.fluidCold.fluidApparatiProps[self.type] = FluidApparatusProps(
            pressureIn=pressureInCold, enthalpyIn=enthalpyInCold)
        self.fluidProps = {"Hot": self.fluidHot.fluidApparatiProps[self.type],
                           "Cold": self.fluidCold.fluidApparatiProps[self.type]}

        #heat exchanger properties
        self.conductivity = conductivity
        self.effectiveLength = effectiveLength
        self.surfaceRoughness = surfaceRoughness
        self.areaWettedHot: float
        self.diameterHydraulicHot: float
        self.areaWettedCold: float
        self.diameterHydraulicCold: float

        #tuning factors
        self.htpColdTuning = htpColdTuning
        self.htpHotTuning = htpHotTuning
        self.hrHotTuning = hrHotTuning
        self.dpColdTuning = dpColdTuning
        self.dpHotTuning = dpHotTuning

        # calculated geometry of hot fluid channel
        self.volumeChannelHot: float
        self.areaFlowHot: float
        self.massFluxAverageHot: float
        self.phaseInHot: float

        # calculated geometry of cold fluid channel
        self.volumeChannelCold: float
        self.areaFlowCold: float
        self.massFluxAverageCold: float
        self.phaseInCold: float
        self.thermalResistanceWall: float

        # calculated outlet properties
        self.qMax: float
        self.heatTransferred: float
        self.qFlux: float

        #aggregates for temperatures and phases
        self.fractions: list
        self.pressureDrops: list
        self.charges: list
        self.heatTransfers: list
        self.cellList: list

    def outputList(self):
        """
            Return a list of parameters for this component for further output

            It is a list of tuples, and each tuple is formed of items:
                [0] Description of value
                [1] Units of value
                [2] The value itself
        """
        outputList = [
            ('Effective Length','m',self.effectiveLength),
            ('Wetted area','m^2',self.areaWettedHot),
            ('Outlet Superheat','K',self.fluidProps["Cold"].tempIn - self.fluidProps["Cold"].tempDew),
            ('Q Total','W',self.heatTransferred),
            ('Charge Total Hot','kg',self.fluidProps["Hot"].charge),
            ('Charge Total Cold','kg',self.fluidProps["Cold"].charge),
            ('Pressure Drop Hot','Pa',self.fluidProps["Hot"].pressureDrop),
            ('Pressure Drop Cold','Pa',self.fluidProps["Cold"].pressureDrop),]
        for temp, phase in product(["Hot", "Cold"], ['Superheated', 'TwoPhase', 'Subcooled',
                                                     'Supercritical', 'Supercrit_liq']):
            outputList.append((f"Q {phase} {temp}", "W", self.heatTransfers[temp][phase]))
            outputList.append((f"Inlet {temp} stream temp", "K", self.fluidProps[temp].tempIn))
            outputList.append((f"Outlet {temp} stream temp", "K", self.fluidProps[temp].tempOut))
            outputList.append((f"Charge {phase} {temp}", "kg", self.charges[temp][phase]))
            outputList.append((f"{temp} Mean HTC {phase}", "W/m^2-K",
                               self.fluidProps[temp].getDict("heatTransferCoeffEffective")[phase]))
            outputList.append((f"Pressure Drop {phase} {temp}", "Pa", self.pressureDrops[temp][phase]))
            outputList.append((f"Area Fraction {phase} {temp}", "-", self.fractions[temp][phase]))
        return outputList

    def setUpCalculation(self, volumeChannels, areasFlow):
        self.logger.debug("In setUpCalculation")
        for temp in ["Hot", "Cold"]:
            tempFluid = getattr(self, f"fluid{temp}")
            tempBubble, densitySatLiquid, tempDew, densitySatVapor, tempSat = self.calculateTempsAndDensities(
                    tempFluid, self.fluidProps[temp].pressureIn)
            self.logger.debug("tempBubble: %g", tempBubble or 0.0)
            self.logger.debug("densitySatLiquid: %g", densitySatLiquid or 0.0)
            self.logger.debug("tempDew: %g", tempDew or 0.0)
            self.logger.debug("densitySatVapor: %g", densitySatVapor or 0.0)
            self.logger.debug("tempSat: %g", tempSat or 0.0)
            tempIn, densityIn, phaseIn = getTempDensityPhaseFromPandH(tempFluid, self.fluidProps[temp].pressureIn,
                                                      self.fluidProps[temp].enthalpyIn, tempBubble,
                                                      tempDew, densitySatLiquid, densitySatVapor)
            self.logger.debug("pressure in %s: %g", temp, self.fluidProps[temp].pressureIn)
            self.logger.debug("temperature in %s: %g", temp, tempIn)
            self.logger.debug("density in %s: %g", temp, densityIn)
            entropyIn = self.calculateEntropyOfFluid(tempFluid, self.fluidProps[temp].pressureIn,
                                                     tempIn, densityIn)
            massFluxAverage = getattr(self, f"massFlow{temp}")/areasFlow[temp]
            for attribute, value in zip(["volumeChannel", "areaFlow"], [volumeChannels[temp], areasFlow[temp]]):
                setattr(self, f"{attribute}{temp}", value)
            attributes = ["tempBubble", "densitySatLiquid", "tempDew", "densitySatVapor", "tempSat",
                          "tempIn", "densityIn", "phaseIn", "entropyIn", "massFluxAverage"]
            values = [tempBubble, densitySatLiquid, tempDew, densitySatVapor, tempSat, tempIn,
                      densityIn, phaseIn, entropyIn, massFluxAverage]
            for attribute, value in zip(attributes, values):
                setattr(self.fluidProps[temp], attribute, value)
            self.logger.debug("conductivity at %s: %s", temp, self.conductivity)
            self.logger.debug("areaWetted at %s: %s", temp, getattr(self, f"areaWetted{temp}"))
        self.qMax = self.determineHTBounds()

    def determineHTBounds(self):
        self.logger.debug("In determineHTBounds")
        # See if each phase could change phase if it were to reach the
        # inlet temperature of the opposite phase
        assert self.fluidProps["Hot"].tempIn > self.fluidProps["Cold"].tempIn, \
            "Hot phase is colder than cold phase"
        # Find the maximum possible rate of heat transfer as the minimum of
        # taking each stream to the inlet temperature of the other stream
        tempMeltHot = self.fluidHot.getMeltingTemperature(self.fluidProps["Hot"].pressureIn)
        temperatureHot = tempMeltHot if self.fluidProps["Cold"].tempIn < tempMeltHot else self.fluidProps["Cold"].tempIn

        enthalpyOutHot = self.fluidHot.calculateEnthalpy(ThermoProps.PT, self.fluidProps["Hot"].pressureIn,
                                                         temperatureHot)
        enthalpyOutCold = self.fluidCold.calculateEnthalpy(ThermoProps.PT, self.fluidProps["Cold"].pressureIn,
                                                           self.fluidProps["Hot"].tempIn)
        qMax = min([self.massFlowCold*(enthalpyOutCold - self.fluidProps["Cold"].enthalpyIn),
                    self.massFlowHot*(self.fluidProps["Hot"].enthalpyIn - enthalpyOutHot)])
        if qMax < 0:
            raise ValueError('qMax in PHE must be > 0')
        # Now we need to check for internal pinch points where the temperature
        # profiles would tend to overlap given the "normal" definitions of
        # maximum heat transfer of taking each stream to the inlet temperature
        # of the other stream
        # First we build the same vectors of enthalpies like below
        enthalpyListCold, enthalpyListHot = self.buildEnthalpyLists(qMax)
        # Then we find the temperature of each stream at each junction
        temperatureListCold = np.zeros_like(enthalpyListCold)
        temperatureListHot = np.zeros_like(enthalpyListHot)
        if len(enthalpyListHot) != len(enthalpyListCold):
            raise ValueError('Length of enthalpy lists for both fluids must be the same')
        #Make the lists of temperatures of each fluid at each cell boundary
        for index, (enthalpyHot, enthalpyCold) in enumerate(zip(enthalpyListHot, enthalpyListCold)):
            temperatureListCold[index] = getTempDensityPhaseFromPandH(self.fluidCold, self.fluidProps["Cold"].pressureIn,
                            enthalpyCold, self.fluidProps["Cold"].tempBubble,
                            self.fluidProps["Cold"].tempDew, self.fluidProps["Cold"].densitySatLiquid,
                            self.fluidProps["Cold"].densitySatVapor)[0]
            temperatureListHot[index] = getTempDensityPhaseFromPandH(self.fluidHot, self.fluidProps["Hot"].pressureIn,
                            enthalpyHot, self.fluidProps["Hot"].tempBubble,
                            self.fluidProps["Hot"].tempDew, self.fluidProps["Hot"].densitySatLiquid,
                            self.fluidProps["Hot"].densitySatVapor)[0]
        #TODO: could do with more generality if both streams can change phase
        # Check if any internal points are pinched
        if (temperatureListCold[1:-1] > temperatureListHot[1:-1]).any():
            # Loop over the internal cell boundaries
            for i in range(1,len(temperatureListCold)-1):
                # If cold stream is hotter than the hot stream
                if temperatureListCold[i] - 1e-9 > temperatureListHot[i]:
                    # Find new enthalpy of cold stream at the hot stream cell boundary
                    hPinch = self.fluidCold.calculateEnthalpy(ThermoProps.PT,
                                self.fluidProps["Cold"].pressureIn, temperatureListHot[i])
                    # Find heat transfer of hot stream in right-most cell
                    qExtra = self.massFlowHot*(enthalpyListHot[i+1] - enthalpyListHot[i])
                    qMax = self.massFlowCold*(hPinch - self.fluidProps["Cold"].enthalpyIn) + qExtra
        return qMax

    def givenQ(self, heat):
        """
        In this function, the heat transfer rate is imposed. Therefore the
        outlet states for both fluids are known, and each element can be solved
        analytically in one shot without any iteration.
        """
        self.logger.debug("In givenQ. heat is %g", heat)
        self.logger.debug("qmax is %g", self.qMax)
        if heat == 0.0:
            return -1
        if heat == self.qMax:
            return np.inf

        enthalpyListCold,enthalpyListHot = self.buildEnthalpyLists(heat)
        self.logger.debug("enthalpyListCold after BuildEnthalpyLists: %s", enthalpyListCold)
        self.logger.debug("enthalpyListHot after BuildEnthalpyLists: %s", enthalpyListHot)

        wList = []
        cellList = []
        qBoundList = self.calculateIncrementalHeatTransfer(enthalpyListHot, enthalpyListCold, 1e-9)
        self.logger.debug("qBoundList: %s", qBoundList)
        for qBound, enthalpyOutHot, enthalpyInCold in zip(qBoundList, enthalpyListHot, enthalpyListCold):
            calcName, calcInputs = self.determineHotAndColdPhases(enthalpyInCold, enthalpyOutHot, qBound)
            outputs = getattr(self, calcName)(calcInputs)
            self.logger.debug((outputs["identifier"], outputs['thermalFraction']))
            wList.append(outputs['thermalFraction'])
            cellList.append(outputs)
        self.cellList = cellList
        self.logger.debug("wlist: %s", wList)
        self.logger.debug('wsum: %s', np.sum(wList))
        return np.sum(wList) - 1.0

    def buildEnthalpyLists(self, qGiven):
        """
        Builds lists of enthalpies for both hot and cold fluids

        Parameters
        ----------
        qGiven : float
            given heat transfer value.

        Returns
        -------
        enthalpyOutCold: list[float]
            list of outlet enthalpies for cold fluid.
        enthalpyOutHot: list[float]
            list of outlet enthalpies for hot fluid.

        """
        self.logger.debug("In buildEnthalpyLists")
        #Start the enthalpy lists with inlet and outlet enthalpies
        #Ordered from lowest to highest enthalpies for both streams
        enthalpyLists = {"Hot": [self.fluidProps["Hot"].enthalpyIn - qGiven/self.massFlowHot,
                                 self.fluidProps["Hot"].enthalpyIn],
                         "Cold": [self.fluidProps["Cold"].enthalpyIn,
                                  self.fluidProps["Cold"].enthalpyIn + qGiven/self.massFlowCold]}
        #Save the value of qGiven and outlet enthalpies
        self.heatTransferred = qGiven
        self.fluidProps["Hot"].enthalpyOut = enthalpyLists["Hot"][0]
        self.fluidProps["Cold"].enthalpyOut = enthalpyLists["Cold"][1]
        eps = 1e-3
        enthalpySatLiquid = {}
        enthalpySatVapor = {}
        for temp in ["Hot", "Cold"]:
            if 'incomp' in getattr(self, f"fluid{temp}").backEnd.lower() \
                or self.fluidProps[temp].pressureIn > getattr(self, f"fluid{temp}").pressureCritical:
                enthalpySatLiquid[temp] = 1e9
                enthalpySatVapor[temp] = 1e9
            else:
                enthalpySatLiquid[temp] = getattr(self, f"fluid{temp}").calculateEnthalpy(ThermoProps.DT,
                        self.fluidProps[temp].densitySatLiquid, self.fluidProps[temp].tempBubble)
                enthalpySatVapor[temp] = getattr(self, f"fluid{temp}").calculateEnthalpy(ThermoProps.DT,
                        self.fluidProps[temp].densitySatVapor, self.fluidProps[temp].tempDew)
            # Check whether the enthalpy boundaries are within the bounds set by
            # the imposed amount of heat transfer
            self.logger.debug("enthalpyLists%s: %s", temp, enthalpyLists[temp])
            self.logger.debug("enthalpySatVapor%s: %g", temp, enthalpySatVapor[temp])
            self.logger.debug("enthalpySatLiquid%s: %g", temp, enthalpySatLiquid[temp])
            if (enthalpyLists[temp][0] + eps) < enthalpySatVapor[temp] < (enthalpyLists[temp][-1] - eps):
                self.logger.debug("enthalpySatVapor%s between first and last enthalpies", temp)
                enthalpyLists[temp].insert(len(enthalpyLists[temp]) - 1, enthalpySatVapor[temp])
            if (enthalpyLists[temp][0] + eps) < enthalpySatLiquid[temp] < (enthalpyLists[temp][-1] - eps):
                self.logger.debug("enthalpySatLiquid%s between first and last enthalpies", temp)
                enthalpyLists[temp].insert(1, enthalpySatLiquid[temp])
        self.calculateIncrementalHeatTransfer(enthalpyLists['Hot'], enthalpyLists['Cold'], 1e-6)
        for temp in ["Hot", "Cold"]:
            self.logger.debug("enthalpyLists%s: %s", temp, enthalpyLists[temp])
            setattr(self.fluidProps[temp], "enthalpySatLiquid", enthalpySatLiquid[temp])
            setattr(self.fluidProps[temp], "enthalpySatVapor", enthalpySatVapor[temp])
        assert(len(enthalpyLists["Cold"]) == len(enthalpyLists["Hot"])), "Length of enthalpy list for \
the cold channel is not equal to the length of the enthalpy list for the hot channel"
        return enthalpyLists["Cold"],enthalpyLists["Hot"]

    def determineHotAndColdPhases(self, enthalpyInCold, enthalpyOutHot, qBound):
        """
        Determines which phase each of the fluids is in

        Hot stream is either single phase or condensing (two phase)
        Cold stream is either single phase or evaporating (two phase)

        Parameters
        ----------
        enthalpyInCold : float
            inlet enthalpy of the cold stream.
        enthalpyOutHot : float
            outlet enthalpy of the hot stream.
        qBound : float
            heat exchanged.

        Raises
        ------
        NotImplementedError
            Some combinations of phases have no calculations to send to.

        Returns
        -------
        str
            name of the method used for the combination of phases.
        inputs : dict
            inputs needed for each method. Possibilites are:
                "heatTransferred": heat transferred between the channels
                'tempMeanHot': average temperature of the hot side
                'tempMeanCold': average temperature of the cold side
                'tempInHot': inlet temperature of the hot side
                'tempInCold': inlet temperature of the cold side
                'phaseHot': phase of the hot side
                'phaseCold': phase of the cold side
                'xInH': vapor pressure in on the hot side (only for two-phase)
                'xOutH': vapor pressure out on the hot side (only for two-phase)
                'xInC': vapor pressure in on the cold side (only for two-phase)
                'xOutC': vapor pressure out on the cold side (only for two-phase)

        """
        #Figure out the inlet and outlet enthalpy for this cell
        enthalpiesIn = {"Cold": enthalpyInCold, "Hot": enthalpyOutHot + qBound/self.massFlowHot}
        enthalpiesOut = {"Hot": enthalpyOutHot, "Cold": enthalpyInCold + qBound/self.massFlowCold}
        assert enthalpiesIn["Hot"] > enthalpiesOut["Hot"], "Hot stream is heating!"
        assert enthalpiesIn["Cold"] < enthalpiesOut["Cold"], "Cold stream is cooling!"
        #Use midpoint enthalpies to figure out the phase in the cell
        phaseHot = getPhaseFromPandH(self.fluidHot, self.fluidProps["Hot"].pressureIn,
                           (enthalpiesIn["Hot"] + enthalpiesOut["Hot"])/2,
                           self.fluidProps["Hot"].tempBubble, self.fluidProps["Hot"].tempDew,
                           self.fluidProps["Hot"].densitySatLiquid, self.fluidProps["Hot"].densitySatVapor)
        phaseCold = getPhaseFromPandH(self.fluidCold, self.fluidProps["Cold"].pressureIn,
                           (enthalpiesIn["Cold"] + enthalpiesOut["Cold"])/2,
                           self.fluidProps["Cold"].tempBubble, self.fluidProps["Cold"].tempDew,
                           self.fluidProps["Cold"].densitySatLiquid, self.fluidProps["Cold"].densitySatVapor)
        tempsIn = {"Hot": getTempDensityPhaseFromPandH(self.fluidHot, self.fluidProps["Hot"].pressureIn,
                                enthalpiesIn["Hot"], self.fluidProps["Hot"].tempBubble,
                                self.fluidProps["Hot"].tempDew, self.fluidProps["Hot"].densitySatLiquid,
                                self.fluidProps["Hot"].densitySatVapor)[0],
                   "Cold": getTempDensityPhaseFromPandH(self.fluidCold, self.fluidProps["Cold"].pressureIn,
                                enthalpiesIn["Cold"], self.fluidProps["Cold"].tempBubble,
                                self.fluidProps["Cold"].tempDew, self.fluidProps["Cold"].densitySatLiquid,
                                self.fluidProps["Cold"].densitySatVapor)[0]}
        tempsOut = {"Hot": getTempDensityPhaseFromPandH(self.fluidHot, self.fluidProps["Hot"].pressureIn,
                                enthalpiesOut["Hot"], self.fluidProps["Hot"].tempBubble,
                                self.fluidProps["Hot"].tempDew, self.fluidProps["Hot"].densitySatLiquid,
                                self.fluidProps["Hot"].densitySatVapor)[0],
                    "Cold": getTempDensityPhaseFromPandH(self.fluidCold, self.fluidProps["Cold"].pressureIn,
                                enthalpiesOut["Cold"], self.fluidProps["Cold"].tempBubble,
                                self.fluidProps["Cold"].tempDew, self.fluidProps["Cold"].densitySatLiquid,
                                self.fluidProps["Cold"].densitySatVapor)[0]}
        self.logger.debug("Q bound: %g", qBound)
        self.logger.debug("temp in cold: %g", tempsIn["Cold"])
        self.logger.debug("temp out cold: %g", tempsOut["Cold"])
        self.logger.debug("enthalpy in cold: %g", enthalpiesIn["Cold"])
        self.logger.debug("enthalpy out cold: %g", enthalpiesOut["Cold"])
        self.logger.debug("pressure in cold: %g", self.fluidProps["Cold"].pressureIn)
        self.logger.debug("tempBubbleCold: %g", self.fluidProps["Cold"].tempBubble or 0.0)
        self.logger.debug("tempDewCold: %g", self.fluidProps["Cold"].tempDew or 0.0)
        self.logger.debug("densitySatLiquidCold: %g", self.fluidProps["Cold"].densitySatLiquid or 0.0)
        self.logger.debug("densitySatVaporCold: %g", self.fluidProps["Cold"].densitySatVapor or 0.0)
        inputs = {
            "heatTransferred": qBound,
            'tempMeanHot': (tempsIn["Hot"] + tempsOut["Hot"])/2,
            'tempMeanCold': (tempsIn["Cold"] + tempsOut["Cold"])/2,
            'tempInHot': tempsIn["Hot"],
            'tempInCold': tempsIn["Cold"],
            'phaseHot': phaseHot,
            'phaseCold': phaseCold
                  }
        for temp, tempOut in tempsOut.items():
            inputs[f"specificHeat{temp}"] = (enthalpiesIn[temp] - enthalpiesOut[temp])/\
                (tempsIn[temp] - tempOut) if tempsIn[temp] != tempOut else (enthalpiesIn[temp] - \
                enthalpiesOut[temp])/1e-5
        for temp, phase in {"Hot": phaseHot, "Cold": phaseCold}.items():
            if phase=='TwoPhase':
                tempDiff = self.fluidProps[temp].enthalpySatVapor - \
                            self.fluidProps[temp].enthalpySatLiquid
                inputs.update({
                    'fractionHigh': min((enthalpiesIn[temp] - \
                            self.fluidProps[temp].enthalpySatLiquid)/tempDiff, 1),
                    'fractionLow': max((enthalpiesOut[temp] - \
                            self.fluidProps[temp].enthalpySatLiquid)/tempDiff, 0)
                    })
                self.logger.debug("enthalpySatVapor%s: %g", temp, self.fluidProps[temp].enthalpySatVapor)
                self.logger.debug("enthalpySatLiquid%s: %g", temp, self.fluidProps[temp].enthalpySatLiquid)
                self.logger.debug("enthalpyIn%s: %g", temp, enthalpiesIn[temp])
                self.logger.debug("enthalpyOut%s: %g", temp, enthalpiesOut[temp])
                self.logger.debug("fractionHigh%s: %g", temp, inputs['fractionHigh'])
                self.logger.debug("enthalpyOut%s: %g", temp, inputs['fractionLow'])
        if all(x in ["Subcooled", "Superheated"] for x in [phaseHot, phaseCold]):
            # Both are single-phase
            return "_onePhaseHOnePhaseCQimposed", inputs
        if phaseCold == 'TwoPhase':
            inputs['xOutC'] = inputs.pop('fractionLow')
            inputs['xInC'] = inputs.pop('fractionHigh')
            if phaseHot in ['Subcooled','Superheated']:
                # Cold stream is evaporating, and hot stream is single-phase (SH or SC)
                return "_onePhaseHTwoPhaseCQimposed", inputs
            if phaseHot in ['Supercritical','Supercrit_liq']:
                # Cold stream is evaporating, and hot stream is transcritical-phase (Supercrit or Supercrit_liq)
                return "_transCritPhaseHTwoPhaseCQimposed", inputs
        if phaseCold in ['Subcooled','Superheated']:
            if phaseHot == 'TwoPhase':
                inputs['xInH'] = inputs.pop('fractionHigh')
                inputs['xOutH'] = inputs.pop('fractionLow')
                # Hot stream is condensing, and cold stream is single-phase (SH or SC)
                # TODO: bounding state can be saturated state if hot stream is condensing
                return "_twoPhaseHOnePhaseCQimposed", inputs
            if phaseHot in ['Supercritical','Supercrit_liq']:
                # Cold stream is single-phase (SH or SC), and hot stream is transcritical-phase
                #(Supercrit or Supercrit_liq)
                inputs.update({
                    'specificHeatCold':(enthalpiesIn["Cold"] - enthalpiesOut["Cold"])/\
                        (tempsIn["Cold"] - tempsOut["Cold"]),
                    'tempOutHot':tempsOut["Hot"]
                    })
                return "_transCritPhaseHOnePhaseCQimposed", inputs
        raise NotImplementedError(f"The case where the cold fluid phase is {phaseCold} and the hot\
                                  fluid phase is {phaseHot} has not been implemented.")

    def _onePhaseHOnePhaseCQimposed(self,inputs):
        """
        Single phase on both sides (hot and cold)
        inputs: dictionary of parameters
        outputs: dictionary of parameters,
        but mainly w, pressure drop and heat transfer coefficient
        This function calculate the fraction of heat exchanger
        that would be required for given thermal duty "w" and DP and h
        """
        self.logger.debug("In super _onePhaseHOnePhaseCQimposed")
        self.logger.debug("inputs: %s", inputs)
        #Evaluate UA [W/K] if entire HX was in this section
        conductanceHot = 1/(inputs['heatTransferCoeffHot']*self.areaWettedHot)
        conductanceCold = 1/(inputs['heatTransferCoeffCold']*self.areaWettedCold)
        conductanceTotal = 1/(conductanceHot + conductanceCold + self.thermalResistanceWall)
        #Get Ntu [-]
        capacitance = [inputs['specificHeatCold']*self.massFlowCold,
                       inputs['specificHeatHot']*self.massFlowHot]
        capacitanceMin = min(capacitance)
        capacitanceRatio = capacitanceMin/max(capacitance)
        #Effectiveness [-]
        qMax = capacitanceMin*(inputs['tempInHot'] - inputs['tempInCold'])
        epsilon = inputs['heatTransferred']/qMax
        #Pure counterflow with capacitanceRatio<1 (Incropera Table 11.4)
        if epsilon > 1.0:
            # In practice this can never happen, but sometimes
            # during bad iterations it is possible
            ntu = 10000
        else:
            ntu = 1/(capacitanceRatio - 1)*np.log((epsilon-1)/(epsilon*capacitanceRatio - 1))
        #Required UA value
        conductanceRequired = capacitanceMin*ntu
        #w is required part of heat exchanger for this duty
        thermalFraction = conductanceRequired/conductanceTotal
        #Determine both charge components
        densityHot = self.fluidHot.calculateDensity(ThermoProps.PT, self.fluidProps["Hot"].pressureIn,
                                                    inputs['tempMeanHot'])
        chargeHot = thermalFraction * self.volumeChannelHot * densityHot
        densityCold = self.fluidCold.calculateDensity(ThermoProps.PT, self.fluidProps["Cold"].pressureIn,
                                                      inputs['tempMeanCold'])
        chargeCold = thermalFraction * self.volumeChannelCold * densityCold

        self.logger.debug("heat transfer coeff hot: %g", inputs['heatTransferCoeffHot'])
        self.logger.debug("heat transfer coeff cold: %g", inputs['heatTransferCoeffCold'])
        self.logger.debug("conductance hot: %g", conductanceHot)
        self.logger.debug("conductance cold: %g", conductanceCold)
        self.logger.debug("total conductance: %g", conductanceTotal)
        self.logger.debug("capacitance: %s", capacitance)
        self.logger.debug("capacitanceMin: %g", capacitanceMin)
        self.logger.debug("capacitanceRatio: %g", capacitanceRatio)
        self.logger.debug("qMax: %g", qMax)
        self.logger.debug("epsilon: %g", epsilon)
        self.logger.debug("ntu: %g", ntu)
        self.logger.debug("conductanceRequired: %g", conductanceRequired)
        self.logger.debug("thermalFraction: %g", thermalFraction)
        self.logger.debug("densityHot: %g", densityHot)
        self.logger.debug("chargeHot: %g", chargeHot)
        self.logger.debug("densityCold: %g", densityCold)
        self.logger.debug("chargeCold: %g", chargeCold)


        #Pack outputs
        outputs = {**inputs, **{
            'thermalFraction': thermalFraction,
            'tempOutHot': inputs['tempInHot'] - inputs['heatTransferred']/\
                (self.massFlowCold*inputs['specificHeatHot']),
            'tempOutCold': inputs['tempInCold'] + inputs['heatTransferred']/\
                (self.massFlowCold*inputs['specificHeatCold']),
            'chargeCold': chargeCold,
            'chargeHot': chargeHot,
            'pressureDropHot': -inputs['pressureDropHot'],
            'pressureDropCold': -inputs['pressureDropCold']
        }}
        return outputs

    def _onePhaseHTwoPhaseCQimposed(self,inputs):
        """
        The hot stream is single phase, and the cold stream is evaporating (two phase)
        inputs: dictionary of parameters
        outputs: dictionary of parameters,
        but mainly w, pressure drop and heat transfer coefficient
        This function calculate the fraction of heat exchanger
        that would be required for given thermal duty "w" and DP and h
        """
        self.logger.debug("In super _onePhaseHTwoPhaseCQimposed")
        #Reduced pressure for Cooper Correlation
        fractionChange = 999
        thermalFraction = 1
        """
        The Cooper Pool boiling relationship is a function of the heat flux,
        therefore the heat flux must be iteratively determined

        According to Cleasson J. PhD Thesis "Thermal and Hydraulic Performance
        of Compact Brazed Plate Heat Exchangers Operating a Evaporators in Domestic
        Heat Pumps", KTH, 2004, pp. 98: the saturated nucleate pool boiling
        correlation by Cooper (1984) works rather well at varying conditions,
        if multiplied by a factor C=1.5.

        To this end, a tuning coefficient, i.e. htpColdTuning, is added to the
        Cooper pool boiling correlation.

        """
        while abs(fractionChange) > 1e-6:
            qFlux = inputs['heatTransferred']/(thermalFraction*self.areaWettedCold)
            #Heat transfer coefficient from Cooper Pool Boiling with
            #correction for the two-phase zone of the cold side
            heatTransferCoeffTwoPhase = self.htpColdTuning*Cooper_PoolBoiling(self.fluidCold,
                                            self.surfaceRoughness, qFlux, str(self.type))
            fractionUpdated = self.calculateFraction("Cold", inputs['heatTransferCoeffHot'],
                                                heatTransferCoeffTwoPhase, inputs['specificHeatHot'],
                                                inputs['tempInHot'], inputs['heatTransferred'])
            self.logger.debug("fractionUpdated: %g", fractionUpdated)
            self.logger.debug("thermalFraction: %g", thermalFraction)
            fractionChange = fractionUpdated - thermalFraction
            thermalFraction = fractionUpdated
        #Refrigerant charge
        densityHot = self.fluidHot.calculateDensity(ThermoProps.PT, self.fluidProps["Hot"].pressureIn,
                                                    inputs['tempMeanHot'])
        chargeHot = thermalFraction*self.volumeChannelHot*densityHot
        densityCold = twoPhaseDensity(self.fluidCold, inputs['xInC'], inputs['xOutC'],
                        self.fluidProps["Cold"].tempDew, self.fluidProps["Cold"].tempBubble)
        chargeCold = densityCold*thermalFraction*self.volumeChannelCold
        pressureDropFriction = LMPressureGradientAvg(inputs['xInC'], inputs['xOutC'], self.fluidCold,
                        self.massFlowCold/self.areaFlowCold, self.diameterHydraulicCold,
                        self.fluidProps["Cold"].tempBubble, self.fluidProps["Cold"].tempDew,
                        C=inputs['C'])*thermalFraction*self.effectiveLength
        #Accelerational pressure drop component
        pressureDropAcceleration = calculateAccelerationalPressureDrop(inputs['xInC'], inputs['xOutC'],
                        self.fluidCold, self.massFlowCold/self.areaFlowCold,
                        self.fluidProps["Cold"].tempBubble,  self.fluidProps["Cold"].tempDew)*\
                        thermalFraction*self.effectiveLength
        outputs = {**inputs, **{
            'thermalFraction': thermalFraction,
            'tempOutHot': inputs['tempInHot'] - inputs['heatTransferred']/\
                (self.massFlowCold*inputs['specificHeatHot']),
            'tempOutCold': inputs['tempInCold'] + inputs['heatTransferred']/\
                (self.massFlowCold*inputs['specificHeatCold']),
            'chargeCold': chargeCold,
            'chargeHot': chargeHot,
            'pressureDropHot': -inputs['pressureDropHot'],
            'pressureDropCold': pressureDropFriction + pressureDropAcceleration,
            'qFlux': qFlux,
            'heatTransferCoeffCold': heatTransferCoeffTwoPhase
        }}
        return outputs

    def _twoPhaseHOnePhaseCQimposed(self,inputs):
        """
        Hot stream is condensing (two phase), cold stream is single phase
        inputs: dictionary of parameters
        outputs: dictionary of parameters,
        but mainly thermalFraction, pressure drop and heat transfer coefficient
        This function calculate the fraction of heat exchanger
        that would be required for given thermal duty "thermalFraction" and DP and h
        """
        self.logger.debug("In super _twoPhaseHOnePhaseCQimposed")
        heatTransferCoeffTwoPhase = LongoCondensation((inputs['xOutH'] + inputs['xInH'])/2,
                                        self.massFlowCold/self.areaFlowHot, self.diameterHydraulicHot,
                                        self.fluidHot, self.fluidProps["Hot"].tempBubble,
                                        self.fluidProps["Hot"].tempDew)*self.htpHotTuning
        thermalFraction = self.calculateFraction("Hot", inputs['heatTranferCoeffCold'],
                                                heatTransferCoeffTwoPhase, inputs['specificHeatCold'],
                                                inputs['tempInCold'], inputs['heatTransferred'])
        #TODO: these can be refactored with onephase_twoPhase calculations
        densityCold = self.fluidCold.calculateDensity(ThermoProps.PT, self.fluidProps["Cold"].pressureIn,
                                                      inputs['tempMeanCold'])
        chargeCold = thermalFraction*self.volumeChannelCold*densityCold
        densityHot = twoPhaseDensity(self.fluidHot,inputs['xOutH'],inputs['xInH'],
                        self.fluidProps["Hot"].tempDew, self.fluidProps["Hot"].tempBubble, slipModel='Zivi')
        chargeHot = densityHot*thermalFraction*self.volumeChannelHot
        pressureDropFriction = LMPressureGradientAvg(inputs['xOutH'], inputs['xInH'], self.fluidHot,
                        self.massFlowCold/self.areaFlowHot, self.diameterHydraulicHot,
                        self.fluidProps["Hot"].tempBubble, self.fluidProps["Hot"].tempDew,
                        C=inputs['C'])*thermalFraction*self.effectiveLength
        #Accelerational pressure drop component
        pressureDropAcceleration = -calculateAccelerationalPressureDrop(inputs['xOutH'], inputs['xInH'],
                            self.fluidHot, self.massFlowCold/self.areaFlowHot,
                            self.fluidProps["Hot"].tempBubble, self.fluidProps["Hot"].tempDew,
                            slipModel='Zivi')*thermalFraction*self.effectiveLength
        outputs = {**inputs, **{
            'thermalFraction': thermalFraction,
            'tempOutCold': inputs['tempInCold'] - inputs['heatTransferred']/\
                (self.massFlowCold*inputs['specificHeatCold']),
            'tempOutHot': inputs['tempInHot'] - inputs['heatTransferred']/\
                (self.massFlowCold*inputs['specificHeatHot']),
            'pressureDropCold': -inputs['pressureDropCold'],
            'pressureDropHot': pressureDropAcceleration + pressureDropFriction,
            'chargeCold': chargeCold,
            'chargeHot': chargeHot,
            'heatTransferCoeffHot': heatTransferCoeffTwoPhase
        }}
        return outputs

    def calculateFraction(self, twoPhaseSide, heatTransferCoeff1Phase, heatTransferCoeff2Phase,
                             specificHeatSinglePhase, tempInSinglePhase, heatTransferred):
        singlePhaseSide = "Cold" if twoPhaseSide == "Hot" else "Hot"
        conductanceTotal = 1/(1/(heatTransferCoeff1Phase*getattr(self, f"areaWetted{singlePhaseSide}")) + \
                              1/(heatTransferCoeff2Phase*getattr(self, f"areaWetted{twoPhaseSide}")) + \
                                  self.thermalResistanceWall)
        capacitanceSinglePhase = specificHeatSinglePhase*getattr(self, f"massFlow{singlePhaseSide}")
        tempDiff = self.fluidProps[twoPhaseSide].tempSat - tempInSinglePhase
        tempDiff = tempDiff  if twoPhaseSide == "Hot" else -tempDiff
        qMax = capacitanceSinglePhase*tempDiff

        epsilon = heatTransferred/qMax if heatTransferred/qMax < 1.0 else 1.0 - 1e-12
        #capacitanceRatio = 0, so ntu is simply
        ntu = -np.log(1 - epsilon)
        conductanceRequired = ntu*capacitanceSinglePhase
        self.logger.debug("areaWettedHot: %g", self.areaWettedHot)
        self.logger.debug("areaWettedCold: %g", self.areaWettedCold)
        self.logger.debug("heatTransferCoeffSinglePhase: %g", heatTransferCoeff1Phase)
        self.logger.debug("heatTransferCoeffTwoPhase: %g", heatTransferCoeff2Phase)
        self.logger.debug("thermalResistanceWall: %g", self.thermalResistanceWall)
        self.logger.debug("conductanceTotal: %g", conductanceTotal)
        self.logger.debug("capacitanceSinglePhase: %g", capacitanceSinglePhase)
        self.logger.debug("tempDiff: %g", tempDiff)
        self.logger.debug("qMax: %g", qMax)
        self.logger.debug("epsilon: %g", epsilon)
        self.logger.debug("ntu: %g", ntu)
        self.logger.debug("conductanceRequired: %g", conductanceRequired)
        return conductanceRequired/conductanceTotal

    def _transCritPhaseHTwoPhaseCQimposed(self,inputs):
        """
        The hot stream is Transcritical phase (supercritical or supercrit_liq), and the cold stream
        is evaporating (two phase)
        inputs: dictionary of parameters
        outputs: dictionary of parameters,
        but mainly thermalFraction, pressure drop and heat transfer coefficient
        This function calculate the fraction of heat exchanger
        that would be required for given thermal duty "thermalFraction" and DP and h
        """
        self.logger.debug("In super _transCritPhaseHTwoPhaseCQimposed")
        #Reduced pressure for Cooper Correlation
        fractionChange = 999
        thermalFraction = 1
        while abs(fractionChange)>1e-6:
            qFlux = inputs['heatTransferred']/(thermalFraction*self.areaWettedCold)
            #Heat transfer coefficient from Cooper Pool Boiling with
            #correction for the two-phase zone of the cold side
            heatTransferCoeffTwoPhase = self.calculateHeatTransferCoeff('Cold', qFlux)*self.htpColdTuning
            #wall heat resistance
            thermalResistanceWall = self.thermalResistanceWall
            #cold-side heat resistance
            resistanceCold = 1/(heatTransferCoeffTwoPhase*self.areaWettedCold)
            #wall temperature calculated from energy balance on the cold-side
            tempWall = (thermalResistanceWall+resistanceCold)*inputs['heatTransferred'] + \
                self.fluidProps["Cold"].tempSat
            #Calculate HTC for the hot Transcritical-phase fluid
            #HTC and friction calculated using Pettersson (2000) correlations
            enthalpyHot, frictionHot, specificHeatHot, densityHot = Petterson_supercritical(
                inputs['tempMeanHot'], tempWall, self.fluidHot, self.massFluxAverageHot,
                self.diameterHydraulicHot, 0, self.diameterHydraulicHot/self.effectiveLength,
                0, self.fluidProps["Hot"].pressureIn, qFlux)
            enthalpyHot = self.hrHotTuning*enthalpyHot #correct HTC for hot-side
            #Evaluate UA [W/K]
            conductanceTotal = 1/(1/(enthalpyHot*self.areaWettedHot) + \
                                1/(heatTransferCoeffTwoPhase*self.areaWettedCold)+self.thermalResistanceWall)
            #cp of cold-side (two-phase) is very large compared to hot-side (trans-phase).
            #Therefore, capacitanceMin is on hot-side
            capacitanceMin = specificHeatHot*self.massFlowCold
            #Effectiveness [-]
            qMax = capacitanceMin*(inputs['tempInHot'] - self.fluidProps["Cold"].tempSat)
            epsilon = inputs['heatTransferred']/qMax
            if epsilon>=1.0:
                epsilon=1.0-1e-12
            #Get Ntu [-]
            ntu=-np.log(1-epsilon)
            #Required UA value
            conductanceRequired=ntu*capacitanceMin
            fractionChange=conductanceRequired/conductanceTotal-thermalFraction
            thermalFraction=conductanceRequired/conductanceTotal
        #Refrigerant charge
        chargeHot = thermalFraction*self.volumeChannelHot*densityHot
        densityCold = twoPhaseDensity(self.fluidCold,inputs['xInC'], inputs['xOutC'],
                    self.fluidProps["Cold"].tempDew, self.fluidProps["Cold"].tempBubble, slipModel='Zivi')
        chargeCold = densityCold*thermalFraction*self.volumeChannelCold

        #Hot-side Pressure gradient using Darcy friction factor
        volumeSpecificHot = 1.0/densityHot
        pressureGradientHot = -frictionHot*volumeSpecificHot*self.massFluxAverageHot**2/\
                                    (2*self.diameterHydraulicHot)
        pressureDropFrictionHot=pressureGradientHot*self.effectiveLength*thermalFraction

        pressureDropFrictionCold = LMPressureGradientAvg(inputs['xInC'], inputs['xOutC'],
                    self.fluidCold, self.massFlowCold/self.areaFlowCold,
                    self.diameterHydraulicCold, self.fluidProps["Cold"].tempBubble,
                    self.fluidProps["Cold"].tempDew,C=4.67)*thermalFraction*self.effectiveLength
        #Accelerational pressure drop component
        pressureDropAccelerationCold = calculateAccelerationalPressureDrop(inputs['xInC'], inputs['xOutC'],
                    self.fluidCold, self.massFlowCold/self.areaFlowCold,
                    self.fluidProps["Cold"].tempBubble, self.fluidProps["Cold"].tempDew, slipModel='Zivi')*\
                    thermalFraction*self.effectiveLength
        outputs = {**inputs, **{
            'thermalFraction': thermalFraction,
            'tempOutHot': inputs['tempInHot']-inputs['heatTransferred']/(self.massFlowCold*specificHeatHot),
            'chargeCold': chargeCold,
            'chargeHot': chargeHot,
            'pressureDropHot': pressureDropFrictionHot,
            'pressureDropCold': pressureDropFrictionCold+pressureDropAccelerationCold,
            'heatTransferCoeffCold': heatTransferCoeffTwoPhase,
            'qFlux': qFlux,
            'specificHeatHot': specificHeatHot,
        }}
        return outputs

    def _transCritPhaseHOnePhaseCQimposed(self, inputs):
        """
        The hot stream is Transcritical phase (supercritical or supercrit_liq), and the cold stream
        is single phase (SC or SH)
        inputs: dictionary of parameters
        outputs: dictionary of parameters,
        but mainly thermalFraction, pressure drop and heat transfer coefficient
        This function calculates the fraction of heat exchanger
        that would be required for given thermal duty "thermalFraction" and DP and h
        """
        self.logger.debug("In super _transCritPhaseHOnePhaseCQimposed")
        #cold-side heat resistance
        resistanceCold = 1/(inputs['heatTransferCoeffCold']*self.areaWettedCold)
        #wall temperature calculate from energy balance on the cold-side
        tempWall = (self.thermalResistanceWall + resistanceCold)*inputs['heatTransferred'] + \
            inputs['tempMeanCold'] #This is just an initial wall temperature
        fractionChange = 999
        thermalFraction = 1
        while abs(fractionChange)>1e-6:
            #heat flux
            qFlux = inputs['heatTransferred']/(thermalFraction*self.areaWettedHot)
            #Calculate HTC for the hot Transcritical-phase fluid
            #HTC and friction calculated using Pettersson (2000) correlations
            heatTransferCoeffHot, frictionHot, specificHeatHot, densityHot = Petterson_supercritical(
                                inputs['tempMeanHot'], tempWall, self.fluidHot,
                                self.massFluxAverageHot, self.diameterHydraulicHot, 0,
                                self.diameterHydraulicHot/self.effectiveLength, 0,
                                self.fluidProps["Hot"].pressureIn, qFlux)*self.hrHotTuning
            #Update wall temperature for the next iteration
            resistanceHot = 1/(heatTransferCoeffHot*self.areaWettedHot) #hot-side heat resistance
            tempOutHot = inputs['tempInHot'] - inputs['heatTransferred']/(self.massFlowCold*specificHeatHot)
            tempWall = tempOutHot - resistanceHot*inputs['heatTransferred']
            conductanceTotal = 1/(1/(heatTransferCoeffHot*self.areaWettedHot + \
                    1/(inputs['heatTransferCoeffCold']*self.areaWettedCold) + self.thermalResistanceWall))
            #Get Ntu [-]
            capacitance = [inputs['specificHeatCold']*self.massFlowCold,specificHeatHot*self.massFlowCold]
            capacitanceMin = min(capacitance)
            capacitanceRatio = capacitanceMin/max(capacitance)
            #Effectiveness [-]
            qMax=capacitanceMin*(inputs['tempInHot']-inputs['tempInCold'])
            epsilon = inputs['heatTransferred']/qMax if inputs['heatTransferred']/qMax < 1.0 else 1.0-1e-12
            #Pure counterflow with capacitanceRatio<1 (Incropera Table 11.4)
            ntu = 1/(capacitanceRatio - 1)*np.log((epsilon - 1)/(epsilon*capacitanceRatio - 1))
            #Required UA value
            conductanceRequired = capacitanceMin*ntu
            fractionChange = conductanceRequired/conductanceTotal-thermalFraction
            thermalFraction = conductanceRequired/conductanceTotal
        #Determine both charge components
        chargeHot = thermalFraction * self.volumeChannelHot * densityHot
        densityCold = self.fluidCold.calculateDensity(ThermoProps.PT, self.fluidProps["Cold"].pressureIn,
                                                      inputs['tempMeanCold'])
        chargeCold = thermalFraction * self.volumeChannelCold * densityCold
        #Hot-side Pressure gradient using Darcy friction factor
        volumeSpecificHot = 1.0/densityHot
        pressureGradientHot = -frictionHot*volumeSpecificHot*self.massFluxAverageHot**2/\
            (2*self.diameterHydraulicHot)
        pressureDropFrictionHot = pressureGradientHot*self.effectiveLength*thermalFraction
        outputs = {**inputs, **{
            'thermalFraction': thermalFraction,
            'tempOutHot': inputs['tempInHot']-inputs['heatTransferred']/\
                (self.massFlowCold*inputs['specificHeatHot']),
            'tempOutCold': inputs['tempInCold']+inputs['heatTransferred']/\
                (self.massFlowCold*inputs['specificHeatCold']),
            'chargeCold': chargeCold,
            'chargeHot': chargeHot,
            'pressureDropHot': pressureDropFrictionHot,
            'pressureDropCold': -inputs['pressureDrop'],
            'heatTransferCoeffHot': heatTransferCoeffHot,
            'qFlux':qFlux,
            'specificHeatHot':specificHeatHot,

        }}
        return outputs

    def calculateHeatTransferCoeff(self, temp, qFlux, xIn=None, xOut=None, massFlux=None):
        """
        Calculates the heat transfer coefficient of a fluid. Should be overridden by child classes

        Parameters
        ----------
        temp : str
            'Hot' or 'Cold', denotes the stream being acted upon.
        qFlux : float
            heat transferred.
        xIn : float, optional
            vapor fraction input on the cold side. The default is None.
        xOut : TYPE, optional
            vapor fraction input on the hot side. The default is None.
        massFlux : float, optional
            mass. The default is None.

        Returns
        -------
        float
            heat transfer coefficient.

        """

    def postProcess(self, cellList):
        """
        Combine all the cells to calculate overall parameters like pressure drop
        and fraction of heat exchanger in two-phase on both sides
        """
        self.logger.debug("In postProcess")
        self.logger.debug("cellList: %s", cellList)
        aggregates = {'pressureDropCold': 0, 'pressureDropHot': 0, 'chargeCold': 0,
                      'chargeHot': 0}
        for cell in cellList:
            for varName in aggregates:
                aggregates[varName] += cell[varName]
        self.fractions = {'Hot':{}, 'Cold':{}}
        self.pressureDrops = {'Hot':{}, 'Cold':{}}
        self.charges = {'Hot':{}, 'Cold':{}}
        self.heatTransfers = {'Hot':{}, 'Cold':{}}
        for temp, phase in product(['Hot', 'Cold'],
                        ['Superheated', 'TwoPhase', 'Subcooled', 'Supercritical', 'Supercrit_liq']):
            phaseFilter = list(filter(lambda x, t=temp, p=phase: x[f"phase{t}"] == p, cellList))
            self.logger.debug("phaseFilter before getting fraction: %s", phaseFilter)
            self.fractions[temp][phase] = sum(map(lambda x: x['thermalFraction'], phaseFilter))
            self.pressureDrops[temp][phase] = sum(map(lambda x, t=temp: x[f"pressureDrop{t}"], phaseFilter))
            self.fluidProps[temp].pressureDrop = sum(self.pressureDrops[temp].values())*\
                                getattr(self, f"dp{temp}Tuning")
            self.charges[temp][phase] = sum(map(lambda x, t=temp: x[f'charge{t}'],phaseFilter))
            self.fluidProps[temp].charge = sum(self.charges[temp].values())
            self.heatTransfers[temp][phase] = sum(map(lambda x: x['heatTransferred'], phaseFilter))
            thermalFraction = list(map(lambda x: x['thermalFraction'], phaseFilter))
            heatTransferCoeff = list(map(lambda x, t=temp: x[f'heatTransferCoeff{t}'], phaseFilter))
            self.logger.debug("thermal fraction: %s", thermalFraction)
            self.logger.debug("heatTransferCoeff: %s", heatTransferCoeff)
            self.fluidProps[temp].addToProperty("heatTransferCoeffEffective", phase, float(
                    sum(np.array(thermalFraction)*np.array(heatTransferCoeff))/\
                    sum(thermalFraction)) if len(thermalFraction) > 0 else 0.0)
        for temp in ['Hot', 'Cold']:
            self.fluidProps[temp].tempOut,self.fluidProps[temp].densityOut = getTempDensityPhaseFromPandH(
                getattr(self, f"fluid{temp}"), self.fluidProps[temp].pressureIn,
                self.fluidProps[temp].enthalpyOut, self.fluidProps[temp].tempBubble,
                self.fluidProps[temp].tempDew,self.fluidProps[temp].densitySatLiquid,
                self.fluidProps[temp].densitySatVapor)[0:2]
        self.logger.debug("fractions: %s", self.fractions)
        self.logger.debug("pressureDrops: %s", self.pressureDrops)
        self.logger.debug("charges: %s", self.charges)
        self.logger.debug("heat transfers: %s", self.heatTransfers)
        self.qFlux = list(map(lambda x: x["qFlux"], filter(lambda x: x["phaseCold"] == 'TwoPhase', cellList)))
        for fluid in [self.fluidHot, self.fluidCold]:
            props = fluid.fluidApparatiProps[self.type]
            props.tempChangeSupercritical = 1e9
            if ('incomp' in fluid.backEnd.lower()):
                props.entropyOut = fluid.calculateEntropy(ThermoProps.PT, props.pressureIn, props.tempOut)
                continue
            else:
                props.entropyOut = fluid.calculateEntropy(ThermoProps.DT, props.densityOut, props.tempOut)
            #TODO: need to take pressurecritical out if it's not found
            if props.pressureIn > fluid.pressureCritical:
                props.tempChangeApproach = props.tempOut - props.tempIn #approach temperature
            else:
                #Effective subcooling for both streams
                enthalpySatLiquid = fluid.calculateEnthalpy(ThermoProps.QT, 0.0, props.tempBubble)
                specificHeatSatLiquid = fluid.calculateHeatCapacity(ThermoProps.QT, 0.0, props.tempBubble)
                if props.enthalpyOut > enthalpySatLiquid:
                    #Outlet is at some quality on cold side
                    props.tempChangeSupercritical =- (props.enthalpyOut - enthalpySatLiquid)/\
                        specificHeatSatLiquid
                else:
                    props.tempChangeSupercritical = props.tempBubble - props.tempOut
                props.tempChangeApproach = 0 #No approach temp in this case


    def calculateTempsAndDensities(self, fluid: Fluid, pressure):
        """
        Calculates the bubble, dew, and saturation temperatures, as well as the
        saturation densities of the vapor and liquid phases of the given fluid
        based on the given pressure

        Parameters
        ----------
        fluid : Fluid
            fluid whose temperatures and densities are to be calculated.
        pressure : float
            pressure of the fluid.

        Returns
        -------
        tempBubble : float
            bubble temperature of the fluid.
        rhoSatLiquid : float
            liquid phase saturation density of the fluid.
        tempDew : float
            dew temperature of the fluid.
        rhoSatVapor : float
            vapor phase saturation density of the fluid.
        tempSat : float
            saturation temperature of the fluid.

        """
        if 'incomp' in fluid.backEnd.lower() or pressure > fluid.pressureCritical:
            tempBubble = rhoSatLiquid = tempDew = rhoSatVapor = tempSat = None
        else:
            tempBubble = fluid.calculateTemperature(ThermoProps.PQ, pressure, 0.0)
            rhoSatLiquid = fluid.calculateDensity(ThermoProps.PQ, pressure, 0.0)
            tempDew = fluid.calculateTemperature(ThermoProps.PQ, pressure, 1.0)
            rhoSatVapor = fluid.calculateDensity(ThermoProps.PQ, pressure, 1.0)
            tempSat = (tempBubble + tempDew)/2.0
        return tempBubble, rhoSatLiquid, tempDew, rhoSatVapor, tempSat

    def calculateEntropyOfFluid(self, fluid: Fluid, pressure, temperature, density):
        """
        Calculates the entropy of the given fluid based on either pressure and temperature
        or density and temperature

        Parameters
        ----------
        fluid : Fluid
            fluid whose entropy is to be calculated.
        pressure : float
            pressure of the fluid in Pa.
        temperature : float
            temperature of the fluid in K.
        density : float
            density of the fluid in kg/m^3.

        Returns
        -------
        entropy : float
            calculated entropy of the fluid in J/kg/K.

        """
        if 'incomp' in fluid.backEnd.lower() or pressure > fluid.pressureCritical:
            entropy = fluid.calculateEntropy(ThermoProps.PT, pressure, temperature)
        else:
            entropy = fluid.calculateEntropy(ThermoProps.DT, density, temperature)
        return entropy

    def calculateIncrementalHeatTransfer(self, enthalpyListHot, enthalpyListCold, factor):
        """
        Calculates the heat transferred between hot and cold fluids at the points in
        enthalpyListHot and enthalpyListCold

        Parameters
        ----------
        enthalpyListHot : list
            list of enthalpies on the hot side.
        enthalpyListCold : list
            list of enthalpies on the cold side.
        factor : float
            factor to either add to or subtract from cold fluid for comparison to hot fluid.

        Returns
        -------
        qBound : float
            heat transferred between the hot and cold plates.

        """
        self.logger.debug("In calculateIncrementalHeatTransfer")
        self.logger.debug("enthalpyListHot: %s", enthalpyListHot)
        self.logger.debug("enthalpyListCold: %s", enthalpyListCold)
        qBound = []
        for index in range(len(enthalpyListHot) - 1):
            hot = self.massFlowHot*(enthalpyListHot[index+1] - enthalpyListHot[index])
            cold = self.massFlowCold*(enthalpyListCold[index+1] - enthalpyListCold[index])
            if hot < (cold - factor):
                self.logger.debug("Qbound_h<Qbound_c-1e-6")
                qBound.append(hot)
                enthalpyListCold.insert(index+1, enthalpyListCold[index] + hot/self.massFlowCold)
            elif hot > (cold + factor):
                self.logger.debug("Qbound_h>Qbound_c+1e-6")
                qBound.append(cold)
                enthalpyListHot.insert(index+1, enthalpyListHot[index] + cold/self.massFlowHot)
            else:
                qBound.append(hot)
        return qBound

class BrazedPlateHEX(HeatExchanger):
    """
    Brazed Plate Heat Exchanger
    """
    def __init__(self, fluidHot, fluidCold, massFlowHot, massFlowCold, pressureInHot, pressureInCold,
                 enthalpyInHot, enthalpyInCold, conductivity, centerlineDistanceShort: float,
                 centerlineDistanceLong, numPlates: int, thickness: float,
                 volumeChannelSingle: float=None, amplitude: float=None, wavelength: float=None,
                 inclinationAngle: float=np.pi/4, moreChannels: str="Cold", surfaceRoughness=1.0,
                 htpColdTuning=1.0, htpHotTuning=1.0, dpHotTuning=1.0, dpColdTuning=1.0):
        assert moreChannels in ["Hot", "Cold"], "moreChannels must be either 'Hot' or 'Cold'"
        self.type = HEXType.PLATE
        super().__init__(fluidHot, fluidCold, massFlowHot, massFlowCold, pressureInHot, pressureInCold,
                    enthalpyInHot, enthalpyInCold, conductivity, centerlineDistanceLong, surfaceRoughness,
                    htpColdTuning, htpHotTuning, dpHotTuning, dpColdTuning)
        self.centerlineDistShort = centerlineDistanceShort
        self.numPlates = numPlates
        self.thickness = thickness
        self.amplitude = amplitude
        self.wavelength = wavelength
        self.inclinationAngle = inclinationAngle # guessing 45 degrees for H, 15 for L for SWEP
        self.moreChannels = moreChannels
        self.volumeChannelSingle = volumeChannelSingle
        #Using Lockhart Martinelli to calculate pressure drop, Claesson found good agreement with C of 4.67
        self.claessonParamC = 4.67
        self.numGapsHot: float
        self.numGapsCold: float

    def calculate(self):
        """
        Calculates everything

        Returns
        -------
        None.

        """
        self.logger.debug("In calculate")
        self.allocateChannels()
        volumeChannels = {}
        areasFlow = {}
        for temp in ['Hot', 'Cold']:
            areaWetted, diameterHydraulic, volumeChannel, areaFlow = self.singlePhaseGeomCorrelations(temp)
            setattr(self, f"diameterHydraulic{temp}", diameterHydraulic)
            setattr(self, f"areaWetted{temp}", areaWetted)
            volumeChannels[temp] = volumeChannel
            areasFlow[temp] = areaFlow
        self.setUpCalculation(volumeChannels, areasFlow)
        low, high = 0, self.qMax
        try:
            brentq(self.givenQ, low, high, xtol=1e-6*self.qMax)
        except ValueError:
            self.logger.debug(self.givenQ(low), self.givenQ(high))
            raise
        # Collect parameters from all the pieces
        self.postProcess(self.cellList)

    def allocateChannels(self):
        """
        Allocates hot and cold channels according to which fluid is given as the
        side with more plates/channels

        Returns
        -------
        None.

        """
        otherChannel = [ele for ele in ['Hot', 'Cold'] if ele != self.moreChannels][0]
        setattr(self, f"numGaps{self.moreChannels}", (self.numPlates - 1)//2 + 1)
        setattr(self, f"numGaps{otherChannel}", self.numPlates - 1 - \
                getattr(self, f"numGaps{self.moreChannels}"))

    def singlePhaseGeomCorrelations(self, temp):
        r"""
        Based on the single-phase pressure drop and heat transfer correlations
        in VDI Heat Atlas Chapter N6: Pressure Drop and Heat Transfer in Plate Heat
        Exchangers by Holger Martin DOI: 10.1007/978-3-540-77877-6_66 Springer Verlag
        outputs: areaPlane, volumeChannel, areaFlow, diameterHydraulic
            =============================
            ||   __               __    ||
            ||  /  \             /  \   ||
            || |    |           |    |  ||  ===
            ||  \__/             \__/   ||   |
            ||                          ||   |
            ||             | <-  B   -> ||   |
            ||                          ||   |
            ||                          ||   |
            ||                          ||
            ||                          ||
            ||             |\           ||
            ||             | \          ||   effectiveLength
            ||             |  \         ||
            ||             |   \        ||
            ||             | A  \       ||
            ||             |     \      ||   |
            ||                          ||   |
            ||   __               __    ||   |
            ||  /  \             /  \   ||   |
            || |    |           |    |  ||  ===
            ||  \__/             \__/   ||
            ||                          ||
            =============================
                 |----------------|
                centerlineDistShort
             A is inclinationAngle
        """
        areaBetweenPorts = self.centerlineDistShort*self.effectiveLength
        if self.volumeChannelSingle is None:
            xValue = 2*np.pi*self.amplitude/self.wavelength
            phi = 1/6*(1 + np.sqrt(1 + xValue**2) + 4*np.sqrt(1 + xValue**2/2))
            areaWetted = phi*areaBetweenPorts*(self.numPlates - 2)
            diameterHydraulic = 4*self.amplitude/phi
            volumeChannel = areaBetweenPorts*2*self.amplitude*getattr(self, f"numGaps{temp}")
            areaFlow = 2*self.amplitude*self.centerlineDistShort*getattr(self, f"numGaps{temp}")
        else:
            areaWetted = 2*areaBetweenPorts*(self.numPlates - 2)
            diameterHydraulic = 2*areaBetweenPorts/(self.centerlineDistShort + self.effectiveLength)
            volumeChannel = self.volumeChannelSingle*getattr(self, f"numGaps{temp}")
            areaFlow = 2*areaBetweenPorts*getattr(self, f"numGaps{temp}")
        self.logger.debug("diameterHydraulic: %g", diameterHydraulic)
        self.thermalResistanceWall = self.thickness/(self.conductivity*areaWetted)
        return areaWetted, diameterHydraulic, volumeChannel, areaFlow

    def singlePhaseThermoCorrelations(self, fluid, temperature, pressure, massFlowGap, diameterHydraulic):
        """
        Based on the single-phase pressure drop and heat transfer correlations
        in VDI Heat Atlas Chapter N6: Pressure Drop and Heat Transfer in Plate Heat
        Exchangers by Holger Martin DOI: 10.1007/978-3-540-77877-6_66 Springer Verlag
        outputs:    heatTransferCoeff, pressureDrop, reynoldsNum, velocity,
                    conductivity, heatCapacity
        """
        #calculate the thermodynamics and pressure drop
        #Single phase Fluid properties
        self.logger.debug("pressure: %g, temperature: %g", pressure, temperature)
        densityGap = fluid.calculateDensity(ThermoProps.PT, pressure, temperature)
        viscosityGap = fluid.calculateViscosity(ThermoProps.PT, pressure, temperature)
        heatCapacityGap = fluid.calculateHeatCapacity(ThermoProps.PT, pressure, temperature)
        conductivityGap = fluid.calculateConductivity(ThermoProps.PT, pressure, temperature)
        prandtlNumGap = heatCapacityGap*viscosityGap/conductivityGap
        viscosityGapW = viscosityGap #TODO: allow for temperature dependence?
        if not self.amplitude:
            velocityGap = massFlowGap*self.effectiveLength/(densityGap*self.volumeChannelSingle)
        else:
            velocityGap = massFlowGap/(densityGap*2*self.amplitude*self.centerlineDistShort)
        reynoldsNumGap = densityGap*velocityGap*diameterHydraulic/viscosityGap
        #Calculate the friction factor zeta
        if reynoldsNumGap < 2000:
            zeta0 = 64/reynoldsNumGap
            zeta1zero = 597/reynoldsNumGap + 3.85
        else:
            zeta0 = (1.8*np.log(reynoldsNumGap) - 1.5)**(-2)
            zeta1zero = 39/reynoldsNumGap**0.289
        varA = 3.8
        varB = 0.18
        varC = 0.36
        zeta1 = varA*zeta1zero
        #RHS from Equation 18
        rhs = np.cos(self.inclinationAngle)/np.sqrt(varB*np.tan(self.inclinationAngle) +\
                                                    varC*np.sin(self.inclinationAngle) +\
                                                    zeta0/np.cos(self.inclinationAngle)) +\
                                                    (1 - np.cos(self.inclinationAngle))/np.sqrt(zeta1)
        zeta = 1/rhs**2
        #Hagen number
        hagenNum = zeta*reynoldsNumGap**2/2
        #Constants for nusseltNum correlation
        constCQ = 0.122
        constQ = 0.374#q=0.39
        #Nusselt number [-]
        nusseltNum = constCQ*prandtlNumGap**(1/3)*(viscosityGap/viscosityGapW)**(1/6)*\
            (2*hagenNum*np.sin(2*self.inclinationAngle))**(constQ)
        #Heat transfer coefficient [W/m^2-K]
        heatTransferCoeff = nusseltNum*conductivityGap/diameterHydraulic
        #Pressure drop
        pressureDrop = hagenNum*viscosityGap**2*self.effectiveLength/(densityGap*diameterHydraulic**3)
        # There are quite a lot of things that might be useful to have access to
        # in outer functions, so pack up parameters into a dictionary
        self.logger.debug("densityGap: %g", densityGap)
        self.logger.debug("viscosityGap: %g", viscosityGap)
        self.logger.debug("heatCapacityGap: %g", heatCapacityGap)
        self.logger.debug("conductivityGap: %g", conductivityGap)
        self.logger.debug("prandtlNumGap: %g", prandtlNumGap)
        self.logger.debug("viscosityGapW: %g", viscosityGapW)
        self.logger.debug("velocityGap: %g", velocityGap)
        self.logger.debug("reynoldsNumGap: %g", reynoldsNumGap)
        self.logger.debug("zeta0: %g", zeta0)
        self.logger.debug("zeta1zero: %g", zeta1zero)
        self.logger.debug("zeta1: %g", zeta1)
        self.logger.debug("rhs: %g", rhs)
        self.logger.debug("zeta: %g", zeta)
        self.logger.debug("hagenNum: %g", hagenNum)
        self.logger.debug("nusseltNum: %g", nusseltNum)
        self.logger.debug("heatTransferCoeff: %g", heatTransferCoeff)
        self.logger.debug("pressureDrop: %g", pressureDrop)
        outputs = {
             'heatTransferCoeff':heatTransferCoeff,     #Heat transfer coeffcient [W/m^2-K]
             'pressureDrop':pressureDrop,                           #Pressure drop [Pa]
             'reynoldsNum': reynoldsNumGap,             #Reynolds number
             'velocity': velocityGap,                   #Velocity of fluid in channel [m/s]
             'conductivity': conductivityGap,           #Thermal conductivity of fluid [W/m-K]
             'heatCapacity': heatCapacityGap,           #Specific heat of fluid [J/kg-K]
        }
        return outputs

    def _onePhaseHOnePhaseCQimposed(self,inputs):
        self.logger.debug("In _onePhaseHOnePhaseCQimposed")
        #Evaluate heat transfer coefficient for both fluids
        outputsHot = self.HTDP(self.fluidHot, inputs['tempMeanHot'], self.fluidProps["Hot"].pressureIn,
                               self.massFlowHot/self.numGapsHot)
        outputsCold = self.HTDP(self.fluidCold, inputs['tempMeanCold'], self.fluidProps["Cold"].pressureIn,
                                self.massFlowCold/self.numGapsCold)
        inputs.update({"identifier": 'w[1-1]: ',
                       "heatTransferCoeffHot": outputsHot['heatTransferCoeff'],
                       "heatTransferCoeffCold": outputsCold['heatTransferCoeff'],
                       "pressureDropHot": outputsHot['pressureDrop'],
                       "pressureDropCold": outputsCold['pressureDrop']})
        return super()._onePhaseHOnePhaseCQimposed(inputs)

    def _onePhaseHTwoPhaseCQimposed(self,inputs):
        self.logger.debug("In _onePhaseHTwoPhaseCQimposed")
        outputsHot = self.HTDP(self.fluidHot, inputs['tempMeanHot'],self.fluidProps["Hot"].pressureIn,
                               self.massFlowHot/self.numGapsHot)
        inputs.update({"identifier": 'w[3-2]: ',
                       "heatTransferCoeffHot": outputsHot['heatTransferCoeff'],
                       "pressureDropHot": outputsHot['pressureDrop'],
                       "C": self.claessonParamC
                       })
        self.logger.debug("inputs: %s", inputs)
        return super()._onePhaseHTwoPhaseCQimposed(inputs)

    def _twoPhaseHOnePhaseCQimposed(self, inputs):
        self.logger.debug("In _twoPhaseHOnePhaseCQimposed")
        outputsCold = self.HTDP(self.fluidCold, inputs['tempMeanCold'], self.fluidProps["Cold"].pressureIn,
                                self.massFlowCold/self.numGapsCold)
        inputs.update({"identifier": 'w[2-1]: ',
                       "heatTransferCoeffCold": outputsCold['heatTransferCoeff'],
                       "specificHeatCold": outputsCold['heatCapacity'],
                       "pressureDropCold": outputsCold['pressureDrop'],
                       "C": self.claessonParamC
                       })
        return super()._twoPhaseHOnePhaseCQimposed(inputs)

    def _transCritPhaseHTwoPhaseCQimposed(self,inputs):
        #TODO: needs to be reworked
        self.logger.debug("In _transCritPhaseHTwoPhaseCQimposed")
        inputs["identifier"] = 'w[3-1]: '
        return super()._transCritPhaseHTwoPhaseCQimposed(inputs)

    def _transCritPhaseHOnePhaseCQimposed(self, inputs):
        self.logger.debug("In _transCritPhaseHOnePhaseCQimposed")
        outputsCold = self.HTDP(self.fluidCold, inputs['tempMeanCold'], self.fluidProps["Cold"].pressureIn,
                            self.massFlowCold/self.numGapsCold)
        inputs.update({"identifier": 'w[1-2]: ',
                       "heatTransferCoeffCold": outputsCold['heatTransferCoeff'],
                       "specificHeatCold": outputsCold['heatCapacity'],
                       "pressureDropCold": outputsCold['pressureDrop']
            })
        return super()._transCritPhaseHOnePhaseCQimposed(inputs)

    def calculateHeatTransferCoeff(self, temp, qFlux, xIn=None, xOut=None, massFlux=None):
        """
        Using the Cooper pool boiling algorithm, calculates the heat transfer
        coefficient given pressure and qFlux


        Parameters
        ----------
        temp : str
            'Hot' or 'Cold', denotes the stream being acted upon.
        qFlux : float
            heat transferred.
        xIn : float, optional
            vapor fraction input on the cold side. The default is None.
        xOut : TYPE, optional
            vapor fraction input on the hot side. The default is None.
        massFlux : float, optional
            mass. The default is None.

        Returns
        -------
        float
            heat transfer coefficient.

        """
        return Cooper_PoolBoiling(getattr(self, f"fluid{temp}"), self.surfaceRoughness, qFlux, str(self.type))

    def HTDP(self, fluid, temperature, pressure, massFlow):
        """
        This function calls mainly the heat transfer and pressure drop
        for single phase fluids of the plate heat exchanger
        Inputs: temperature [K] and pressure [Pa]
        outputs: h [W/m^2-K] and cp [J/kg-K]
        Note: There are several other output. Check "singlePhaseThermoCorrelations" function
        for more details.


        Parameters
        ----------
        fluid : Fluid
            fluid to perform the calculation on.
        temperature : float
            temperature of fluid.
        pressure : float
            pressure of fluid.
        massFlow : float
            mass flow of fluid.

        Returns
        -------
        outputs
            dict of outputs containing the following:
                'heatTransferCoeff': Heat transfer coeffcient in W/m^2/K
                'pressureDrop': Pressure drop in Pa
                'reynoldsNum':  Reynolds number
                'velocity': Velocity of fluid in channel in m/s
                'conductivity': Thermal conductivity of fluid in W/m/K
                'heatCapacity': Heat capacity of fluid in J/kg/K

        """
        self.logger.debug("In HTDP")
        #using diameterHydraulicHot because both are equal
        return self.singlePhaseThermoCorrelations(fluid, temperature, pressure, massFlow,
                                                     self.diameterHydraulicHot)

class CoaxialHEX(HeatExchanger):
    """
    Coaxial Heat Exchanger
    """
    def __init__(self, fluidHot, fluidCold, massFlowHot, massFlowCold, pressureInHot, pressureInCold,
                 enthalpyInHot, enthalpyInCold, conductivity, innerTubeInnerDiameter: float,
                 innerTubeOuterDiameter: float, outerTubeInnerDiameter: float, length: float,
                 surfaceRoughness=1.0, htpColdTuning=1.0, htpHotTuning=1.0, dpHotTuning=1.0,
                 dpColdTuning=1.0):
        self.type = HEXType.COAXIAL
        super().__init__(fluidHot, fluidCold, massFlowHot, massFlowCold, pressureInHot, pressureInCold,
                    enthalpyInHot, enthalpyInCold, conductivity, length, surfaceRoughness, htpColdTuning,
                    htpHotTuning, dpHotTuning, dpColdTuning)
        self.innerTubeID = innerTubeInnerDiameter
        self.innerTubeOD = innerTubeOuterDiameter
        self.outerTubeID = outerTubeInnerDiameter

    def calculate(self):
        """
        Calculates everything

        Returns
        -------
        None.

        """
        self.logger.debug("In calculate")
        volumeChannels = {}
        areasFlow = {}
        for temp in ['Hot', 'Cold']:
            self.logger.debug("pressure in {temp}: %g", self.fluidProps[temp].pressureIn)
            areaWetted, diameterHydraulic, volumeChannel, areaFlow = self.singlePhaseGeomCorrelations(temp)
            setattr(self, f"diameterHydraulic{temp}", diameterHydraulic)
            setattr(self, f"areaWetted{temp}", areaWetted)
            volumeChannels[temp] = volumeChannel
            areasFlow[temp] = areaFlow
        self.setUpCalculation(volumeChannels, areasFlow)
        low, high = 0, self.qMax
        try:
            brentq(self.givenQ, low, high, xtol=1e-6*self.qMax)
        except ValueError:
            self.logger.debug(self.givenQ(low), self.givenQ(high))
            raise
        # Collect parameters from all the pieces
        self.postProcess(self.cellList)

    def singlePhaseGeomCorrelations(self, temp):
        if temp == 'Hot':
            areaWetted = np.pi*self.innerTubeID*self.effectiveLength
            areaFlow = np.pi*self.innerTubeID**2/4.0
            diameterHydraulic = self.innerTubeID
        else:
            areaWetted = np.pi*self.innerTubeOD*self.effectiveLength
            areaFlow = np.pi*(self.outerTubeID**2 - self.innerTubeOD**2)/4.0
            diameterHydraulic = self.outerTubeID - self.innerTubeOD
        volumeChannel = self.effectiveLength*areaFlow
        self.thermalResistanceWall = np.log(self.innerTubeOD/self.innerTubeID)/\
            (2*np.pi*self.conductivity*self.effectiveLength)
        return areaWetted, diameterHydraulic, volumeChannel, areaFlow


    def HTDP(self, fluid, temperature, pressure, massFlow, side):
        heatCapacity = fluid.calculateHeatCapacity(ThermoProps.PT, pressure, temperature)
        specificVolume = 1/fluid.calculateDensity(ThermoProps.PT, pressure, temperature)
        if side == "Hot":
            frictionFactor, heatTransferCoeff, reynoldsNum = f_h_1phase_Tube(massFlow,
                                            self.innerTubeID, temperature, pressure, fluid)
            diameterHydraulic = self.innerTubeID
        elif side == 'Cold':
            frictionFactor, heatTransferCoeff, reynoldsNum = f_h_1phase_Annulus(massFlow,
                                self.outerTubeID, self.innerTubeOD, temperature, pressure, fluid)
            diameterHydraulic = self.outerTubeID - self.innerTubeOD
        pressureGradient = frictionFactor*specificVolume*self.fluidProps[side].massFluxAverage**2/\
            (2.*diameterHydraulic)
        pressureDrop = pressureGradient*self.effectiveLength
        outputs = {
            'diameterHydraulic': diameterHydraulic,
            'heatTransferCoeff': heatTransferCoeff,
            'pressureDrop': pressureDrop,
            'reynoldsNum': reynoldsNum,
            'heatCapacity': heatCapacity,
        }
        return outputs

    def calculateHeatTransferCoeff(self, temp, qFlux, xIn=None, xOut=None, massFlux=None):
        """
        Using the Kandlikar Evaporation algorithm, calculates the heat transfer
        coefficient of fluid with given temperature (Hot or Cold) given qFlux,
        input and output vapor fraction, and mass flux


        Parameters
        ----------
        temp : str
            'Hot' or 'Cold', denotes the stream being acted upon.
        qFlux : float
            heat transferred.
        xIn : float, optional
            vapor fraction input on the cold side. The default is None.
        xOut : TYPE, optional
            vapor fraction input on the hot side. The default is None.
        massFlux : float, optional
            mass. The default is None.

        Returns
        -------
        float
            heat transfer coefficient.

        """
        return KandlikarEvaporation_average(xIn, xOut, getattr(self, f"fluid{temp}"), massFlux,
                        getattr(self, f"diameterHydraulic{temp}"), qFlux, self.fluidProps[temp].tBubble,
                        self.fluidProps[temp].tempDew)

    def _onePhaseHOnePhaseCQimposed(self,inputs):
        self.logger.debug("In _onePhaseHOnePhaseCQimposed")
        #Evaluate heat transfer coefficient for both fluids
        outputs = {}
        inputs.update({"identifier": 'w[1-1]: '})
        for temp in ['Hot', 'Cold']:
            outputs[temp] = self.HTDP(getattr(self, f"fluid{temp}"), inputs[f'tempMean{temp}'],
                          self.fluidProps[temp].pressureIn, getattr(self, f"massFlow{temp}"), side=temp)
            inputs.update({f"heatTransferCoeff{temp}": outputs[temp]['heatTransferCoeff'],
                           f"pressureDrop{temp}": outputs[temp]['pressureDrop']})
        return super()._onePhaseHOnePhaseCQimposed(inputs)

    def _onePhaseHTwoPhaseCQimposed(self,inputs):
        self.logger.debug("In _onePhaseHTwoPhaseCQimposed")
        outputsHot = self.HTDP(self.fluidHot, inputs['tempMeanHot'],self.fluidProps["Hot"].pressureIn,
                               self.massFlowHot, side="Hot")
        inputs.update({"identifier": 'w[3-2]: ',
                       "heatTransferCoeffHot": outputsHot['heatTransferCoeff'],
                       "pressureDropHot": outputsHot['pressureDrop'],
                       "C": None
                       })
        self.logger.debug("inputs: %s", inputs)
        return super()._onePhaseHTwoPhaseCQimposed(inputs)

    def _twoPhaseHOnePhaseCQimposed(self, inputs):
        self.logger.debug("In _twoPhaseHOnePhaseCQimposed")
        outputsCold = self.HTDP(self.fluidCold, inputs['tempMeanCold'], self.fluidProps["Cold"].pressureIn,
                                self.massFlowCold, side="Cold")
        inputs.update({"identifier": 'w[2-1]: ',
                       "heatTransferCoeffCold": outputsCold['heatTransferCoeff'],
                       "specificHeatCold": outputsCold['heatCapacity'],
                       "pressureDropCold": outputsCold['pressureDrop'],
                       "C": None
                       })
        return super()._twoPhaseHOnePhaseCQimposed(inputs)

    def _transCritPhaseHTwoPhaseCQimposed(self,inputs):
        #TODO: needs to be re-worked
        self.logger.debug("In _transCritPhaseHTwoPhaseCQimposed")
        inputs["identifier"] = 'w[3-1]: '
        return super()._transCritPhaseHTwoPhaseCQimposed(inputs)

    def _transCritPhaseHOnePhaseCQimposed(self, inputs):
        self.logger.debug("In _transCritPhaseHOnePhaseCQimposed")
        outputsCold = self.HTDP(self.fluidCold, inputs['tempMeanCold'], self.fluidProps["Cold"].pressureIn,
                            self.massFlowCold, side="Cold")
        inputs.update({"identifier": 'w[1-2]: ',
                       "heatTransferCoeffCold": outputsCold['heatTransferCoeff'],
                       "specificHeatCold": outputsCold['heatCapacity'],
                       "pressureDropCold": outputsCold['pressureDrop']
            })
        return super()._transCritPhaseHOnePhaseCQimposed(inputs)

if __name__=='__main__':
    logging.basicConfig(filename="ACHPlog.log", level=logging.DEBUG, encoding='utf-8',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

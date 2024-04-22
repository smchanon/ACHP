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
from ACHP.calculations.Correlations import getTempFromPandH, getDensityFromPandH, getPhaseFromPandH,\
    Cooper_PoolBoiling, twoPhaseDensity, lmPressureGradientAvg, calculateAccelerationalPressureDrop,\
    LongoCondensation, PettersonSupercritical, f_h_1phase_Tube, f_h_1phase_Annulus,\
    kandlikarEvaporationAvg, FluidMechanics

class HEXType(StrEnum):
    """
    Heat Exchanger type enum for all available plate types
    """
    PLATE = "Plate-HX"
    COAXIAL = "Coaxial-HX"
    DEFAULT = "Default"

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
        self.type = self.type if hasattr(self, 'type') else HEXType.DEFAULT
        self.logger = logging.getLogger(str(self.type))
        logging.LoggerAdapter(self.logger, {"methodname": "%(funcName)s"})

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
        self.thermalResistanceWall: float
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
        self.phaseInHot: float

        # calculated geometry of cold fluid channel
        self.volumeChannelCold: float
        self.areaFlowCold: float
        self.phaseInCold: float

        # calculated outlet properties
        self.qMax: float
        self.heatTransferred: float
        self.heatFlux: float

        #aggregates for temperatures and phases
        self.thermalFraction: list
        self.pressureDrop: list
        self.charge: list
        self.heatTransferred: list
        self.cellList = []

    def logLocalVars(self, funcName, localDict):
        """
        Logs local variables in methods that call this. Should be moved to a log
        class

        Parameters
        ----------
        funcName : string
            name of method/function whose variables are being logged.
        localDict : dict
            dict of local variables from method/function.

        Returns
        -------
        None.

        """
        #TODO: move to logging file
        del localDict["self"]
        for localKey, localVal in localDict.items():
            if isinstance(localVal, (str, dict, list)):
                self.logger.debug("%s: %s", localKey, localVal, extra={"methodname": funcName})
            elif isinstance(localVal, (float, int)):
                self.logger.debug("%s: %g", localKey, localVal or 0.0, extra={"methodname": funcName})
            else:
                continue

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
            ('Outlet Superheat','K',self.fluidProps["Cold"].get('tempIn') - self.fluidProps["Cold"].get('tempDew')),
            ('Q Total','W',self.heatTransferred),
            ('Charge Total Hot','kg',self.fluidProps["Hot"].get('charge')),
            ('Charge Total Cold','kg',self.fluidProps["Cold"].get('charge')),
            ('Pressure Drop Hot','Pa',self.fluidProps["Hot"].get('pressureDrop')),
            ('Pressure Drop Cold','Pa',self.fluidProps["Cold"].get('pressureDrop')),]
        for temp, phase in product(["Hot", "Cold"], ['Superheated', 'TwoPhase', 'Subcooled',
                                                     'Supercritical', 'Supercrit_liq']):
            outputList.append((f"Q {phase} {temp}", "W", self.heatTransferred[temp][phase]))
            outputList.append((f"Inlet {temp} stream temp", "K", self.fluidProps[temp].get('tempIn')))
            outputList.append((f"Outlet {temp} stream temp", "K", self.fluidProps[temp].get('tempOut')))
            outputList.append((f"Charge {phase} {temp}", "kg", self.charge[temp][phase]))
            outputList.append((f"{temp} Mean HTC {phase}", "W/m^2-K",
                               self.fluidProps[temp].getDict("heatTransferCoeffEffective")[phase]))
            outputList.append((f"Pressure Drop {phase} {temp}", "Pa", self.pressureDrop[temp][phase]))
            outputList.append((f"Area Fraction {phase} {temp}", "-", self.thermalFraction[temp][phase]))
        return outputList

    def setUpCalculation(self, volumeChannels, areasFlow):
        """
        Calculates heat exchanger/fluid shared variables that will be needed for further
        calculations.

        Parameters
        ----------
        volumeChannels : dict
            volumes of the hot and cold channels.
        areasFlow : dict
            areas perpendicular to flow of the hot and cold channels.

        Returns
        -------
        None.

        """
        for temp in ["Hot", "Cold"]:
            tempFluid = getattr(self, f"fluid{temp}")
            tempBubble, densitySatLiquid, tempDew, densitySatVapor, tempSat = self.calculateTempsAndDensities(
                    tempFluid, self.fluidProps[temp].get('pressureIn'))
            self.fluidProps[temp].set('enthalpySatLiquid', self.fluid.calculateEnthalpy(ThermoProps.DT, densitySatLiquid, tempBubble))
            self.fluidProps[temp].set('enthalpySatVapor', self.fluid.calculateEnthalpy(ThermoProps.DT, densitySatVapor, tempDew))
            tempIn = getTempFromPandH(self.fluid, self.fluidProps[temp].get('pressureIn'), self.fluidProps[temp].get('enthalpyIn'), self.type)
            densityIn = getDensityFromPandH(self.fluid, self.fluidProps[temp].get('pressureIn'), self.fluidProps[temp].get('enthalpyIn'), self.type)
            phaseIn = getPhaseFromPandH(self.fluid, self.fluidProps[temp].get('pressureIn'), self.fluidProps[temp].get('enthalpyIn'), self.type)
            # tempIn, densityIn, phaseIn = getTempDensityPhaseFromPandH(tempFluid,
            #                                     self.fluidProps[temp].get('pressureIn'),
            #                                     self.fluidProps[temp].get('enthalpyIn'), tempBubble,
            #                                     tempDew, densitySatLiquid, densitySatVapor)
            entropyIn = self.calculateEntropyOfFluid(tempFluid, self.fluidProps[temp].get('pressureIn'),
                                                     tempIn, densityIn)
            massFluxAverage = getattr(self, f"massFlow{temp}")/areasFlow[temp]
            self.logLocalVars(self.setUpCalculation.__name__, locals())
            for attribute, value in zip(["volumeChannel", "areaFlow"], [volumeChannels[temp], areasFlow[temp]]):
                setattr(self, f"{attribute}{temp}", value)
            attributes = ["tempBubble", "densitySatLiquid", "tempDew", "densitySatVapor", "tempSat",
                          "tempIn", "densityIn", "phaseIn", "entropyIn", "massFluxAverage"]
            values = [tempBubble, densitySatLiquid, tempDew, densitySatVapor, tempSat, tempIn,
                      densityIn, phaseIn, entropyIn, massFluxAverage]
            for attribute, value in zip(attributes, values):
                setattr(self.fluidProps[temp], attribute, value)
            self.logger.debug("conductivity at %s: %s", temp, self.conductivity,
                              extra={"methodname": self.setUpCalculation.__name__})
            self.logger.debug("areaWetted at %s: %s", temp, getattr(self, f"areaWetted{temp}"),
                              extra={"methodname": self.setUpCalculation.__name__})
        self.qMax = self.determineHTBounds()

    def determineHTBounds(self):
        # See if each phase could change phase if it were to reach the
        # inlet temperature of the opposite phase
        assert self.fluidProps["Hot"].get('tempIn') > self.fluidProps["Cold"].get('tempIn'), \
            "Hot phase is colder than cold phase"
        # Find the maximum possible rate of heat transfer as the minimum of
        # taking each stream to the inlet temperature of the other stream
        tempMeltHot = self.fluidHot.getMeltingTemperature(self.fluidProps["Hot"].get('pressureIn'))
        temperatureHot = tempMeltHot if self.fluidProps["Cold"].get('tempIn') < tempMeltHot else self.fluidProps["Cold"].get('tempIn')

        enthalpyOutHot = self.fluidHot.calculateEnthalpy(ThermoProps.PT,
                            self.fluidProps["Hot"].get('pressureIn'), temperatureHot)
        enthalpyOutCold = self.fluidCold.calculateEnthalpy(ThermoProps.PT,
                            self.fluidProps["Cold"].get('pressureIn'), self.fluidProps["Hot"].get('tempIn'))
        qMax = min([self.massFlowCold*(enthalpyOutCold - self.fluidProps["Cold"].get('enthalpyIn')),
                    self.massFlowHot*(self.fluidProps["Hot"].get('enthalpyIn') - enthalpyOutHot)])
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
        for index, (enthalpyHot, enthalpyCold) in enumerate(zip(enthalpyListHot, enthalpyListCold)):
            temperatureListCold[index] = getTempFromPandH(self.fluidCold, self.fluidProps["Cold"].get('pressureIn'), enthalpyCold, self.type)
            temperatureListHot[index] = getTempFromPandH(self.fluidHot, self.fluidProps["Hot"].get('pressureIn'), enthalpyHot, self.type)
            # temperatureListCold[index] = getTempDensityPhaseFromPandH(self.fluidCold,
            #             self.fluidProps["Cold"].get('pressureIn'),
            #             enthalpyCold,
            #             self.fluidProps["Cold"].get('tempBubble'),
            #             self.fluidProps["Cold"].get('tempDew'),
            #             self.fluidProps["Cold"].get('densitySatLiquid'),
            #             self.fluidProps["Cold"].get('densitySatVapor'))[0]
            # temperatureListHot[index] = getTempDensityPhaseFromPandH(self.fluidHot,
            #             self.fluidProps["Hot"].get('pressureIn'),
            #             enthalpyHot,
            #             self.fluidProps["Hot"].get('tempBubble'),
            #             self.fluidProps["Hot"].get('tempDew'),
            #             self.fluidProps["Hot"].get('densitySatLiquid'),
            #             self.fluidProps["Hot"].get('densitySatVapor'))[0]
        #TODO: could do with more generality if both streams can change phase
        qMaxPinch = self.checkPinchPoints(temperatureListHot, temperatureListCold, enthalpyListHot)
        qMax = qMaxPinch or qMax
        self.logLocalVars(self.determineHTBounds.__name__, locals())
        return qMax

    def checkPinchPoints(self, temperatureListHot, temperatureListCold, enthalpyListHot):
        """
        Checks for pinch points

        Parameters
        ----------
        temperatureListHot : list
            list of temperatures on the hot side.
        temperatureListCold : list
            list of temperatures on the cold side.
        enthalpyListHot : list
            list of enthalpies on the hot side.

        Returns
        -------
        float or None
            if there is a pinch point, returns the qMax at that point. If there is none, returns None

        """
        if (temperatureListCold[1:-1] > temperatureListHot[1:-1]).any():
            # Loop over the internal cell boundaries
            for i in range(1,len(temperatureListCold)-1):
                # If cold stream is hotter than the hot stream
                if temperatureListCold[i] - 1e-9 > temperatureListHot[i]:
                    # Find new enthalpy of cold stream at the hot stream cell boundary
                    hPinch = self.fluidCold.calculateEnthalpy(ThermoProps.PT,
                                self.fluidProps["Cold"].get('pressureIn'), temperatureListHot[i])
                    # Find heat transfer of hot stream in right-most cell
                    qExtra = self.massFlowHot*(enthalpyListHot[i+1] - enthalpyListHot[i])
                    return self.massFlowCold*(hPinch - self.fluidProps["Cold"].get('enthalpyIn')) + qExtra
        return None

    def givenQ(self, heat):
        """
        In this function, the heat transfer rate is imposed. Therefore the
        outlet states for both fluids are known, and each element can be solved
        analytically in one shot without any iteration.
        """
        if heat == 0.0:
            return -1
        if heat == self.qMax:
            return np.inf

        enthalpyListCold,enthalpyListHot = self.buildEnthalpyLists(heat)

        wList = []
        self.cellList = []
        qBoundList = self.calculateIncrementalHeatTransfer(enthalpyListHot, enthalpyListCold, 1e-9)
        self.logLocalVars(self.givenQ.__name__, locals())
        for qBound, enthalpyOutHot, enthalpyInCold in zip(qBoundList, enthalpyListHot, enthalpyListCold):
            calcName, calcInputs = self.determineHotAndColdPhases(enthalpyInCold, enthalpyOutHot, qBound)
            outputs = getattr(self, calcName)(calcInputs)
            self.logger.debug("%s%g", outputs["identifier"], outputs['thermalFraction'],
                              extra={"methodname": self.givenQ.__name__})
            wList.append(outputs['thermalFraction'])
            self.cellList.append(outputs)
        self.logger.debug("wlist: %s", wList, extra={"methodname": self.givenQ.__name__})
        self.logger.debug('wsum: %s', np.sum(wList), extra={"methodname": self.givenQ.__name__})
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
        #Start the enthalpy lists with inlet and outlet enthalpies
        #Ordered from lowest to highest enthalpies for both streams
        enthalpyLists = {"Hot": [self.fluidProps["Hot"].get('enthalpyIn') - qGiven/self.massFlowHot,
                                 self.fluidProps["Hot"].get('enthalpyIn')],
                         "Cold": [self.fluidProps["Cold"].get('enthalpyIn'),
                                  self.fluidProps["Cold"].get('enthalpyIn') + qGiven/self.massFlowCold]}
        #Save the value of qGiven and outlet enthalpies
        self.heatTransferred = qGiven
        self.fluidProps["Hot"].set('enthalpyOut', enthalpyLists["Hot"][0])
        self.fluidProps["Cold"].set('enthalpyOut', enthalpyLists["Cold"][1])
        eps = 1e-3
        enthalpySatLiquid = {}
        enthalpySatVapor = {}
        for temp in ["Hot", "Cold"]:
            if 'incomp' in getattr(self, f"fluid{temp}").backEnd.lower() \
                or self.fluidProps[temp].get('pressureIn') > getattr(self, f"fluid{temp}").pressureCritical:
                enthalpySatLiquid[temp] = 1e9
                enthalpySatVapor[temp] = 1e9
            else:
                enthalpySatLiquid[temp] = getattr(self, f"fluid{temp}").calculateEnthalpy(ThermoProps.DT,
                        self.fluidProps[temp].get('densitySatLiquid'), self.fluidProps[temp].get('tempBubble'))
                enthalpySatVapor[temp] = getattr(self, f"fluid{temp}").calculateEnthalpy(ThermoProps.DT,
                        self.fluidProps[temp].get('densitySatVapor'), self.fluidProps[temp].get('tempDew'))
            self.logLocalVars(self.buildEnthalpyLists.__name__, locals())
            # Check whether the enthalpy boundaries are within the bounds set by
            # the imposed amount of heat transfer
            if (enthalpyLists[temp][0] + eps) < enthalpySatVapor[temp] < (enthalpyLists[temp][-1] - eps):
                self.logger.debug("enthalpySatVapor%s between first and last enthalpies", temp,
                                  extra={"methodname": self.buildEnthalpyLists.__name__})
                enthalpyLists[temp].insert(len(enthalpyLists[temp]) - 1, enthalpySatVapor[temp])
            if (enthalpyLists[temp][0] + eps) < enthalpySatLiquid[temp] < (enthalpyLists[temp][-1] - eps):
                self.logger.debug("enthalpySatLiquid%s between first and last enthalpies", temp,
                                  extra={"methodname": self.buildEnthalpyLists.__name__})
                enthalpyLists[temp].insert(1, enthalpySatLiquid[temp])
        self.calculateIncrementalHeatTransfer(enthalpyLists['Hot'], enthalpyLists['Cold'], 1e-6)
        for temp in ["Hot", "Cold"]:
            self.logger.debug("enthalpyLists%s: %s", temp, enthalpyLists[temp],
                              extra={"methodname": self.buildEnthalpyLists.__name__})
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
        phaseHot = getPhaseFromPandH(self.fluidHot, self.fluidProps["Hot"].get('pressureIn'),
                                     (enthalpiesIn["Hot"] + enthalpiesOut["Hot"])/2, self.type)
        phaseCold = getPhaseFromPandH(self.fluidCold, self.fluidProps["Cold"].get('pressureIn'),
                                     (enthalpiesIn["Cold"] + enthalpiesOut["Cold"])/2, self.type)
        tempsIn = {"Hot": getTempFromPandH(self.fluidHot, self.fluidProps["Hot"].get('pressureIn'),
                                enthalpiesIn["Hot"], self.type),
                   "Cold": getTempFromPandH(self.fluidCold, self.fluidProps["Cold"].get('pressureIn'),
                                enthalpiesIn["Cold"], self.type)}
        tempsOut = {"Hot": getTempFromPandH(self.fluidHot, self.fluidProps["Hot"].get('pressureIn'),
                                enthalpiesOut["Hot"], self.type),
                    "Cold": getTempFromPandH(self.fluidCold, self.fluidProps["Cold"].get('pressureIn'),
                                enthalpiesOut["Cold"], self.type)}
        # phaseHot = getPhaseFromPandH(self.fluidHot,
        #                         self.fluidProps["Hot"].get('pressureIn'),
        #                         (enthalpiesIn["Hot"] + enthalpiesOut["Hot"])/2,
        #                         self.fluidProps["Hot"].get('tempBubble'),
        #                         self.fluidProps["Hot"].get('tempDew'),
        #                         self.fluidProps["Hot"].get('densitySatLiquid'),
        # #                         self.fluidProps["Hot"].get('densitySatVapor'))
        # phaseCold = getPhaseFromPandH(self.fluidCold,
        #                         self.fluidProps["Cold"].get('pressureIn'),
        #                         (enthalpiesIn["Cold"] + enthalpiesOut["Cold"])/2,
        #                         self.fluidProps["Cold"].get('tempBubble'),
        #                         self.fluidProps["Cold"].get('tempDew'),
        #                         self.fluidProps["Cold"].get('densitySatLiquid'),
        #                         self.fluidProps["Cold"].get('densitySatVapor'))
        # tempsIn = {"Hot": getTempDensityPhaseFromPandH(self.fluidHot,
        #                         self.fluidProps["Hot"].get('pressureIn'),
        #                         enthalpiesIn["Hot"],
        #                         self.fluidProps["Hot"].get('tempBubble'),
        #                         self.fluidProps["Hot"].get('tempDew'),
        #                         self.fluidProps["Hot"].get('densitySatLiquid'),
        #                         self.fluidProps["Hot"].get('densitySatVapor'))[0],
        #            "Cold": getTempDensityPhaseFromPandH(self.fluidCold,
        #                         self.fluidProps["Cold"].get('pressureIn'),
        #                         enthalpiesIn["Cold"],
        #                         self.fluidProps["Cold"].get('tempBubble'),
        #                         self.fluidProps["Cold"].get('tempDew'),
        #                         self.fluidProps["Cold"].get('densitySatLiquid'),
        #                         self.fluidProps["Cold"].get('densitySatVapor'))[0]}
        # tempsOut = {"Hot": getTempDensityPhaseFromPandH(self.fluidHot,
        #                         self.fluidProps["Hot"].get('pressureIn'),
        #                         enthalpiesOut["Hot"],
        #                         self.fluidProps["Hot"].get('tempBubble'),
        #                         self.fluidProps["Hot"].get('tempDew'),
        #                         self.fluidProps["Hot"].get('densitySatLiquid'),
        #                         self.fluidProps["Hot"].get('densitySatVapor'))[0],
        #             "Cold": getTempDensityPhaseFromPandH(self.fluidCold,
        #                         self.fluidProps["Cold"].get('pressureIn'),
        #                         enthalpiesOut["Cold"],
        #                         self.fluidProps["Cold"].get('tempBubble'),
        #                         self.fluidProps["Cold"].get('tempDew'),
        #                         self.fluidProps["Cold"].get('densitySatLiquid'),
        #                         self.fluidProps["Cold"].get('densitySatVapor'))[0]}
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
                tempDiff = self.fluidProps[temp].get('enthalpySatVapor') - \
                            self.fluidProps[temp].get('enthalpySatLiquid')
                inputs.update({
                    'fractionHigh': min((enthalpiesIn[temp] - \
                            self.fluidProps[temp].get('enthalpySatLiquid'))/tempDiff, 1),
                    'fractionLow': max((enthalpiesOut[temp] - \
                            self.fluidProps[temp].get('enthalpySatLiquid'))/tempDiff, 0)
                    })
        self.logLocalVars(self.determineHotAndColdPhases.__name__, locals())
        if all(x in ["Subcooled", "Superheated"] for x in [phaseHot, phaseCold]):
            # Both are single-phase
            return "_onePhaseHOnePhaseCQimposed", inputs
        if phaseCold == 'TwoPhase':
            inputs['xOutC'] = inputs.pop('fractionLow')
            inputs['xInC'] = inputs.pop('fractionHigh')
            if phaseHot in ['Subcooled','Superheated']:
                # Cold stream is evaporating, and hot stream is single-phase
                return "_onePhaseHTwoPhaseCQimposed", inputs
            if phaseHot in ['Supercritical','Supercrit_liq']:
                # Cold stream is evaporating, and hot stream is transcritical-phase
                return "_transCritPhaseHTwoPhaseCQimposed", inputs
        if phaseCold in ['Subcooled','Superheated']:
            if phaseHot == 'TwoPhase':
                inputs['xInH'] = inputs.pop('fractionHigh')
                inputs['xOutH'] = inputs.pop('fractionLow')
                # Hot stream is condensing, and cold stream is single-phase
                # TODO: bounding state can be saturated state if hot stream is condensing
                return "_twoPhaseHOnePhaseCQimposed", inputs
            if phaseHot in ['Supercritical','Supercrit_liq']:
                # Cold stream is single-phase, and hot stream is transcritical-phase
                inputs.update({
                    'specificHeatCold':(enthalpiesIn["Cold"] - enthalpiesOut["Cold"])/\
                        (tempsIn["Cold"] - tempsOut["Cold"]),
                    'tempOutHot':tempsOut["Hot"]
                    })
                return "_transCritPhaseHOnePhaseCQimposed", inputs
        raise NotImplementedError(f"The case where the cold fluid phase is {phaseCold} and the hot\
                                  fluid phase is {phaseHot} has not been implemented.")

    def _onePhaseHOnePhaseCQimposed(self, inputs):
        """
        Single phase on both sides (hot and cold)
        inputs: dictionary of parameters
        outputs: dictionary of parameters,
        but mainly w, pressure drop and heat transfer coefficient
        This function calculate the fraction of heat exchanger
        that would be required for given thermal duty "w" and DP and h
        """
        self.logger.debug("inputs: %s", inputs,
                          extra={"methodname": self._onePhaseHOnePhaseCQimposed.__name__})
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
        densityHot = self.fluidHot.calculateDensity(ThermoProps.PT,
                                    self.fluidProps["Hot"].get('pressureIn'), inputs['tempMeanHot'])
        chargeHot = thermalFraction*self.volumeChannelHot*densityHot
        densityCold = self.fluidCold.calculateDensity(ThermoProps.PT,
                                    self.fluidProps["Cold"].get('pressureIn'), inputs['tempMeanCold'])
        chargeCold = thermalFraction*self.volumeChannelCold*densityCold
        self.logLocalVars(self._onePhaseHOnePhaseCQimposed.__name__, locals())
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
        This function calculates the fraction of heat exchanger
        that would be required for given thermal duty "w" and DP and h
        """
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
            heatFlux = inputs['heatTransferred']/(thermalFraction*self.areaWettedCold)
            #Heat transfer coefficient from Cooper Pool Boiling with
            #correction for the two-phase zone of the cold side
            heatTransferCoeffTwoPhase = self.htpColdTuning*Cooper_PoolBoiling(self.fluidCold,
                                            self.surfaceRoughness, heatFlux, str(self.type))
            fractionUpdated = self.calculateFraction("Cold", inputs['heatTransferCoeffHot'],
                                                heatTransferCoeffTwoPhase, inputs['specificHeatHot'],
                                                inputs['tempInHot'], inputs['heatTransferred'])
            fractionChange = fractionUpdated - thermalFraction
            thermalFraction = fractionUpdated
            self.logLocalVars(self._onePhaseHTwoPhaseCQimposed.__name__, locals())
        #Refrigerant charge
        densityHot = self.fluidHot.calculateDensity(ThermoProps.PT, self.fluidProps["Hot"].get('pressureIn'),
                                                    inputs['tempMeanHot'])
        chargeHot = thermalFraction*self.volumeChannelHot*densityHot
        densityCold = twoPhaseDensity(self.fluidCold, inputs['xInC'], inputs['xOutC'],
                        self.fluidProps["Cold"].get('tempDew'), self.fluidProps["Cold"].get('tempBubble'))
        chargeCold = densityCold*thermalFraction*self.volumeChannelCold
        pressureDropFriction = lmPressureGradientAvg(inputs['xInC'], inputs['xOutC'], self.fluidCold,
                        self.massFlowCold/self.areaFlowCold, self.diameterHydraulicCold,
                        self.fluidProps["Cold"].get('tempBubble'), self.fluidProps["Cold"].get('tempDew'),
                        coeff=inputs['C'])*thermalFraction*self.effectiveLength
        #Accelerational pressure drop component
        pressureDropAcceleration = calculateAccelerationalPressureDrop(inputs['xInC'], inputs['xOutC'],
                        self.fluidCold, self.massFlowCold/self.areaFlowCold,
                        self.fluidProps["Cold"].get('tempBubble'),  self.fluidProps["Cold"].get('tempDew'))*\
                        thermalFraction*self.effectiveLength
        outputs = {
            'thermalFraction': thermalFraction,
            'tempOutHot': inputs['tempInHot'] - inputs['heatTransferred']/\
                (self.massFlowCold*inputs['specificHeatHot']),
            'tempOutCold': inputs['tempInCold'] + inputs['heatTransferred']/\
                (self.massFlowCold*inputs['specificHeatCold']),
            'chargeCold': chargeCold,
            'chargeHot': chargeHot,
            'pressureDropHot': -inputs['pressureDropHot'],
            'pressureDropCold': pressureDropFriction + pressureDropAcceleration,
            'heatFlux': heatFlux,
            'heatTransferCoeffCold': heatTransferCoeffTwoPhase
        }
        self.logLocalVars(self._onePhaseHTwoPhaseCQimposed.__name__, locals())
        return outputs

    def _twoPhaseHOnePhaseCQimposed(self,inputs):
        """
        Hot stream is condensing (two phase), cold stream is single phase
        inputs: dictionary of parameters
        outputs: dictionary of parameters, but mainly thermalFraction, pressure drop
            and heat transfer coefficient
        This function calculates the fraction of heat exchanger
        that would be required for given thermal duty "thermalFraction" and pressure drop
        and heat transfer coefficient
        """
        heatTransferCoeffTwoPhase = LongoCondensation((inputs['xOutH'] + inputs['xInH'])/2,
                                        self.massFlowCold/self.areaFlowHot, self.diameterHydraulicHot,
                                        self.fluidHot, self.fluidProps["Hot"].get('tempBubble'),
                                        self.fluidProps["Hot"].get('tempDew'))*self.htpHotTuning
        thermalFraction = self.calculateFraction("Hot", inputs['heatTranferCoeffCold'],
                                                heatTransferCoeffTwoPhase, inputs['specificHeatCold'],
                                                inputs['tempInCold'], inputs['heatTransferred'])
        #TODO: these can be refactored with onephase_twoPhase calculations
        densityCold = self.fluidCold.calculateDensity(ThermoProps.PT,
                                        self.fluidProps["Cold"].get('pressureIn'), inputs['tempMeanCold'])
        chargeCold = thermalFraction*self.volumeChannelCold*densityCold
        densityHot = twoPhaseDensity(self.fluidHot,inputs['xOutH'],inputs['xInH'],
                        self.fluidProps["Hot"].get('tempDew'), self.fluidProps["Hot"].get('tempBubble'),
                        slipModel='Zivi')
        chargeHot = densityHot*thermalFraction*self.volumeChannelHot
        pressureDropFriction = lmPressureGradientAvg(inputs['xOutH'], inputs['xInH'], self.fluidHot,
                        self.massFlowCold/self.areaFlowHot, self.diameterHydraulicHot,
                        self.fluidProps["Hot"].get('tempBubble'), self.fluidProps["Hot"].get('tempDew'),
                        coeff=inputs['C'])*thermalFraction*self.effectiveLength
        #Accelerational pressure drop component
        pressureDropAcceleration = -calculateAccelerationalPressureDrop(inputs['xOutH'], inputs['xInH'],
                            self.fluidHot, self.massFlowCold/self.areaFlowHot,
                            self.fluidProps["Hot"].get('tempBubble'), self.fluidProps["Hot"].get('tempDew'),
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
        """
        Calculates the thermal fraction of the given two-phase side

        Parameters
        ----------
        twoPhaseSide : string
            Side of heat exchanger that has two phases. Can be "hot" or "cold".
        heatTransferCoeff1Phase : float
            Heat transfer coefficient of the fluid on the one-phase side.
        heatTransferCoeff2Phase : float
            Heat transfer coefficient of the fluid on the two-phase side.
        specificHeatSinglePhase : float
            Specific heat of the fluid on the one-phase side.
        tempInSinglePhase : float
            Inlet temperature of the fluid on the one-phase side.
        heatTransferred : float
            Heat transferred between the two sides.

        Returns
        -------
        thermalFraction
            Thermal fraction of two-phase side.

        """
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
        self.logLocalVars(self.calculateFraction.__name__, locals())
        return conductanceRequired/conductanceTotal

    def _transCritPhaseHOnePhaseCQimposed(self, inputs):
        """
        The hot stream is Transcritical phase (supercritical or supercrit_liq), and the cold stream
        is single phase (subcooled or superheated)
        inputs: dictionary of parameters
        outputs: dictionary of parameters, but mainly thermal fraction, pressure drop and heat
            transfer coefficient
        This function calculates the fraction of heat exchanger that would be required for given
        thermal duty "thermalFraction" and pressure drop and heat transfer coefficient
        """
        #cold-side heat resistance
        conductanceCold = 1/(inputs['heatTransferCoeffCold']*self.areaWettedCold)
        # if inputs['heatTransferred'] > inputs['tempMeanHot']:
        #     tempWall = (inputs['tempMeanHot'] + inputs['tempMeanCold'])/2
        # else:
            # #wall temperature calculate from energy balance on the cold-side
            # tempWall = (self.thermalResistanceWall + conductanceCold)*inputs['heatTransferred'] + \
            #     inputs['tempMeanCold'] #This is just an initial wall temperature
        tempWall = (self.thermalResistanceWall + conductanceCold)*inputs['heatTransferred'] + \
             inputs['tempMeanCold'] #This is just an initial wall temperature
        fractionChange = 999
        thermalFraction = 1
        while abs(fractionChange) > 1e-6:
            #heat flux
            heatFlux = inputs['heatTransferred']/(thermalFraction*self.areaWettedHot)
            heatTransferCoeffHot, frictionHot, specificHeatHot, densityHot = PettersonSupercritical(
                                inputs['tempMeanHot'], tempWall, self.fluidHot,
                                self.fluidProps["Hot"].get('massFluxAverage'), self.diameterHydraulicHot,
                                self.diameterHydraulicHot/self.effectiveLength,
                                self.fluidProps["Hot"].get('pressureIn'), heatFlux)
            heatTransferCoeffHot *= self.hrHotTuning
            #Update wall temperature for the next iteration
            conductanceHot = 1/(heatTransferCoeffHot*self.areaWettedHot) #hot-side heat resistance
            tempWall = tempWall + conductanceHot*inputs['heatTransferred']
            conductanceTotal = 1/(conductanceHot + conductanceCold + (self.thickness or 1)/self.thermalResistanceWall)
            #Get Ntu [-]
            capacitance = [inputs['specificHeatCold']*self.massFlowCold,specificHeatHot*self.massFlowCold]
            capacitanceMin = min(capacitance)
            capacitanceRatio = capacitanceMin/max(capacitance)
            #Effectiveness [-]
            qMax = capacitanceMin*(inputs['tempInHot'] - inputs['tempInCold'])
            epsilon = inputs['heatTransferred']/qMax if inputs['heatTransferred']/qMax < 1.0 else 1.0-1e-12
            #Pure counterflow with capacitanceRatio<1 (Incropera Table 11.4)
            ntu = 1/(capacitanceRatio - 1)*np.log((epsilon - 1)/(epsilon*capacitanceRatio - 1))
            #Required UA value
            conductanceRequired = capacitanceMin*ntu
            fractionChange = conductanceRequired/conductanceTotal - thermalFraction
            thermalFraction = conductanceRequired/conductanceTotal
            self.logLocalVars(self._transCritPhaseHOnePhaseCQimposed.__name__, locals())
        #Determine both charge components
        chargeHot = thermalFraction*self.volumeChannelHot*densityHot
        densityCold = self.fluidCold.calculateDensity(ThermoProps.PT,
                                self.fluidProps["Cold"].get('pressureIn'), inputs['tempMeanCold'])
        chargeCold = thermalFraction*self.volumeChannelCold*densityCold
        #Hot-side Pressure gradient using Darcy friction factor
        volumeSpecificHot = 1.0/densityHot
        pressureGradientHot = -frictionHot*volumeSpecificHot*self.fluidProps["Hot"].get('massFluxAverage')**2/\
            (2*self.diameterHydraulicHot)
        pressureDropFrictionHot = pressureGradientHot*self.effectiveLength*thermalFraction
        outputs = {**inputs, **{
            'thermalFraction': thermalFraction,
            'tempOutHot': inputs['tempInHot'] - inputs['heatTransferred']/\
                (self.massFlowCold*inputs['specificHeatHot']),
            'tempOutCold': inputs['tempInCold'] + inputs['heatTransferred']/\
                (self.massFlowCold*inputs['specificHeatCold']),
            'chargeCold': chargeCold,
            'chargeHot': chargeHot,
            'pressureDropHot': pressureDropFrictionHot,
            'pressureDropCold': -inputs['pressureDropCold'],
            'heatTransferCoeffHot': heatTransferCoeffHot,
            'heatFlux':heatFlux,
            'specificHeatHot':specificHeatHot,

        }}
        return outputs

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
        #Reduced pressure for Cooper Correlation
        fractionChange = 999
        thermalFraction = 1
        while abs(fractionChange) > 1e-6:
            heatFlux = inputs['heatTransferred']/(thermalFraction*self.areaWettedCold)
            #Heat transfer coefficient from Cooper Pool Boiling with
            #correction for the two-phase zone of the cold side
            #TODO: This will not work for coaxial
            heatTransferCoeffTwoPhase = self.calculateHeatTransferCoeff('Cold', heatFlux)*self.htpColdTuning
            #cold-side heat resistance
            resistanceCold = 1/(heatTransferCoeffTwoPhase*self.areaWettedCold)
            #wall temperature calculated from energy balance on the cold-side
            tempWall = (self.thermalResistanceWall + resistanceCold)*inputs['heatTransferred'] + \
                self.fluidProps["Cold"].get('tempSat')
            #Calculate HTC for the hot Transcritical-phase fluid
            #HTC and friction calculated using Pettersson (2000) correlations
            enthalpyHot, frictionHot, specificHeatHot, densityHot = PettersonSupercritical(
                inputs['tempMeanHot'], tempWall, self.fluidHot, self.fluidProps["Hot"].get('massFluxAverage'),
                self.diameterHydraulicHot, self.diameterHydraulicHot/self.effectiveLength,
                self.fluidProps["Hot"].get('pressureIn'), heatFlux)
            enthalpyHot = self.hrHotTuning*enthalpyHot #correct HTC for hot-side
            #Evaluate UA [W/K]
            conductanceTotal = 1/(1/(enthalpyHot*self.areaWettedHot) + \
                                1/(heatTransferCoeffTwoPhase*self.areaWettedCold)+self.thermalResistanceWall)
            #cp of cold-side (two-phase) is very large compared to hot-side (trans-phase).
            #Therefore, capacitanceMin is on hot-side
            capacitanceMin = specificHeatHot*self.massFlowCold
            #Effectiveness [-]
            qMax = capacitanceMin*(inputs['tempInHot'] - self.fluidProps["Cold"].get('tempSat'))
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
                    self.fluidProps["Cold"].get('tempDew'), self.fluidProps["Cold"].get('tempBubble'),
                    slipModel='Zivi')
        chargeCold = densityCold*thermalFraction*self.volumeChannelCold

        #Hot-side Pressure gradient using Darcy friction factor
        volumeSpecificHot = 1.0/densityHot
        pressureGradientHot = -frictionHot*volumeSpecificHot*self.fluidProps["Hot"].get('massFluxAverage')**2/\
                                    (2*self.diameterHydraulicHot)
        pressureDropFrictionHot=pressureGradientHot*self.effectiveLength*thermalFraction

        pressureDropFrictionCold = lmPressureGradientAvg(inputs['xInC'], inputs['xOutC'],
                    self.fluidCold, self.massFlowCold/self.areaFlowCold,
                    self.diameterHydraulicCold, self.fluidProps["Cold"].get('tempBubble'),
                    self.fluidProps["Cold"].get('tempDew'), coeff=4.67)*thermalFraction*self.effectiveLength
        #Accelerational pressure drop component
        pressureDropAccelerationCold = calculateAccelerationalPressureDrop(inputs['xInC'], inputs['xOutC'],
                    self.fluidCold, self.massFlowCold/self.areaFlowCold,
                    self.fluidProps["Cold"].get('tempBubble'), self.fluidProps["Cold"].get('tempDew'),
                    slipModel='Zivi')*thermalFraction*self.effectiveLength
        outputs = {**inputs, **{
            'thermalFraction': thermalFraction,
            'tempOutHot': inputs['tempInHot']-inputs['heatTransferred']/(self.massFlowCold*specificHeatHot),
            'chargeCold': chargeCold,
            'chargeHot': chargeHot,
            'pressureDropHot': pressureDropFrictionHot,
            'pressureDropCold': pressureDropFrictionCold+pressureDropAccelerationCold,
            'heatTransferCoeffCold': heatTransferCoeffTwoPhase,
            'heatFlux': heatFlux,
            'specificHeatHot': specificHeatHot,
        }}
        return outputs

    def calculateHeatTransferCoeff(self, temp, heatFlux, xIn=None, xOut=None, massFlux=None):
        """
        Calculates the heat transfer coefficient of a fluid. Should be overridden by child classes

        Parameters
        ----------
        temp : str
            'Hot' or 'Cold', denotes the stream being acted upon.
        heatFlux : float
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
        self.logger.debug("cellList: %s", cellList, extra={"methodname": self.postProcess.__name__})
        aggregates = {'pressureDropCold': 0, 'pressureDropHot': 0, 'chargeCold': 0,
                      'chargeHot': 0}
        for cell in cellList:
            for varName in aggregates:
                aggregates[varName] += cell[varName]
        phaseProperties = ["thermalFraction", "pressureDrop", "charge", "heatTransferred"]
        for phaseProperty in phaseProperties:
            setattr(self, phaseProperty, {'Hot':{}, 'Cold':{}})
        for temp, phase in product(['Hot', 'Cold'],
                        ['Superheated', 'TwoPhase', 'Subcooled', 'Supercritical', 'Supercrit_liq']):
            phaseFilter = list(filter(lambda x, t=temp, p=phase: x[f"phase{t}"] == p, cellList))
            if phaseFilter:
                self.logger.debug("phaseFilter %s %s before getting fraction: %s", temp, phase, phaseFilter,
                                  extra={"methodname": self.postProcess.__name__})
                for phaseProperty in phaseProperties:
                    setattr(self, f"{phaseProperty}[{temp}][{phase}]", sum(map(lambda x, t=temp,
                        p=phaseProperty: x[f"{p}{t}"] if f"{p}{t}" in x.keys() else x[f"{p}"], phaseFilter)))
                self.fluidProps[temp].set('pressureDrop',sum(self.pressureDrop[temp].values())*\
                                    getattr(self, f"dp{temp}Tuning"))
                # self.charge[temp][phase] = sum(map(lambda x, t=temp: x[f'charge{t}'],phaseFilter))
                self.fluidProps[temp].set('charge',sum(self.charge[temp].values()))
                # self.heatTransferred[temp][phase] = sum(map(lambda x: x['heatTransferred'], phaseFilter))
                thermalFraction = list(map(lambda x: x['thermalFraction'], phaseFilter))
                heatTransferCoeff = list(map(lambda x, t=temp: x[f'heatTransferCoeff{t}'], phaseFilter))
                self.logger.debug("thermal fraction: %s", thermalFraction,
                                  extra={"methodname": self.postProcess.__name__})
                self.logger.debug("heatTransferCoeff: %s", heatTransferCoeff,
                                  extra={"methodname": self.postProcess.__name__})
                self.fluidProps[temp].addToProperty("heatTransferCoeffEffective", phase, float(
                    sum(np.array(thermalFraction)*np.array(heatTransferCoeff))/\
                    sum(thermalFraction)) if len(thermalFraction) > 0 else 0.0)
        for temp in ['Hot', 'Cold']:
            tempOut, densityOut = getTempDensityPhaseFromPandH(getattr(self, f"fluid{temp}"),
                self.fluidProps[temp].get('pressureIn'), self.fluidProps[temp].get('enthalpyOut'),
                self.fluidProps[temp].get('tempBubble'), self.fluidProps[temp].get('tempDew'),
                self.fluidProps[temp].get('densitySatLiquid'),
                self.fluidProps[temp].get('densitySatVapor'))[0:2]
            self.fluidProps[temp].set('tempOut', tempOut)
            self.fluidProps[temp].set('densityOut', densityOut)
        self.logger.debug("thermalFraction: %s", self.thermalFraction,
                          extra={"methodname": self.postProcess.__name__})
        self.logger.debug("pressureDrop: %s", self.pressureDrop,
                          extra={"methodname": self.postProcess.__name__})
        self.logger.debug("charge: %s", self.charge, extra={"methodname": self.postProcess.__name__})
        self.logger.debug("heat transfers: %s", self.heatTransferred,
                          extra={"methodname": self.postProcess.__name__})
        self.heatFlux = list(map(lambda x: x["heatFlux"], filter(lambda x: x["phaseCold"] == 'TwoPhase',
                                                                 cellList)))
        for fluid in [self.fluidHot, self.fluidCold]:
            props = fluid.fluidApparatiProps[self.type]
            props.tempChangeSupercritical = 1e9
            if 'incomp' in fluid.backEnd.lower():
                props.entropyOut = fluid.calculateEntropy(ThermoProps.PT, props.pressureIn, props.tempOut)
                continue
            props.entropyOut = fluid.calculateEntropy(ThermoProps.DT, props.densityOut, props.tempOut)
            #TODO: need to take pressureCritical out if it's not found
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
        self.logger.debug("enthalpyListHot: %s", enthalpyListHot,
                          extra={"methodname": self.calculateIncrementalHeatTransfer.__name__})
        self.logger.debug("enthalpyListCold: %s", enthalpyListCold,
                          extra={"methodname": self.calculateIncrementalHeatTransfer.__name__})
        qBound = []
        for index in range(len(enthalpyListHot) - 1):
            hot = self.massFlowHot*(enthalpyListHot[index+1] - enthalpyListHot[index])
            cold = self.massFlowCold*(enthalpyListCold[index+1] - enthalpyListCold[index])
            if hot < (cold - factor):
                self.logger.debug("Qbound_h<Qbound_c-1e-6",
                                  extra={"methodname": self.calculateIncrementalHeatTransfer.__name__})
                qBound.append(hot)
                enthalpyListCold.insert(index+1, enthalpyListCold[index] + hot/self.massFlowCold)
            elif hot > (cold + factor):
                self.logger.debug("Qbound_h>Qbound_c+1e-6",
                                  extra={"methodname": self.calculateIncrementalHeatTransfer.__name__})
                qBound.append(cold)
                enthalpyListHot.insert(index+1, enthalpyListHot[index] + cold/self.massFlowHot)
            else:
                qBound.append(hot)
        return qBound

class BrazedPlateHEX(HeatExchanger):
    r"""
    Brazed Plate Heat Exchanger

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
        self.allocateChannels()
        volumeChannels = {}
        areasFlow = {}
        for temp in ['Hot', 'Cold']:
            areaBetweenPorts = self._calculateAreaBetweenPorts()
            volumeChannels[temp] = self.calculateChannelVolume(temp, areaBetweenPorts)
            setattr(self, f"diameterHydraulic{temp}", self.calculateHydraulicDiameter(volumeChannels[temp],
                                                                                      areaBetweenPorts))
            setattr(self, f"areaWetted{temp}", self.calculateAreaWetted(areaBetweenPorts))
            areasFlow[temp] = self.calculateAreaFlow(temp, areaBetweenPorts)
            self.thermalResistanceWall = self.thickness/(self.conductivity*getattr(self, f"areaWetted{temp}"))
        self.logger.debug("thermalResistanceWall: %g", self.thermalResistanceWall,
                      extra={"methodname": self.calculate.__name__})
        self.setUpCalculation(volumeChannels, areasFlow)
        low, high = 0, self.qMax
        try:
            brentq(self.givenQ, low, high, xtol=1e-6*self.qMax)
        except ValueError:
            self.logger.error(self.givenQ(low), self.givenQ(high),
                              extra={"methodname": self.calculate.__name__})
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
        setattr(self, f"numGaps{self.moreChannels}", np.ceil((self.numPlates - 1)/2))
        setattr(self, f"numGaps{otherChannel}", np.floor((self.numPlates - 1)/2))

    def _calculateAreaBetweenPorts(self):
        """
        Calculates the area between inlet and outlet ports

        Returns
        -------
        float
            area between ports.

        """
        return self.centerlineDistShort*self.effectiveLength

    def _calculatePhi(self):
        """
        Calculates phi value.

        Returns
        -------
        float
            phi.

        """
        xValue = 2*np.pi*self.amplitude/self.wavelength
        return 1/6*(1 + np.sqrt(1 + xValue**2) + 4*np.sqrt(1 + xValue**2/2))

    def calculateAreaWetted(self, areaBetweenPorts):
        """
        Calculates the wetted surface area.

        Parameters
        ----------
        areaBetweenPorts : float
            area between inlet and outlet ports.

        Returns
        -------
        float
            wetted surface area.

        """
        if self.volumeChannelSingle is None:
            return self._calculatePhi()*areaBetweenPorts*(self.numPlates - 2)
        return 2*areaBetweenPorts*(self.numPlates - 2)

    def calculateChannelVolume(self, temp, areaBetweenPorts):
        """
        Calculates the channel volume of the "Hot" or "Cold" side, designated by
        the temp variable.

        Parameters
        ----------
        temp : string
            "Hot" or "Cold" channel.
        areaBetweenPorts : float
            area between inlet and outlet ports.

        Returns
        -------
        float
            volume of entire channel on the side described by temp.

        """
        if self.volumeChannelSingle is None:
            return areaBetweenPorts*2*self.amplitude*getattr(self, f"numGaps{temp}")
        return self.volumeChannelSingle*getattr(self, f"numGaps{temp}")

    def calculateHydraulicDiameter(self, volumeChannel, areaBetweenPorts):
        """
        Calculates the hydraulic diameter of the channel.

        Parameters
        ----------
        volumeChannel : float
            volume of the channel.
        areaBetweenPorts : float
            area between inlet and outlet ports.

        Returns
        -------
        float
            hydraulic diameter.

        """
        if self.volumeChannelSingle is None:
            return 4*self.amplitude/self._calculatePhi()
        return 4*volumeChannel/areaBetweenPorts

    def calculateAreaFlow(self, temp, areaBetweenPorts):
        """
        Calculates the flow area

        Parameters
        ----------
        temp : string
            "Hot" or "Cold" channel.
        areaBetweenPorts : float
            area between inlet and outlet ports.

        Returns
        -------
        float
            area perpendicular to flow in given channel.

        """
        if self.volumeChannelSingle is None:
            return 2*self.amplitude*self.centerlineDistShort*getattr(self, f"numGaps{temp}")
        return 2*areaBetweenPorts*getattr(self, f"numGaps{temp}")



    def singlePhaseThermoCorrelations(self, fluid, temperature, pressure, massFlow, diameterHydraulic):
        """
        Calculates the pressure drop and heat transfer correlations
        Based on the single-phase pressure drop and heat transfer correlations
        in VDI Heat Atlas Chapter N6: Pressure Drop and Heat Transfer in Plate Heat
        Exchangers by Holger Martin DOI: 10.1007/978-3-540-77877-6_66 Springer Verlag
        outputs:    heatTransferCoeff, pressureDrop, reynoldsNum, velocity,
                    conductivity, heatCapacity
        """
        self.logger.debug("pressure: %g, temperature: %g", pressure, temperature,
                          extra={"methodname": self.singlePhaseThermoCorrelations.__name__})
        density = fluid.calculateDensity(ThermoProps.PT, pressure, temperature)
        viscosity = fluid.calculateViscosity(ThermoProps.PT, pressure, temperature)
        heatCapacity = fluid.calculateHeatCapacity(ThermoProps.PT, pressure, temperature)
        conductivity = fluid.calculateConductivity(ThermoProps.PT, pressure, temperature)
        viscosityW = viscosity #TODO: allow for temperature dependence?
        if not self.amplitude:
            velocity = massFlow*self.effectiveLength/(density*self.volumeChannelSingle)
        else:
            velocity = massFlow/(2*density*self.amplitude*self.centerlineDistShort)
        reynoldsNum = FluidMechanics.calculateReynoldsNumber(velocity, diameterHydraulic,
                                                                viscosity, density)
        hagenNum = FluidMechanics.calculateHagenNumber(reynoldsNum, self.inclinationAngle)
        prandtlNum = FluidMechanics.calculatePrandtlNumber(heatCapacity, viscosity, conductivity)
        #Constants for nusseltNum correlation
        constCQ = 0.122
        constQ = 0.374 #q=0.39
        #Nusselt number [-]
        nusseltNum = constCQ*prandtlNum**(1/3)*(viscosity/viscosityW)**(1/6)*\
            (2*hagenNum*np.sin(2*self.inclinationAngle))**(constQ)
        #Heat transfer coefficient [W/m^2-K]
        heatTransferCoeff = nusseltNum*conductivity/diameterHydraulic
        #Pressure drop
        pressureDrop = hagenNum*viscosity**2*self.effectiveLength/(density*diameterHydraulic**3)
        self.logLocalVars(self.singlePhaseThermoCorrelations.__name__, locals())
        outputs = {
             'heatTransferCoeff': heatTransferCoeff,    #[W/m^2-K]
             'pressureDrop': pressureDrop,              #[Pa]
             'heatCapacity': heatCapacity,              #[J/kg-K]
        }
        return outputs

    def _onePhaseHOnePhaseCQimposed(self,inputs):
        outputsHot = self.htdp(self.fluidHot, inputs['tempMeanHot'],
                               self.fluidProps["Hot"].get('pressureIn'),
                               self.massFlowHot/self.numGapsHot)
        outputsCold = self.htdp(self.fluidCold, inputs['tempMeanCold'],
                                self.fluidProps["Cold"].get('pressureIn'),
                                self.massFlowCold/self.numGapsCold)
        inputs.update({"identifier": 'w[1-1]: ',
                       "heatTransferCoeffHot": outputsHot['heatTransferCoeff'],
                       "heatTransferCoeffCold": outputsCold['heatTransferCoeff'],
                       "pressureDropHot": outputsHot['pressureDrop'],
                       "pressureDropCold": outputsCold['pressureDrop']})
        return {**inputs, **super()._onePhaseHOnePhaseCQimposed(inputs)}

    def _onePhaseHTwoPhaseCQimposed(self,inputs):
        outputsHot = self.htdp(self.fluidHot, inputs['tempMeanHot'],
                               self.fluidProps["Hot"].get('pressureIn'),
                               self.massFlowHot/self.numGapsHot)
        inputs.update({"identifier": 'w[3-2]: ',
                       "heatTransferCoeffHot": outputsHot['heatTransferCoeff'],
                       "pressureDropHot": outputsHot['pressureDrop'],
                       "C": self.claessonParamC
                       })
        return {**inputs, **super()._onePhaseHTwoPhaseCQimposed(inputs)}

    def _twoPhaseHOnePhaseCQimposed(self, inputs):
        outputsCold = self.htdp(self.fluidCold, inputs['tempMeanCold'],
                                self.fluidProps["Cold"].get('pressureIn'),
                                self.massFlowCold/self.numGapsCold)
        inputs.update({"identifier": 'w[2-1]: ',
                       "heatTransferCoeffCold": outputsCold['heatTransferCoeff'],
                       "specificHeatCold": outputsCold['heatCapacity'],
                       "pressureDropCold": outputsCold['pressureDrop'],
                       "C": self.claessonParamC
                       })
        return {**inputs, **super()._twoPhaseHOnePhaseCQimposed(inputs)}

    def _transCritPhaseHOnePhaseCQimposed(self, inputs):
        outputsCold = self.htdp(self.fluidCold, inputs['tempMeanCold'],
                                self.fluidProps["Cold"].get('pressureIn'),
                            self.massFlowCold/self.numGapsCold)
        inputs.update({"identifier": 'w[1-2]: ',
                       "heatTransferCoeffCold": outputsCold['heatTransferCoeff'],
                       "specificHeatCold": outputsCold['heatCapacity'],
                       "pressureDropCold": outputsCold['pressureDrop']
            })
        return {**inputs, **super()._transCritPhaseHOnePhaseCQimposed(inputs)}

    def _transCritPhaseHTwoPhaseCQimposed(self,inputs):
        #TODO: needs to be reworked
        inputs["identifier"] = 'w[3-1]: '
        return {**inputs, **super()._transCritPhaseHTwoPhaseCQimposed(inputs)}

    def calculateHeatTransferCoeff(self, temp, heatFlux, xIn=None, xOut=None, massFlux=None):
        """
        Using the Cooper pool boiling algorithm, calculates the heat transfer
        coefficient given pressure and heatFlux


        Parameters
        ----------
        temp : str
            'Hot' or 'Cold', denotes the stream being acted upon.
        heatFlux : float
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
        return Cooper_PoolBoiling(getattr(self, f"fluid{temp}"), self.surfaceRoughness, heatFlux,
                                  str(self.type))

    def htdp(self, fluid, temperature, pressure, massFlow):
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
        volumeChannels = {}
        areasFlow = {}
        for temp in ['Hot', 'Cold']:
            self.logger.debug("pressure in {temp}: %g", self.fluidProps[temp].get('pressureIn'))
            setattr(self, f"diameterHydraulic{temp}", self.calculateHydraulicDiameter(temp))
            setattr(self, f"areaWetted{temp}", self.calculateAreaWetted(temp))
            areasFlow[temp] = self.calculateAreaFlow(temp)
            volumeChannels[temp] = self.calculateChannelVolume(areasFlow[temp])
        self.setThermalResistanceWall()
        self.setUpCalculation(volumeChannels, areasFlow)
        low, high = 0, self.qMax
        try:
            brentq(self.givenQ, low, high, xtol=1e-6*self.qMax)
        except ValueError:
            self.logger.error(self.givenQ(low), self.givenQ(high))
            raise
        self.postProcess(self.cellList)

    def calculateAreaWetted(self, side):
        """
        Calculates the wetted area of the given side

        Parameters
        ----------
        side : string
            'Hot' or 'Cold' side.

        Returns
        -------
        float
            wetted area of side given by 'side' variable.

        """
        if side == 'Hot':
            return np.pi*self.innerTubeID*self.effectiveLength
        return np.pi*self.innerTubeOD*self.effectiveLength

    def calculateChannelVolume(self, areaFlow):
        """
        Calculates the volume of the channel

        Parameters
        ----------
        areaFlow : TYPE
            DESCRIPTION.

        Returns
        -------
        float
            DESCRIPTION.

        """
        return self.effectiveLength*areaFlow

    def calculateHydraulicDiameter(self, side):
        """
        Calculares the hydraulic diameter of the channel

        Parameters
        ----------
        side : string
            'Hot' or 'Cold' side.

        Returns
        -------
        float
            hydraulic diameter of channel.

        """
        if side == 'Hot':
            return self.innerTubeID
        return self.outerTubeID - self.innerTubeOD

    def calculateAreaFlow(self, side):
        """
        Calculates the area of the channel perpendicular to the flow of the fluid

        Parameters
        ----------
        side : string
            'Hot' or 'Cold' side.

        Returns
        -------
        float
            channel area perpendicular to flow.

        """
        if side == 'Hot':
            return np.pi*self.innerTubeID**2/4.0
        return np.pi*(self.outerTubeID**2 - self.innerTubeOD**2)/4.0

    def setThermalResistanceWall(self):
        """
        Sets the thermal resistance of the wall between the "Hot" and "Cold" sides

        Returns
        -------
        None.

        """
        self.thermalResistanceWall = np.log(self.innerTubeOD/self.innerTubeID)/\
            (2*np.pi*self.conductivity*self.effectiveLength)

    def _onePhaseHOnePhaseCQimposed(self,inputs):
        outputs = {}
        inputs.update({"identifier": 'w[1-1]: '})
        for temp in ['Hot', 'Cold']:
            outputs[temp] = self.htdp(getattr(self, f"fluid{temp}"), inputs[f'tempMean{temp}'],
                          self.fluidProps[temp].pressureIn, getattr(self, f"massFlow{temp}"), side=temp)
            inputs.update({f"heatTransferCoeff{temp}": outputs[temp]['heatTransferCoeff'],
                           f"pressureDrop{temp}": outputs[temp]['pressureDrop']})
        return {**inputs, **super()._onePhaseHOnePhaseCQimposed(inputs)}

    def _onePhaseHTwoPhaseCQimposed(self,inputs):
        outputsHot = self.htdp(self.fluidHot, inputs['tempMeanHot'],self.fluidProps["Hot"].get('pressureIn'),
                               self.massFlowHot, side="Hot")
        inputs.update({"identifier": 'w[3-2]: ',
                       "heatTransferCoeffHot": outputsHot['heatTransferCoeff'],
                       "pressureDropHot": outputsHot['pressureDrop'],
                       "C": None
                       })
        return {**inputs, **super()._onePhaseHTwoPhaseCQimposed(inputs)}

    def _twoPhaseHOnePhaseCQimposed(self, inputs):
        outputsCold = self.htdp(self.fluidCold, inputs['tempMeanCold'],
                                self.fluidProps["Cold"].get('pressureIn'),
                                self.massFlowCold, side="Cold")
        inputs.update({"identifier": 'w[2-1]: ',
                       "heatTransferCoeffCold": outputsCold['heatTransferCoeff'],
                       "specificHeatCold": outputsCold['heatCapacity'],
                       "pressureDropCold": outputsCold['pressureDrop'],
                       "C": None
                       })
        return {**inputs, **super()._twoPhaseHOnePhaseCQimposed(inputs)}

    def _transCritPhaseHTwoPhaseCQimposed(self,inputs):
        #TODO: needs to be re-worked
        inputs["identifier"] = 'w[3-1]: '
        return {**inputs, **super()._transCritPhaseHTwoPhaseCQimposed(inputs)}

    def _transCritPhaseHOnePhaseCQimposed(self, inputs):
        outputsCold = self.htdp(self.fluidCold, inputs['tempMeanCold'],
                                self.fluidProps["Cold"].get('pressureIn'),
                                self.massFlowCold, side="Cold")
        inputs.update({"identifier": 'w[1-2]: ',
                       "heatTransferCoeffCold": outputsCold['heatTransferCoeff'],
                       "specificHeatCold": outputsCold['heatCapacity'],
                       "pressureDropCold": outputsCold['pressureDrop']
            })
        return {**inputs, **super()._transCritPhaseHOnePhaseCQimposed(inputs)}


    def htdp(self, fluid, temperature, pressure, massFlow, side):
        """
        This function calls mainly the heat transfer and pressure drop
        for single phase fluids of the plate heat exchanger
        Inputs: temperature [K] and pressure [Pa]
        outputs: h [W/m^2-K] and cp [J/kg-K]
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
                'heatTransferCoeff': Heat transfer coefficient in W/m^2/K
                'pressureDrop': Pressure drop in Pa
                'reynoldsNum':  Reynolds number
                'velocity': Velocity of fluid in channel in m/s
                'conductivity': Thermal conductivity of fluid in W/m/K
                'heatCapacity': Heat capacity of fluid in J/kg/K

        """
        heatCapacity = fluid.calculateHeatCapacity(ThermoProps.PT, pressure, temperature)
        specificVolume = 1/fluid.calculateDensity(ThermoProps.PT, pressure, temperature)
        if side == 'Hot':
            frictionFactor, heatTransferCoeff, reynoldsNum = f_h_1phase_Tube(massFlow,
                                            self.innerTubeID, temperature, pressure, fluid)
        else:
            frictionFactor, heatTransferCoeff, reynoldsNum = f_h_1phase_Annulus(massFlow,
                                self.outerTubeID, self.innerTubeOD, temperature, pressure, fluid)
        pressureGradient = frictionFactor*specificVolume*self.fluidProps[side].get('massFluxAverage')**2/\
            (2.*self.calculateHydraulicDiameter(side))
        pressureDrop = pressureGradient*self.effectiveLength
        outputs = {
            'heatTransferCoeff': heatTransferCoeff,
            'pressureDrop': pressureDrop,
            'heatCapacity': heatCapacity,
        }
        return outputs

    def calculateHeatTransferCoeff(self, temp, heatFlux, xIn=None, xOut=None, massFlux=None):
        """
        Using the Kandlikar Evaporation algorithm, calculates the heat transfer
        coefficient of fluid with given temperature (Hot or Cold) given heatFlux,
        input and output vapor fraction, and mass flux


        Parameters
        ----------
        temp : str
            'Hot' or 'Cold', denotes the stream being acted upon.
        heatFlux : float
            heat transferred.
        xIn : float, optional
            vapor fraction input on the cold side. The default is None.
        xOut : float, optional
            vapor fraction input on the hot side. The default is None.
        massFlux : float, optional
            mass. The default is None.

        Returns
        -------
        float
            heat transfer coefficient.

        """
        return kandlikarEvaporationAvg(xIn, xOut, getattr(self, f"fluid{temp}"), massFlux,
                    getattr(self, f"diameterHydraulic{temp}"), heatFlux, self.fluidProps[temp].get('tBubble'),
                    self.fluidProps[temp].get('tempDew'))

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 08:30:49 2024

@author: SMCANANA
"""

import pytest
from ACHP.models.HeatExchangers import HeatExchanger, BrazedPlateHEX, CoaxialHEX, HEXType
from ACHP.models.Fluid import Fluid

def pytest_generate_tests(metafunc):
    if metafunc.function.__name__ in metafunc.cls.params.keys():
        funcarglist = metafunc.cls.params[metafunc.function.__name__]
        argnames = sorted(funcarglist[0])
        metafunc.parametrize(
            argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
        )

@pytest.fixture
def brazedPlate():
    return BrazedPlateHEX(fluidHot=Fluid("R744", "HEOS"),
                          fluidCold=Fluid("Water","HEOS"),
                          massFlowHot=1.1,
                          massFlowCold=1.2,
                          pressureInHot=1.3,
                          pressureInCold=1.4,
                          enthalpyInHot=1.5,
                          enthalpyInCold=1.6,
                          conductivity=1.7,
                          centerlineDistanceShort=1.8,
                          centerlineDistanceLong=1.9,
                          numPlates=5,
                          thickness=1.12,
                          volumeChannelSingle=1.13,
                          amplitude=1.14,
                          wavelength=3.14159/4)

@pytest.fixture
def coaxial():
    return CoaxialHEX(fluidHot=Fluid("R744", "HEOS"),
                      fluidCold=Fluid("Water","HEOS"),
                      massFlowHot=1.1,
                      massFlowCold=1.2,
                      pressureInHot=1.3,
                      pressureInCold=1.4,
                      enthalpyInHot=1.5,
                      enthalpyInCold=1.6,
                      conductivity=1.7,
                      innerTubeInnerDiameter=1.8,
                      innerTubeOuterDiameter=1.9,
                      outerTubeInnerDiameter=2.1,
                      length=2.2)

@pytest.fixture
def heatExchanger():
    return HeatExchanger(fluidHot=Fluid("R744", "HEOS"),
                      fluidCold=Fluid("Water","HEOS"),
                      massFlowHot=1.1,
                      massFlowCold=1.2,
                      pressureInHot=1.3,
                      pressureInCold=1.4,
                      enthalpyInHot=1.5,
                      enthalpyInCold=1.6,
                      conductivity=1.7,
                      effectiveLength=1.8)

@pytest.fixture
def fluidR744():
    return Fluid("R744", "HEOS")

class TestHeatExchanger:
    params = {}
    def test_logLocalVars(self, heatExchanger):
        assert(True)

    def test_outputList(self, heatExchanger):
        assert(True)

    def test_setUpCalculation(self, heatExchanger):
        assert(True)

    def test_determineHTBounds(self, heatExchanger):
        assert(True)

    def test_checkPinchPoints(self, heatExchanger):
        assert(True)

    def test_givenQ(self, heatExchanger):
        assert(True)

    def test_buildEnthalpyLists(self, heatExchanger):
        assert(True)

    def test_determineHotAndColdPhases(self, heatExchanger):
        assert(True)

    def test__onePhaseHOnePhaseCQimposed(self, heatExchanger):
        assert(True)

    def test__onePhaseHTwoPhaseCQimposed(self, heatExchanger):
        assert(True)

    def test__twoPhaseHOnePhaseCQimposed(self, heatExchanger):
        assert(True)

    def test_calculateFraction(self, heatExchanger):
        assert(True)

    def test__transCritPhaseHTwoPhaseCQimposed(self, heatExchanger):
        assert(True)

    def test__transCritPhaseHOnePhaseCQimposed(self, heatExchanger):
        assert(True)

    def test_calculateHeatTransferCoeff(self, heatExchanger):
        assert(True)

    def test_postProcess(self, heatExchanger):
        assert(True)

    def test_calculateTempsAndDensities(self, heatExchanger):
        assert(True)

    def test_calculateEntropyOfFluid(self, heatExchanger):
        assert(True)

    def test_calculateIncrementalHeatTransfer(self, heatExchanger):
        assert(True)

class TestBrazedPlateHEX:
    params = {"test_BrazedPlateHEX_Instantiation":  [dict(variable="type", value=HEXType.PLATE, innerVar=None, innerVar2=None),
                                                     dict(variable="fluidHot", innerVar="name", value="R744", innerVar2=None),
                                                     dict(variable="fluidCold", innerVar="name", value="Water", innerVar2=None),
                                                     dict(variable="massFlowHot", value=1.1, innerVar=None, innerVar2=None),
                                                     dict(variable="massFlowCold", value=1.2, innerVar=None, innerVar2=None),
                                                     dict(variable="fluidProps", innerVar2="Hot", innerVar="pressureIn", value=1.3),
                                                     dict(variable="fluidProps", innerVar2="Cold", innerVar="pressureIn", value=1.4),
                                                     dict(variable="fluidProps", innerVar2="Hot", innerVar="enthalpyIn", value=1.5),
                                                     dict(variable="fluidProps", innerVar2="Cold", innerVar="enthalpyIn", value=1.6),
                                                     dict(variable="conductivity", value=1.7, innerVar=None, innerVar2=None),
                                                     dict(variable="centerlineDistShort", value=1.8, innerVar=None, innerVar2=None),
                                                     dict(variable="effectiveLength", value=1.9, innerVar=None, innerVar2=None),
                                                     dict(variable="numPlates", value=5, innerVar=None, innerVar2=None),
                                                     dict(variable="thickness", value=1.12, innerVar=None, innerVar2=None)],
              "test_allocateChannels":              [dict(temp="Hot", value=2, numPlates = 5),
                                                     dict(temp="Cold", value=2, numPlates = 5),
                                                     dict(temp="Cold", value=3, numPlates = 6),
                                                     dict(temp="Cold", value=3, numPlates = 6)],
              "test_calculateAreaWetted":           [dict(volume=None, calculatedAreaWetted=0.42967944609579034, areaBetweenPorts=0.072*0.329),
                                                     dict(volume=0.236, calculatedAreaWetted=0.142128, areaBetweenPorts=0.072*0.329)],
              "test_calculateChannelVolume":        [dict(volume=None, calculatedVol=0.10801728, temp="Hot", numGaps=2),
                                                     dict(volume=0.236, calculatedVol=0.472, temp="Hot", numGaps=2),
                                                     dict(volume=None, calculatedVol=0.16202592, temp="Cold", numGaps=3),
                                                     dict(volume=0.236, calculatedVol=0.708, temp="Cold", numGaps=3)],
               "test_calculateHydraulicDiameter":    [dict(volume=None, calculatedDh=0.7541711453606689),
                                                      dict(volume=0.236, calculatedDh=0.010300574130361364)],
               "test_calculateAreaFlow":             [dict(volume=None, calculatedAreaFlow=8.208, temp="Hot", numGaps=2),
                                                      dict(volume=0.236, calculatedAreaFlow=0.094752, temp="Hot", numGaps=2),
                                                      dict(volume=None, calculatedAreaFlow=12.312000000000001, temp="Cold", numGaps=3),
                                                      dict(volume=0.236, calculatedAreaFlow=0.142128, temp="Cold", numGaps=3)]
        }

    def test_BrazedPlateHEX_Instantiation(self, brazedPlate, variable, value, innerVar, innerVar2):
        if innerVar2 is not None:
            assert(getattr(getattr(brazedPlate, variable)[innerVar2], innerVar) == value)
        elif innerVar is not None:
            assert(getattr(getattr(brazedPlate, variable), innerVar) == value)
        else:
            assert(getattr(brazedPlate, variable) == value)

    def test_calculate(self, brazedPlate):
        assert(True)

    def test_allocateChannels(self, temp, value, numPlates, brazedPlate):
        temps = ["Hot", "Cold"]
        temps.remove(temp)
        brazedPlate.moreChannels = temp
        brazedPlate.numPlates = numPlates
        brazedPlate.allocateChannels()
        assert(getattr(brazedPlate, f"numGaps{temp}") == value)
        assert(getattr(brazedPlate, f"numGaps{temps[0]}") == value if value%2==0 else value-1)

    def test__calculateAreaBetweenPorts(self, brazedPlate):
        areaBetweenPorts = brazedPlate._calculateAreaBetweenPorts()
        assert(areaBetweenPorts == 3.42)

    def test__calculatePhi(self, brazedPlate):
        phi = brazedPlate._calculatePhi()
        print(phi)
        assert(phi == 6.046372932790025)

    def test_calculateAreaWetted(self, brazedPlate, volume, calculatedAreaWetted, areaBetweenPorts):
        brazedPlate.volumeChannelSingle = volume
        areaWetted = brazedPlate.calculateAreaWetted(areaBetweenPorts)
        assert(areaWetted == calculatedAreaWetted)

    def test_calculateChannelVolume(self, brazedPlate, volume, temp, calculatedVol, numGaps):
        brazedPlate.volumeChannelSingle = volume
        setattr(brazedPlate, f"numGaps{temp}", numGaps)
        channelVol = brazedPlate.calculateChannelVolume(temp, 0.072*0.329)
        assert(channelVol == calculatedVol)

    def test_calculateHydraulicDiameter(self, brazedPlate, volume, calculatedDh):
        brazedPlate.volumeChannelSingle = volume
        hydraulicDiam = brazedPlate.calculateHydraulicDiameter(0.000061, 0.072*0.329)
        assert(hydraulicDiam == calculatedDh)

    def test_calculateAreaFlow(self, brazedPlate, volume, temp, calculatedAreaFlow, numGaps):
        brazedPlate.volumeChannelSingle = volume
        setattr(brazedPlate, f"numGaps{temp}", numGaps)
        areaFlow = brazedPlate.calculateAreaFlow(temp, 0.072*0.329)
        assert(areaFlow == calculatedAreaFlow)

    def test_singlePhaseThermoCorrelations(self, brazedPlate):
        assert(True)

    def test__onePhaseHOnePhaseCQimposed(self, brazedPlate, mocker):
        mocker.patch('ACHP.models.HeatExchangers.HeatExchanger._onePhaseHOnePhaseCQimposed',
                      return_value={})
        mocker.patch('ACHP.models.HeatExchangers.BrazedPlateHEX.htdp',
                      return_value={"heatTransferCoeff": 2.1, "pressureDrop": 2.3})
        mocker.patch('ACHP.models.Fluid.FluidApparatusProps.get')
        brazedPlate.numGapsHot = brazedPlate.numGapsCold = 1
        inputs = brazedPlate._onePhaseHOnePhaseCQimposed({'tempMeanHot': 0.1, 'tempMeanCold': 0.2})
        assert(inputs == {"identifier": 'w[1-1]: ',
                           "heatTransferCoeffHot": 2.1,
                           "heatTransferCoeffCold": 2.1,
                           "pressureDropHot": 2.3,
                           "pressureDropCold": 2.3,
                           "tempMeanHot": 0.1,
                           "tempMeanCold": 0.2})

    def test__onePhaseHTwoPhaseCQimposed(self, brazedPlate, mocker):
        mocker.patch('ACHP.models.HeatExchangers.HeatExchanger._onePhaseHTwoPhaseCQimposed',
                      return_value={})
        mocker.patch('ACHP.models.HeatExchangers.BrazedPlateHEX.htdp',
                      return_value={"heatTransferCoeff": 2.1, "pressureDrop": 2.3})
        brazedPlate.numGapsHot = 1
        brazedPlate.claessonParamC = 0.3
        inputs = brazedPlate._onePhaseHTwoPhaseCQimposed({'tempMeanHot': 0.1,})
        assert(inputs == {"identifier": 'w[3-2]: ',
                           "heatTransferCoeffHot": 2.1,
                           "pressureDropHot": 2.3,
                           "C": 0.3,
                           "tempMeanHot": 0.1})

    def test__twoPhaseHOnePhaseCQimposed(self, brazedPlate, mocker):
        mocker.patch('ACHP.models.HeatExchangers.HeatExchanger._twoPhaseHOnePhaseCQimposed',
                      return_value={})
        mocker.patch('ACHP.models.HeatExchangers.BrazedPlateHEX.htdp',
                      return_value={"heatTransferCoeff": 2.1, "pressureDrop": 2.3, "heatCapacity": 0.4})
        brazedPlate.numGapsCold = 1
        brazedPlate.claessonParamC = 0.3
        inputs = brazedPlate._twoPhaseHOnePhaseCQimposed({'tempMeanCold': 0.2})
        assert(inputs == {"identifier": 'w[2-1]: ',
                           "heatTransferCoeffCold": 2.1,
                           "specificHeatCold": 0.4,
                           "pressureDropCold": 2.3,
                           "C": 0.3,
                           "tempMeanCold": 0.2})

    def test__transCritPhaseHOnePhaseCQimposed(self, brazedPlate, mocker):
        mocker.patch('ACHP.models.HeatExchangers.HeatExchanger._transCritPhaseHOnePhaseCQimposed',
                      return_value={})
        mocker.patch('ACHP.models.HeatExchangers.BrazedPlateHEX.htdp',
                      return_value={"heatTransferCoeff": 2.1, "pressureDrop": 2.3, "heatCapacity": 0.4})
        brazedPlate.numGapsCold = 1
        inputs = brazedPlate._transCritPhaseHOnePhaseCQimposed({'tempMeanCold': 0.2})
        assert(inputs == {"identifier": 'w[1-2]: ',
                           "heatTransferCoeffCold": 2.1,
                           "specificHeatCold": 0.4,
                           "pressureDropCold": 2.3,
                           "tempMeanCold": 0.2})

    def test__transCritPhaseHTwoPhaseCQimposed(self, brazedPlate, mocker):
        mocker.patch('ACHP.models.HeatExchangers.HeatExchanger._transCritPhaseHTwoPhaseCQimposed',
                      return_value={})
        inputs = brazedPlate._transCritPhaseHTwoPhaseCQimposed({})
        assert(inputs == {"identifier": 'w[3-1]: '})

    def test_calculateHeatTransferCoeff(self, brazedPlate, mocker):
        #TODO: This should be an integration test
        mocker.patch('ACHP.models.HeatExchangers.Cooper_PoolBoiling',
                     return_value=1.2345)
        mocker.patch('ACHP.models.HeatExchangers.getattr')
        result = brazedPlate.calculateHeatTransferCoeff("Hot", 2.0, xIn=0.25, xOut=0.75, massFlux=5.0)
        assert(result == 1.2345)

    def test_htdp(self, brazedPlate):
        #TODO: This should be an integration test
        assert(True)

class TestCoaxialHEX:
    params = {"test_CoaxialHEX_Instantiation": [dict(variable="type", value=HEXType.COAXIAL, innerVar=None, innerVar2=None),
                                                   dict(variable="fluidHot", innerVar="name", value="R744", innerVar2=None),
                                                   dict(variable="fluidCold", innerVar="name", value="Water", innerVar2=None),
                                                   dict(variable="massFlowHot", value=1.1, innerVar=None, innerVar2=None),
                                                   dict(variable="massFlowCold", value=1.2, innerVar=None, innerVar2=None),
                                                   dict(variable="fluidProps", innerVar2="Hot", innerVar="pressureIn", value=1.3),
                                                   dict(variable="fluidProps", innerVar2="Cold", innerVar="pressureIn", value=1.4),
                                                   dict(variable="fluidProps", innerVar2="Hot", innerVar="enthalpyIn", value=1.5),
                                                   dict(variable="fluidProps", innerVar2="Cold", innerVar="enthalpyIn", value=1.6),
                                                   dict(variable="conductivity", value=1.7, innerVar=None, innerVar2=None),
                                                   dict(variable="innerTubeID", value=1.8, innerVar=None, innerVar2=None),
                                                   dict(variable="innerTubeOD", value=1.9, innerVar=None, innerVar2=None),
                                                   dict(variable="outerTubeID", value=2.1, innerVar=None, innerVar2=None),
                                                   dict(variable="effectiveLength", value=2.2, innerVar=None, innerVar2=None)],
              "test_calculateAreaWetted": [dict(side='Hot', expected=12.440706908215581),
                                           dict(side='Cold', expected=13.131857292005336)],
              "test_calculateChannelVolume": [dict(areaFlow=1.2, expected=2.64)],
              "test_calculateHydraulicDiameter": [dict(side='Hot', expected=1.8),
                                         dict(side='Cold', expected=0.2)],
              "test_calculateAreaFlow": [dict(side='Hot', expected=2.5446900494077327),
                                     dict(side='Cold', expected=0.6283185307179588)],
              "test_setThermalResistanceWall": [dict(expected=0.0023008196589325283)],
              "test_htdp_calculates_pressureDrop": [dict(side='Hot', pd=2.7204878048780485),
                            dict(side='Cold', pd=48.968780487804835)]
        }

    def test_CoaxialHEX_Instantiation(self, coaxial, variable, value, innerVar, innerVar2):
        if innerVar2 is not None:
            assert(getattr(getattr(coaxial, variable)[innerVar2], innerVar) == value)
        elif innerVar is not None:
            assert(getattr(getattr(coaxial, variable), innerVar) == value)
        else:
            assert(getattr(coaxial, variable) == value)

    def test_calculate(self, coaxial):
        assert(True)

    def test_calculateAreaWetted(self, coaxial, side, expected):
        actual = coaxial.calculateAreaWetted(side)
        assert(actual == expected)

    def test_calculateChannelVolume(self, coaxial, areaFlow, expected):
        actual = coaxial.calculateChannelVolume(areaFlow)
        assert(actual == expected)

    def test_calculateHydraulicDiameter(self, coaxial, side, expected):
        actual = coaxial.calculateHydraulicDiameter(side)
        #rounding to one digit because float math
        assert(round(actual, 1) == expected)

    def test_calculateAreaFlow(self, coaxial, side, expected):
        actual = coaxial.calculateAreaFlow(side)
        assert(actual == expected)

    def test_setThermalResistanceWall(self, coaxial, expected):
        coaxial.setThermalResistanceWall()
        assert(coaxial.thermalResistanceWall == expected)

    def test__onePhaseHOnePhaseCQimposed(self, coaxial, mocker):
        mocker.patch('ACHP.models.HeatExchangers.HeatExchanger._onePhaseHOnePhaseCQimposed',
                      return_value={})
        mocker.patch('ACHP.models.HeatExchangers.CoaxialHEX.htdp',
                      return_value={"heatTransferCoeff": 2.1, "pressureDrop": 2.3})
        inputs = coaxial._onePhaseHOnePhaseCQimposed({'tempMeanHot': 0.1, 'tempMeanCold': 0.2})
        assert(inputs == {"identifier": 'w[1-1]: ',
                           "heatTransferCoeffHot": 2.1,
                           "heatTransferCoeffCold": 2.1,
                           "pressureDropHot": 2.3,
                           "pressureDropCold": 2.3,
                           "tempMeanHot": 0.1,
                           "tempMeanCold": 0.2})

    def test__onePhaseHTwoPhaseCQimposed(self, coaxial, mocker):
        mocker.patch('ACHP.models.HeatExchangers.HeatExchanger._onePhaseHTwoPhaseCQimposed',
                      return_value={})
        mocker.patch('ACHP.models.HeatExchangers.CoaxialHEX.htdp',
                      return_value={"heatTransferCoeff": 2.1, "pressureDrop": 2.3})
        inputs = coaxial._onePhaseHTwoPhaseCQimposed({'tempMeanHot': 0.1,})
        assert(inputs == {"identifier": 'w[3-2]: ',
                           "heatTransferCoeffHot": 2.1,
                           "pressureDropHot": 2.3,
                           "C": None,
                           "tempMeanHot": 0.1})

    def test__twoPhaseHOnePhaseCQimposed(self, coaxial, mocker):
        mocker.patch('ACHP.models.HeatExchangers.HeatExchanger._twoPhaseHOnePhaseCQimposed',
                      return_value={})
        mocker.patch('ACHP.models.HeatExchangers.CoaxialHEX.htdp',
                      return_value={"heatTransferCoeff": 2.1, "pressureDrop": 2.3, "heatCapacity": 0.4})
        inputs = coaxial._twoPhaseHOnePhaseCQimposed({'tempMeanCold': 0.2})
        assert(inputs == {"identifier": 'w[2-1]: ',
                           "heatTransferCoeffCold": 2.1,
                           "specificHeatCold": 0.4,
                           "pressureDropCold": 2.3,
                           "C": None,
                           "tempMeanCold": 0.2})

    def test__transCritPhaseHOnePhaseCQimposed(self, coaxial, mocker):
        mocker.patch('ACHP.models.HeatExchangers.HeatExchanger._transCritPhaseHOnePhaseCQimposed',
                      return_value={})
        mocker.patch('ACHP.models.HeatExchangers.CoaxialHEX.htdp',
                      return_value={"heatTransferCoeff": 2.1, "pressureDrop": 2.3, "heatCapacity": 0.4})
        inputs = coaxial._transCritPhaseHOnePhaseCQimposed({'tempMeanCold': 0.2})
        assert(inputs == {"identifier": 'w[1-2]: ',
                           "heatTransferCoeffCold": 2.1,
                           "specificHeatCold": 0.4,
                           "pressureDropCold": 2.3,
                           "tempMeanCold": 0.2})

    def test__transCritPhaseHTwoPhaseCQimposed(self, coaxial, mocker):
        mocker.patch('ACHP.models.HeatExchangers.HeatExchanger._transCritPhaseHTwoPhaseCQimposed',
                      return_value={})
        inputs = coaxial._transCritPhaseHTwoPhaseCQimposed({})
        assert(inputs == {"identifier": 'w[3-1]: '})

    def test_calculateHeatTransferCoeff(self, mocker, coaxial):
        #TODO: This should be an integration test
        mocker.patch('ACHP.models.HeatExchangers.kandlikarEvaporationAvg',
                     return_value=1.2345)
        mocker.patch('ACHP.models.HeatExchangers.getattr')
        mocker.patch('ACHP.models.Fluid.FluidApparatusProps.get')
        result = coaxial.calculateHeatTransferCoeff("Hot", 2.0, xIn=0.25, xOut=0.75, massFlux=5.0)
        assert(result == 1.2345)

    def test_htdp_calculates_pressureDrop(self, coaxial, mocker, fluidR744, side, pd):
        mocker.patch('ACHP.models.HeatExchangers.f_h_1phase_Tube', return_value=(1.0, 1.0, 1.0))
        mocker.patch('ACHP.models.HeatExchangers.f_h_1phase_Annulus', return_value=(2.0, 2.0, 2.0))
        mocker.patch('ACHP.models.Fluid.Fluid.calculateHeatCapacity', return_value=123456)
        mocker.patch('ACHP.models.Fluid.Fluid.calculateDensity', return_value=1.23)
        mocker.patch('ACHP.models.Fluid.FluidApparatusProps.get', return_value=2.34)
        outputs = coaxial.htdp(fluidR744, 273.15, 100000, 80, side)
        assert(outputs['pressureDrop'] == pd)

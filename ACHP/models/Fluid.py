# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:35:16 2024

@author: smcanana
"""
from enum import StrEnum
from ACHP.wrappers.CoolPropWrapper import AbstractStateWrapper

class FluidPhase(StrEnum):
    """
    Fluid phase enum. Used to determine or set the phase of a fluid.
    """
    SUBCOOLED = "Subcooled"
    SUPERHEATED = "Superheated"
    TWOPHASE = "Two phase"
    SUPERCRITICAL = "Supercritical"
    SUPERCRITLIQ = "Supercrit_liq"

class ThermoProps(StrEnum):
    """
    Thermodynamic properties pair enum. Used to determine which two thermodynamic
    properties will be used in calculation.
    """
    DH = 'density (kg/m^3) and enthalpy (J/kg)'
    DP = 'density (kg/m^3) and pressure (Pa)'
    DQ = 'density (kg/m^3) and vapor quality'
    DS = 'density (kg/m^3) and entropy (J/kg/K)'
    DT = 'density (kg/m^3) and temperature (K)'
    DU = 'density (kg/m^3) and internal energy (J/kg)'
    HP = 'enthalpy (J/kg) and pressure (Pa)'
    HQ = 'enthalpy (J/kg) and vapor quality'
    HS = 'enthalpy (J/kg) and entropy (J/kg/K)'
    HT = 'enthalpy (J/kg) and temperature (K)'
    PQ = 'pressure (Pa) and vapor quality'
    PS = 'pressure (Pa) and entropy (J/kg/K)'
    PT = 'pressure (Pa) and temperature (K)'
    PU = 'pressure (Pa) and internal energy (J/kg)'
    QS = 'vapor quality and entropy (J/kg/K)'
    QT = 'vapor quality and temperature (K)'
    ST = 'entropy (J/kg/K) and temperature (K)'
    SU = 'entropy (J/kg/K) and internal energy (J/kg)'
    TU = 'temperature (K) and internal energy (J/kg)'

class Fluid():
    """
    This class represents a fluid used in the library. It is used to calculate necessary
    properties of the fluid.
    """
    def __init__(self, name: str, backEnd: str, massFraction: float=1.0, volumeFraction: float=1.0):
        self.name = name
        self.backEnd = backEnd
        self.abstractState = AbstractStateWrapper(backEnd, name, massFraction, volumeFraction)
        self.pressureCritical = self.calculateCriticalPressure()
        self.massMolar = self.calculateMassMolar()
        self.fluidApparatiProps = {str: FluidApparatusProps}

    def getMeltingTemperature(self, pressure):
        return self.abstractState.getMeltingTemperature(pressure)
    
    def getIsobaricExpansionCoefficient(self):
        return self.abstractState.getIsobaricExpansionCoefficient()

    def calculateCriticalPressure(self):
        """
        Calculates the critical pressure of the fluid in Pa.

        Returns
        -------
        pressure : float
            critical pressure of the fluid in Pa.

        """
        return self.abstractState.calculateCriticalPressure()

    def calculateCriticalTemperature(self):
        """
        Calculates the critical temperature of the fluid in K.

        Returns
        -------
        temperature : float
            critical temperature of the fluid in K.

        """
        return self.abstractState.calculateCriticalTemperature()
    
    def calculateMassMolar(self):
        """
        Calculates the molar mass of the fluid in kg/mol.

        Returns
        -------
        massMolar : float
            molar mass of the fluid in kg/mol.

        """
        return self.abstractState.calculateMassMolar()

    def calculatePressure(self, properties: ThermoProps, variable1, variable2):
        """
        Calculates the pressure of the fluid in Pa.

        Parameters
        ----------
        properties : ThermoProps
            Enum for pair of thermodynamic properties to be used in calculation.
        variable1 : float
            First thermodynamic property value in the pair.
        variable2 : float
            Second thermodynamic property value in the pair.

        Raises
        ------
        NotImplementedError
            The calculation with this particular pair of thermodynamic properties has not been
            transcribed in the wrapper class.

        Returns
        -------
        pressure : float
            pressure of the fluid in Pa.

        """
        match properties:
            case ThermoProps.QT:
                pressure = self.abstractState.calculatePressureFromQandT(variable1, variable2)
            case _:
                raise NotImplementedError('This calculation has not yet been implemented')
        return pressure
    
    def calculateSurfaceTension(self, properties: ThermoProps, variable1, variable2):
        """
        Calculates the surface tension of the fluid in N/m.

        Parameters
        ----------
        properties : ThermoProps
            Enum for pair of thermodynamic properties to be used in calculation.
        variable1 : float
            First thermodynamic property value in the pair.
        variable2 : float
            Second thermodynamic property value in the pair.

        Raises
        ------
        NotImplementedError
            The calculation with this particular pair of thermodynamic properties has not been
            transcribed in the wrapper class.

        Returns
        -------
        surface tension : float
            surface tension of the fluid in N/m.

        """
        match properties:
            case ThermoProps.PQ:
                surfaceTension = self.abstractState.calculateSurfaceTensionFromPandQ(variable1, variable2)
            case _:
                raise NotImplementedError('This calculation has not yet been implemented')
        return surfaceTension
    

    def calculateTemperature(self, properties: ThermoProps, variable1, variable2):
        """
        Calculates the temperature of the fluid in K.

        Parameters
        ----------
        properties : ThermoProps
            Enum for pair of thermodynamic properties to be used in calculation.
        variable1 : float
            First thermodynamic property value in the pair.
        variable2 : float
            Second thermodynamic property value in the pair.

        Raises
        ------
        NotImplementedError
            The calculation with this particular pair of thermodynamic properties has not been
            transcribed in the wrapper class.

        Returns
        -------
        temperature : float
            temperature of the fluid in K.

        """
        match properties:
            case ThermoProps.PQ:
                temperature = self.abstractState.calculateTempFromPandQ(variable1, variable2)
            case ThermoProps.HP:
                temperature = self.abstractState.calculateTempFromHandP(variable1, variable2)
            case _:
                raise NotImplementedError('This calculation has not yet been implemented')
        return temperature

    def calculateSpecificVolume(self, properties: ThermoProps, variable1, variable2):
        """
        Calculates the specific volume of the fluid in m^3/kg.

        Parameters
        ----------
        properties : ThermoProps
            Enum for pair of thermodynamic properties to be used in calculation.
        variable1 : float
            First thermodynamic property value in the pair.
        variable2 : float
            Second thermodynamic property value in the pair.

        Raises
        ------
        NotImplementedError
            The calculation with this particular pair of thermodynamic properties has not been
            transcribed in the wrapper class.

        Returns
        -------
        specific volume : float
            specific volume of the fluid in m^3/kg.

        """
        match properties:
            case ThermoProps.PT:
                specificVolume = self.abstractState.calculateSpecificVolumeFromPandT(variable1, variable2)
            case _:
                raise NotImplementedError('This calculation has not yet been implemented')
        return specificVolume

    def calculateEnthalpy(self, properties: ThermoProps, variable1, variable2):
        """
        Calculates the enthalpy of the fluid in J/kg.

        Parameters
        ----------
        properties : ThermoProps
            Enum for pair of thermodynamic properties to be used in calculation.
        variable1 : float
            First thermodynamic property value in the pair.
        variable2 : float
            Second thermodynamic property value in the pair.

        Raises
        ------
        NotImplementedError
            The calculation with this particular pair of thermodynamic properties has not been
            transcribed in the wrapper class.

        Returns
        -------
        enthalpy : float
            enthalpy of the fluid in J/kg.

        """
        match properties:
            case ThermoProps.DT:
                enthalpy = self.abstractState.calculateEnthalpyFromDandT(variable1, variable2)
            case ThermoProps.PQ:
                enthalpy = self.abstractState.calculateEnthalpyFromPandQ(variable1, variable2)
            case ThermoProps.PS:
                enthalpy = self.abstractState.calculateEnthalpyFromPandS(variable1, variable2)
            case ThermoProps.PT:
                enthalpy = self.abstractState.calculateEnthalpyFromPandT(variable1, variable2)
            case ThermoProps.QT:
                enthalpy = self.abstractState.calculateEnthalpyFromQandT(variable1, variable2)
            case _:
                raise NotImplementedError('This calculation has not yet been implemented')
        return enthalpy

    def calculateEntropy(self, properties: ThermoProps, variable1, variable2):
        """
        Calculates the entropy of the fluid in J/kg/K.

        Parameters
        ----------
        properties : ThermoProps
            Enum for pair of thermodynamic properties to be used in calculation.
        variable1 : float
            First thermodynamic property value in the pair.
        variable2 : float
            Second thermodynamic property value in the pair.

        Raises
        ------
        NotImplementedError
            The calculation with this particular pair of thermodynamic properties has not been
            transcribed in the wrapper class.

        Returns
        -------
        entropy : float
            entropy of the fluid in J/kg/K.

        """
        match properties:
            case ThermoProps.DT:
                entropy = self.abstractState.calculateEntropyFromDandT(variable1, variable2)
            case ThermoProps.HP:
                entropy = self.abstractState.calculateEntropyFromHandP(variable1, variable2)
            case ThermoProps.PT:
                entropy = self.abstractState.calculateEntropyFromPandT(variable1, variable2)
            case ThermoProps.QT:
                entropy = self.abstractState.calculateEntropyFromQandT(variable1, variable2)
            case _:
                raise NotImplementedError('This calculation has not yet been implemented')
        return entropy

    def calculateDensity(self, properties: ThermoProps, variable1, variable2):
        """
        Calculates the density of the fluid in kg/m^3.

        Parameters
        ----------
        properties : ThermoProps
            Enum for pair of thermodynamic properties to be used in calculation.
        variable1 : float
            First thermodynamic property value in the pair.
        variable2 : float
            Second thermodynamic property value in the pair.

        Raises
        ------
        NotImplementedError
            The calculation with this particular pair of thermodynamic properties has not been
            transcribed in the wrapper class.

        Returns
        -------
        density : float
            density of the fluid in kg/m^3.

        """
        match properties:
            case ThermoProps.HP:
                density = self.abstractState.calculateDensityFromHandP(variable1, variable2)
            case ThermoProps.PT:
                density = self.abstractState.calculateDensityFromPandT(variable1, variable2)
            case ThermoProps.PQ:
                density = self.abstractState.calculateDensityFromPandQ(variable1, variable2)
            case ThermoProps.QT:
                density = self.abstractState.calculateDensityFromQandT(variable1, variable2)
            case _:
                raise NotImplementedError('This calculation has not yet been implemented')
        return density

    def calculateHeatCapacity(self, properties: ThermoProps, variable1, variable2):
        """
        Calculates the heat capacity of the fluid in J/kg/K.

        Parameters
        ----------
        properties : ThermoProps
            Enum for pair of thermodynamic properties to be used in calculation.
        variable1 : float
            First thermodynamic property value in the pair.
        variable2 : float
            Second thermodynamic property value in the pair.

        Raises
        ------
        NotImplementedError
            The calculation with this particular pair of thermodynamic properties has not been
            transcribed in the wrapper class.

        Returns
        -------
        heatCapacity : float
            Heat capacity of the fluid in J/kg/K.

        """
        match properties:
            case ThermoProps.PT:
                heatCapacity = self.abstractState.calculateHeatCapacityFromPandT(variable1, variable2)
            case ThermoProps.PQ:
                heatCapacity = self.abstractState.calculateHeatCapacityFromPandQ(variable1, variable2)
            case ThermoProps.QT:
                heatCapacity = self.abstractState.calculateHeatCapacityFromQandT(variable1, variable2)
            case _:
                raise NotImplementedError('This calculation has not yet been implemented')
        return heatCapacity

    def calculateViscosity(self, properties: ThermoProps, variable1, variable2):
        """
        Calculates the viscosity of the fluid in Pa*s.

        Parameters
        ----------
        properties : ThermoProps
            Enum for pair of thermodynamic properties to be used in calculation.
        variable1 : float
            First thermodynamic property value in the pair.
        variable2 : float
            Second thermodynamic property value in the pair.

        Raises
        ------
        NotImplementedError
            The calculation with this particular pair of thermodynamic properties has not been
            transcribed in the wrapper class.

        Returns
        -------
        viscosity : float
            viscosity of the fluid in Pa*s.

        """
        match properties:
            case ThermoProps.PT:
                viscosity = self.abstractState.calculateViscosityFromPandT(variable1, variable2)
            case ThermoProps.QT:
                viscosity = self.abstractState.calculateViscosityFromQandT(variable1, variable2)
            case _:
                raise NotImplementedError('This calculation has not yet been implemented')
        return viscosity

    def calculateConductivity(self, properties: ThermoProps, variable1, variable2):
        """
        Calculates the conductivity of the fluid in W/m/K.

        Parameters
        ----------
        properties : ThermoProps
            Enum for pair of thermodynamic properties to be used in calculation.
        variable1 : float
            First thermodynamic property value in the pair.
        variable2 : float
            Second thermodynamic property value in the pair.

        Raises
        ------
        NotImplementedError
            The calculation with this particular pair of thermodynamic properties has not been
            transcribed in the wrapper class.

        Returns
        -------
        conductivity : float
            conductivity of the fluid in W/m/K.

        """
        match properties:
            case ThermoProps.PT:
                conductivity = self.abstractState.calculateConductivityFromPandT(variable1, variable2)
            case _:
                raise NotImplementedError('This calculation has not yet been implemented')
        return conductivity
    
class FluidApparatusProps():
    def __init__(self, pressureIn: float=0.0, temperatureIn: float=0.0,
             enthalpyIn: float=0.0, fractionIn: float=0.0):
        self.pressureIn = pressureIn
        self.temperatureIn = temperatureIn
        self.enthalpyIn = enthalpyIn
        self.fractionIn = fractionIn
        
        self.pressureOut: float
        self.temperatureOut: float
        self.enthalpyOut: float
        self.fractionOut: float
        
    def get(self, fluidProperty):
        if not hasattr(self, fluidProperty):
            setattr(self, fluidProperty, 0.0)
        return getattr(self, fluidProperty)
    
    def getDict(self, fluidProperty):
        if not hasattr(self, fluidProperty):
            setattr(self, fluidProperty, {})
        return getattr(self, fluidProperty)
    
    def addToProperty(self, fluidProperty, key, value):
        if hasattr(self, fluidProperty):
            getattr(self,fluidProperty)[key] = value
        else:
            setattr(self, fluidProperty, {key: value})

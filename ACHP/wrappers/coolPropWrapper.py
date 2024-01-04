# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:39:35 2024

@author: SMCANANA
"""
import CoolProp as CP
from CoolProp.CoolProp import HAPropsSI, PropsSI

class HumidAirPropertiesWrapper():
    """
    Wrapper for HAPropsSI
    """

    def calculateHumidityRatio(self, temperature, pressure, relativeHumidity):
         """
         Given a temperature, pressure, and relative humidity, calculates the humidity
         ratio.

         Parameters
         ----------
         temperature : float
             temperature in K
         pressure : float
             pressure in Pa
         relativeHumidity : float
             relative humidity

         Returns
         -------
         float
             humidity ratio

         """
         return HAPropsSI('W','T',temperature,'P',pressure,'R',relativeHumidity)

    def calculateDryAirEnthalpy(self, temperature, pressure, humidityRatio):
         """
         Given a temperature, pressure, and humidity ratio, calculates enthalpy

         Parameters
         ----------
         temperature : float
             temperature in K
         pressure : float
             pressure in Pa
         humidityRatio : float
             humidity ratio

         Returns
         -------
         float
             enthalpy in J/kg

         """
         return HAPropsSI('H','T',temperature,'P',pressure,'W',humidityRatio)

    def calculateDryAirVolume(self, temperature, pressure, humidityRatio):
        """
        Given a temperature, pressure, and humidity ratio, calculates the volume of dry
        air

        Parameters
        ----------
        temperature : float
            temperature in K
        pressure : float
            pressure in Pa
        humidityRatio : float
            humidity ratio

        Returns
        -------
        float
            volume of dry air in m^3

        """
        return HAPropsSI('V','T',temperature,'P',pressure,'W',humidityRatio)

    def calculateViscosity(self, temperature, pressure, humidityRatio):
        """
        Given a temperature, pressure, and humidity ratio, calculates air viscosity

        Parameters
        ----------
        temperature : float
            temperature in K
        pressure : float
            pressure in Pa
        humidityRatio : float
            humidity ratio

        Returns
        -------
        float
            air viscosity

        """
        return HAPropsSI('M', 'T', temperature, 'P', pressure, 'W', humidityRatio)

    def calculateThermalConductivity(self, temperature, pressure, humidityRatio):
        """
        Given a temperature, pressure, and humidity ratio, calculates the thermal
        conductivity of air

        Parameters
        ----------
        temperature : float
            temperature in K
        pressure : float
            pressure in Pa
        humidityRatio : float
            humidity ratio

        Returns
        -------
        float
            thermal conductivity

        """
        return HAPropsSI('K', 'T', temperature, 'P', pressure, 'W', humidityRatio)

class AbstractStateWrapper():
    """
    Wrapper for CoolProp AbstractState class
    """
    def __init__(self, backEnd: str, fluid: str):
        self.fluid = fluid
        self.abstractState = CP.AbstractState(backEnd, fluid)

    def calculateSpecificVolumeFromPandT(self, pressure, temperature):
        """
        CoolProp wrapper method to calculate the specific volume of the fluid
        given the pressure and temperature

        Parameters
        ----------
        pressure : float
            pressure of the state in Pa
        temperature : float
            temperature of the state in K

        Returns
        -------
        specificVolume : float
            specific volume of the state in m^3/kg

        """
        self.abstractState.update(CP.PT_INPUTS, pressure, temperature)
        return 1/self.abstractState.rhomass()

    def calculateTempFromPandH(self, pressure, enthalpy):
        """
        CoolProp wrapper method to calculate the temperature of the fluid
        given the pressure and enthalpy

        Parameters
        ----------
        pressure : float
            pressure of the state in Pa
        enthalpy : float
            enthalpy of the state in J/kg

        Returns
        -------
        float
            temperature of the state in K

        """
        self.abstractState.update(CP.HmassP_INPUTS, enthalpy, pressure)
        return self.abstractState.T()

    def calculateTempFromPandQ(self, pressure, quality):
        """
        CoolProp wrapper method to calculate the temperature of the fluid
        given the pressure and vapor quality

        Parameters
        ----------
        pressure : float
            pressure of the state in Pa
        quality : float
            vapor quality, unitless

        Returns
        -------
        float
            temperature of the state in K

        """
        self.abstractState.update(CP.PQ_INPUTS, pressure, quality)
        return self.abstractState.T()

    def calculateEnthalpyFromPandS(self, pressure, entropy):
        """
        CoolProp wrapper method to calculate enthalpy of the fluid 
        given the pressure and entropy

        Parameters
        ----------
        abstractState : CoolProp AbstractState class
            abstract state to calculate state properties
        pressure : float
            pressure of the state in Pa
        entropy : float
            entropy of the state in J/kg/K

        Returns
        -------
        float
            enthalpy of the state in J/kg

        """
        self.abstractState.update(CP.PSmass_INPUTS, pressure, entropy)
        return self.abstractState.hmass()

    def calculateEnthalpyFromPandQ(self, pressure, quality):
        """
        CoolProp wrapper method to calculate enthalpy of the fluid 
        given the pressure and vapor quality

        Parameters
        ----------
        abstractState : CoolProp AbstractState class
            abstract state to calculate state properties
        pressure : float
            pressure of the state in Pa
        quality : float
            vapor quality of the state, unitless

        Returns
        -------
        float
            enthalpy of the state in J/kg

        """
        self.abstractState.update(CP.PQ_INPUTS, pressure, quality)
        return self.abstractState.hmass()

    def calculateEnthalpyFromPandT(self, pressure, temperature):
        """
        CoolProp wrapper method to calculate the enthalpy of the fluid
        given the pressure and temperature

        Parameters
        ----------
        pressure : float
            pressure of the state in Pa
        temperature : float
            temperature of the state in K

        Returns
        -------
        enthalpy : float
            enthalpy of the state in J/kg

        """
        self.abstractState.update(CP.PT_INPUTS, pressure, temperature)
        return self.abstractState.hmass()

    def calculateEnthalpyFromQandT(self, quality, temperature):
        """
        CoolProp wrapper method to calculate the enthalpy of the fluid
        given the pressure and temperature

        Parameters
        ----------
        quality : float
            vapor quality of the state, unitless
        temperature : float
            temperature of the state in K

        Returns
        -------
        enthalpy : float
            enthalpy of the state in J/kg

        """
        self.abstractState.update(CP.QT_INPUTS, quality, temperature)
        return self.abstractState.hmass()

    def calculateEntropyFromPandH(self, pressure, enthalpy):
        """
        CoolProp wrapper method to calculate the entropy of the fluid 
        given the pressure and enthalpy

        Parameters
        ----------
        pressure : float
            pressure of the state in Pa
        enthalpy : float
            enthalpy of the state in J/kg

        Returns
        -------
        float
            entropy of the state in J/kg/K

        """
        self.abstractState.update(CP.HmassP_INPUTS, enthalpy, pressure)
        return self.abstractState.smass()

    def calculateEntropyFromPandT(self, pressure, temperature):
        """
        CoolProp wrapper method to calculate the entropy of the fluid
        given the pressure and temperature

        Parameters
        ----------
        pressure : float
            pressure of the state in Pa
        temperature : float
            temperature of the state in K

        Returns
        -------
        entropy : float
            entropy of the state in J/kg/K

        """
        self.abstractState.update(CP.PT_INPUTS, pressure, temperature)
        return self.abstractState.smass()

    def calculateEntropyFromQandT(self, quality, temperature):
        """
        CoolProp wrapper method to calculate the entropy of the fluid
        given the pressure and temperature

        Parameters
        ----------
        quality : float
            vapor quality of the state, unitless
        temperature : float
            temperature of the state in K

        Returns
        -------
        entropy : float
            entropy of the state in J/kg/K

        """
        self.abstractState.update(CP.QT_INPUTS, quality, temperature)
        return self.abstractState.smass()
    
    def calculateDensityFromPandT(self, pressure, temperature):
        """
        CoolProp wrapper method to calculate the density of the fluid
        given the pressure and temperature

        Parameters
        ----------
        pressure : float
            pressure of the state in Pa
        temperature : float
            temperature of the state in K

        Returns
        -------
        density : float
            density of the state in kg/m^3

        """
        self.abstractState.update(CP.PT_INPUTS, pressure, temperature)
        return self.abstractState.rhomass()
    
    def calculateHeatCapacityFromPandT(self, pressure, temperature):
        """
        CoolProp wrapper method to calculate the heat capacity of the fluid
        given the pressure and temperature

        Parameters
        ----------
        pressure : float
            pressure of the state in Pa
        quality : float
            vapor quality of the state, unitless

        Returns
        -------
        density : float
            density of the state in kg/m^3

        """
        self.abstractState.update(CP.PT_INPUTS, pressure, temperature)
        return self.abstractState.cpmass()
    
    def calculateHeatCapacityFromPandQ(self, pressure, quality):
        """
        CoolProp wrapper method to calculate the heat capacity of the fluid
        given the pressure and vapor quality

        Parameters
        ----------
        pressure : float
            pressure of the state in Pa
        temperature : float
            temperature of the state in K

        Returns
        -------
        density : float
            density of the state in kg/m^3

        """
        self.abstractState.update(CP.PQ_INPUTS, pressure, quality)
        return self.abstractState.cpmass()
    
    def calculateHeatCapacityFromQandT(self, quality, temperature):
        """
        CoolProp wrapper method to calculate the heat capacity of the fluid
        given the pressure and vapor quality

        Parameters
        ----------
        quality : float
            vapor quality of the state, unitless
        temperature : float
            temperature of the state in K

        Returns
        -------
        density : float
            density of the state in kg/m^3

        """
        self.abstractState.update(CP.QT_INPUTS, quality, temperature)
        return self.abstractState.cpmass()


class PropsSIWrapper():
    """
    Wrapper for CoolProp PropsSI class
    """
    def __init__(self, fluid):
        self.fluid = fluid

    def calculatePressureFromTandQ(self, temperature, quality):
        """
        Calculates pressure of saved fluid from temperature and vapor quality

        Parameters
        ----------
        temperature : float
            temperature of fluid in K
        quality : float
            vapor quality of fluid (unitless)

        Returns
        -------
        float
            pressure of fluid in Pa

        """
        return PropsSI('P','T',temperature,'Q',quality, self.fluid)
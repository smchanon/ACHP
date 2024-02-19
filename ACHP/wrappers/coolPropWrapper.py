# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:39:35 2024

@author: SMCANANA
"""
from enum import StrEnum
import logging
import CoolProp as CP
from CoolProp.CoolProp import HAPropsSI, PropsSI

class BackEnd(StrEnum):
    #TODO: add incompressible?
    HEOS = "HEOS"
    TTSEHEOS = "TTSE&HEOS"
    BICUBICHEOS = "BICUBIC&HEOS"
    REFPROP = "REFPROP"
    SRK = "SRK"
    PR = "PR"

class HumidAirPropertiesWrapper():
    """
    Wrapper for HAPropsSI
    """
    def __init__(self):
        self.logger = logging.getLogger("HumidAirPropertiesWrapper")
    
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
    def __init__(self, backEnd: str, fluid: str, massFraction: float=1.0, volumeFraction: float=1.0):
        self.logger = logging.getLogger("AbstractStateWrapper")
        self.name = fluid
        self.abstractState = CP.AbstractState(backEnd, fluid)
        if massFraction < 1.0: self.setMassFraction(massFraction)
        if volumeFraction < 1.0: self.setVolumeFraction(volumeFraction)
        self.propsSIName = f"{backEnd}::{fluid}-{massFraction*100}%"

    def getMeltingTemperature(self, pressure):
        if self.abstractState.has_melting_line():
            return self.abstractState.melting_line(CP.iT, CP.iP, pressure)
        
    def setMassFraction(self, massFraction):
        """
        CoolProp wrapper method to set the mass fraction of an incompressible fluid

        Parameters
        ----------
        massFraction : float
            Fraction of the solution according to its mass.

        Returns
        -------
        None.

        """
        self.massFractions = {self.name: massFraction, 'Water': 1.0 - massFraction}
        self.abstractState.set_mass_fractions([massFraction])

    def setVolumeFraction(self, volumeFraction):
        """
        CoolProp wrapper method to set the volume fraction of an incompressible fluid

        Parameters
        ----------
        volumeFraction : float
            Fraction of the solution according to its volume.

        Returns
        -------
        None.

        """
        self.volumeFractions = {self.name: volumeFraction, 'Water': 1.0 - volumeFraction}
        self.abstractState.set_volu_fractions([volumeFraction])

    def calculateCriticalPressure(self):
        """
        CoolProp wrapper method to calculate the critical pressure of the fluid

        Returns
        -------
        criticalPressure : float
            Critical pressure of the fluid.

        """
        try:
            self.logger.debug("Critical pressure for %s: %g", self.name, self.abstractState.p_critical())
            return self.abstractState.p_critical()
        except ValueError:
            self.logger.info("Could not find critical pressure for %s. Giving critical pressure for water.",
                                  self.name)
            return PropsSIWrapper().calculatePCritical('Water')
    
    def calculateMassMolar(self):
        """
        Calculates the molar mass of the fluid in kg/mol.

        Returns
        -------
        massMolar : float
            molar mass of the fluid in kg/mol.

        """ 
        try:
            return self.abstractState.molar_mass()*1000
        except ValueError:
            self.logger.info("Molar mass cannot be calculated by CoolProp. Calculating molar mass by fraction.")
            propsSI = PropsSIWrapper()
            massMolarWater = propsSI.calculateMolarMass('Water')*self.massFractions['Water']
            if self.name == "MEG": massMolarFluid = 62.068*self.massFractions[self.name]
            else: massMolarFluid = propsSI.calculateMolarMass(self.propsSIName)*self.massFractions[self.name]
            return massMolarFluid + massMolarWater

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

    def calculateTempFromHandP(self, enthalpy, pressure):
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

    def calculateEnthalpyFromDandT(self, density, temperature):
        """
        CoolProp wrapper method to calculate enthalpy of the fluid
        given the density and temperature

        Parameters
        ----------
        density : float
            density of the state in kg/m^3
        temperature : float
            temperature of the state in K

        Returns
        -------
        float
            enthalpy of the state in J/kg

        """
        self.abstractState.update(CP.DmassT_INPUTS, density, temperature)
        return self.abstractState.hmass()

    def calculateEnthalpyFromPandQ(self, pressure, quality):
        """
        CoolProp wrapper method to calculate enthalpy of the fluid
        given the pressure and vapor quality

        Parameters
        ----------
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

    def calculateEnthalpyFromPandS(self, pressure, entropy):
        """
        CoolProp wrapper method to calculate enthalpy of the fluid
        given the pressure and entropy

        Parameters
        ----------
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

    def calculateEntropyFromDandT(self, density, temperature):
        """
        CoolProp wrapper method to calculate the entropy of the fluid
        given the density and temperature

        Parameters
        ----------
        density : float
            density of the state in kg/m^3
        temperature : float
            temperature of the state in K

        Returns
        -------
        entropy : float
            entropy of the state in J/kg/K

        """
        self.abstractState.update(CP.DmassT_INPUTS, density, temperature)
        return self.abstractState.smass()

    def calculateEntropyFromHandP(self, enthalpy, pressure):
        """
        CoolProp wrapper method to calculate the entropy of the fluid
        given the pressure and enthalpy

        Parameters
        ----------
        enthalpy : float
            enthalpy of the state in J/kg
        pressure : float
            pressure of the state in Pa

        Returns
        -------
        entropy : float
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

    def calculateDensityFromHandP(self, enthalpy, pressure):
        """
        CoolProp wrapper method to calculate the density of the fluid
        given the enthalpy and pressure

        Parameters
        ----------
        enthalpy : float
            enthalpy of the state in J/kg
        pressure : float
            pressure of the state in Pa

        Returns
        -------
        density : float
            density of the state in kg/m^3

        """
        self.abstractState.update(CP.HmassP_INPUTS, enthalpy, pressure)
        return self.abstractState.rhomass()

    def calculateDensityFromPandQ(self, pressure, quality):
        """
        CoolProp wrapper method to calculate the density of the fluid
        given the pressure and vapor quality

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
        self.abstractState.update(CP.PQ_INPUTS, pressure, quality)
        return self.abstractState.rhomass()

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
        return self.abstractState.rhomass()

    def calculateDensityFromQandT(self, quality, temperature):
        """
        CoolProp wrapper method to calculate the density of the fluid
        given the quality and temperature

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
        quality : float
            vapor quality of the state, unitless

        Returns
        -------
        heatCapacity : float
            heat capacity/specific heat of the state in J/kg/K

        """
        self.abstractState.update(CP.PQ_INPUTS, pressure, quality)
        return self.abstractState.cpmass()

    def calculateHeatCapacityFromQandT(self, quality, temperature):
        """
        CoolProp wrapper method to calculate the heat capacity of the fluid
        given the vapor quality and temperature

        Parameters
        ----------
        quality : float
            vapor quality of the state, unitless
        temperature : float
            temperature of the state in K

        Returns
        -------
        heatCapacity : float
            heat capacity/specific heat of the state in J/kg/K

        """
        self.abstractState.update(CP.QT_INPUTS, quality, temperature)
        return self.abstractState.cpmass()
    
    def calculatePressureFromQandT(self, quality, temperature):
        """
        CoolProp wrapper method to calculate the pressure of the fluid
        given the vapor quality and temperature

        Parameters
        ----------
        quality : float
            vapor quality of the state, unitless
        temperature : float
            temperature of the state in K

        Returns
        -------
        pressure : float
            pressure of the state in Pa

        """
        self.abstractState.update(CP.QT_INPUTS, quality, temperature)
        return self.abstractState.p()
    
    def calculateSurfaceTensionFromPandQ(self, pressure, quality):
        """
        CoolProp wrapper method to calculate the pressure of the fluid
        given the vapor quality and temperature

        Parameters
        ----------
        quality : float
            vapor quality of the state, unitless
        temperature : float
            temperature of the state in K

        Returns
        -------
        pressure : float
            pressure of the state in Pa

        """
        self.abstractState.update(CP.PQ_INPUTS, pressure, quality)
        return self.abstractState.surface_tension()
    

    def calculateViscosityFromPandT(self, pressure, temperature):
        """
        CoolProp wrapper method to calculate the viscosity of the fluid
        given the pressure and temperature

        Parameters
        ----------
        pressure : float
            pressure of the state in Pa
        temperature : float
            temperature of the state in K

        Returns
        -------
        viscosity : float
            viscosity of the state in Pa*s

        """
        self.abstractState.update(CP.PT_INPUTS, pressure, temperature)
        return self.abstractState.viscosity()

    def calculateViscosityFromQandT(self, quality, temperature):
        """
        CoolProp wrapper method to calculate the viscosity of the fluid
        given the pressure and temperature

        Parameters
        ----------
        quality : float
            vapor quality of the state, unitless
        temperature : float
            temperature of the state in K

        Returns
        -------
        viscosity : float
            viscosity of the state in Pa*s

        """
        self.abstractState.update(CP.QT_INPUTS, quality, temperature)
        return self.abstractState.viscosity()

    def calculateConductivityFromPandT(self, pressure, temperature):
        """
        CoolProp wrapper method to calculate the conductivity of the fluid
        given the pressure and temperature

        Parameters
        ----------
        pressure : float
            pressure of the state in Pa
        temperature : float
            temperature of the state in K

        Returns
        -------
        conductivity : float
            conductivity of the state in W/m/K

        """
        self.abstractState.update(CP.PT_INPUTS, pressure, temperature)
        return self.abstractState.conductivity()


class PropsSIWrapper():
    """
    Wrapper for CoolProp PropsSI class
    """
    def __init__(self):
        self.logger = logging.getLogger("PropsSIWrapper")
    
    def calculatePCritical(self, name):
        return PropsSI("PCRIT", name)
    
    def calculateMolarMass(self, name):
        return PropsSI("MOLARMASS", name)

    def calculateEnthalpyFromPandQ(self, fluid, pressure, quality):
        """
        Calculates enthalpy of saved fluid from temperature and pressure

        Parameters
        ----------
        fluid: string
            name of fluid
        pressure: float
            pressure of fluid in Pa
        quality : float
            vapor quality of fluid (unitless)

        Returns
        -------
        enthalpy: float
            enthalpy of fluid in J/kg

        """
        return PropsSI('H', 'P', pressure, 'Q', quality, fluid.name)
    
    def calculateEnthalpyFromTandP(self, fluid, temperature, pressure):
        """
        Calculates enthalpy of saved fluid from temperature and pressure

        Parameters
        ----------
        fluid: string
            name of fluid
        temperature : float
            temperature of fluid in K
        pressure: float
            pressure of fluid in Pa

        Returns
        -------
        enthalpy: float
            enthalpy of fluid in J/kg

        """
        return PropsSI('H', 'T', temperature, 'P', pressure, fluid.name)
    
    def calculateEnthalpyFromTandQ(self, fluid, temperature, quality):
        """
        Calculates enthalpy of saved fluid from temperature and vapor quality

        Parameters
        ----------
        fluid: string
            name of fluid
        temperature : float
            temperature of fluid in K
        quality : float
            vapor quality of fluid (unitless)

        Returns
        -------
        enthalpy: float
            enthalpy of fluid in J/kg

        """
        return PropsSI('H', 'T', temperature, 'Q', quality, fluid.name)
    
    def calculatePressureFromTandQ(self, fluid, temperature, quality):
        """
        Calculates pressure of saved fluid from temperature and vapor quality

        Parameters
        ----------
        fluid: string
            name of fluid
        temperature : float
            temperature of fluid in K
        quality : float
            vapor quality of fluid (unitless)

        Returns
        -------
        pressure: float
            pressure of fluid in Pa

        """
        return PropsSI('P', 'T', temperature, 'Q', quality, fluid.name)

    def calculateTemperatureFromPandQ(self, fluid, pressure, quality):
        """
        Calculates temperature of saved fluid from pressure and vapor quality

        Parameters
        ----------
        fluid: string
            name of fluid
        pressure : float
            pressure of fluid in Pa
        quality : float
            vapor quality of fluid (unitless)

        Returns
        -------
        temperature : float
            temperature of fluid in K

        """
        return PropsSI('T', 'P', pressure, 'Q', quality, fluid.name)

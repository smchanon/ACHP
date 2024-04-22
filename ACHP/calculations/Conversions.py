'''This module is used to convert values between different units,
   as needed commonly for ACHP'''
from enum import Enum

class TemperatureConversions():
    """
    Class for all temperature unit conversions
    """

    class Unit(Enum):
        """
        Temperature units enum
        """
        K = ("Kelvin", 0)
        C = ("Celsius", -273.15)
        F = ("Fahrenheit", -459.67)

        def __new__(cls, value, minVal):
            obj = object.__new__(cls)
            obj._value_ = value
            obj.minVal = minVal
            return obj

    def __init__(self):
        self.multiplicationFactorF = 9.0/5.0
        self.additionFactorFToC = 32.0

    def _checkForUnrealTemp(self, unit: Unit, temperature):
        """
        Checks if given temperature is possible. If not, raises ValueError

        Parameters
        ----------
        unit : Unit
            temperature unit. One of Kelvin, Celsius, Fahrenheit.
        temperature : float
            DESCRIPTION.

        Raises
        ------
        ValueError
            Temperature is not physically possible.

        Returns
        -------
        None.

        """
        if temperature < unit.minVal:
            raise ValueError(f"Temperature in %{unit.value} cannot be lower than %{unit.minVal}")

    def convertTemperature(self, unitFrom, unitTo, temperature):
        """
        Convert temperature among Celsius, Kelvin, and Fahrenheit

        Parameters
        ----------
        unitFrom : Unit
            temperature unit you are converting from.
        unitTo : Unit
            temperature unit you are converting to.
        temperature : float
            temperature to be converted.

        Returns
        -------
        float
            temperature in the unit you are converting to.

        """
        self._checkForUnrealTemp(unitFrom, temperature)
        if unitFrom == self.Unit.F:
            multiplicationFactor = 1/self.multiplicationFactorF
            if unitTo == self.Unit.C:
                additionFactor = self.additionFactorFToC
            else: additionFactor = unitFrom.minVal
            return multiplicationFactor*(temperature - additionFactor)
        if unitTo == self.Unit.F:
            multiplicationFactor = self.multiplicationFactorF
            if unitFrom == self.Unit.C:
                additionFactor = self.additionFactorFToC
            else: additionFactor = unitTo.minVal
            return multiplicationFactor*temperature + additionFactor
        return temperature + unitTo.minVal - unitFrom.minVal

class MassFlowConversions():
    """
    class for all mass flow unit conversions
    """

    class Unit(Enum):
        """
        Mass flow units enum
        """
        LBH = ("pounds per hour", 3600)
        KGS = ("kgs per second", 0.45359237)
        LBM = ("pounds per minute", 60)

        def __new__(cls, units, factor):
            obj = object.__new__(cls)
            obj._value_ = units
            obj.factor = factor
            return obj

    def convertMassFlow(self, unitFrom: Unit, unitTo: Unit, massFlow):
        """
        Convert mass flow among pounds per hour (LBH), kilograms per second (KGS),
        and pounds per minute (LBM)

        Parameters
        ----------
        unitFrom : Unit
            mass flow unit you are converting from.
        unitTo : Unit
            mass flow unit you are converting to.
        massFlow : float
            mass flow to be converted.

        Returns
        -------
        float
            mass flow in the unit you are converting to.

        """
        return unitTo.factor*massFlow/unitFrom.factor

class VolumetricFlowConversions():
    """
    class for all volumetric flow unit conversions
    """

    class Unit(Enum):
        """
        Volumetric flow units enum
        """
        CFM = ("cubic feet per minute", 0.0283168466)
        CMS = ("cubic meters per second", 60)
        GPM = ("gallons per minute", 264.17287472922)

        def __new__(cls, units, factor):
            obj = object.__new__(cls)
            obj._value_ = units
            obj.factor = factor
            return obj

    def convertVolumetricFlow(self, unitFrom: Unit, unitTo: Unit, volumetricFlow):
        """
        Convert volumetric flow among cubic feet per minute (CFM), cubic meters per second (CMS),
        and gallons per minute (GPM)

        Parameters
        ----------
        unitFrom : Unit
            volumetric flow unit you are converting from.
        unitTo : Unit
            volumetric flow unit you are converting to.
        volumetricFlow : float
            volumetric flow to be converted.

        Returns
        -------
        float
            volumetric flow in the unit you are converting to.

        """
        if unitTo == self.Unit.GPM:
            return unitFrom.factor*volumetricFlow*unitTo.factor
        if unitFrom == self.Unit.GPM:
            return volumetricFlow/(unitTo.factor*unitFrom.factor)
        return unitFrom.factor*volumetricFlow/unitTo.factor

class PowerConversions():
    """
    Class for all power unit conversions
    """

    class Unit(Enum):
        """
        Power units enum
        """
        BTUH = "Btu/h"
        W = "Watts"
        HP = "Horsepower"

    def __init__(self):
        self.conversions = {(self.Unit.BTUH, self.Unit.W): 0.2930710702,
                            (self.Unit.HP, self.Unit.W): 745.699872,
                            (self.Unit.HP, self.Unit.BTUH): 2544.4336113065}

    def convertPower(self, unitFrom: Unit, unitTo: Unit, power):
        """
        Convert power among Btu/h (BTUH), Watts (W),
        and horsepower (HP)

        Parameters
        ----------
        unitFrom : Unit
            power unit you are converting from.
        unitTo : Unit
            power unit you are converting to.
        power : float
            power in the unit you are converting from.

        Returns
        -------
        float
            power in the unit you are converting to.

        """
        if (unitFrom, unitTo) in self.conversions:
            return power*self.conversions[(unitFrom, unitTo)]
        return power/self.conversions[(unitTo, unitFrom)]

class PressureConversions():
    """
    class for all pressure unit conversions
    """
    class Unit(Enum):
        """
        Pressure units enum
        """
        PA = "Pascal"
        PSI = "psi"
        BAR = "bar"

    def __init__(self):
        self.conversions = {(self.Unit.BAR, self.Unit.PA): 1e5,
                       (self.Unit.PSI, self.Unit.PA): 6894.757293,
                       (self.Unit.BAR, self.Unit.PSI): 14.503773773}

    def convertPressure(self, unitFrom: Unit, unitTo: Unit, pressure):
        """
        Convert pressure among Pascals (PA), bar (BAR), and PSI

        Parameters
        ----------
        unitFrom : Unit
            pressure unit you are converting from (PA, BAR, PSI).
        unitTo : Unit
            pressure unit you are converting to (PA, BAR, PSI).
        pressure : float
            pressure in the unit you are converting from.

        Returns
        -------
        pressure
            pressure in the unit you are converting to.

        """
        if (unitFrom, unitTo) in self.conversions:
            return pressure*self.conversions[(unitFrom, unitTo)]
        return pressure/self.conversions[(unitTo, unitFrom)]

class GeometricConversions():
    """
    Class for all geometric unit conversions
    """
    class Unit(Enum):
        """
        Geometric units enum
        """
        M = "meters"
        IN = "inches"
        FT = "feet"

    def __init__(self):
        self.conversions = {(self.Unit.IN, self.Unit.M): 0.0254,
                            (self.Unit.FT, self.Unit.M): 0.3048,
                            (self.Unit.FT, self.Unit.IN): 12}

    def convertLength(self, unitFrom: Unit, unitTo: Unit, length):
        """
        Convert length among meters (M), inches (IN), and feet (FT)

        Parameters
        ----------
        unitFrom : Unit
            length unit you are converting from (M, IN, FT).
        unitTo : Unit
            length unit you are converting from (M, IN, FT).
        length : float
            length in the unit you are converting from.

        Returns
        -------
        float
            length in the unit you are converting to.

        """
        if (unitFrom, unitTo) in self.conversions:
            return length*self.conversions[(unitFrom, unitTo)]
        return length/self.conversions[(unitTo, unitFrom)]

    def convertArea(self, unitFrom: Unit, unitTo: Unit, area):
        """
        Convert area among meters^2 (M), inches^2 (IN), and feet^2 (FT)

        Parameters
        ----------
        unitFrom : Unit
            area unit you are converting from (M, IN, FT).
        unitTo : Unit
            area unit you are converting to (M, IN, FT).
        area : float
            area in the unit you are converting from.

        Returns
        -------
        float
            area in the unit you are converting to.

        """
        if (unitFrom, unitTo) in self.conversions:
            return area*self.conversions[(unitFrom, unitTo)]**2
        return area/self.conversions[(unitTo, unitFrom)]**2

    def convertVolume(self, unitFrom: Unit, unitTo: Unit, volume):
        """
        Convert area among meters^3 (M), inches^3 (IN), and feet^3 (FT)

        Parameters
        ----------
        unitFrom : Unit
            volume unit you are converting from (M, IN, FT).
        unitTo : Unit
            volume unit you are converting to (M, IN, FT).
        volume : TYPE
            volume in the unit you are converting from.

        Returns
        -------
        float
            volume in the unit you are converting to.

        """
        if (unitFrom, unitTo) in self.conversions:
            return volume*self.conversions[(unitFrom, unitTo)]**3
        return volume/self.conversions[(unitTo, unitFrom)]**3

class MassConversions():
    """
    Class for all mass unit conversions
    """

    class Unit(Enum):
        """
        Mass units enum
        """
        OZ = "ounces"
        KG = "kilograms"

    def __init__(self):
        self.conversions = {(self.Unit.OZ, self.Unit.KG): 0.0283495}

    def convertMass(self, unitFrom: Unit, unitTo: Unit, mass):
        """
        Convert mass units among ounces (OZ), kilograms (KG)

        Parameters
        ----------
        unitFrom : Unit
            mass unit you are converting from (OZ, KG).
        unitTo : Unit
            mass unit you are converting to (OZ, KG).
        mass : TYPE
            mass in the unit you are converting from.

        Returns
        -------
        float
            mass in the unit you are converting to.

        """
        if (unitFrom, unitTo) in self.conversions:
            return mass*self.conversions[(unitFrom, unitTo)]
        return mass/self.conversions[(unitTo, unitFrom)]

class ComposedPropertyConversions():
    """
    Class for all composed property conversions
    """

    class Unit(Enum):
        """
        Composed property units enum
        """
        IPK = "imperial K-value"
        SIK = "SI K-value"

    def __init__(self):
        self.conversions = {(self.Unit.IPK, self.Unit.SIK): 1.730735}

    def convertComposedProperty(self, unitFrom: Unit, unitTo: Unit, composedProperty):
        """
        Convert composed property units among imperial K-value (IPK), SI K-value (SIK)

        Parameters
        ----------
        unitFrom : Unit
            composed property unit you are converting from (IPK, SIK).
        unitTo : Unit
            composed property unit you are converting to (IPK, SIK).
        mass : TYPE
            composed property in the unit you are converting from.

        Returns
        -------
        float
            composed property in the unit you are converting to.

        """
        if (unitFrom, unitTo) in self.conversions:
            return composedProperty*self.conversions[(unitFrom, unitTo)]
        return composedProperty/self.conversions[(unitTo, unitFrom)]

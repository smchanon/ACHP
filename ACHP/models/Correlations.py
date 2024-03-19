# -*- coding: utf-8 -*-
import logging
from scipy.integrate import quad,simps
from scipy.constants import g
import numpy as np
import CoolProp as CP
from ACHP.models.Fluid import ThermoProps

try:
    import psyco
    psyco.full()
except ImportError:
    pass

#Machine precision
machineEps = np.finfo(np.float64).eps

class FluidMechanics():
    """
    Class for calculations of dimensionless numbers used to characterize fluids
    """
    def calculateReynoldsNumber(flowSpeed, length, viscosity, density=None):
        """
        Calculates the Reynolds number of a fluid given its flow speed, characteristic length,
        viscosity, and density. For the sake of conciseness, the kinematic and dynamic
        viscosities have both been labeled viscosity.

        Parameters
        ----------
        flowSpeed : float
            flow speed of the fluid in m/s.
        length : float
            characteristic length of the fluid in m.
        viscosity : float
            for the case where there is no density, the kinematic viscosity of the fluid
            in kg/(m·s). For the case where density is included, the dynamic viscosity
            of the fluid in m^s/s
        density : float, optional
            density of the fluid in kg/m^3. The default is None.

        Returns
        -------
        reynoldsNum: float
            Reynolds number of the fluid.

        """
        if not density:
            return flowSpeed*length/viscosity
        return density*flowSpeed*length/viscosity

    def calculatePrandtlNumber(specificHeat, viscosity, conductivity):
        """
        Calculates the Prandtl number of a fluid given its specific heat, viscosity and conductivity

        Parameters
        ----------
        specificHeat : float
            specific heat of fluid.
        viscosity : float
            viscosity of fluid.
        conductivity : float
            thermal conductivity of fluid.

        Returns
        -------
        prandtlNum: float
            Prandtl number of the fluid.

        """
        return specificHeat*viscosity/conductivity

    def calculateNusseltNumber(frictionFactorDarcy, reynoldsNum, prandtlNum, frictionFactor=None):
        """
        Calculates the Nusselt number of a fluid give a calculated Darcy friction factor,
        Reynolds number, and Prandtl number

        Parameters
        ----------
        frictionFactorDarcy : float
            Darcy friction factor.
        reynoldsNum : float
            Reynolds number of fluid.
        prandtlNum : float
            Prandtl number of fluid.

        Returns
        -------
        nusseltNum : float
            Nusselt number of fluid (dimensionless).

        """
        if not frictionFactor:
            return (frictionFactorDarcy/8.0)*(reynoldsNum - 1000)*prandtlNum/\
                (1 + 12.7*np.sqrt(frictionFactor/8.0)*(prandtlNum**(2/3) - 1))
        return (frictionFactorDarcy/8.0)*reynoldsNum*prandtlNum/\
                (1.07 + 12.7*np.sqrt(frictionFactor/8.0)*(prandtlNum**(2/3) - 1))

    def calculateChurchillFrictionFactor(reynoldsNum):
        """
        Calculates friction factor of Churchill (Darcy Friction factor where f_laminar=64/reynoldsNum).

        Parameters
        ----------
        reynoldsNum : float
            Reynolds number of fluid.

        Returns
        -------
        churchillFrictionFactor: float
            Churchill friction factor.

        """
        roughnessRatio = 0.0 #pipe roughness divided by pipe diameter
        varA = ((-2.457*np.log((7.0/reynoldsNum)**(0.9) + 0.27*roughnessRatio)))**16
        varB = (37530.0 / reynoldsNum)**16
        return 8.0*((8.0/reynoldsNum)**12.0 + 1/(varA + varB)**(1.5))**(1/12)

    def calculateGrashofNumber(beta, tempSurface, tempBulk, length, viscosity):
        """
        Calculates the Grashof number of a fluid.

        Parameters
        ----------
        beta : float
            coefficient of volume expansion.
        tempSurface : float
            surface temperature.
        tempBulk : float
            fluid bulk temperature.
        length : float
            length.
        viscosity : float
            kinematic viscosity.

        Returns
        -------
        grashofNum : float
            Grashof number of fluid.

        """
        grashofNum = g*beta*(tempSurface-tempBulk)*length**3/\
                viscosity**2
        return grashofNum

def getPhaseFromPandH(fluid,pressure,enthalpy,tBubble,tDew,rhosatL,rhosatV):
    """
    Convenience function to return just the Phase without temperature or density
    """
    return getTempDensityPhaseFromPandH(fluid,pressure,enthalpy,tBubble,tDew,rhosatL,rhosatV)[2]

def getTempDensityPhaseFromPandH(fluid,pressure,enthalpy,tBubble,tDew,rhosatL=None,rhosatV=None):
    """
    Convenience function to find temperature, density, and phase of fluid as a
    function of pressure and enthalpy
    """
    logger = logging.getLogger("Correlations")
    logger.debug("fluid %s backend: %s", fluid.name, fluid.backEnd,
                 extra={"methodname": "getTempDensityPhaseFromPandH"})
    if 'incomp' in fluid.backEnd.lower() or pressure >= fluid.pressureCritical:
        temperature = fluid.calculateTemperature(ThermoProps.HP, enthalpy, pressure)
        density = fluid.calculateDensity(ThermoProps.HP, enthalpy, pressure)
        if 'incomp' in fluid.backEnd.lower():
            return temperature, density, 'Subcooled'
        if temperature >= fluid.calculateCriticalTemperature():
            return temperature, density, 'Supercritical'
        return temperature, density, 'Supercrit_liq'
    if not rhosatL:
        rhosatL = fluid.calculateDensity(ThermoProps.QT, 0.0, tBubble)
        rhosatV = fluid.calculateDensity(ThermoProps.QT, 1.0, tDew)
    hsatL = fluid.calculateEnthalpy(ThermoProps.DT, rhosatL, tBubble)
    hsatV = fluid.calculateEnthalpy(ThermoProps.DT, rhosatV, tDew)
    if enthalpy > hsatV or enthalpy < hsatL:
        temperature = fluid.calculateTemperature(ThermoProps.HP, enthalpy, pressure)
        density = fluid.calculateDensity(ThermoProps.HP, enthalpy, pressure)
        if enthalpy > hsatV:
            return temperature,density,'Superheated'
        return temperature,density,'Subcooled'
    fraction = (enthalpy - hsatL)/(hsatV - hsatL) #[-]
    volumeSpecific = fraction/rhosatV + (1 - fraction)/rhosatL #[m^3/kg]
    temperature = fraction*tDew + (1 - fraction)*tBubble #[K]
    density = 1/volumeSpecific #[kg/m^3]
    return temperature, density, 'TwoPhase'

def twoPhaseDensity(fluid, xMin, xMax, tDew, tBubble, slipModel='Zivi'):
    """
    function to obtain the average density in the two-phase region
    """
    rhog = fluid.calculateDensity(ThermoProps.QT, 1.0, tDew)
    rhof = fluid.calculateDensity(ThermoProps.QT, 0.0, tBubble)
    if slipModel == 'Zivi':
        sVal = pow(rhof/rhog,0.3333)
    elif slipModel == 'Homogeneous':
        sVal = 1
    else:
        raise ValueError("slipModel must be either 'Zivi' or 'Homogeneous'")
    cVal = sVal*rhog/rhof
    if xMin + 5*machineEps < 0 or xMax - 10*machineEps > 1.0:
        raise ValueError('Quality must be between 0 and 1, ' + 'xMin: ' + str(xMin + 5*machineEps) +
                         ', xMax: ' + str(xMax - 10*machineEps))
    #Avoid the zero and one qualities (undefined integral)
    if xMin == xMax:
        alphaAverage = 1/(1 + cVal*(1 - xMin)/xMin)
    else:
        if xMin >= 1.0:
            alphaAverage = 1.0
        elif xMax <= 0.0:
            alphaAverage = 0.0
        else:
            alphaAverage =- (cVal*(np.log(((xMax - 1.0)*cVal - xMax)/((xMin - 1.0)*cVal - xMin)) +\
                                   xMax - xMin) - xMax + xMin)/(cVal**2 - 2*cVal + 1)/(xMax - xMin)
    return alphaAverage*rhog + (1 - alphaAverage)*rhof

def calculateAccelerationalPressureDrop(xMin, xMax, fluid, massFlux, tBubble, tDew, diameter=None,
                                        rhosatL=None, rhosatV=None, slipModel='Zivi'):
    """
    Accelerational pressure drop

    From -dpdz|A=massFlux^2*d[x^2v_g/alpha+(1-x)^2*v_f/(1-alpha)]/dz

    Integrating over z from 0 to L where x=x_1 at z=0 and x=x_2 at z=L

    Maxima code:
        alpha:1/(1+slip*rho_g/rho_f*(1-x)/x)$
        num1:x^2/rho_g$
        num2:(1-x)^2/rho_f$
        subst(num1/alpha+num2/(1-alpha),x,1);
        subst(num1/alpha+num2/(1-alpha),x,0);
    """
    if rhosatL is None or rhosatV is None:
        rhosatV = fluid.calculateDensity(ThermoProps.QT, 1.0, tDew)
        rhosatL = fluid.calculateDensity(ThermoProps.QT, 0.0, tBubble)

    def slipFunction(xVal):
        if abs(xVal) < 1e-12:
            return 1/rhosatL
        if abs(1 - xVal) < 1e-12:
            return 1/rhosatV
        if slipModel == 'Premoli':
            slip = calculatePremoliSlipFlowFactor(xVal,fluid,massFlux,diameter,tBubble,tDew,rhosatL,rhosatV)
        elif slipModel == 'Zivi':
            slip = pow(rhosatL/rhosatV,1/3)
        elif slipModel == 'Homogeneous':
            slip = 1
        else:
            raise ValueError("slipModel must be either 'Premoli', 'Zivi' or 'Homogeneous'")
        alpha = 1/(1 + slip*rhosatV/rhosatL*(1 - xVal)/xVal)
        return xVal**2/rhosatV/alpha + (1 - xVal)**2/rhosatL/(1 - alpha)

    return massFlux**2*(slipFunction(xMin)-slipFunction(xMax))

def calculatePremoliSlipFlowFactor(xVal,fluid,massFlux,diameter,tBubble,tDew,rhoL=None,rhoV=None):
    '''
    return Premoli (1970) slip flow factor
    function copied from ACMODEL souce code
    same correlations can be found in the Appendix A2 of Petterson (2000)
    '''
    if rhoL is None or rhoV is None:
        rhoV = fluid.calculateDensity(ThermoProps.QT, 1.0, tDew)
        rhoL = fluid.calculateDensity(ThermoProps.QT, 0.0, tBubble)
    muL = fluid.calculateViscosity(ThermoProps.QT, 0.0, tBubble)
    psat = fluid.calculatePressure(ThermoProps.QT, 0.0, tBubble)
    sigma = fluid.calculateSurfaceTension(ThermoProps.PQ, psat, xVal)
    densityFraction = rhoV/rhoL
    weberNum = pow(massFlux,2)*diameter/(sigma*rhoL)
    reynoldsNum = FluidMechanics.calculateReynoldsNumber(massFlux, diameter, muL)
    fParam1 = 1.578*pow(reynoldsNum,-0.19)*pow(densityFraction,-0.22)
    fParam2 = 0.0273*weberNum*pow(reynoldsNum,-0.51)*pow(densityFraction,0.08)
    yParam = (xVal/(1 - xVal))/densityFraction
    #TODO: check this is to the 0.5 power
    slip = 1 + fParam1*pow((yParam/(1 + fParam2*yParam) - fParam2*yParam),0.5)
    return slip

def lmPressureGradientAvg(xMin, xMax, fluid, massFlux, diameter, tBubble,tDew,coeff=None,satTransport=None):
    """
    Returns the average pressure gradient between qualities of xMin and xMax.

    To obtain the pressure gradient for a given value of x, pass it in as xMin and xMax

    Required parameters:
    * xMin : The minimum quality for the range [-]
    * xMax : The maximum quality for the range [-]
    * fluid : fluid to do the calculation on
    * massFlux : Mass flux [kg/m^2/s]
    * diameter : Diameter of tube [m]
    * tBubble : Bubblepoint temperature of refrigerant [K]
    * tDew : Dewpoint temperature of refrigerant [K]

    Optional parameters:
    * coeff : The coefficient in the pressure drop
    * satTransport : A dictionary with the keys 'mu_f','mu_g,'v_f','v_g' for the saturation
    properties. So they can be calculated once and passed in for a slight improvement in efficiency
    """
    ## Use Simpson's Rule to calculate the average pressure gradient
    ## Can't use adapative quadrature since function is not sufficiently smooth
    ## Not clear why not sufficiently smooth at x>0.9
    if xMin == xMax:
        return lockhartMartinelli(fluid,massFlux,diameter,xMin,tBubble,tDew,coeff,satTransport)[0]
    #Calculate the tranport properties once
    satTransport = {'volumeSpecificLiquid': fluid.calculateDensity(ThermoProps.QT, 0.0, tBubble),
                  'viscosityLiquid': fluid.calculateViscosity(ThermoProps.QT, 0.0, tBubble),
                  'volumeSpecificVapor': 1/fluid.calculateDensity(ThermoProps.QT, 1.0, tDew),
                  'viscosityVapor': fluid.calculateViscosity(ThermoProps.QT, 1.0, tDew)}
    xx = np.linspace(xMin, xMax, 30)
    pressureDrop = np.zeros_like(xx)
    for i, xxVal in enumerate(xx):
        pressureDrop[i] = lockhartMartinelli(fluid, massFlux, diameter, xxVal, tBubble, tDew,
                                             coeff, satTransport)[0]
    return -simps(pressureDrop,xx)/(xMax-xMin)

def lockhartMartinelli(fluid, massFlux, diameter, quality, tBubble, tDew, constant=None, satTransport=None):
    """
    Following the method laid out in ME506 notes on separated Flow pressure drop calculations

    Parameters
    ----------
    fluid : TYPE
        DESCRIPTION.
    massFlux : TYPE
        DESCRIPTION.
    diameter : TYPE
        DESCRIPTION.
    quality : TYPE
        DESCRIPTION.
    tBubble : TYPE
        DESCRIPTION.
    tDew : TYPE
        DESCRIPTION.
    constant : TYPE, optional
        DESCRIPTION. The default is None.
    satTransport : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    #Convert the quality, which might come in as a single numpy float value, to a float
    #With the conversion, >20x speedup in the lockhartMartinelli function, not clear why
    quality = float(quality)

    def calculatePressureGradient(phase, fraction):
        if phase == 'Liquid':
            qualitySat, temperature = 0.0, tBubble
        else:
            qualitySat, temperature = 1.0, tDew
        if satTransport is None:
            volumeSpecific = 1/fluid.calculateDensity(ThermoProps.QT, qualitySat, temperature)
            viscosity = fluid.calculateViscosity(ThermoProps.QT, qualitySat, temperature)
        else:
            volumeSpecific = satTransport[f"volumeSpecific{phase}"]
            viscosity = satTransport[f"viscosity{phase}"]
        reynoldsNum = FluidMechanics.calculateReynoldsNumber(massFlux*fraction, diameter, viscosity)
        if quality == 1.0 - qualitySat:
            frictionFactor = 0
        elif reynoldsNum < 1000:
            frictionFactor = 16.0/reynoldsNum
        elif reynoldsNum > 2000:
            frictionFactor = 0.046/(reynoldsNum**0.2)
        else:
            weight = (reynoldsNum-1000)/(2000-1000)
            frictionFactor = (1 - weight)*16.0/reynoldsNum + weight*0.046/(reynoldsNum**0.2)
        return reynoldsNum, 2*frictionFactor*massFlux**2*fraction**2*volumeSpecific/diameter

    for phase in ['Liquid', 'Vapor']:
        if phase == 'Liquid':
            reynoldsNumLiq, pressureGradientLiq = calculatePressureGradient(phase, 1 - quality)
        if phase == 'Vapor':
            reynoldsNumVap, pressureGradientVap = calculatePressureGradient(phase, quality)
    if quality <= 0:
        return pressureGradientLiq, 0.0
    if quality >= 1:
        return pressureGradientVap, 1.0
    # Lockhart-Martinelli parameter
    paramLM = np.sqrt(pressureGradientLiq/pressureGradientVap)
    # Find the Constant based on the flow reynoldsNum of each phase
    #    (using 1500 as the transitional reynoldsNum to ensure continuity)
    #Calculate constant if not passed in:
    if constant is None:
        if reynoldsNumLiq > 1500:
            constant = 20.0 if reynoldsNumVap > 1500 else 10.0
        else:
            constant = 12.0 if reynoldsNumVap > 1500 else 5.0
    # Two-phase multipliers for each phase
    phiVap2 = 1 + constant*paramLM + paramLM**2
    phiLiq2 = 1 + constant/paramLM + 1/paramLM**2
    # Find gradient
    dpdz = max(pressureGradientVap*phiVap2, pressureGradientLiq*phiLiq2)
    # Void Fraction
    alpha = 1 - paramLM/np.sqrt(paramLM*paramLM + 20*paramLM + 1)
    return dpdz, alpha

def ShahEvaporation_Average(xMin,xMax,fluid,massFlux,diameter,q_flux,tBubble,tDew):
    """
    Returns the average heat transfer coefficient between qualities of xMin and xMax.

    Required parameters:
    * xMin : The minimum quality for the range [-]
    * xMax : The maximum quality for the range [-]
    * AS : AbstractState with the refrigerant name and backend
    * massFlux : Mass flux [kg/m^2/s]
    * diameter : Diameter of tube [m]
    * q_flux : Heat transfer flux [W/m^2]
    * tBubble : Bubblepoint temperature of refrigerant [K]
    * tDew : Dewpoint temperature of refrigerant [K]
    """
    AS = fluid.abstractState.abstractState
    # ********************************
    #        Necessary Properties
    # ********************************
    AS.update(CP.QT_INPUTS,0.0,tBubble)
    rho_f = AS.rhomass() #[kg/m^3]
    mu_f = AS.viscosity() #[Pa-s OR kg/m-s]
    cp_f = AS.cpmass() #[J/kg-K]
    k_f = AS.conductivity() #[W/m-K]
    h_l = AS.hmass() #[J/kg]

    AS.update(CP.QT_INPUTS,1.0,tDew)
    rho_g = AS.rhomass() #[kg/m^3]
    mu_g = AS.viscosity() #[Pa-s OR kg/m-s]
    cp_g = AS.cpmass() #[J/kg-K]
    k_g = AS.conductivity() #[W/m-K]
    h_v = AS.hmass() #[J/kg]

    h_fg = h_v - h_l #[J/kg]
    Pr_f = cp_f * mu_f / k_f #[-]
    Pr_g = cp_g * mu_g / k_g #[-]

    g_grav = 9.81 #[m/s^2]

    # Shah evaporation correlation
    Fr_L = massFlux**2 / (rho_f*rho_f * g_grav * diameter) #[-]
    Bo = q_flux / (massFlux * h_fg) #[-]

    if Bo < 0:
        raise ValueError('Heat flux for Shah Evaporation must be positive')

    if Bo > 0.0011:
        F = 14.7
    else:
        F = 15.43
    #Pure vapor single-phase heat transfer coefficient
    h_g = 0.023 * (massFlux*diameter/mu_g)**(0.8) * Pr_g**(0.4) * k_g / diameter #[W/m^2-K]
    def ShahEvaporation(x):
        if abs(1-x)<5*machineEps:
            return h_g

        #If the quality is above 0.999, linearly interpolate to avoid division by zero
        if x>0.999:
            h_1=ShahEvaporation(1.0) #Fully fry
            h_999=ShahEvaporation(0.999) #At a quality of 0.999
            return (h_1-h_999)/(1.0-0.999)*(x-0.999)+h_999 #Linear interpolation
        if abs(x) < 5*machineEps:
            h_L = 0.023 * (massFlux*(1 - x)*diameter/mu_f)**(0.8) * Pr_f**(0.4) * k_f / diameter #[W/m^2-K]
            return h_L
        h_L = 0.023 * (massFlux*(1 - x)*diameter/mu_f)**(0.8) * Pr_f**(0.4) * k_f / diameter #[W/m^2-K]
        Co = (1 / x - 1)**(0.8) * (rho_g / rho_f)**(0.5) #[-]

        if Fr_L >= 0.04:
            N = Co
        else:
            N = 0.38 * Fr_L**(-0.3) * Co

        psi_cb = 1.8 / N**(0.8)
        if (0.1 < N and N <= 1.0):
            psi_bs = F * (Bo)**(0.5) * np.exp(2.74 * N**(-0.1))
            psi = max([psi_bs, psi_cb])
        else:
            if N > 1.0:
                if Bo > 0.00003:
                    psi_nb = 230 * (Bo)**(0.5)
                else:
                    psi_nb = 1.0 + 46.0 * (Bo)**(0.5)
                psi = max([psi_nb,psi_cb])
            else:
                psi_bs = F * (Bo)**(0.5) * np.exp(2.47 * N**(-0.15))
                psi = max([psi_bs, psi_cb])
        return psi * h_L #[W/m^2-K]

    #Calculate h over the range of x
    x=np.linspace(xMin,xMax,100)
    h=np.zeros_like(x)
    for i, xVal in enumerate(x):
        h[i]=ShahEvaporation(xVal)

    #if xMin == xMax, or they are really really close to being the same
    if abs(xMax-xMin)<5*machineEps:
        #return just one of the edge values
        return h[0]
    #Use Simpson's rule to carry out numerical integration to get average
    return simps(h,x)/(xMax-xMin)

def KandlikarEvaporation_average(xMin,xMax,fluid,massFlux,diameter,q_flux,tBubble,tDew):
    """
    Kandlikar (1990) recommended by Petterson et al. (2000) for CO2, Heat transfer and pressure
    drop for flow supercritical and subcritical CO2 in microchannel tubes
    All details for this correlation are available in Ding Li Thesis (Appendix C):
    "INVESTIGATION OF AN EJECTOR-EXPANSION DEVICE IN A TRANSCRITICAL CARBON DIOXIDE CYCLE FOR
    MILITARY ECU APPLICATIONS"

    Returns the average heat transfer coefficient between qualities of xMin and xMax.

    Required parameters:
    * xMin : The minimum quality for the range [-]
    * xMax : The maximum quality for the range [-]
    * AS : AbstractState with the refrigerant name and backend
    * massFlux : Mass flux [kg/m^2/s]
    * diameter : Diameter of tube [m]
    * q_flux : Heat transfer flux [W/m^2]
    * tBubble : Bubblepoint temperature of refrigerant [K]
    * tDew : Dewpoint temperature of refrigerant [K]
    """
    AS = fluid.abstractState.abstractState
    # ********************************
    #        Necessary Properties
    # ********************************
    AS.update(CP.QT_INPUTS,0.0,tBubble)
    rho_f = AS.rhomass() #[kg/m^3]
    mu_f = AS.viscosity() #[Pa-s OR kg/m-s]
    cp_f = AS.cpmass() #[J/kg-K]
    k_f = AS.conductivity() #[W/m-K]
    h_l = AS.hmass() #[J/kg]

    AS.update(CP.QT_INPUTS,1.0,tDew)
    rho_g = AS.rhomass() #[kg/m^3]
    mu_g = AS.viscosity() #[Pa-s OR kg/m-s]
    cp_g = AS.cpmass() #[J/kg-K]
    k_g = AS.conductivity() #[W/m-K]
    h_v = AS.hmass() #[J/kg]

    h_fg = h_v - h_l #[J/kg]
    Pr_f = cp_f * mu_f / k_f #[-]
    Pr_g = cp_g * mu_g / k_g #[-]

    g_grav = 9.81 #[m/s^2]

    # Petterson evaporation correlation
    Fr_L = massFlux**2 / (rho_f*rho_f * g_grav * diameter) #[-]
    Bo = q_flux / (massFlux * h_fg) #[-]

    if Bo < 0:
        raise ValueError('Heat flux for Petterson Evaporation must be positive')

    F_fl = 1 #Forster and Zuber multiplier depend on fluid float. CO2 is not available, therefore F_fl=1 (for water) is selected.

    #Kandlikar correlation constants for CO2
    c_c_1 = 1.1360
    c_c_2 = -0.9
    c_c_3 = 667.2
    c_c_4 = 0.7
    c_n_1 = 0.6683
    c_n_2 = -0.2
    c_n_3 = 1058.0
    c_n_4 = 0.7
    if Fr_L > 0.4:
        c_c_5 = 0.0
        c_n_5 = 0.0
    else:
        c_c_5 = 0.3
        c_n_5 = 0.3

    #Pure vapor single-phase heat transfer coefficient
    h_g = 0.023 * (massFlux*diameter/mu_g)**(0.8) * Pr_g**(0.4) * k_g / diameter #[W/m^2-K]

    def KandlikarEvaporation(x):
        if abs(1-x)<5*machineEps:
            return h_g

        #If the quality is above 0.999, linearly interpolate to avoid division by zero
        if x>0.999:
            h_1=KandlikarEvaporation(1.0) #Fully fry
            h_999=KandlikarEvaporation(0.999) #At a quality of 0.999
            return (h_1-h_999)/(1.0-0.999)*(x-0.999)+h_999 #Linear interpolation
        if abs(x)<5*machineEps:
            h_L = 0.023 * (massFlux*(1 - x)*diameter/mu_f)**(0.8) * Pr_f**(0.4) * k_f / diameter #[W/m^2-K]
            return h_L
        h_L = 0.023 * (massFlux*(1 - x)*diameter/mu_f)**(0.8) * Pr_f**(0.4) * k_f / diameter #[W/m^2-K]

        Co = (1 / x - 1)**(0.8) * (rho_g / rho_f)**(0.5) #[-]

        #HTC due to convective boiling
        h_c = h_L*(c_c_1*pow(Co, c_c_2)*pow((25.0*Fr_L), c_c_5) + c_c_3*pow(Bo, c_c_4)*F_fl)
        #HTC due to nucleate boiling
        h_n = h_L*(c_n_1*pow(Co, c_n_2)*pow((25.0*Fr_L), c_n_5) + c_n_3*pow(Bo, c_n_4)*F_fl)

        #This was found in ACCO2 model, however Petterson (2000) recommends to take the max of h_n and h_c
        #if (Co < 0.65):
        #    h = h_c
        #else:
        #    h = h_n
        h = max(h_c,h_n)
        return h

    #Calculate h over the range of x
    x=np.linspace(xMin,xMax,100)
    h=np.zeros_like(x)
    for i, xVal in x:
        h[i]=KandlikarEvaporation(xVal)

    #if xMin == xMax, or they are really really close to being the same
    if abs(xMax-xMin)<5*machineEps:
        #return just one of the edge values
        return h[0]
    #Use Simpson's rule to carry out numerical integration to get average
    return simps(h,x)/(xMax-xMin)

def LongoCondensation(x_avg,massFlux,diameterHydraulic,fluid,TsatL,TsatV):

    AS = fluid.abstractState.abstractState
    AS.update(CP.QT_INPUTS,1.0,TsatV)
    rho_V = AS.rhomass() #[kg/m^3]
    AS.update(CP.QT_INPUTS,0.0,TsatL)
    rho_L = AS.rhomass() #[kg/m^3]
    mu_L = AS.viscosity() #[Pa-s OR kg/m-s]
    cp_L = AS.cpmass() #[J/kg-K]
    k_L = AS.conductivity() #[W/m-K]
    Pr_L = cp_L * mu_L / k_L #[-]

    Re_eq=massFlux*((1-x_avg)+x_avg*np.sqrt(rho_L/rho_V))*diameterHydraulic/mu_L

    if Re_eq<1750:
        Nu=60*Pr_L**(1/3)
    else:
        Nu=((75-60)/(3000-1750)*(Re_eq-1750)+60)*Pr_L**(1/3)
    h=Nu*k_L/diameterHydraulic
    return h

def ShahCondensation_Average(xMin,xMax,fluid,massFlux,D,pressure,TsatL):
    # ********************************
    #        Necessary Properties
    #    Calculated outside the quadrature integration for speed
    # ********************************
    AS = fluid.abstractState.abstractState
    AS.update(CP.QT_INPUTS,0.0,TsatL)
    mu_f = AS.viscosity() #[Pa-s OR kg/m-s]
    cp_f = AS.cpmass() #[J/kg-K]
    k_f = AS.conductivity() #[W/m-K]
    Pr_f = cp_f * mu_f / k_f #[-]
    pcrit = AS.p_critical() #[Pa]
    Pstar = pressure / pcrit
    h_L = 0.023 * (massFlux*D/mu_f)**(0.8) * Pr_f**(0.4) * k_f / D #[W/m^2-K]
    def ShahCondensation(x):
        return h_L * ((1 - x)**(0.8) + (3.8 * x**(0.76) * (1 - x)**(0.04)) / (Pstar**(0.38)) )

    if not xMin==xMax:
        return quad(ShahCondensation,xMin,xMax)[0]/(xMax-xMin)
    return ShahCondensation(xMin)

def PettersonSupercriticalAverage(tempOut, tempIn, tempWall, fluid, massFluxAverage, diamOuter,
                                  diamHydraulicOverLength, pressure, heatFluxWall, diamInner=0, massFlow=0):
    '''
    Petterson et al. (2000), Heat transfer and pressure drop for flow supercritical and subcritical
    CO2 in microchannel tubes
    All details for this correlation are available in Ding Li Thesis (Appendix B):
    "INVESTIGATION OF AN EJECTOR-EXPANSION DEVICE IN A TRANSCRITICAL CARBON DIOXIDE CYCLE FOR
    MILITARY ECU APPLICATIONS"
    '''

    def SuperCriticalCondensation_h(temperature, tempWall, fluid, massFluxAverage, diamOuter,
                                    diamHydraulicOverLength, pressure, heatFluxWall, diamInner, massFlow):
        '''return enthalpy value'''
        return PettersonSupercritical(temperature, tempWall, fluid, massFluxAverage, diamOuter,
                                      diamHydraulicOverLength, pressure, heatFluxWall, diamInner, massFlow)[0]
    def SuperCriticalCondensation_f(temperature, tempWall, fluid, massFluxAverage, diamOuter,
                                    diamHydraulicOverLength, pressure, heatFluxWall, diamInner, massFlow):
        '''return frictionFactor value'''
        return PettersonSupercritical(temperature, tempWall, fluid, massFluxAverage, diamOuter,
                                      diamHydraulicOverLength, pressure, heatFluxWall, diamInner, massFlow)[1]
    def SuperCriticalCondensation_cp(temperature, tempWall, fluid, massFluxAverage, diamOuter,
                                     diamHydraulicOverLength, pressure, heatFluxWall, diamInner, massFlow):
        '''return specificHeat value'''
        return PettersonSupercritical(temperature, tempWall, fluid, massFluxAverage, diamOuter,
                                      diamHydraulicOverLength, pressure, heatFluxWall, diamInner, massFlow)[2]
    def SuperCriticalCondensation_rho(temperature, tempWall, fluid, massFluxAverage, diamOuter,
                                      diamHydraulicOverLength, pressure, heatFluxWall, diamInner, massFlow):
        '''return density value'''
        return PettersonSupercritical(temperature, tempWall, fluid, massFluxAverage, diamOuter,
                                      diamHydraulicOverLength, pressure, heatFluxWall, diamInner, massFlow)[3]

    if not tempOut == tempIn:
        enthalpy = quad(SuperCriticalCondensation_h,tempIn,tempOut,
                        args=(tempWall,fluid,massFluxAverage,diamOuter,diamInner,diamHydraulicOverLength,
                              massFlow,pressure,heatFluxWall))[0]/(tempOut-tempIn)
        frictionFactor = quad(SuperCriticalCondensation_f,tempIn,tempOut,
                              args=(tempWall,fluid,massFluxAverage,diamOuter,diamInner,
                            diamHydraulicOverLength,massFlow,pressure,heatFluxWall))[0]/(tempOut-tempIn)
        specificHeat = quad(SuperCriticalCondensation_cp,tempIn,tempOut,
                            args=(tempWall,fluid,massFluxAverage,diamOuter,diamInner,diamHydraulicOverLength,
                                  massFlow,pressure,heatFluxWall))[0]/(tempOut-tempIn)
        density = quad(SuperCriticalCondensation_rho,tempIn,tempOut,
                       args=(tempWall,fluid,massFluxAverage,diamOuter,diamInner,diamHydraulicOverLength,
                             massFlow,pressure,heatFluxWall))[0]/(tempOut-tempIn)
        return (enthalpy,frictionFactor,specificHeat,density)
    return PettersonSupercritical(tempOut,tempWall,fluid,massFluxAverage,diamOuter,diamHydraulicOverLength,
                                  pressure,heatFluxWall,diamInner,massFlow)

def PettersonSupercritical(temperature, tempWall, fluid, massFluxAverage, diamOuter, diamHydraulicOverLength,
                           pressure, heatFluxWall, diamInner=0, massFlow=0):
    '''
    Petterson et al. (2000), Heat transfer and pressure drop for flow supercritical and subcritical
    CO2 in microchannel tubes
    All details for this correlation are available in Ding Li Thesis (Appendix B):
    "INVESTIGATION OF AN EJECTOR-EXPANSION DEVICE IN A TRANSCRITICAL CARBON DIOXIDE CYCLE FOR
    MILITARY ECU APPLICATIONS"
    '''
    viscosityWall = fluid.calculateViscosity(ThermoProps.PT, pressure, tempWall)
    specificHeatWall = fluid.calculateHeatCapacity(ThermoProps.PT, pressure, tempWall)
    conductivityWall = fluid.calculateConductivity(ThermoProps.PT, pressure, tempWall)
    prandtlNumWall = FluidMechanics.calculatePrandtlNumber(specificHeatWall, viscosityWall,
                                                                  conductivityWall)

    viscosity = fluid.calculateViscosity(ThermoProps.PT, pressure, temperature)
    specificHeat = fluid.calculateHeatCapacity(ThermoProps.PT, pressure, temperature)
    conductivity = fluid.calculateConductivity(ThermoProps.PT, pressure, temperature)
    density = fluid.calculateDensity(ThermoProps.PT, pressure, temperature)
    prandtlNum = FluidMechanics.calculatePrandtlNumber(specificHeat, viscosity, conductivity)

    if massFlow == 0: #For the case of Micro-channel
        diameterHydraulic = diamOuter
        reynoldsNum = FluidMechanics.calculateReynoldsNumber(massFluxAverage, diameterHydraulic, viscosity)
        reynoldsNumWall = FluidMechanics.calculateReynoldsNumber(massFluxAverage, diameterHydraulic,
                                                                        viscosityWall)
    else: #for the case of fin-and-tube
        diameterHydraulic = diamOuter - diamInner
        area = np.pi*(diamOuter**2-diamInner**2)/4.0
        velocity = massFlow/(area*density)
        reynoldsNum = FluidMechanics.calculateReynoldsNumber(velocity, diameterHydraulic,
                                                                    viscosity, density=density)
        reynoldsNumWall = reynoldsNum #rho_w*velocity*diameterHydraulic/viscosityWall

    if massFluxAverage > 350:
        #from the Petterson paper
        roughnessRatio = 0 #smooth pipe
        frictionFactor = (-1.8*np.log10(6.9/reynoldsNum + (1/3.7*roughnessRatio)**1.11))**(-2)/4
        nusseltNumMean = FluidMechanics.calculateNusseltNumber(frictionFactor, reynoldsNum, prandtlNum)*\
                (1+(diamHydraulicOverLength)**(2/3))
        nusseltNum = nusseltNumMean*(prandtlNum/prandtlNumWall)**0.11
    else:
        enthalpyWall = fluid.calculateEnthalpy(ThermoProps.PT, pressure, tempWall)
        enthalpy = fluid.calculateEnthalpy(ThermoProps.PT, pressure, temperature)
        M = 0.001 #[kg/J]
        K = 0.00041 #[kg/J]
        specificHeatAvg = (enthalpy - enthalpyWall)/(temperature - tempWall)
        nConstant = 0.66 if specificHeatAvg/specificHeatWall <= 1 else 0.9
        n = nConstant - K*(heatFluxWall/massFluxAverage)
        frictionFactorDefault = (0.79*np.log(reynoldsNum) - 1.64)**(-2)
        isobaricExpansionCoeff = fluid.getIsobaricExpansionCoefficient()
        grashofNum = FluidMechanics.calculateGrashofNumber(isobaricExpansionCoeff, tempWall,
                                            temperature, diameterHydraulic, viscosity/density)
        # grashof number/reynolds number**2 = Richardson number (Ri)
        # Ri << 1: can ignore forced convection
        # Ri ~= 1: mixture of free and forced convection
        # Ri << 1: can ignore free convection
        if grashofNum/reynoldsNum**2 < 5e-4:
            frictionFactor = frictionFactorDefault*(viscosityWall/viscosity)**0.22
        elif grashofNum/reynoldsNum**2 >= 5e-4 and massFluxAverage/reynoldsNum**2 < 0.3:
            frictionFactor = 2.15*frictionFactorDefault*(viscosityWall/viscosity)**0.22*\
                (grashofNum/reynoldsNum)**0.1
        else: #use frictionFactorDefault for friction factor
            frictionFactor = frictionFactorDefault
        nusseltNumWall = FluidMechanics.calculateNusseltNumber(frictionFactorDefault,
                            reynoldsNumWall + 1000, prandtlNumWall, frictionFactor=frictionFactor)
        nusseltNum = nusseltNumWall*(1 - M*heatFluxWall/massFluxAverage)*(specificHeatAvg/specificHeatWall)**n
    heatTransferCoeff = conductivity*nusseltNum/diameterHydraulic #[W/m^2-K]
    for localKey, localVal in locals().items():
        if isinstance(localVal, (str, dict, list)):
            logging.debug("%s: %s", localKey, localVal, extra={"methodname": "PettersonSupercritical"})
        elif isinstance(localVal, (float, int)):
            logging.debug("%s: %g", localKey, localVal or 0.0, extra={"methodname": "PettersonSupercritical"})
        else:
            continue
    return heatTransferCoeff, frictionFactor, specificHeat, density

def f_h_1phase_Tube(massFlow, innerDiam, temperature, pressure, fluid, Phase='Single'):
    """
    Convenience function to run annular model for tube. Tube is a degenerate case of
    annulus with inner diameter of 0

    """
    return f_h_1phase_Annulus(massFlow, innerDiam, 0.0, temperature, pressure, fluid, Phase)

def f_h_1phase_Annulus(massFlow, outerDiam, innerDiam, temperature, pressure, fluid, Phase='Single'):
    """
    This function return the friction factor, heat transfer coefficient,
    and reynold's number for single phase fluid inside annular pipe
    """
    if "Sat" in Phase:
        thermoProps = ThermoProps.QT
        firstVar = 1.0 if Phase == "SatVap" else 0.0
    else:
        thermoProps = ThermoProps.PT
        firstVar = pressure
    viscosity = fluid.calculateViscosity(thermoProps, firstVar, temperature) #[Pa-s OR kg/m-s]
    specificHeat = fluid.calculateHeatCapacity(thermoProps, firstVar, temperature) #[J/kg-K]
    conductivity = fluid.calculateConductivity(thermoProps, firstVar, temperature) #[W/m-K]
    density = fluid.calculateDensity(thermoProps, firstVar, temperature)

    prandtlNum = FluidMechanics.calculatePrandtlNumber(specificHeat, viscosity, conductivity) #[-]

    diameterHydraulic = outerDiam - innerDiam
    area = np.pi*(outerDiam**2-innerDiam**2)/4.0
    velocity = massFlow/(area*density)
    reynoldsNum = FluidMechanics.calculateReynoldsNumber(velocity, diameterHydraulic,
                                                                viscosity, density=density)

    darcyFrictionFactor = FluidMechanics.calculateChurchillFrictionFactor(reynoldsNum)

    # Heat Transfer coefficient of Gnielinski
    nusseltNum = FluidMechanics.calculateNusseltNumber(darcyFrictionFactor, reynoldsNum, prandtlNum)
    heatTransferCoeff = conductivity*nusseltNum/diameterHydraulic #W/m^2-K
    return (darcyFrictionFactor, heatTransferCoeff, reynoldsNum)

def Cooper_PoolBoiling(fluid, surfaceRoughness, qFlux, apparatus):
    """
    Cooper M.G., 1984, "Heat flow rates in saturated nucleate boiling - A wide-ranging
    examination using reduced properties. Advances in Heat Transfer. Vol. 16,
    Eds. J.P. Harnett and T.F. Irvine Jr., Academic Press, Orlando, Florida. pp 157-239"

    Rp : surface roughness in microns
    """
    pressureCorrected = fluid.fluidApparatiProps[apparatus].pressureIn/fluid.pressureCritical
    return 55*pressureCorrected**(0.12-0.2*np.log10(surfaceRoughness))*\
        (-np.log10(pressureCorrected))**(-0.55)*qFlux**(0.67)*fluid.massMolar**(-0.5)

def KandlikarPHE(fluid,xmean,massFlux,D,q,tBubble,tDew):
    """
    From http://www.rit.edu/kgcoe/mechanical/taleme/Papers/Conference%20Papers/C041.pdf

    Not recommended for fluids other than R134a
    """
    AS = fluid.abstractState.abstractState
    AS.update(CP.QT_INPUTS,0.0,tBubble)
    rhoL = AS.rhomass() #[kg/m^3]
    mu_f = AS.viscosity() #[Pa-s OR kg/m-s]
    cp_f = AS.cpmass() #[J/kg-K]
    k_f = AS.conductivity() #[W/m/K]
    h_L = AS.hmass() #[J/kg]

    AS.update(CP.QT_INPUTS,1.0,tDew)
    rhoG = AS.rhomass() #[kg/m^3]
    h_G = AS.hmass() #[J/kg]

    Pr_f = cp_f * mu_f / k_f #[-]

    h_LG = h_G-h_L #[J/kg]
    alpha_L = 0.023 * (massFlux*D/mu_f)**(0.8) * Pr_f**(0.4) * k_f / D #[W/m^2-K]
    Co=(rhoG/rhoL)**(0.5)*((1-xmean)/xmean)**(0.8)
    Bo=q/(massFlux*h_LG)
#    E_CB=0.512
#    E_NB=0.338
#    F_fl=1.0
    #alpha_r=(2.312*Co**(-0.3)*E_CB+667.3*Bo**(2.8)*F_fl*E_NB)*(1-xmean)**(0.003)*alpha_L
    alpha_r=1.055*(1.056*Co**(-0.4)+1.02*Bo**(0.9))*xmean**(-0.12)*alpha_L**(0.98)
    return alpha_r

def Bertsch_MC(x,fluid,massFlux,diameterHydraulic,q_flux,L,tBubble,tDew):
    """
    This function return the heat transfer coefficient for two phase fluid
    inside Micro-channel tube
    Correlatation is based on Bertsch (2009)
    """
    AS = fluid.abstractState.abstractState
    #Define properties
    AS.update(CP.QT_INPUTS,0.0,tBubble)
    k_L=AS.conductivity() #[W/m/K]
    cp_L=AS.cpmass() #[J/kg-K]
    mu_L=AS.viscosity() #[Pa-s OR kg/m-s]
    rho_L=AS.rhomass() #[kg/m^3]

    AS.update(CP.QT_INPUTS,1.0,tDew)
    k_G=AS.conductivity() #[W/m/K]
    cp_G=AS.cpmass() #[J/kg-K]
    mu_G=AS.viscosity() #[Pa-s OR kg/m-s]
    rho_G=AS.rhomass() #[kg/m^3]

    pressure=AS.p() #saturation pressure [Pa] @ tDew
    pc=AS.p_critical() #critical pressure [Pa]
    pr=pressure/pc
    M=AS.molar_mass() #molar mass [kg/mol]

    AS.update(CP.PQ_INPUTS,pressure,x)
    sig = AS.surface_tension() #surface tension [N/m]

    #if Ref=='R290':
    #    sig=55.28*(1-tDew/369.818)**(1.258)/1000.
    #elif Ref=='R410A':
        ## From Okada 1999 "Surface Tension of HFC Refrigerant Mixtures"
    #    sig=62.38*(1-tDew/344.56)**(1.246)/1000.

    Re_L=massFlux*diameterHydraulic/mu_L
    Re_G=massFlux*diameterHydraulic/mu_G
    Pr_L=cp_L*mu_G/k_L
    Pr_G=cp_G*mu_G/k_G

    h_nb=55*(pr)**(0.12)*(-np.log10(pr))**(-0.55)*M**(-0.5)*q_flux**(0.67)
    h_conv_l=(3.66+(0.0668*diameterHydraulic/L*Re_L*Pr_L)/(1+0.04*(diameterHydraulic/L*Re_L*Pr_L)**(2.0/3.0)))*k_L/diameterHydraulic
    h_conv_g=(3.66+(0.0668*diameterHydraulic/L*Re_G*Pr_G)/(1+0.04*(diameterHydraulic/L*Re_G*Pr_G)**(2.0/3.0)))*k_G/diameterHydraulic
    h_conv_tp=h_conv_l*(1-x)+h_conv_g*x
    Co=np.sqrt(sig/(g*(rho_L-rho_G)*diameterHydraulic**2))
    h_TP=h_nb*(1-x)+h_conv_tp*(1.0+80.0*(x**2-x**6)*np.exp(-0.6*Co))
    return h_TP

def Bertsch_MC_Average(xMin,xMax,fluid,massFlux,diameterHydraulic,q_flux,L,TsatL,TsatV):
    '''
    Returns the average heat transfer coefficient
    between qualities of xMin and xMax.
    for Bertsch two-phase evaporation in mico-channel HX
    '''
    AS = fluid.abstractState.abstractState
    if not xMin==xMax:
        return quad(Bertsch_MC,xMin,xMax,args=(AS,massFlux,diameterHydraulic,q_flux,L,TsatL,TsatV))[0]/(xMax-xMin)
    return Bertsch_MC(xMin,AS,massFlux,diameterHydraulic,q_flux,L,TsatL,TsatV)

def f_h_1phase_MicroTube(massFlux, diameterHydraulic, temperature, pressure, fluid, Phase='Single'):
    """
    This function return the friction factor, heat transfer coefficient,
    and Reynold's number for single phase fluid inside flat plate tube
    Micro-channel HX
    """
    if "Sat" in Phase:
        thermoProps = ThermoProps.QT
        firstVar = 1.0 if Phase == "SatVap" else 0.0
    else:
        thermoProps = ThermoProps.PT
        firstVar = pressure
    viscosity = fluid.calculateViscosity(thermoProps, firstVar, temperature) #[Pa-s OR kg/m-s]
    specificHeat = fluid.calculateHeatCapacity(thermoProps, firstVar, temperature) #[J/kg-K]
    conductivity = fluid.calculateConductivity(thermoProps, firstVar, temperature) #[W/m-K]

    prandtlNum = FluidMechanics.calculatePrandtlNumber(specificHeat, viscosity, conductivity) #[-]

    reynoldsNum = FluidMechanics.calculateReynoldsNumber(massFlux, diameterHydraulic, viscosity)
    reynoldsNum = massFlux*diameterHydraulic/viscosity

    darcyFrictionFactor = FluidMechanics.calculateChurchillFrictionFactor(reynoldsNum)

    # Heat Transfer coefficient of Gnielinski
    nusseltNum = FluidMechanics.calculateNusseltNumber(darcyFrictionFactor, reynoldsNum, prandtlNum)
    h = conductivity*nusseltNum/diameterHydraulic #W/m^2-K
    return (darcyFrictionFactor, h, reynoldsNum)

def KM_Cond_Average(xMin,xMax,fluid,massFlux,diameterHydraulic,tBubble,tDew,pressure,beta,coeff=None,satTransport=None):
    """
    Returns the average pressure gradient and average heat transfer coefficient
    between qualities of xMin and xMax.
    for Kim&Mudawar two-phase condensation in mico-channel HX

    To obtain the pressure gradient for a given value of x, pass it in as xMin and xMax

    Required parameters:
    * xMin : The minimum quality for the range [-]
    * xMax : The maximum quality for the range [-]
    * AS : AbstractState with the refrigerant name and backend
    * massFlux : Mass flux [kg/m^2/s]
    * diameterHydraulic : Hydraulic diameter of tube [m]
    * tBubble : Bubblepoint temperature of refrigerant [K]
    * tDew : Dewpoint temperature of refrigerant [K]
    * beta: channel aspect ratio (=width/height)

    Optional parameters:
    * satTransport : A dictionary with the keys 'mu_f','mu_g,'rho_f','rho_g', 'sigma' for the
      saturation properties.  So they can be calculated once and passed in for a slight improvement
      in efficiency
    """
    AS = fluid.abstractState.abstractState
    def KMFunc(x):
        dpdz, h = Kim_Mudawar_condensing_DPDZ_h(AS,massFlux,diameterHydraulic,x,tBubble,tDew,pressure,beta,coeff,satTransport)
        return dpdz , h

    ## Use Simpson's Rule to calculate the average pressure gradient
    ## Can't use adapative quadrature since function is not sufficiently smooth
    ## Not clear why not sufficiently smooth at x>0.9
    if xMin==xMax:
        return KMFunc(xMin)
    #Calculate the tranport properties once
    satTransport={}
    AS.update(CP.QT_INPUTS,0.0,tBubble)
    satTransport['rho_f']=AS.rhomass() #[kg/m^3]
    satTransport['mu_f']=AS.viscosity() #[Pa-s OR kg/m-s]
    satTransport['cp_f']=AS.cpmass() #[J/kg-K]
    satTransport['k_f']=AS.conductivity() #[W/m-K]
    AS.update(CP.QT_INPUTS,1.0,tDew)
    satTransport['rho_g']=AS.rhomass() #[kg/m^3]
    satTransport['mu_g']=AS.viscosity() #[Pa-s OR kg/m-s]

    #Calculate Dp and h over the range of xx
    xx=np.linspace(xMin,xMax,100)
    DP=np.zeros_like(xx)
    h=np.zeros_like(xx)
    for i, xVal in enumerate(xx):
        DP[i], h[i]=KMFunc(xVal)

    #Use Simpson's rule to carry out numerical integration to get average DP and average h
    if abs(xMax-xMin)<5*machineEps:
        #return just one of the edge values
        return -DP[0], h[0]
    #Use Simpson's rule to carry out numerical integration to get average DP and average h
    return -simps(DP,xx)/(xMax-xMin), simps(h,xx)/(xMax-xMin)

def Kim_Mudawar_condensing_DPDZ_h(fluid, massFlux, diameterHydraulic, x, tBubble, tDew, pressure, beta, coeff=None, satTransport=None):
    """
    This function return the pressure gradient and heat transfer coefficient for
    two phase fluid inside Micro-channel tube while CONDENSATION
    Correlations Based on:
    Kim and Mudawar (2012) "Universal approach to predicting two-phase
    frictional pressure drop and condensing mini/micro-channel flows", Int. J Heat Mass, 55, 3246-3261
    and
    Kim and Mudawar (2013) "Universal approach to predicting heat transfer coefficient
    for condensing min/micro-channel flow", Int. J Heat Mass, 56, 238-250
    """
    AS = fluid.abstractState.abstractState
    #Convert the quality, which might come in as a single numpy float value, to a float
    #With the conversion, >20x speedup in the lockhartMartinelli function, not clear why
    x=float(x)

    if satTransport is None:
        # Calculate Necessary saturation properties
        AS.update(CP.QT_INPUTS,0.0,tBubble)
        rho_f=AS.rhomass() #[kg/m^3]
        mu_f=AS.viscosity() #[Pa-s OR kg/m-s]
        cp_f=AS.cpmass() #[J/kg-K]
        k_f=AS.conductivity() #[W/m-K]
        AS.update(CP.QT_INPUTS,1.0,tDew)
        rho_g=AS.rhomass() #[kg/m^3]
        mu_g=AS.viscosity() #[Pa-s OR kg/m-s]
    else:
        #Pull out of the dictionary
        rho_f=satTransport['rho_f']
        rho_g=satTransport['rho_g']
        mu_f=satTransport['mu_f']
        mu_g=satTransport['mu_g']
        cp_f=satTransport['cp_f']
        k_f=satTransport['k_f']

    AS.update(CP.PQ_INPUTS,pressure,x)
    sigma=AS.surface_tension() #surface tesnion [N/m]

    Pr_f = cp_f * mu_f / k_f #[-]

    reynoldsNumLiq = massFlux*(1-x)*diameterHydraulic/mu_f
    reynoldsNumVap = massFlux*x*diameterHydraulic/mu_g


    if x==1: #No liquid
        frictionFactorLiq = 0 #Just to be ok until next step
    elif reynoldsNumLiq<2000: #Laminar
        frictionFactorLiq = 16.0/reynoldsNumLiq
        if beta<1:
            frictionFactorLiq = 24*(1 - 1.3553*beta + 1.9467*beta**2 - 1.7012*pow(beta,3) + \
                                    0.9564*pow(beta,4) - 0.2537*pow(beta,5))/reynoldsNumLiq
    elif reynoldsNumLiq>=20000: #Fully-Turbulent
        frictionFactorLiq = 0.046*pow(reynoldsNumLiq,-0.2)
    else: #Transient
        frictionFactorLiq = 0.079*pow(reynoldsNumLiq,-0.25)

    if x==0: #No gas
        frictionFactorVap = 0 #Just to be ok until next step
    elif reynoldsNumVap<2000: #Laminar
        frictionFactorVap=16.0/reynoldsNumVap
        if beta<1:
            frictionFactorVap = 24*(1 - 1.3553*beta + 1.9467*beta**2 - 1.7012*pow(beta,3) + \
                                    0.9564*pow(beta,4) - 0.2537*pow(beta,5))/reynoldsNumVap
    elif reynoldsNumVap>=20000: #Fully-Turbulent
        frictionFactorVap = 0.046*pow(reynoldsNumVap,-0.2)
    else: #Transient
        frictionFactorVap = 0.079*pow(reynoldsNumVap,-0.25)

    Re_fo = massFlux*diameterHydraulic/mu_f
    Su_go = rho_g*sigma*diameterHydraulic/pow(mu_g,2)

    dpdz_f = 2*frictionFactorLiq/rho_f*pow(massFlux*(1-x),2)/diameterHydraulic
    dpdz_g = 2*frictionFactorVap/rho_g*pow(massFlux*x,2)/diameterHydraulic

    if x<=0:
        # Entirely liquid
        dpdz = dpdz_f
        AS.update(CP.QT_INPUTS,0.0,tBubble)
        psat = AS.p() #pressure [Pa]
        h = f_h_1phase_MicroTube(massFlux, diameterHydraulic, tBubble, psat, AS, Phase='SatLiq')[1]
        return dpdz, h
    if x>=1:
        #Entirely vapor
        dpdz = dpdz_g
        AS.update(CP.QT_INPUTS,1.0,tDew)
        psat = AS.p() #pressure [Pa]
        h = f_h_1phase_MicroTube(massFlux, diameterHydraulic, tDew, psat, AS, Phase='SatVap')[1]
        return dpdz, h

    paramLM = np.sqrt(dpdz_f/dpdz_g)

    # Find the C coefficient (Calculate C if not passed, otherwise use the set value of C)
    if coeff is None:
        if reynoldsNumLiq < 2000 and reynoldsNumVap < 2000:
            coeff = 3.5e-5*pow(Re_fo,0.44)*pow(Su_go,0.50)*pow(rho_f/rho_g,0.48)
        elif reynoldsNumLiq < 2000 <= reynoldsNumVap:
            coeff = 0.0015*pow(Re_fo,0.59)*pow(Su_go,0.19)*pow(rho_f/rho_g,0.36)
        elif reynoldsNumLiq >= 2000 > reynoldsNumVap:
            coeff = 8.7e-4*pow(Re_fo,0.17)*pow(Su_go,0.50)*pow(rho_f/rho_g,0.14)
        else:
            coeff = 0.39*pow(Re_fo,0.03)*pow(Su_go,0.10)*pow(rho_f/rho_g,0.35)
    else:
        pass

    # Two-phase multiplier
    phi_f_square = 1.0 + coeff/paramLM + 1.0/paramLM**2
    phi_g_square = 1.0 + coeff*paramLM + paramLM**2

    # Find Condensing pressure drop griendient
    if dpdz_g*phi_g_square > dpdz_f*phi_f_square:
        dpdz=dpdz_g*phi_g_square
    else:
        dpdz=dpdz_f*phi_f_square

    #Use calculated Lockhart-Martinelli parameter
    Xtt = paramLM
    # Simplified Lockhart-Martinelli paramter from Kim & Mudawar(2013) "Universal approach to predict HTC for condensing mini/micro-channel flow"
    #Xtt = pow(mu_f/mu_g,0.1) * pow((1-x)/x,0.9) * pow(rho_g/rho_f,0.5)

    # Modified Weber number
    if reynoldsNumLiq <= 1250:
        We_star = 2.45 * pow(reynoldsNumVap,0.64) / (pow(Su_go,0.3) * pow(1 + 1.09*pow(Xtt,0.039),0.4))
    else:
        We_star = 0.85*pow(reynoldsNumVap,0.79)*pow(Xtt,0.157)/\
            (pow(Su_go,0.3)*pow(1 + 1.09*pow(Xtt,0.039),0.4))*pow(pow(mu_g/mu_f,2)*(rho_f/rho_g),0.084)

    # Condensation Heat transfer coefficient
    if We_star > 7*Xtt**0.2: ##for annual flow (smooth-annular, wavy-annular, transition)
        h = k_f/diameterHydraulic * 0.048 * pow(reynoldsNumLiq,0.69) * pow(Pr_f,0.34) * np.sqrt(phi_g_square) / Xtt
    else: ##for slug and bubbly flow
        h = k_f/diameterHydraulic*pow((0.048*pow(reynoldsNumLiq,0.69)*pow(Pr_f,0.34)*np.sqrt(phi_g_square)/Xtt)**2 + \
                       (3.2e-7*pow(reynoldsNumLiq,-0.38)*pow(Su_go,1.39))**2,0.5)

    return dpdz, h

def KM_Evap_Average(xMin,xMax,fluid,massFlux,diameterHydraulic,tBubble,tDew,pressure,beta,q_fluxH,PH_PF=1,coeff=None,satTransport=None):
    """
    Returns the average pressure gradient and average heat transfer coefficient
    between qualities of xMin and xMax.
    for Kim&Mudawar two-phase evaporation in mico-channel HX

    To obtain the pressure gradient for a given value of x, pass it in as xMin and xMax

    Required parameters:
    * xMin : The minimum quality for the range [-]
    * xMax : The maximum quality for the range [-]
    * AS : AbstractState with the refrigerant name and backend
    * massFlux : Mass flux [kg/m^2/s]
    * diameterHydraulic : Hydraulic diameter of tube [m]
    * tBubble : Bubblepoint temperature of refrigerant [K]
    * tDew : Dewpoint temperature of refrigerant [K]
    * pressure : pressure [Pa]
    * beta: channel aspect ratio (=width/height)
    * q_fluxH: heat flux [W/m^2]
    * PH_PF: ratio of PH over PF where PH: heated perimeter of channel, PF: wetted perimeter of channel

    Optional parameters:
    * satTransport : A dictionary with the keys 'mu_f','mu_g,'rho_f','rho_g', 'sigma' for the
      saturation properties.  So they can be calculated once and passed in for a slight improvement
      in efficiency
    """
    AS = fluid.abstractState.abstractState
    def KMFunc(x):
        dpdz, h = Kim_Mudawar_boiling_DPDZ_h(AS,massFlux,diameterHydraulic,x,tBubble,tDew,pressure,beta,q_fluxH,PH_PF,coeff,satTransport)
        return dpdz , h

    ## Use Simpson's Rule to calculate the average pressure gradient
    ## Can't use adapative quadrature since function is not sufficiently smooth
    ## Not clear why not sufficiently smooth at x>0.9
    if xMin==xMax:
        return KMFunc(xMin)
    #Calculate the tranport properties once
    satTransport={}
    AS.update(CP.QT_INPUTS,0.0,tBubble)
    satTransport['rho_f']=AS.rhomass() #[kg/m^3]
    satTransport['mu_f']=AS.viscosity() #[Pa-s OR kg/m-s]
    h_f=AS.hmass() #[J/kg]
    satTransport['cp_f']=AS.cpmass() #[J/kg-K]
    satTransport['k_f']=AS.conductivity() #[W/m-K]
    AS.update(CP.QT_INPUTS,1.0,tDew)
    satTransport['rho_g']=AS.rhomass() #[kg/m^3]
    satTransport['mu_g']=AS.viscosity() #[Pa-s OR kg/m-s]
    h_g=AS.hmass() #[J/kg]
    satTransport['h_fg'] = h_g - h_f #[J/kg]

    #Calculate Dp and h over the range of xx
    xx=np.linspace(xMin,xMax,100)
    DP=np.zeros_like(xx)
    h=np.zeros_like(xx)
    for i, xxVal in enumerate(xx):
        DP[i], h[i]=KMFunc(xxVal)

    #Use Simpson's rule to carry out numerical integration to get average DP and average h
    if abs(xMax-xMin)<5*machineEps:
        #return just one of the edge values
        return -DP[0], h[0]
    #Use Simpson's rule to carry out numerical integration to get average DP and average h
    return -simps(DP,xx)/(xMax-xMin), simps(h,xx)/(xMax-xMin)

def Kim_Mudawar_boiling_DPDZ_h(fluid, massFlux, diameterHydraulic, x, tBubble, tDew, pressure, beta, q_fluxH, PH_PF=1,
                               coeff=None, satTransport=None):
    """
    This function return the pressure gradient and heat transfer coefficient for
    two phase fluid inside Micro-channel tube while BOILING (EVAPORATION)

    Correlations of DPDZ Based on: Kim and Mudawar (2013) "Universal approach to predicting
    two-phase frictional pressure drop for mini/micro-channel saturated flow boiling"

    Correlations of HTC Based on: Kim and Mudawar (2013) "Universal approach to predicting
    saturated flow boiling heat transfer in mini/micro-channels - Part II. Two-heat heat transfer coefficient"
    """
    #Convert the quality, which might come in as a single numpy float value, to a float
    #With the conversion, >20x speedup in the lockhartMartinelli function, not clear why
    x=float(x)

    AS = fluid.abstractState.abstractState
    if satTransport is None:
        # Calculate Necessary saturation properties
        AS.update(CP.QT_INPUTS,0.0,tBubble)
        rho_f=AS.rhomass() #[kg/m^3]
        mu_f=AS.viscosity() #[Pa-s OR kg/m-s]
        h_f=AS.hmass() #[J/kg]
        cp_f=AS.cpmass() #[J/kg-K]
        k_f=AS.conductivity() #[W/m-K]
        AS.update(CP.QT_INPUTS,1.0,tDew)
        rho_g=AS.rhomass() #[kg/m^3]
        mu_g=AS.viscosity() #[Pa-s OR kg/m-s]
        h_g=AS.hmass() #[J/kg]
        h_fg = h_g - h_f #[J/kg]
    else:
        #Pull out of the dictionary
        rho_f=satTransport['rho_f']
        rho_g=satTransport['rho_g']
        mu_f=satTransport['mu_f']
        mu_g=satTransport['mu_g']
        h_fg=satTransport['h_fg']
        cp_f=satTransport['cp_f']
        k_f=satTransport['k_f']

    pc=AS.p_critical() #critical pressure [Pa]
    pr=pressure/pc #reducred pressure [-]
    AS.update(CP.PQ_INPUTS,pressure,x)
    sigma = AS.surface_tension() #surface tesnion [N/m]

    reynoldsNumLiq = massFlux*(1-x)*diameterHydraulic/mu_f
    reynoldsNumVap = massFlux*x*diameterHydraulic/mu_g
    Pr_f = cp_f*mu_f/k_f

    if x==1: #No liquid
        frictionFactorLiq = 0 #Just to be ok until next step
    elif reynoldsNumLiq<2000: #Laminar
        frictionFactorLiq = 16.0/reynoldsNumLiq
        if beta<1:
            frictionFactorLiq = 24*(1 - 1.3553*beta + 1.9467*beta**2 - 1.7012*pow(beta,3) + \
                                    0.9564*pow(beta,4) - 0.2537*pow(beta,5))/reynoldsNumLiq
    elif reynoldsNumLiq>=20000: #Fully-Turbulent
        frictionFactorLiq = 0.046*pow(reynoldsNumLiq,-0.2)
    else: #Transient
        frictionFactorLiq = 0.079*pow(reynoldsNumLiq,-0.25)

    if x==0: #No gas
        frictionFactorVap = 0 #Just to be ok until next step
    elif reynoldsNumVap<2000: #Laminar
        frictionFactorVap=16.0/reynoldsNumVap
        if beta<1:
            frictionFactorVap = 24*(1 - 1.3553*beta + 1.9467*beta**2 - 1.7012*pow(beta,3) + \
                                    0.9564*pow(beta,4) - 0.2537*pow(beta,5))/reynoldsNumVap
    elif reynoldsNumVap>=20000: #Fully-Turbulent
        frictionFactorVap = 0.046*pow(reynoldsNumVap,-0.2)
    else: #Transient
        frictionFactorVap = 0.079*pow(reynoldsNumVap,-0.25)

    Re_fo = massFlux*diameterHydraulic/mu_f
    Su_go = rho_g*sigma*diameterHydraulic/pow(mu_g,2)

    dpdz_f = 2*frictionFactorLiq/rho_f*pow(massFlux*(1-x),2)/diameterHydraulic
    dpdz_g = 2*frictionFactorVap/rho_g*pow(massFlux*x,2)/diameterHydraulic

    if x<=0:
        # Entirely liquid
        dpdz = dpdz_f
        AS.update(CP.QT_INPUTS,0.0,tBubble)
        psat = AS.p() #pressure [Pa]
        h = f_h_1phase_MicroTube(massFlux, diameterHydraulic, tBubble, psat, AS, Phase='SatLiq')[1]
        return dpdz, h
    if x>=1:
        #Entirely vapor
        dpdz = dpdz_g
        AS.update(CP.QT_INPUTS,1.0,tDew)
        psat = AS.p() #pressure [Pa]
        h = f_h_1phase_MicroTube(massFlux, diameterHydraulic, tDew, psat, AS, Phase='SatVap')[1]
        return dpdz, h

    paramLM = np.sqrt(dpdz_f/dpdz_g)

    We_fo = massFlux*massFlux*diameterHydraulic/rho_f/sigma
    Bo = q_fluxH/(massFlux*h_fg)

    # Find the C coefficient (Calculate C if not passed, otherwise use the set value of C)
    if coeff is None:
        # Calculate C (non boiling)
        if reynoldsNumLiq < 2000 and reynoldsNumVap < 2000:
            Cnon_boiling = 3.5e-5*pow(Re_fo,0.44)*pow(Su_go,0.50)*pow(rho_f/rho_g,0.48)
        elif reynoldsNumLiq < 2000 <= reynoldsNumVap:
            Cnon_boiling = 0.0015*pow(Re_fo,0.59)*pow(Su_go,0.19)*pow(rho_f/rho_g,0.36)
        elif reynoldsNumLiq >= 2000 > reynoldsNumVap:
            Cnon_boiling = 8.7e-4*pow(Re_fo,0.17)*pow(Su_go,0.50)*pow(rho_f/rho_g,0.14)
        elif reynoldsNumLiq >= 2000 and reynoldsNumVap >= 2000:
            Cnon_boiling = 0.39*pow(Re_fo,0.03)*pow(Su_go,0.10)*pow(rho_f/rho_g,0.35)
        # Calculate actual C
        if reynoldsNumLiq >= 2000:
            coeff = Cnon_boiling*(1+60*pow(We_fo,0.32)*pow(Bo*PH_PF,0.78))
        else:
            coeff = Cnon_boiling*(1+530*pow(We_fo,0.52)*pow(Bo*PH_PF,1.09))
    else:
        pass

    #Two-phase multiplier
    phi_f_square = 1 + coeff/paramLM + 1/paramLM**2
    phi_g_square = 1 + coeff*paramLM + paramLM**2

    #Find Boiling pressure drop griendient
    if dpdz_g*phi_g_square > dpdz_f*phi_f_square:
        dpdz=dpdz_g*phi_g_square
    else:
        dpdz=dpdz_f*phi_f_square

    #Use calculated Lockhart-Martinelli parameter
    Xtt = paramLM
    #Simplified X_tt from Kim and Mudawar(2013) "Universal approach to predicting .... Part II. Two-phase heat transfer coefficient"
    #Xtt = pow(mu_f/mu_g,0.1)*pow((1-x)/x,0.9)*pow(rho_g/rho_f,0.5)

    #Pre-dryout saturated flow boiling Heat transfer coefficient
    h_nb = (2345*pow(Bo*PH_PF,0.7)*pow(pr,0.38)*pow(1-x,-0.51))*(0.023*pow(reynoldsNumLiq,0.8)*\
                                                                 pow(Pr_f,0.4)*k_f/diameterHydraulic)
    h_cb = (5.2*pow(Bo*PH_PF,0.08)*pow(We_fo,-0.54) + 3.5*pow(1/Xtt,0.94)*pow(rho_g/rho_f,0.25))*\
        (0.023*pow(reynoldsNumLiq,0.8)*pow(Pr_f,0.4)*k_f/diameterHydraulic)
    h = pow(h_nb**2 +h_cb**2,0.5)

    return dpdz, h

def NaturalConv_HTC(fluid,HTCat,T_wall,T_inf,P_film,L,D_pipe=None,PlateNum=None):
    """
    Nusselt number for different heat transfer categories;
    find heat transfer coefficient based on Nusselt number;
    characteristic length, A/P;
    Based on: Incropera et al. "Fundamentals of Heat and Mass Transfer"

    Return heat transfer rate by natural convection

    Parameters
    ----------
    AS : AbstractState with the refrigerant name and backend
    HTCat : 'horizontal_pipe' or 'vertical_pipe' or 'vertical_plate' or 'horizontal_plate'
    T_wall [K]: surface temperature
    P_film [Pa]: pressure at the film
    Tinf [K]: surrounding temperature
    L [m]: characteristc length
    D_pipe [m]: pipe diameter
    PlateNum : 'upper_surface' or 'lower_surface'
    ----------
    Return
    h [W/m^2 K]: Heat transfer coefficient
    """
    AS = fluid.abstractState.abstractState
    # Gravity acceleration

    # Film temperature, used to calculate thermal propertiers
    T_film = (T_wall + T_inf)/2 # [K]

    # thermal expansion coefficient, assumed ideal gas
    beta = 1/T_film # [1/K]

    # Transport properties calcualtion film
    AS.update(CP.PT_INPUTS,P_film,T_film)# use the film temperature to find the outer convective coefficient

    rho_film = AS.rhomass() #[kg/m3]
    k_film = AS.conductivity() #[W/m-K]
    mu_film = AS.viscosity() #[Pa-s OR kg/m-s]
    nu_film = mu_film/rho_film  #[m^2/s]
    cp_film = AS.cpmass() #[J/kg/K]
    Pr_film = cp_film*mu_film/k_film #[-]

    grashofNum = FluidMechanics.calculateGrashofNumber(beta, max(T_inf,T_wall), min(T_inf, T_wall), L, nu_film)

    # Rayleigh number
    RaL = grashofNum*Pr_film #[-]

    if HTCat == 'horizontal_pipe':
        RaD =  RaL*D_pipe**3/L**3 #[-]

    if RaL < 1e-3:
        Nu = 0.0 #[-]
    else:
        if HTCat == 'vertical_plate':
            if RaL > 1e9:
                Nu = (0.825 + 0.387*RaL**(1/6) / ( 1 + (0.492/Pr_film)**(9/16) )**(8/27))**2 #[-]
            else:
                Nu = 0.68 + 0.670*RaL**(1/4) / ( 1 + (0.492/Pr_film)**(9/16) )**(4/9) #[-]

        elif HTCat == 'horizontal_plate':
            if PlateNum == 'upper_surface':
                if T_wall > T_inf: #hot plate
                    if 1e4 <= RaL <= 1e7:
                        Nu = 0.54*RaL**(1/4) #[-]
                    elif 1e7 <= RaL <= 1e11:
                        Nu = 0.15*RaL**(1/3) #[-]
                    else:
                        Nu = 0.71*RaL**(1/4) #[-]

                else: # cold plate
                    if 1e5 <= RaL <= 1e10:
                        Nu = 0.27*RaL**(1/4)
                    else:
                        Nu = 0.71*RaL**(1/4) #[-]

            elif PlateNum == 'lower_surface':
                if T_wall > T_inf: # hot plate
                    if 1e5 <= RaL <= 1e10:
                        Nu = 0.27*RaL**(1/4) #[-]
                    else:
                        Nu = 0.25*RaL**(1/4) #[-]

                else: # cold plate
                    if 1e4 <= RaL <= 1e7:
                        Nu = 0.54*RaL**(1/4) #[-]
                    elif 1e7 <= RaL <= 1e11:
                        Nu = 0.15*RaL**(1/3) #[-]
                    else:
                        Nu = 0.25*RaL**(1/4) #[-]
            else:
                raise ValueError('PlateNum must be either upper_surface or lower_surface')


        elif HTCat == 'horizontal_pipe':
            if RaD <= 1e12:
                # Churchill and Chu, 1975, RaL->RaD
                Nu = (0.60+0.387*RaD**(1/6)/(1 + (0.559/Pr_film)**(9/16))**(8/27))**2 #[-]

            else: # Kuehn and Goldstein, 1976.
                temp = (( 0.518*(RaD**0.25)*(1+(0.559/Pr_film)**0.6)**(-5/12) )**15 + \
                        (0.1*RaD**(1/3))**15)**(1/15)
                Nu = 2/(np.log(1 + 2/temp)) #[-]

        elif HTCat == 'vertical_pipe':
            if (D_pipe/L) < 35/grashofNum**(1/4):
                F = 1/3*((L/D_pipe)/(1/grashofNum))**(1/4)+1 #[-]
            else:
                F = 1.0 #[-]
            Nu = F*(0.825 + 0.387*RaL**(1/6)/(1+(0.492/Pr_film)**(9/16))**(8/27))**2 #[-]

        else:
            raise NotImplementedError('not implemented')

    # Convective heat transfer coefficient
    if HTCat == 'horizontal_pipe':
        h = Nu*k_film/D_pipe #[W/m^2 K]
    else:
        h = Nu*k_film/L #[W/m^2 K]

    return h

if __name__=='__main__':
    DP_vals_acc=[]
    DP_vals_fric=[]
    x_vals=[]
    import pylab
    abstractState = CP.AbstractState("HEOS", "R410A")
    for xValue in np.linspace(0.1,1.0,10):
        DP_vals_acc.append(calculateAccelerationalPressureDrop(xValue-0.1,xValue,abstractState,2,250,250))
        DP_vals_fric.append(lmPressureGradientAvg(xValue-0.1,xValue,abstractState,0.1,0.01,250,250)*1*1)
        x_vals.append(xValue)

    print("plot shows accelerational pressure drop as f(x) for 0.1 x segments")
    pylab.plot(x_vals, DP_vals_acc)
    pylab.show()
    print("plot shows frictional pressure drop as f(x) for 0.1 x segments of a fictional tube\
          with unit length")
    pylab.plot(x_vals, DP_vals_fric)
    pylab.show()

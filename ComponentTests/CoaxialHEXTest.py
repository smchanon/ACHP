# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 07:25:51 2024

@author: smcanana
"""
import numpy as np
import pylab
import logging
from ACHP.models.Fluid import Fluid
from ACHP.models.HeatExchangers import CoaxialHEX
from ACHP.wrappers.CoolPropWrapper import PropsSIWrapper
        
if __name__=='__main__':
    logging.basicConfig(filename="ACHPlog.log", level=logging.DEBUG, encoding='utf-8',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    TT=[]
    QQ=[]
    Q1=[]
    #cold
    refrigerant = Fluid("R290", "HEOS")
    #hot/glycol
    coolant = Fluid("Water", "HEOS")
    propsSI = PropsSIWrapper()
    for Tdew_evap in np.linspace(270,290.4):
        Tdew_cond=317.73
#        Tdew_evap=285.42
        pdew_cond = propsSI.calculatePressureFromTandQ(refrigerant, Tdew_cond, 1.0)
        h = propsSI.calculateEnthalpyFromTandP(refrigerant, Tdew_cond-7, pdew_cond)
        params = {
                'fluidCold': refrigerant,
                'fluidHot': coolant,
                'massFlowCold': 0.040,
                'massFlowHot': 0.38,
                'pressureInCold': propsSI.calculatePressureFromTandQ(refrigerant, Tdew_evap, 1.0),
                'pressureInHot': 300000,
                'enthalpyInCold': h,
                'enthalpyInHot': propsSI.calculateEnthalpyFromTandP(coolant, 290.52, 300000),
                #'tempInCold': 290.52, #TODO: this is enthalpyInCold right now
                'innerTubeInnerDiameter': 0.0278,      #inner tube, Internal Diameter (ID)
                'innerTubeOuterDiameter': 0.03415,     #inner tube, Outer Diameter (OD)
                'outerTubeInnerDiameter': 0.045,       #outer tube (annulus), Internal Diameter (ID)
                'length': 50,
                'conductivity' : 237, #[W/m-K]
                }
        coaxialHX = CoaxialHEX(**params)
        coaxialHX.calculate()
        
        TT.append(Tdew_evap)
        QQ.append(coaxialHX.fluidProps["Cold"].heatTransferCoeffEffective["TwoPhase"])
        Q1.append(coaxialHX.fluidProps["Cold"].heatTransferCoeffEffective["Superheated"])
        print (coaxialHX.heatTransferred)
    pylab.plot(TT,QQ)
    pylab.plot(TT,Q1)
    pylab.show()
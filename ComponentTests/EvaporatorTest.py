import CoolProp as CP
import numpy as np
import pylab
import logging
from ACHP.models.FinnedTube import FinnedTube, Tubes, WavyLouveredFins, Air
from ACHP.models.Evaporator import Evaporator
from ACHP.models.Fluid import Fluid
from ACHP.wrappers.CoolPropWrapper import PropsSIWrapper
from ACHP.FinCorrelations import FinInputs
from ACHP.Evaporator import EvaporatorClass

def Evaporator1():
    FinsTubes=FinInputs()
    
    FinsTubes.Tubes.NTubes_per_bank=32
    FinsTubes.Tubes.Ncircuits=5
    FinsTubes.Tubes.Nbank=3
    FinsTubes.Tubes.Ltube=0.452
    FinsTubes.Tubes.OD=0.009525
    FinsTubes.Tubes.ID=0.0089154
    FinsTubes.Tubes.Pl=0.0254
    FinsTubes.Tubes.Pt=0.0219964
    FinsTubes.Tubes.kw=237                   #Wall thermal conductivity
    
    FinsTubes.Fins.FPI=14.5
    FinsTubes.Fins.Pd=0.001
    FinsTubes.Fins.xf=0.001
    FinsTubes.Fins.t=0.00011
    FinsTubes.Fins.k_fin=237
    
    FinsTubes.Air.Vdot_ha=0.5663
    FinsTubes.Air.Tdb=299.8
    FinsTubes.Air.p=101325
    FinsTubes.Air.RH=0.51
    FinsTubes.Air.FanPower=438
    
    Ref = 'R410A'
    Backend = 'TTSE&HEOS' #choose between: 'HEOS','TTSE&HEOS','BICUBIC&HEOS','REFPROP','SRK','PR'
    AS = CP.AbstractState(Backend, Ref)
    
    kwargs={'AS': AS,
            'mdot_r': 0.0708,
            'psat_r': PropsSI('P','T',282,'Q',1.0,Ref),
            'Fins': FinsTubes,
            'FinsType': 'WavyLouveredFins', #WavyLouveredFins, HerringboneFins, PlainFins
            'hin_r': PropsSI('H','P',PropsSI('P','T',282,'Q',1.0,Ref),'Q',0.15,Ref), #[J/kg]
            'Verbosity': 0,
            }
    
    Evap=EvaporatorClass(**kwargs)
    Evap.Update(**kwargs)
    Evap.Calculate()
    
    print ('Evaporator heat transfer rate is',Evap.Q,'W')
    print ('Evaporator capacity (less fan power) is',Evap.Capacity,'W')
    print ('Evaporator fraction of length in two-phase section',Evap.w_2phase,'W')
    print ('Evaporator sensible heat ratio',Evap.SHR)

if __name__=='__main__':
    logging.basicConfig(filename="ACHPlog.log", level=logging.DEBUG, encoding='utf-8',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #Example usage for a parametric study

    num_points= 101
    T_dews= np.linspace(270,299.7,num_points)
    TT= np.empty(num_points)
    Q_2p= np.empty(num_points)
    w_2p= np.empty(num_points)
    w_sh= np.empty(num_points)
    Q_tot= np.empty(num_points)
    h_2p= np.empty(num_points)
    h_sh= np.empty(num_points)

    evapTubes = Tubes(32, 3, 5, 0.452, 0.0089154, 0.009525, 0.0254, 0.0219964, 237)
    evapFins = WavyLouveredFins(14.5, 0.00011, 237, 0.001, 0.001)
    evapAir = Air(0.5663, 299.9, 101325, 0.51, 438)
    finsTubes = FinnedTube(evapTubes, evapFins, evapAir)

    #Abstract State
    ref = Fluid('R410A', 'HEOS')
    
    propsSI = PropsSIWrapper()
    evaporator = Evaporator(finsTubes, ref, 0.0708, propsSI.calculatePressureFromTandQ(ref,T_dews[0],1.0), 
                    enthalpyInR=propsSI.calculateEnthalpyFromPandQ(ref, propsSI.calculatePressureFromTandQ(ref, 282,1.0),0.15))

    # kwargs={'AS': AS,
    #         'mdot_r':  0.0708,
    #         'psat_r':  PropsSI('P','T',T_dews[0],'Q',1.0,Ref),
    #         'Fins': FinsTubes,
    #         'FinsType': 'WavyLouveredFins',  #Choose fin Type: 'WavyLouveredFins' or 'HerringboneFins'or 'PlainFins'
    #         'hin_r': PropsSI('H','P',PropsSI('P','T',282,'Q',1.0,Ref),'Q',0.15,Ref),
    #         'Verbosity': 8,
    #         'h_a_tuning':1,
    #         'h_tp_tuning':1,
    #         'DP_tuning':1,
    #     }

    # Evap=Evaporator(**kwargs) #generate new evaporator instance and update kwargs

    for i in range(0, len(T_dews)):
        evaporator = Evaporator(finsTubes, ref, 0.0708, propsSI.calculatePressureFromTandQ(ref,T_dews[0],1.0), 
                        enthalpyInR=propsSI.calculateEnthalpyFromPandQ(ref,propsSI.calculatePressureFromTandQ(ref,282,1.0),0.15))
        evaporator.calculate()
        Q_tot[i] = evaporator.Q
        Q_2p[i]= evaporator.Q_2phase
        w_2p[i]= evaporator.w_2phase
        w_sh[i]= evaporator.w_superheat
        h_2p[i]= evaporator.h_r_2phase
        h_sh[i]= evaporator.h_r_superheat

    print ("Demonstrate output list")
    #print (Evap.OutputList())
    for id, unit, value in evaporator.OutputList():
        print (str(id) + ' = ' + str(value) + ' ' + str(unit))

    pylab.plot(T_dews,Q_2p,T_dews,Q_tot)
    pylab.title('Parametric Study With Fixed flowrates - Capacity')
    pylab.legend(['two-phase','total'],loc='best')
    pylab.title('Parametric Study With Fixed flowrates - Capacity')
    pylab.xlabel('Evaporation Dew Temperature in Kelvin')
    pylab.ylabel('Capacity in Watt')
    #pylab.savefig('Evaporator_py_capacity.pdf')
    pylab.show()
    pylab.plot(T_dews,h_2p,T_dews, h_sh)
    pylab.title('Parametric Study with fixed Flowrates - Heat Transfer Coefficients')
    pylab.legend(['two-phase','superheat'],loc='best')
    pylab.xlabel('Evaporation Dew Temperature in Kelvin')
    pylab.ylabel('Heat Transfer Coefficient in W/m2-K')
    #pylab.savefig('Evaporator_py_HTC.pdf')
    pylab.show()
    pylab.plot(T_dews,w_2p, T_dews, w_sh)
    pylab.title('Parametric Study with fixed Flowrates - Area Fraction')
    pylab.legend(['two-phase', 'superheat'],loc='best')
    pylab.xlabel('Evaporation Dew Temperature in Kelvin')
    pylab.ylabel('Two-phase Wetted Area Fraction')
    pylab.ylim(-0.01,1.01)
    #pylab.savefig('Evaporator_py_wetted_area.pdf')
    pylab.show()

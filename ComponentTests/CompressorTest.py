from __future__ import division, print_function, absolute_import
from ACHP.models.Compressor import Compressor
from ACHP.models.Fluid import Fluid
from ACHP.wrappers.CoolPropWrapper import PropsSIWrapper

REFRIGERANT = 'R134a'
BACKEND = 'HEOS' #choose between: 'HEOS','TTSE&HEOS','BICUBIC&HEOS','REFPROP','SRK','PR'
ref = Fluid(REFRIGERANT, BACKEND)
propsSI = PropsSIWrapper()
for j in range(1):
    kwds={
          'massFlowCoeffs':[217.3163128,5.094492028,-0.593170311,4.38E-02,-2.14E-02,
                            1.04E-02,7.90E-05,-5.73E-05,1.79E-04,-8.08E-05],
          'powerCoeffs':[-561.3615705,-15.62601841,46.92506685,-0.217949552,0.435062616,
                         -0.442400826,2.25E-04,2.37E-03,-3.32E-03,2.50E-03],
          'refrigerant': ref,
          'tempInR':280,
          'pressureInR':propsSI.calculatePressureFromTandQ(REFRIGERANT, 279,1),
          'pressureOutR':propsSI.calculatePressureFromTandQ(REFRIGERANT, 315,1),
          'ambientPowerLoss':0.15, #Fraction of electrical power lost as heat to ambient
          'vDotRatio': 1.0, #Displacement Scale factor
          'shellPressure': 'low-pressure',
          'oil': 'POE32',
          'volumeOilSump': 0.0,
          }
Comp = Compressor(**kwds)
Comp.calculate()
print ('Electrical Power:', Comp.power,'W')
print ('Flow rate:',Comp.vDotPumped,'m^3/s')
print ('Heat loss rate:', Comp.ambientHeatLoss, 'W')
print ('Refrigerant dissolved in oil sump:', Comp.refrigerantChangeOilSump,'kg')
print ('Actual mass flow rate is: ' + str(Comp.massFlowR) + ' kg/s')
print ('Isentropic Efficiency is: ' + str(Comp.overallIsentropicEfficiency))
print ('Discharge Refrigerant Temperature is: ' + str(Comp.tempOutR) + ' K')
print (' ')

'''to print all the output, uncomment the next 2 lines'''
# for id, unit, value in Comp.OutputList():                
#     print (str(id) + ' = ' + str(value) + ' ' + str(unit))
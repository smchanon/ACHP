# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:26:43 2024

@author: SMCANANA
"""
import pytest
from ACHP.calculations.Conversions import *

def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )

@pytest.fixture
def tc():
    return TemperatureConversions()

@pytest.fixture
def mfc():
    return MassFlowConversions()

@pytest.fixture
def vfc():
    return VolumetricFlowConversions()

@pytest.fixture
def wc():
    return PowerConversions()

@pytest.fixture
def pc():
    return PressureConversions()

@pytest.fixture
def gc():
    return GeometricConversions()

@pytest.fixture
def mc():
    return MassConversions()

@pytest.fixture
def cpc():
    return ComposedPropertyConversions()

class TestTemperatureConversions:
    params = {
        "test_convertTemperature_F2C": [dict(tempF=32, tempC=0),
                                        dict(tempF=-40, tempC=-40),
                                        dict(tempF=100, tempC=37.7778),
                                        dict(tempF=12, tempC=-11.1111)],
        "test_convertTemperature_F2K": [dict(tempF=32, tempK=273.15),
                                        dict(tempF=-40, tempK=233.15),
                                        dict(tempF=100, tempK=310.9278),
                                        dict(tempF=12, tempK=262.0389)],
        "test_convertTemperature_C2F": [dict(tempF=32, tempC=0),
                                        dict(tempF=-40, tempC=-40),
                                        dict(tempF=100, tempC=37.7778),
                                        dict(tempF=12, tempC=-11.1111)],
        "test_convertTemperature_C2K": [dict(tempC=0, tempK=273.15),
                                        dict(tempC=-40, tempK=233.15),
                                        dict(tempC=100, tempK=373.15),
                                        dict(tempC=-5, tempK=268.15)],
        "test_convertTemperature_K2F": [dict(tempK=0, tempF=-459.67),
                                        dict(tempK=300, tempF=80.33),
                                        dict(tempK=233, tempF=-40.27),
                                        dict(tempK=574.5875, tempF=574.5875)],
        "test_convertTemperature_K2C": [dict(tempK=0, tempC=-273.15),
                                        dict(tempK=300, tempC=26.85),
                                        dict(tempK=233, tempC=-40.15),
                                        dict(tempK=574.5875, tempC=301.4375)]
        }

    def test_convertTemperature_F2C(self, tempF, tempC, tc):
        assert(round(tc.convertTemperature(tc.Unit.F, tc.Unit.C, tempF), 4) == tempC)

    def test_convertTemperature_F2K(self, tempF, tempK, tc):
        assert(round(tc.convertTemperature(tc.Unit.F, tc.Unit.K, tempF), 4) == tempK)

    def test_convertTemperature_C2F(self, tempC, tempF, tc):
        assert(round(tc.convertTemperature(tc.Unit.C, tc.Unit.F, tempC), 4) == tempF)

    def test_convertTemperature_C2K(self, tempC, tempK, tc):
        assert(round(tc.convertTemperature(tc.Unit.C, tc.Unit.K, tempC), 4) == tempK)

    def test_convertTemperature_K2F(self, tempK, tempF, tc):
        assert(round(tc.convertTemperature(tc.Unit.K, tc.Unit.F, tempK), 4) == tempF)

    def test_convertTemperature_K2C(self, tempK, tempC, tc):
        assert(round(tc.convertTemperature(tc.Unit.K, tc.Unit.C, tempK), 4) == tempC)

class TestMassFlowConversions:
    params = {
        "test_convertMassFlow_LBH2KGS": [dict(lbh=0, kgs=0),
                                        dict(lbh=125, kgs=0.0157),
                                        dict(lbh=125732, kgs=15.8420),
                                        dict(lbh=15600, kgs=1.9656)],
        "test_convertMassFlow_LBM2KGS": [dict(lbm=0, kgs=0),
                                        dict(lbm=125, kgs=0.9450),
                                        dict(lbm=1050, kgs=7.9379),
                                        dict(lbm=0.16, kgs=0.0012)],
        "test_convertMassFlow_KGS2LBH": [dict(kgs=0, lbh=0),
                                        dict(kgs=0.0015, lbh=11.9050),
                                        dict(kgs=1, lbh=7936.6414),
                                        dict(kgs=0.59, lbh=4682.6184)],
        "test_convertMassFlow_KGS2LBM": [dict(kgs=0, lbm=0),
                                        dict(kgs=0.0015, lbm=0.1984),
                                        dict(kgs=1, lbm=132.2774),
                                        dict(kgs=0.59, lbm=78.0436)],
        "test_convertMassFlow_LBH2LBM": [dict(lbh=0, lbm=0),
                                        dict(lbh=125, lbm=2.0833),
                                        dict(lbh=125732, lbm=2095.5333),
                                        dict(lbh=15600, lbm=260)],
        "test_convertMassFlow_LBM2LBH": [dict(lbm=0, lbh=0),
                                        dict(lbm=125, lbh=7500),
                                        dict(lbm=1050, lbh=63000),
                                        dict(lbm=0.16, lbh=9.6)]
        }

    def test_convertMassFlow_LBH2KGS(self, lbh, kgs, mfc):
        assert(round(mfc.convertMassFlow(mfc.Unit.LBH, mfc.Unit.KGS, lbh), 4) == kgs)

    def test_convertMassFlow_LBM2KGS(self, lbm, kgs, mfc):
        assert(round(mfc.convertMassFlow(mfc.Unit.LBM, mfc.Unit.KGS, lbm), 4) == kgs)

    def test_convertMassFlow_KGS2LBH(self, kgs, lbh, mfc):
        assert(round(mfc.convertMassFlow(mfc.Unit.KGS, mfc.Unit.LBH, kgs), 4) == lbh)

    def test_convertMassFlow_KGS2LBM(self, kgs, lbm, mfc):
        assert(round(mfc.convertMassFlow(mfc.Unit.KGS, mfc.Unit.LBM, kgs), 4) == lbm)

    def test_convertMassFlow_LBH2LBM(self, lbh, lbm, mfc):
        assert(round(mfc.convertMassFlow(mfc.Unit.LBH, mfc.Unit.LBM, lbh), 4) == lbm)

    def test_convertMassFlow_LBM2LBH(self, lbm, lbh, mfc):
        assert(round(mfc.convertMassFlow(mfc.Unit.LBM, mfc.Unit.LBH, lbm), 4) == lbh)

class TestVolumetricFlowConversions:
    params = {
        "test_convertVolumetricFlow_CFM2CMS": [dict(cfm=0, cms=0),
                                        dict(cfm=125, cms=0.05900),
                                        dict(cfm=125732, cms=59.3389),
                                        dict(cfm=15600, cms=7.3624)],
        "test_convertVolumetricFlow_CFM2GPM": [dict(cfm=0, gpm=0),
                                        dict(cfm=125, gpm=935.0678),
                                        dict(cfm=125732, gpm=940543.6035),
                                        dict(cfm=0.16, gpm=1.1969)],
        "test_convertVolumetricFlow_CMS2CFM": [dict(cms=0, cfm=0),
                                        dict(cms=0.0015, cfm=3.1783),
                                        dict(cms=1, cfm=2118.8800),
                                        dict(cms=0.59, cfm=1250.1392)],
        "test_convertVolumetricFlow_CMS2GPM": [dict(cms=0, gpm=0),
                                        dict(cms=0.0015, gpm=23.7756),
                                        dict(cms=1, gpm=15850.3725),
                                        dict(cms=0.59, gpm=9351.7198)],
        "test_convertVolumetricFlow_GPM2CMS": [dict(gpm=0, cms=0),
                                        dict(gpm=125, cms=0.0079),
                                        dict(gpm=1050, cms=0.0662),
                                        dict(gpm=125732, cms=7.9324)],
        "test_convertVolumetricFlow_GPM2CFM": [dict(gpm=0, cfm=0),
                                        dict(gpm=125, cfm=16.7100),
                                        dict(gpm=1050, cfm=140.3641),
                                        dict(gpm=0.16, cfm=0.0214)]
        }

    def test_convertVolumetricFlow_CFM2CMS(self, cfm, cms, vfc):
        assert(round(vfc.convertVolumetricFlow(vfc.Unit.CFM, vfc.Unit.CMS, cfm), 4) == cms)

    def test_convertVolumetricFlow_CFM2GPM(self, cfm, gpm, vfc):
        assert(round(vfc.convertVolumetricFlow(vfc.Unit.CFM, vfc.Unit.GPM, cfm), 4) == gpm)

    def test_convertVolumetricFlow_CMS2CFM(self, cms, cfm, vfc):
        assert(round(vfc.convertVolumetricFlow(vfc.Unit.CMS, vfc.Unit.CFM, cms), 4) == cfm)

    def test_convertVolumetricFlow_CMS2GPM(self, cms, gpm, vfc):
        assert(round(vfc.convertVolumetricFlow(vfc.Unit.CMS, vfc.Unit.GPM, cms), 4) == gpm)

    def test_convertVolumetricFlow_GPM2CMS(self, gpm, cms, vfc):
        assert(round(vfc.convertVolumetricFlow(vfc.Unit.GPM, vfc.Unit.CMS, gpm), 4) == cms)

    def test_convertVolumetricFlow_GPM2CFM(self, gpm, cfm, vfc):
        assert(round(vfc.convertVolumetricFlow(vfc.Unit.GPM, vfc.Unit.CFM, gpm), 4) == cfm)

class TestPowerConversions:
    params = {
        "test_convertPower_BTUH2W": [dict(btuh=0, watts=0),
                                        dict(btuh=125, watts=36.6339),
                                        dict(btuh=125732, watts=36848.4118),
                                        dict(btuh=0.59, watts=0.1729)],
        "test_convertPower_BTUH2HP": [dict(btuh=0, hp=0),
                                        dict(btuh=125, hp=0.0491),
                                        dict(btuh=125732, hp=49.4145),
                                        dict(btuh=0.59, hp=0.0002)],
        "test_convertPower_W2BTUH": [dict(watts=0, btuh=0),
                                        dict(watts=35, btuh=119.425),
                                        dict(watts=100, btuh=341.2142),
                                        dict(watts=0.59, btuh=2.0132)],
        "test_convertPower_W2HP": [dict(watts=0, hp=0),
                                        dict(watts=35, hp=0.0469),
                                        dict(watts=100, hp=0.1341),
                                        dict(watts=0.59, hp=0.0008)],
        "test_convertPower_HP2W": [dict(hp=0, watts=0),
                                        dict(hp=125, watts=93212.484),
                                        dict(hp=0.136, watts=101.4152),
                                        dict(hp=12, watts=8948.3985)],
        "test_convertPower_HP2BTUH": [dict(hp=0, btuh=0),
                                        dict(hp=125, btuh=318054.2014),
                                        dict(hp=0.136, btuh=346.0430),
                                        dict(hp=12, btuh=30533.2033)]
        }

    def test_convertPower_BTUH2W(self, btuh, watts, wc):
        assert(round(wc.convertPower(wc.Unit.BTUH, wc.Unit.W, btuh), 4) == watts)

    def test_convertPower_BTUH2HP(self, btuh, hp, wc):
        assert(round(wc.convertPower(wc.Unit.BTUH, wc.Unit.HP, btuh), 4) == hp)

    def test_convertPower_W2BTUH(self, watts, btuh, wc):
        assert(round(wc.convertPower(wc.Unit.W, wc.Unit.BTUH, watts), 4) == btuh)

    def test_convertPower_W2HP(self, watts, hp, wc):
        assert(round(wc.convertPower(wc.Unit.W, wc.Unit.HP, watts), 4) == hp)

    def test_convertPower_HP2W(self, hp, watts, wc):
        assert(round(wc.convertPower(wc.Unit.HP, wc.Unit.W, hp), 4) == watts)

    def test_convertPower_HP2BTUH(self, hp, btuh, wc):
        assert(round(wc.convertPower(wc.Unit.HP, wc.Unit.BTUH, hp), 4) == btuh)

class TestPressureConversions:
    params = {
        "test_convertPressure_PA2PSI": [dict(pa=0, psi=0),
                                        dict(pa=125, psi=0.0181),
                                        dict(pa=125732, psi=18.2359),
                                        dict(pa=15600, psi=2.2626)],
        "test_convertPressure_PA2BAR": [dict(pa=0, bar=0),
                                        dict(pa=125, bar=0.0013),
                                        dict(pa=125732, bar=1.2573),
                                        dict(pa=5700000, bar=57)],
        "test_convertPressure_PSI2PA": [dict(psi=0, pa=0),
                                        dict(psi=0.0015, pa=10.3421),
                                        dict(psi=1, pa=6894.7573),
                                        dict(psi=500, pa=3447378.6465)],
        "test_convertPressure_PSI2BAR": [dict(psi=0, bar=0),
                                        dict(psi=0.0015, bar=0.0001),
                                        dict(psi=1, bar=0.0689),
                                        dict(psi=500, bar=34.4738)],
        "test_convertPressure_BAR2PSI": [dict(bar=0, psi=0),
                                        dict(bar=125, psi=1812.9717),
                                        dict(bar=0.43, psi=6.2366),
                                        dict(bar=87.6, psi=1270.5306)],
        "test_convertPressure_BAR2PA": [dict(bar=0, pa=0),
                                        dict(bar=125, pa=1.25e7),
                                        dict(bar=0.43, pa=4.3e4),
                                        dict(bar=87.6, pa=8.76e6)]
        }

    def test_convertPressure_PA2PSI(self, pa, psi, pc):
        assert(round(pc.convertPressure(pc.Unit.PA, pc.Unit.PSI, pa), 4) == psi)

    def test_convertPressure_PA2BAR(self, pa, bar, pc):
        assert(round(pc.convertPressure(pc.Unit.PA, pc.Unit.BAR, pa), 4) == bar)

    def test_convertPressure_PSI2PA(self, psi, pa, pc):
        assert(round(pc.convertPressure(pc.Unit.PSI, pc.Unit.PA, psi), 4) == pa)

    def test_convertPressure_PSI2BAR(self, psi, bar, pc):
        assert(round(pc.convertPressure(pc.Unit.PSI, pc.Unit.BAR, psi), 4) == bar)

    def test_convertPressure_BAR2PSI(self, bar, psi, pc):
        assert(round(pc.convertPressure(pc.Unit.BAR, pc.Unit.PSI, bar), 4) == psi)

    def test_convertPressure_BAR2PA(self, bar, pa, pc):
        assert(round(pc.convertPressure(pc.Unit.BAR, pc.Unit.PA, bar), 4) == pa)

class TestGeometricConversions:
    params = {
        "test_convertLength_M2IN": [dict(meter=0, inch=0),
                                        dict(meter=1, inch=39.3701),
                                        dict(meter=563, inch=22165.3543),
                                        dict(meter=0.3048, inch=12)],
        "test_convertLength_M2FT": [dict(meter=0, feet=0),
                                        dict(meter=1, feet=3.2808),
                                        dict(meter=563, feet=1847.1129),
                                        dict(meter=0.3048, feet=1)],
        "test_convertLength_IN2M": [dict(inch=0, meter=0),
                                        dict(inch=2.5, meter=0.0635),
                                        dict(inch=12, meter=0.3048),
                                        dict(inch=500, meter=12.7)],
        "test_convertLength_IN2FT": [dict(inch=0, feet=0),
                                        dict(inch=2.5, feet=0.2083),
                                        dict(inch=12, feet=1),
                                        dict(inch=500, feet=41.6667)],
        "test_convertLength_FT2IN": [dict(feet=0, inch=0),
                                        dict(feet=1, inch=12),
                                        dict(feet=0.43, inch=5.16),
                                        dict(feet=87.6, inch=1051.2)],
        "test_convertLength_FT2M": [dict(feet=0, meter=0),
                                        dict(feet=1, meter=0.3048),
                                        dict(feet=0.43, meter=0.1311),
                                        dict(feet=87.6, meter=26.7005)],
        "test_convertArea_M2IN": [dict(meter=0, inch=0),
                                        dict(meter=1, inch=1550.0031),
                                        dict(meter=563, inch=872651.7453),
                                        dict(meter=0.3048, inch=472.4409)],
        "test_convertArea_M2FT": [dict(meter=0, feet=0),
                                        dict(meter=1, feet=10.7639),
                                        dict(meter=563, feet=6060.0816),
                                        dict(meter=0.0929034, feet=1)],
        "test_convertArea_IN2M": [dict(inch=0, meter=0),
                                        dict(inch=144, meter=0.0929),
                                        dict(inch=1296, meter=0.8361),
                                        dict(inch=50000, meter=32.258)],
        "test_convertArea_IN2FT": [dict(inch=0, feet=0),
                                        dict(inch=144, feet=1),
                                        dict(inch=1296, feet=9),
                                        dict(inch=500, feet=3.4722)],
        "test_convertArea_FT2IN": [dict(feet=0, inch=0),
                                        dict(feet=1, inch=144),
                                        dict(feet=0.43, inch=61.92),
                                        dict(feet=87.6, inch=12614.4)],
        "test_convertArea_FT2M": [dict(feet=0, meter=0),
                                        dict(feet=1, meter=0.0929),
                                        dict(feet=563, meter=52.3044),
                                        dict(feet=87.6, meter=8.1383)],
        "test_convertVolume_M2IN": [dict(meter=0, inch=0),
                                        dict(meter=1, inch=61023.7441),
                                        dict(meter=563, inch=34356367.9253),
                                        dict(meter=0.028316846592, inch=1728)],
        "test_convertVolume_M2FT": [dict(meter=0, feet=0),
                                        dict(meter=1, feet=35.3147),
                                        dict(meter=563, feet=19882.1574),
                                        dict(meter=0.028316846592, feet=1)],
        "test_convertVolume_IN2M": [dict(inch=0, meter=0),
                                        dict(inch=25, meter=0.0004),
                                        dict(inch=1728, meter=0.0283),
                                        dict(inch=500000, meter=8.1935)],
        "test_convertVolume_IN2FT": [dict(inch=0, feet=0),
                                        dict(inch=25, feet=0.0145),
                                        dict(inch=1728, feet=1),
                                        dict(inch=50000, feet=28.9352)],
        "test_convertVolume_FT2IN": [dict(feet=0, inch=0),
                                        dict(feet=1, inch=1728),
                                        dict(feet=0.43, inch=743.04),
                                        dict(feet=87.6, inch=151372.8)],
        "test_convertVolume_FT2M": [dict(feet=0, meter=0),
                                        dict(feet=1, meter=0.0283),
                                        dict(feet=0.43, meter=0.0122),
                                        dict(feet=87.6, meter=2.4806)]
        }

    def test_convertLength_M2IN(self, meter, inch, gc):
        assert(round(gc.convertLength(gc.Unit.M, gc.Unit.IN, meter), 4) == inch)

    def test_convertLength_M2FT(self, meter, feet, gc):
        assert(round(gc.convertLength(gc.Unit.M, gc.Unit.FT, meter), 4) == feet)

    def test_convertLength_IN2M(self, inch, meter, gc):
        assert(round(gc.convertLength(gc.Unit.IN, gc.Unit.M, inch), 4) == meter)

    def test_convertLength_IN2FT(self, inch, feet, gc):
        assert(round(gc.convertLength(gc.Unit.IN, gc.Unit.FT, inch), 4) == feet)

    def test_convertLength_FT2IN(self, feet, inch, gc):
        assert(round(gc.convertLength(gc.Unit.FT, gc.Unit.IN, feet), 4) == inch)

    def test_convertLength_FT2M(self, feet, meter, gc):
        assert(round(gc.convertLength(gc.Unit.FT, gc.Unit.M, feet), 4) == meter)

    def test_convertArea_M2IN(self, meter, inch, gc):
        assert(round(gc.convertArea(gc.Unit.M, gc.Unit.IN, meter), 4) == inch)

    def test_convertArea_M2FT(self, meter, feet, gc):
        assert(round(gc.convertArea(gc.Unit.M, gc.Unit.FT, meter), 4) == feet)

    def test_convertArea_IN2M(self, inch, meter, gc):
        assert(round(gc.convertArea(gc.Unit.IN, gc.Unit.M, inch), 4) == meter)

    def test_convertArea_IN2FT(self, inch, feet, gc):
        assert(round(gc.convertArea(gc.Unit.IN, gc.Unit.FT, inch), 4) == feet)

    def test_convertArea_FT2IN(self, feet, inch, gc):
        assert(round(gc.convertArea(gc.Unit.FT, gc.Unit.IN, feet), 4) == inch)

    def test_convertArea_FT2M(self, feet, meter, gc):
        assert(round(gc.convertArea(gc.Unit.FT, gc.Unit.M, feet), 4) == meter)

    def test_convertVolume_M2IN(self, meter, inch, gc):
        assert(round(gc.convertVolume(gc.Unit.M, gc.Unit.IN, meter), 4) == inch)

    def test_convertVolume_M2FT(self, meter, feet, gc):
        assert(round(gc.convertVolume(gc.Unit.M, gc.Unit.FT, meter), 4) == feet)

    def test_convertVolume_IN2M(self, inch, meter, gc):
        assert(round(gc.convertVolume(gc.Unit.IN, gc.Unit.M, inch), 4) == meter)

    def test_convertVolume_IN2FT(self, inch, feet, gc):
        assert(round(gc.convertVolume(gc.Unit.IN, gc.Unit.FT, inch), 4) == feet)

    def test_convertVolume_FT2IN(self, feet, inch, gc):
        assert(round(gc.convertVolume(gc.Unit.FT, gc.Unit.IN, feet), 4) == inch)

    def test_convertVolume_FT2M(self, feet, meter, gc):
        assert(round(gc.convertVolume(gc.Unit.FT, gc.Unit.M, feet), 4) == meter)

class TestMassConversions:
    params = {
        "test_convertMass_OZ2KG": [dict(oz=0, kg=0),
                                        dict(oz=0.23, kg=0.0065),
                                        dict(oz=1, kg=0.0283),
                                        dict(oz=500, kg=14.1747)],
        "test_convertMass_KG2OZ": [dict(kg=0, oz=0),
                                        dict(kg=0.15, oz=5.2911),
                                        dict(kg=1, oz=35.2740),
                                        dict(kg=65, oz=2292.8094)]
        }

    def test_convertMass_OZ2KG(self, oz, kg, mc):
        assert(round(mc.convertMass(mc.Unit.OZ, mc.Unit.KG, oz), 4) == kg)

    def test_convertMass_KG2OZ(self, kg, oz, mc):
        assert(round(mc.convertMass(mc.Unit.KG, mc.Unit.OZ, kg), 4) == oz)

class TestComposedPropertyConversions:
    params = {
        "test_convertComposedProperty_IPK2SIK": [dict(ipk=0, sik=0),
                                        dict(ipk=0.23, sik=0.3981),
                                        dict(ipk=1, sik=1.7307),
                                        dict(ipk=500, sik=865.3675)],
        "test_convertComposedProperty_SIK2IPK": [dict(sik=0, ipk=0),
                                        dict(sik=0.15, ipk=0.0867),
                                        dict(sik=1, ipk=0.5778),
                                        dict(sik=65, ipk=37.5563)]
        }

    def test_convertComposedProperty_IPK2SIK(self, ipk, sik, cpc):
        assert(round(cpc.convertComposedProperty(cpc.Unit.IPK, cpc.Unit.SIK, ipk), 4) == sik)

    def test_convertComposedProperty_SIK2IPK(self, sik, ipk, cpc):
        assert(round(cpc.convertComposedProperty(cpc.Unit.SIK, cpc.Unit.IPK, sik), 4) == ipk)

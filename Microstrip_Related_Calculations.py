#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:08:48 2020

@author: wenninger
"""
import numpy as np
import scipy.constants as const
import sys

sys.path.insert(1, '../Superconductivity')

from Fundamental_BCS_Equations import conductivity_BCS

def zIn_lossless(z0,zL,betaL):
    '''This function computes the input impedance of a terminated ossless transmission line.
    
    inputs
    ------
    z0: complex
        The impedance of the transmission line.
    zL: complex
        The impedance of the load at the end of the transmission line.
    betaL: float
        The angle of the signal which is given by beta times the length of the transmission line. Beta is 2pi/wavelength.
        The unit is radiants
        
    returns
    -------
    complex
        The impedance seen at the input of the transmission line.
    '''
    dividend = zL+ 1j*z0*np.tan(betaL)
    divisor = z0 + 1j*zL*np.tan(betaL)
    return z0*dividend/divisor

def zIn(z0,zL,gammaL):
    '''This function computes the input impedance of a terminated ossless transmission line.
    
    inputs
    ------
    z0: complex
        The impedance of the transmission line.
    zL: complex
        The impedance of the load at the end of the transmission line.
    gammaL: complex float
        The angle of the signal which is given by gamma times the length of the transmission line including the attenuation.
        The unit is radiants
        
    returns
    -------
    complex
        The impedance seen at the input of the transmission line.
    '''
    dividend = zL+ 1j*z0*np.tanh(gammaL)
    divisor = z0 + 1j*zL*np.tanh(gammaL)
    return z0*dividend/divisor


class Microstrip():
    '''This class is a wrapper around the functions for calculations related to microstrips.
    '''
    def effective_Permittivity(self,epsilonDielectric, d, w):
        '''This function computes the effective permittivity of a microstrip following Pozar equation 3.195. 
        Modified of Schneider 1969, which has a factor 10 instead of 12.
        
        inputs
        ------
        epsilonDielectric: float
            The permittivity of the dielectric material.
        d: float
            The thickness of the dielectric in meter.
        w: float 
            The width of the microstrip in meter.
            
        returns
        -------
        float
            The effective permittivity of the microstrip configuration.
        '''
        return np.divide(epsilonDielectric+1,2)+np.divide(epsilonDielectric-1,2)*np.reciprocal(np.sqrt(1 + 12*d/w))
    
    def propagation_constant(self,freq, *args):
        '''This function computes the propagation constant following Pozar equation 3.194.
        
        inputs
        ------
        freq: float
            The frequency of the signal.
        args: 1 or 3 argument
            1 argument: float
                The effective permittivity of the microstrip.
            3 arguments:
                The arguments of :func: effective_Permittivity
        
        returns
        -------
        float
            The propagation constant beta.
        '''
        if len(args)==3:
            effective_Permittivity = self.effective_Permittivity(args[0],args[1],args[2])
        else:
            effective_Permittivity = args[0]
        return np.multiply(np.divide(2*np.pi*freq,const.c),np.sqrt(effective_Permittivity))

    def characteristic_Impedance_Z0(self,d, w, effective_Permittivity):
        '''This function computes the characteristic impedance Z_0 of a microstrip line following Pozar equation 3.196.
        
        inputs
        ------
        d: float
            The thickness of the dielectric in meter.
        w: float 
            The width of the microstrip in meter.
        effective_Permittivity: float
            The effective permittivity of the microstrip configuration.
            
        returns
        -------
        float
            The characteristic impedance obtained from the geometry as stated in Pozar.
        '''
        if w<=d:
            return np.divide(60,np.sqrt(effective_Permittivity))*np.log(np.divide(8*d,w)+np.divide(w,4*d))
        else:
            return np.divide(120*np.pi,np.multiply(np.sqrt(effective_Permittivity),np.divide(w,d)+1.393+.667*np.log(np.divide(w,d)+1.444)))

M = Microstrip()


class Yassin_Withington():
    '''This is a wrapper around the functions defined in Yassin Withington 1995 'Electromagnetic models for superconducting millimetre-wave and sub-millimetre-wave microstrip transmission lines'.
    '''
#    def effectivePermittivity(self,epsilonDielectric,h,w):
#        '''The static (frequency independent) effective permittivity presented in Whitaker 1988 which cites Gupta et al. 1979.
#        The equation is not directly specified by Yassin and Withington 1995 and is different to the equation given for the effective permittivity in Pozar 2005.
#        
#        inputs
#        ------
#        epsilonDielectric: float
#            The permittivity of the dielectric material.
#        h: float
#            The thickness of the dielectric.
#        w: float
#            The width of the superconducting microstrip line  
#        
#        returns
#        -------
#        float
#        '''
#        k = np.divide(h,h+2*w)
#        return np.multiply(np.divide(epsilonDielectric+1,2),
#                           np.add(np.tanh(1.785*np.log10(np.divide(h,w))+1.75) , #TODO log10 or log
#                                  np.multiply(np.divide(k*w,h),
#                                              np.add(.04-.7*k,
#                                                     .01*np.multiply(1 - .1*epsilonDielectric,
#                                                                     .25+k)
#                                                     )
#                                          )
#                                  )
#                           )
#    
    def b(self,h,t):
        '''A wrapper for the function 1+t/h defined as function 14 in the paper.
        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        t: float 
            The thickness of the superconducting film.
            
        returns
        -------
        float
             1+t/h
        '''
        return  1+np.divide(t,h)
    
    def p(self,h,t):
        '''This function is defined as p in the paper in equation 13.
        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        t: float 
            The thickness of the superconducting film.
        
        returns
        -------
        float
        '''
        b = self.b(h,t)
        return 2*b*b - 1 + 2*b*np.sqrt(b*b-1)
    
    def nu(self,h,t,w):
        '''This function is defined as nu in the paper in equation 20.
        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        t: float 
            The thickness of the superconducting film.
        w: float
            The width of the superconducting microstrip line
        
        returns
        -------
        float
        '''
        p = self.p(h,t)
        return np.multiply(np.sqrt(p),
                           np.add(np.divide(np.pi*w,2*h),
                                  np.subtract(np.multiply(np.divide(p+1,2*np.sqrt(p)),
                                                         1+np.log(np.divide(4,p-1))),
                                              2*np.arctanh(np.reciprocal(np.sqrt(p)))
                                              )
                                  )
                          )
                                  
    def lnra(self,h,t,w):
        '''This function is defined as ln(ra) in the paper in equation 16.
    
        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        t: float 
            The thickness of the superconducting film.
        w: float
            The width of the superconducting microstrip line
        
        returns
        -------
        float
        '''
        p = self.p(h,t)
        return -1-np.divide(np.pi*w,2*h)-np.multiply(np.divide(p+1,np.sqrt(p)),np.arctanh(np.reciprocal(np.sqrt(p))))-np.log(np.divide(p-1,4*p))        
        
    def rbo(self,h,t,w):
        '''This function is defined as rbo in the paper in equation 19.
    
        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        t: float 
            The thickness of the superconducting film.
        w: float
            The width of the superconducting microstrip line
        
        returns
        -------
        float
        '''
        nu = self.nu(h,t,w)
        p = self.p(h,t)
        if nu>p: delta = nu
        else: delta = p
        return nu + np.multiply(np.divide(p+1,2),np.log(delta))
    
    def rb(self,h,t,w):
        '''This function is defined as rb in the paper in equation 17 and 18.
    
        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        t: float 
            The thickness of the superconducting film.
        w: float
            The width of the superconducting microstrip line
        
        returns
        -------
        float
        '''
        p = self.p(h,t)
        rbo = self.rbo(h,t,w)
        if w>=5*h: 
            return rbo
        else:
            return rbo + np.add( - np.sqrt(np.multiply(rbo-1,rbo-p)),
                                np.add(np.multiply(p+1,np.arctanh(np.sqrt(np.divide(rbo-p,rbo-1)))),
                                        np.add(- np.multiply(2*np.sqrt(p),np.arctanh(np.sqrt(np.divide(rbo-p,np.multiply(p,rbo-1))))),
                                               np.divide(np.pi*w*np.sqrt(p),2*h)
                                                )
                                        )
                                )
                                        
    def kf(self,h,t,w):
        '''This function is defined as Kf in the paper in equation 15.
        TODO only for w>5 h
        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        t: float 
            The thickness of the superconducting film.
        w: float
            The width of the superconducting microstrip line
        
        returns
        -------
        float
        '''
        rb = self.rb(h,t,w)
        lnra = self.lnra(h,t,w)
        
        return np.multiply(np.divide(2*h,np.pi*w),
                           np.subtract(np.log(2*rb),lnra)
                           )

                                   
    def g1(self,h,t,w):
        '''This function is defined as g1 in the paper in equation 3.
        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        t: float 
            The thickness of the superconducting film.
        w: float
            The width of the superconducting microstrip line
        
        returns
        -------
        float
        '''
        return np.divide(h,np.multiply(w,self.kf(h,t,w)))
    
    def modal_Impedance_nu_m(self,h,t,w,effective_Permittivity):
        '''This function is defined as nu_m in the paper in equation 10.
        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        t: float 
            The thickness of the superconducting film.
        w: float
            The width of the superconducting microstrip line
        effective_Permittivity: float
            The effective permittivity of the microstrip configuration.
        
        returns
        -------
        float
        '''
        return np.divide(np.multiply(120*np.pi,self.g1(h,t,w)),np.sqrt(effective_Permittivity))
    
    
    def impedance_nu_ideal(self,h,t,w,effective_Permittivity,xi=1,londonPenetrationDepth=0.1e-9):
        '''This function computes the impedance away from the superconductor's limits. The function is described in the paper in equation 36.
        The limitations are thatthe strip thickness is several times the London penetration depth, ...
        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        t: float 
            The thickness of the superconducting film.
        w: float
            The width of the superconducting microstrip line
        effective_Permittivity: float
            The effective permittivity of the microstrip configuration.
        xi: float
            The penetration factor xi introduced in the paper
        londonPenetrationDepth: float
            The london penetration depth of the superconductor.
        
        returns
        -------
        float
        '''
        print('The London penetration depth is %.1f times the thickness of the conductor layer.'%(londonPenetrationDepth/h))
        return np.multiply(self.modal_Impedance_nu_m(h,t,w,effective_Permittivity),
                           np.sqrt(1+np.divide(2*xi*londonPenetrationDepth,h)))
        
    def impedance_nu(self,h,t,w,freq,effective_Permittivity,surface_impedance,xi=1):
        '''This function computes the impedance from the surface impedance. The function is described in the paper in equation 9.
        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        t: float 
            The thickness of the superconducting film.
        w: float
            The width of the superconducting microstrip line
        freq: float
            The frequency of the signal.
        effective_Permittivity: float
            The effective permittivity of the microstrip configuration.
        surface_impedance: complex float
            The surface impedance of the superconductor.
        xi: float
            The penetration factor xi introduced in the paper.
        
        
        returns
        -------
        float
        '''
        return np.multiply(self.modal_Impedance_nu_m(h,t,w,effective_Permittivity),
                           self.factor_surface_impedance(h,freq, xi,surface_impedance).real)
        
    def modal_propagation_constant_beta_m(self,freq,effective_Permittivity):
        '''This function is the modal propagation constant beta_m defined in the paper as equation 8.
        
        inputs
        ------
        freq: float
            The frequency of the signal.
        effective_Permittivity: float
            The effective permittivity of the microstrip configuration.
        
        returns
        ------
        float
        '''
        return np.multiply(np.divide(2*np.pi*freq,const.c),np.sqrt(effective_Permittivity))
    
    def propagation_constant_beta_ideal(self,h,freq,effective_Permittivity,xi=1,londonPenetrationDepth=0.1e-9):
        '''This function computes the propagation constant away from the superconductor's limits. The function is described in the paper in equation 36.
        The limitations are thatthe strip thickness is several times the London penetration depth, ...
        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        freq: float
            The frequency of the signal.
        effective_Permittivity: float
            The effective permittivity of the microstrip configuration.
        xi: float
            The penetration factor xi introduced in the paper.
        londonPenetrationDepth: float
            The london penetration depth of the superconductor.
        
        returns
        -------
        float
        '''
        print('The London penetration depth is %.1f times the thickness of the conductor layer.'%(londonPenetrationDepth/h))
        return np.multiply(self.modal_propagation_constant_beta_m(freq,effective_Permittivity),
                           np.sqrt(1+np.divide(2*xi*londonPenetrationDepth,h)))
        
    def propagation_constant_beta(self,h,freq,effective_Permittivity,surface_impedance,xi=1):
        '''This function computes the propagation constant depending on the surface impedance of the superconductor. The function is described in the paper in equation 9.        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        freq: float
            The frequency of the signal.
        effective_Permittivity: float
            The effective permittivity of the microstrip configuration.
        surface_impedance: complex float
            The surface impedance of the superconductor.
        xi: float
            The penetration factor xi introduced in the paper.
        
        returns
        -------
        float
        '''
        return np.multiply(self.modal_propagation_constant_beta_m(freq,effective_Permittivity),
                           self.factor_surface_impedance(h,freq, xi,surface_impedance).real)
        
    def attenuation_alpha(self,h,freq,effective_Permittivity,surface_impedance,xi=1):
        '''This function computes the propagation constant depending on the surface impedance of the superconductor. The function is described in the paper in equation 9.        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        freq: float
            The frequency of the signal.
        effective_Permittivity: float
            The effective permittivity of the microstrip configuration.
        surface_impedance: complex float
            The surface impedance of the superconductor.
        xi: float
            The penetration factor xi introduced in the paper.
        
        returns
        -------
        float
        '''
        return np.multiply(-self.modal_propagation_constant_beta_m(freq,effective_Permittivity),
                           self.factor_surface_impedance(h,freq, xi,surface_impedance).imag)
    
        
    def factor_surface_impedance(self,h,freq,surface_impedance,xi=1):
        '''This function computes the factor containing the surface impedance in equation 6, 9 and 11.
        The computed value corresponds with the complex value in the bracket of each equation, where in each equation either the real or imaginary part is taken into account.
        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        freq: float
            The frequency of the signal.
        surface_impedance: complex float
            The surface impedance of the superconductor.
        xi: float
            The penetration factor xi introduced in the paper.
        
        returns
        -------
        complex
        '''
        return np.sqrt(1-np.divide(2*xi*surface_impedance*const.c,2*np.pi*freq*120*np.pi*h))
        
    def superconductor_surface_Impedance(self,t,freq,conductivity):
        '''The complex surface impedance of a superconductor requires the application of the BCS theory.
        In the paper, the equation is adopted from Kautz 1978, and shown in equation 37.
        
        inputs
        ------
        t: float 
            The thickness of the superconducting film.
        freq: float
            The frequency of the signal.
        conductivity: complex float
            The complex conductivity of the superconductor.
            
        returns
        -------
        complex float
        '''
        #coth is 1/tanh
        return np.multiply(np.sqrt(np.divide(2j*np.pi*const.mu_0,conductivity)),
                           np.reciprocal(np.tanh(np.multiply(np.sqrt(np.multiply(2j*np.pi*const.mu_0,conductivity)),t))))
        
    def is1(self,h,t,w):
        '''This function is defined as Is1 in the paper in equation 26.
    
        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        t: float 
            The thickness of the superconducting film.
        w: float
            The width of the superconducting microstrip line
        
        returns
        -------
        float
        '''
        p = self.p(h,t)
        ra = np.exp(self.lnra(h,t,w)) #convert the ln(ra) to ra
        return np.log(np.divide(2*p-np.multiply(p+1,ra)+2*np.sqrt(np.multiply(p,self.isRaRbsubequation(ra,p))),
                                np.multiply(ra,p-1)))
        
    def is2(self,h,t,w):
        '''This function is defined as Is2 in the paper in equation 27.
    
        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        t: float 
            The thickness of the superconducting film.
        w: float
            The width of the superconducting microstrip line
        
        returns
        -------
        float
        '''
        p = self.p(h,t)
        rb = self.rb(h,t,w)
        return -np.log(np.divide(np.multiply(p+1,rb)-2*p-2*np.sqrt(np.multiply(p,self.isRaRbsubequation(rb,p))),
                                np.multiply(rb,p-1)))
        
    def isRaRbsubequation(self,rarb,p):
        '''This function is the definition of Ra and Rb dependent on the parameter ra and rb respectively. 
        The function is defined as subfunciton in equation 26 and 27.
        
        inputs 
        ------
        rarb: float
            ra or rb defined in equation 16, 17, 18. To compute Ra insert ra, to compute Rb insert rb.
        p: float
            This variable is defined in equation 3.
            
        returns
        -------
        float
        '''
        return np.multiply(1 - rarb,p - rarb)
    
    def ig1(self,h,t,w):
        '''This function is defined as Is2 in the paper in equation 28.
    
        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        t: float 
            The thickness of the superconducting film.
        w: float
            The width of the superconducting microstrip line
        
        returns
        -------
        float
        '''
        p = self.p(h,t)
        rb = self.rb(h,t,w)
        return -np.log(np.divide(np.multiply(p+1,rb)+2*p+2*np.sqrt(np.multiply(p,self.isRaRbPrimesubequation(rb,p))),
                                np.multiply(rb,p-1)))
        
    def ig2(self,h,t,w):
        '''This function is defined as Is1 in the paper in equation 28.
    
        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        t: float 
            The thickness of the superconducting film.
        w: float
            The width of the superconducting microstrip line
        
        returns
        -------
        float
        '''
        p = self.p(h,t)
        ra = np.exp(self.lnra(h,t,w)) #convert the ln(ra) to ra
        return np.log(np.divide(2*p+np.multiply(p+1,ra)+2*np.sqrt(np.multiply(p,self.isRaRbPrimesubequation(ra,p))),
                                np.multiply(ra,p-1)))
        
    def isRaRbPrimesubequation(self,rarb,p):
        '''This function is the definition of Ra' and Rb' dependent on the parameter ra and rb respectively. 
        The function is defined as subfunciton in equation 28 and 29.
        
        inputs 
        ------
        rarb: float
            ra or rb defined in equation 16, 17, 18. To compute Ra insert ra, to compute Rb insert rb.
        p: float
            This variable is defined in equation 3.
            
        returns
        -------
        float
        '''
        return np.multiply(1 + rarb,p + rarb)
        
    def xi(self,h,t,w):
        '''This function is defined as xi in the paper in equation 24 and 25.
    
        
        inputs
        ------
        h: float
            The thickness of the dielectric.
        t: float 
            The thickness of the superconducting film.
        w: float
            The width of the superconducting microstrip line
        
        returns
        -------
        float
        '''
        ra = np.exp(self.lnra(h,t,w)) #convert the ln(ra) to ra
        rb = self.rb(h,t,w)
        divisor = self.is1(h,t,w)+self.is2(h,t,w)+self.ig1(h,t,w)+self.ig2(h,t,w)+np.pi
        if w<2*h:
            return np.divide(divisor,2*np.log(np.divide(rb,ra)))
        else:
            return np.divide(divisor,2*np.log(np.divide(2*rb,ra)))
        
    
Y = Yassin_Withington()

sigmacomplex = conductivity_BCS(freq=480e9,sigmaN=1.57e7,te=4,delta= 1.5e-3*const.e)
xi = Y.xi(150e-9,350e-9,8e-6)
surfaceImpedance =Y.superconductor_surface_Impedance(350e-9,480e9,sigmacomplex)
z0 = Y.impedance_nu(150e-9,350e-9,8e-6,480e9,5.6,surfaceImpedance,xi)
beta = Y.propagation_constant_beta(150e-9,480e9,5.6,surfaceImpedance,xi)
alpha = Y.attenuation_alpha(150e-9,480e9,5.6,surfaceImpedance,xi)
gamma = alpha+1j*beta
zInResult = zIn(z0,6-.88j,gamma*12e-6)

attentuation = np.exp(alpha*12e-6)

#Comparison with Kerr 1999
#w= 2, 4, 6 um
#h = 300 nm
#epsilon r =3.8
#t = 300 nm
#ondon penetration depth 100 nm

#Reproduce Z, but the epsilon eff in the paper make no sense.
        
                                   











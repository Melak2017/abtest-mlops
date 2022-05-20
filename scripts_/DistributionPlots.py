# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:33:18 2022
@author: kiiru
"""

import sys
sys.path.insert(0, './AB_Testing')

import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as scs
from HypothesisPlot import HypothesisPlot
HPP = HypothesisPlot() 
#hypo_plot

class DistributionPlots:
    def _init_(self):
        """
        Initializing DistributionPlots class
        """
        
    def cont_distribution(self, C_aware, C_total, C_cr, E_cr) -> None:
        fig, ax = plt.subplots(figsize=(12,6))
        x = np.linspace(C_aware-49, C_aware+50, 100)
        y = scs.binom(C_total, C_cr).pmf(x)
        ax.bar(x, y, alpha=0.5)
        ax.axvline(x=E_cr * C_total, c='blue', alpha=0.75, linestyle='--')
        plt.xlabel('Aware')
        plt.ylabel('probability')
        plt.show()
        
    def cont_exp_distribution(self, C_aware, E_aware, C_total, E_total, C_cr, E_cr) -> None:
        fig, ax = plt.subplots(figsize=(12,6))
        xC = np.linspace(C_aware-49, C_aware+50, 100)
        yC = scs.binom(C_total, C_cr).pmf(xC)
        ax.bar(xC, yC, alpha=0.5)
        xE = np.linspace(E_aware-49, E_aware+50, 100)
        yE = scs.binom(E_total, E_cr).pmf(xE)
        ax.bar(xE, yE, alpha=0.5)
        plt.xlabel('Aware')
        plt.ylabel('probability')
        #plt.show()
        
    def null_alt_distribution(self, C_total, E_total, C_cr, E_cr) -> None:
        bcr = C_cr
        mde = E_cr - C_cr
        HPP.hypo_plot(C_total, E_total, bcr, mde, show_power=True, show_beta=True, show_alpha=True, show_p_value=True)
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 17:03:43 2014

@author: wangronin
"""

import pdb
import numpy as np
import hello as h
import cPickle as cp
from boundary_handling import boundary_handling
from numpy.linalg import norm, cholesky, LinAlgError
from numpy.random import randn, rand
from numpy import sqrt, eye, exp, dot, inf, zeros, outer, triu, isinf, isreal, ones


class optimizer(object):
    pass

class one_plus_one_cma_es(optimizer):
    
    def __init__(self, dim, parent_init, fitnessfunc, opts,
                 is_cholesky=False, is_boundary_handling=False, mode=0, is_register=False):
        
        self.parent = eval(parent_init) if isinstance(parent_init, basestring) else parent_init 
        self.eval_budget = int(eval(opts['eval_budget'])) if isinstance(opts['eval_budget'], basestring) else int(opts['eval_budget'])
        
        self.dim = dim
        self.sigma_init = opts['sigma_init']
        self.sigma = opts['sigma_init']
        self.f_target = opts['f_target']
        self.lb = eval(opts['lb'])
        self.ub = eval(opts['ub'])
        self.fitness = fitnessfunc
        self.is_boundary_handling = is_boundary_handling
        self.is_cholesky = is_cholesky
    
        # Exogenous strategy parameters 
        self.p_threshold = 0.44
        self.p_succ_target = 2. / 11.
        self.p_succ = self.p_succ_target
        self.c_p = 1. / 12.
        self.ccov = 2. / (dim**2 + 6.)
        self.d = 1.0 + dim / 2.0
        
        if self.is_cholesky:
            self.A = eye(dim)
            self.c_a = sqrt(1 - self.ccov)
        else:
            self.C = eye(dim)
            self.A = eye(dim)
            self.pc = zeros((dim, 1))
            self.cc = 2. / (dim + 2.)
        
        # Parameters for evolution loop
        self.evalcount = 0
        self.xopt = self.parent
        self.fopt, self.f_parent = inf, inf
        self.exception_info = 0
        self.is_verbose = False
        
        # setup working mode
        self.mode = mode
        if self.mode == 70:
            self.T = 1e5
            self.alpha = 0.95
        
        self.stop_list = []
        self.is_register = is_register
        
        if self.is_register:
            self.hist_sigma = zeros(self.eval_budget)
            self.hist_fbest = zeros(self.eval_budget)
            self.hist_xbest = zeros((self.dim, self.eval_budget))
            self.hist_dist = zeros(self.eval_budget)
            self.hist_parent = [self.parent[:, 0].tolist() + [0]]
    
    def optimize(self):
        
        # Evolution loop       
        while len(self.stop_list) == 0:
            self.step()
            self.exception_handle()
            self.restart_criteria()
            
        if self.is_verbose:  
            print "optimization stop"
        
        if self.is_register:
            return self.xopt, self.fopt, self.evalcount, self.stop_list, self.hist_dist, self.hist_sigma, self.hist_parent
        else:
            return self.xopt, self.fopt, self.evalcount, self.stop_list
            
    def step(self):
        
        # Mutation
        offspring, z = self.mutation()
        
        # Evaluation
        f_offspring = self.fitness(offspring)
        self.evalcount += 1
        
        # selection
        if self.mode == 70:
            accept_prob = max(1, exp((self.f_parent - f_offspring) / self.T))
            is_success = accept_prob > rand()
    
            self.T *= 0.95
            
        is_success = f_offspring < self.f_parent
        
        
        # Parameter adaptation
        self.p_succ = (1 - self.c_p) * self.p_succ + self.c_p * is_success
        self.update_step_size(is_success)

        if is_success:
            self.f_parent = self.fopt = f_offspring
            self.parent = self.xopt = offspring
             
            if self.is_cholesky:
                self.update_cov_cholesky(z)
            else:
                self.update_cov(z)
                
                # Cholesky decomposition
                if np.any(isinf(self.C)):
                    self.exception_info ^= 2**0
                else:
                    try:
                        A = cholesky(self.C)
                        if np.any(~isreal(A)):
                            self.exception_info ^= 2**1
                        else:
                            self.A = A
                    except LinAlgError:
                        self.exception_info ^= 2**3
        
        if self.is_register:
            self.hist_sigma[self.evalcount-1] = self.sigma
            self.hist_xbest[:, [self.evalcount-1]] = offspring
            self.hist_fbest[self.evalcount-1] = f_offspring
            self.hist_dist[self.evalcount-1] = norm(self.parent)
            self.hist_parent.append(self.parent[:, 0].tolist() + [0])
                        
    def mutation(self):
        
        z = randn(self.dim, 1)
        offspring = self.parent + self.sigma * dot(self.A, z)
        if self.is_boundary_handling:
            offspring = boundary_handling(offspring, self.lb, self.ub) 
            
        return offspring, z
                            
    def exception_handle(self):
        """
        
        Handling warings: Internally rectification of strategy paramters
        """
        if (self.sigma < 1e-16) or (self.sigma > 1e16):
            self.exception_info ^= 2**4
        
        if self.exception_info != 0:
            if not self.is_cholesky:
                self.C = eye(self.dim)
                self.pc = zeros((self.dim, 1))
            self.A = eye(self.dim)
            self.sigma = self.sigma_init
            
    def update_step_size(self, is_success):
        
        self.sigma *= exp((self.p_succ - self.p_succ_target) / ((1 - self.p_succ_target) * self.d))
        
    def update_cov(self, z):
        
        cc = self.cc
        ccov = self.ccov
        if self.p_succ < self.p_threshold:
            self.pc = (1 - cc) * self.pc + sqrt(cc*(2-cc)) * dot(self.A, z)
            self.C = (1 - ccov) * self.C + ccov * outer(self.pc, self.pc)
        else:
            self.pc = (1 - cc) * self.pc
            self.C = (1 - ccov) * self.C + ccov * (\
                     outer(self.pc, self.pc) + cc*(2-cc) * self.C)
        self.C = triu(self.C) + triu(self.C, 1).T 
    
    def update_cov_cholesky(self, z):
        
        if self.p_succ < self.p_threshold:
            c_a = self.c_a
            coeff = c_a *(sqrt(1 + (1-c_a**2) * norm(z)**2 / c_a**2) - 1.0) / norm(z)**2
            self.A = c_a * self.A + coeff * dot(dot(self.A, z), z.T)
    
    def restart_criteria(self):
        
        if self.fopt <= self.f_target:
            self.stop_list.append('ftarget')
            
        if self.evalcount >= self.eval_budget:
            self.stop_list.append('maxfevals')
        
        
        # restart criteria to be implemented
        
    
    def dump(self):
        pass
#        fdata = file('test' + '.dat', 'w')
#    cp.dump([histsigma, \
#             hist_condition_number, \
#             hist_e_value], \
#             fdata)
#    fdata.close()
    
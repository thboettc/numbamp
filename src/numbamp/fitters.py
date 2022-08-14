from numba import cuda
import cupy as cp
import numpy as np
import math
import pyext

def pdf_kernel_decorator(f):
    '''
    Helper function that takes an input PDF function and creates a CUDA kernel.
    '''
    dev_f = cuda.jit(f, device=True)
    @cuda.jit
    def kernel(data, pars, vals):
        loc = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
        while loc < data.size:
            vals[loc] = dev_f(data[loc], pars)
            loc += cuda.blockDim.x * cuda.gridDim.x
    return kernel

class PDFKernel:
    '''
    Class that creates a CUDA kernel from a PDF.
    '''
    def __init__(self, func, integral, bound):
        self.func = func
        self.bound = bound
        self.integral = integral
        self.kernel = pdf_kernel_decorator(func)

    def __call__(self, data, pars):
        vals = cp.zeros(data.size, dtype='float64')
        threads_per_block = 32
        blocks_per_grid = (data.size + (threads_per_block - 1)) // threads_per_block
        self.kernel[blocks_per_grid, threads_per_block](data, pars, vals)
        vals = vals / self.integral(self.bound, pars)
        return vals

    def generate(self, pars, size):
        '''
        Sample from the PDF using accept-reject. For now only works for 1D
        functions.
        '''
        vals = cp.array([], dtype='float64')
        while vals.size < size:
            tmp_vals = cp.random.uniform(self.bound[0], self.bound[1], size, dtype='float64')
            pdf_vals = self.__call__(tmp_vals, pars)
            pdf_max = 1.2*cp.amax(pdf_vals)
            rnd_vals = cp.random.uniform(0, pdf_max, size=pdf_vals.size, dtype='float64')
            tmp_vals = tmp_vals[rnd_vals<pdf_vals]
            vals = cp.concatenate((vals, tmp_vals))
        return vals[:size]

def amp_kernel_decorator(f):
    '''
    Helper function that takes an input amplitude and creates a CUDA kernel.
    '''
    # dev_f = cuda.jit(f, device=True)
    @cuda.jit
    def kernel(m12, m13, m23, mp, mc1, mc2, mc3, pars, vals):
        loc = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
        # pars = cuda.const.array_like(pars)
        while loc < m12.size:
            vals[loc] = f(m12[loc], m13[loc], m23[loc], mp, mc1, mc2, mc3, pars)
            loc += cuda.blockDim.x * cuda.gridDim.x
    return kernel

class AmpKernel:
    '''
    Creates a kernel that evaluates a 3-body amplitude.
    '''
    def __init__(self, amp, n_pars):
        self.amp = amp
        self.dev_amp = cuda.jit(amp, device=True)
        self.n_pars = n_pars
        self.kernel = amp_kernel_decorator(self.dev_amp)

    def __call__(self, m12, m13, m23, mp, mc1, mc2, mc3, pars):
        amp = cp.zeros(m12.size, dtype='complex128')
        threads_per_block = 32
        blocks_per_grid = (m12.size + (threads_per_block - 1)) // threads_per_block
        self.kernel[blocks_per_grid, threads_per_block](
            m12, m13, m23, mp, mc1, mc2, mc3, pars, amp)
        return amp

class Amp3BodyPDF:
    '''
    Class that creates a PDF from a collection of amplitudes.
    '''
    def __init__(self, amp_kernels, mp, mc1, mc2, mc3):
        self.amp_kernels = amp_kernels
        self.n_pars = [amp_kernel.n_pars for amp_kernel in self.amp_kernels]
        self.dev_amps = [amp_kernel.dev_amp for amp_kernel in self.amp_kernels]
        self.amps = [amp_kernel.amp for amp_kernel in self.amp_kernels]
        self.mp = mp
        self.mc1 = mc1
        self.mc2 = mc2
        self.mc3 = mc3
        self.code = \
"""
from numba import cuda
def make_device_function(amps):
"""
        for i in range(len(self.dev_amps)):
            self.code += \
"""
    amp_{0} = cuda.jit(amps[{0}], device=True)
""".format(i)
        self.code += \
"""
    @cuda.jit(device=True)
    def f(m12, m13, m23, mp, mc1, mc2, mc3, pars):
        pars_offset = 0
        ret = complex(0., 0.)
"""
        for i in range(len(self.dev_amps)):
            self.code += \
"""
        tmp_pars = pars[pars_offset:pars_offset + 2 + {1}]
        ret += complex(tmp_pars[0], tmp_pars[1])*amp_{0}(m12, m13, m23, mp, mc1, mc2, mc3, tmp_pars[2:])
        pars_offset += 2 + {1}
""".format(i, self.n_pars[i])
        self.code += \
"""
        return ret
    return f
"""     
        self.dev_f_maker = pyext.RuntimeModule.from_string('dev_f_maker', self.code)
        self.dev_f = self.dev_f_maker.make_device_function(self.amps)
        self.kernel = amp_kernel_decorator(self.dev_f)

    def __call__(self, data, pars):
        m12 = data[:,0]
        m13 = data[:,1]
        m23 = self.mp**2 + self.mc1**2 + self.mc2**2 + self.mc3**2 - m12 - m13
        amp = cp.zeros(m12.size, dtype='complex128')
        threads_per_block = 32
        blocks_per_grid = (m12.size + (threads_per_block - 1)) // threads_per_block
        loc_pars = cp.array(pars, dtype='float64')
        self.kernel[blocks_per_grid, threads_per_block](
            m12, m13, m23, self.mp, self.mc1, self.mc2, self.mc3, loc_pars, amp)
        return amp.real**2 + amp.imag**2

    def norm(self, n_bins, pars):
        m12_edges = cp.linspace((self.mc1 + self.mc2)**2, (self.mp - self.mc3)**2, n_bins, dtype='float64')
        m13_edges = cp.linspace((self.mc1 + self.mc2)**2, (self.mp - self.mc2)**2, n_bins, dtype='float64')
        m12_centers = 0.5*(m12_edges[1:] + m12_edges[:-1])
        m13_centers = 0.5*(m13_edges[1:] + m13_edges[:-1])
        m12, m13 = cp.meshgrid(m12_centers, m13_centers)
        m12 = m12.flatten()
        m13 = m13.flatten()
        sqrt_m12 = cp.sqrt(m12)
        e1star = 0.5*(m12 - self.mc2**2 + self.mc1**2)/sqrt_m12
        e3star = 0.5*(self.mp**2 - m12 - self.mc3**2)/sqrt_m12
        rte1mdm11 = cp.sqrt(e1star**2 - self.mc1**2)
        rte3mdm33 = cp.sqrt(e3star**2 - self.mc3**2)
        minimum = (e1star + e3star)**2 - (rte1mdm11 + rte3mdm33)**2
        maximum = (e1star + e3star)**2 - (rte1mdm11 - rte3mdm33)**2
        m12 = m12[(m13>minimum) & (m13<maximum)]
        m13 = m13[(m13>minimum) & (m13<maximum)]

        data = cp.vstack((m12, m13)).T
        vals = self.__call__(data, pars)
        m12_bin_size = m12_edges[1] - m12_edges[0]
        m13_bin_size = m13_edges[1] - m13_edges[0]
        integral = vals.sum()*m12_bin_size*m13_bin_size
        return integral

    def gen_phsp(self, size):
        ret_m12s = cp.array([])
        ret_m13s = cp.array([])
        while ret_m12s.size < size:
            m12 = cp.random.uniform((self.mc1 + self.mc2)**2, (self.mp - self.mc3)**2, size=size, dtype='float64')
            m13 = cp.random.uniform((self.mc1 + self.mc2)**2, (self.mp - self.mc2)**2, size=size, dtype='float64')
            sqrt_m12 = cp.sqrt(m12)
            e1star = 0.5*(m12 - self.mc2**2 + self.mc1**2)/sqrt_m12
            e3star = 0.5*(self.mp**2 - m12 - self.mc3**2)/sqrt_m12
            rte1mdm11 = cp.sqrt(e1star**2 - self.mc1**2)
            rte3mdm33 = cp.sqrt(e3star**2 - self.mc3**2)
            minimum = (e1star + e3star)**2 - (rte1mdm11 + rte3mdm33)**2
            maximum = (e1star + e3star)**2 - (rte1mdm11 - rte3mdm33)**2
            m12 = m12[(m13>minimum) & (m13<maximum)]
            m13 = m13[(m13>minimum) & (m13<maximum)]
            ret_m12s = cp.concatenate((ret_m12s, m12))
            ret_m13s = cp.concatenate((ret_m13s, m13))
        return ret_m12s, ret_m13s

    def generate(self, size, pars):
        m12s = cp.array([])
        m13s = cp.array([])
        while m12s.size < size:
            m12, m13 = self.gen_phsp(size)
            data = cp.vstack((m12, m13)).T
            tmp_vals = self.__call__(data, pars)
            pdf_max = 1.01*cp.amax(tmp_vals)
            rnd = cp.random.uniform(0, pdf_max, size=m12.size, dtype='float64')
            m12 = m12[rnd < tmp_vals]
            m13 = m13[rnd < tmp_vals]
            m12s = cp.concatenate((m12s, m12))
            m13s = cp.concatenate((m13s, m13))
        return m12s[:size], m13s[:size]

def td_kernel_decorator(f):
    # dev_f = cuda.jit(f, device=True)
    @cuda.jit
    def kernel(t, amp_dcs, amp_cf, pars, vals):
        loc = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
        while loc < t.size:
            vals[loc] = f(t[loc], amp_dcs[loc], amp_cf[loc], pars)
            loc += cuda.blockDim.x * cuda.gridDim.x
    return kernel

class Amp3BodyTDPDF:
    def __init__(self, dcs_model, cf_model):
        self.dcs_model = dcs_model
        self.cf_model = cf_model
        @cuda.jit(device=True)
        def t_amp(t, amp_dcs, amp_cf, pars):
            g, x, y = pars
            amp_mix = amp_dcs.conjugate()*amp_cf
            a_dcs2 = amp_dcs.real**2 + amp_dcs.imag**2
            a_cf2 = amp_cf.real**2 + amp_cf.imag**2
            # ret = a_dcs2 + a_cf2*(x**2+y**2)*(g*t)**2/4 + amp_mix.real*y*g*t - amp_mix.imag*x*g*t
            ret = (a_dcs2 + a_cf2)*math.cosh(y*g*t) + (a_dcs2 - a_cf2)*math.cos(x*g*t)
            ret += 2*amp_mix.real*math.sinh(y*g*t) - 2*amp_mix.imag*math.sin(x*g*t)
            return ret*math.exp(-g*t)
        def t_amp_integral(t, amp_dcs, amp_cf, g, x, y):
            amp_mix = amp_dcs.conjugate()*amp_cf
            a_dcs2 = amp_dcs.real**2 + amp_dcs.imag**2
            a_cf2 = amp_cf.real**2 + amp_cf.imag**2
            ret = a_dcs2
            ret += a_cf2*(x**2 + y**2)*((g*t)**2 + 2*g*t + 2)/4.
            ret += (amp_mix.real*y - amp_mix.imag*x)*(g*t + 1)
            return -1.*ret*math.exp(-g*t)/g
        def t_amp_norm(bound, amp_dcs, amp_cf, g, x, y):
            return t_amp_integral(bound[1], amp_dcs, amp_cf, g, x, y) - t_amp_integral(bound[0], amp_dcs, amp_cf, g, x, y)
        self.t_amp = t_amp
        self.td_kernel = td_kernel_decorator(self.t_amp)
        self.t_amp_integral = t_amp_integral
        self.t_amp_norm = t_amp_norm
        
    def __call__(self, data, pars):
        m12 = data[:,0]
        m13 = data[:,1]
        t = data[:,2]
        amp_dcs = cp.zeros(m12.size, dtype='complex128')
        amp_cf = cp.zeros(m12.size, dtype='complex128')
        m23 = self.dcs_model.mp**2 + self.dcs_model.mc1**2 + self.dcs_model.mc2**2 + self.dcs_model.mc3**2 - m12 - m13
        threads_per_block = 32
        blocks_per_grid = (m12.size + (threads_per_block - 1)) // threads_per_block
        n_pars_dcs = np.sum(self.dcs_model.n_pars) + 2*len(self.dcs_model.amp_kernels)
        n_pars_cf = np.sum(self.cf_model.n_pars) + 2*len(self.cf_model.amp_kernels)
        dcs_pars = pars[:n_pars_dcs]
        cf_pars = pars[n_pars_dcs:n_pars_dcs+n_pars_cf]
        t_pars = pars[n_pars_dcs+n_pars_cf:n_pars_dcs+n_pars_cf+3]
        self.dcs_model.kernel[blocks_per_grid, threads_per_block](
            m12, m13, m23, self.dcs_model.mp, self.dcs_model.mc1, self.dcs_model.mc2, self.dcs_model.mc3, dcs_pars, amp_dcs)
        self.cf_model.kernel[blocks_per_grid, threads_per_block](
            m12, m13, m23, self.dcs_model.mp, self.dcs_model.mc1, self.dcs_model.mc2, self.dcs_model.mc3, cf_pars, amp_cf)
        vals = cp.zeros(m12.size, dtype='float64')
        self.td_kernel[blocks_per_grid, threads_per_block](
            t, amp_dcs, amp_cf, t_pars, vals)
        return vals

    def gen_exp(self, gamma, size):
        vals = cp.random.uniform(0, 1, size)
        exp = -cp.log(vals)/gamma
        return exp

    def generate(self, size, pars, batch=None):
        m12s = cp.array([])
        m13s = cp.array([])
        ts = cp.array([])
        if batch==None: batch = size
        while m12s.size < size:
            print(m12s.size)
            m12, m13 = self.dcs_model.gen_phsp(batch)
            gamma, x, y = pars[-3:]
            gammamin = gamma*(1. - y)
            
            # Calculate the maximum weight.
            t_start = cp.zeros(m12.size)
            data = cp.vstack((m12, m13, t_start)).T
            weights = self.__call__(data, pars)
            wmax = 1.1*cp.amax(weights)

            # Calculate initial PDF values for accept/reject.
            t = self.gen_exp(gammamin, m12.size)
            data = cp.vstack((m12, m13, t)).T
            tmp_vals = self.__call__(data, pars)

            # Perform accept/reject.
            rnd = cp.random.uniform(0, wmax, size=m12.size, dtype='float64')
            rnd *= cp.exp(-gammamin*t)
            m12 = m12[rnd < tmp_vals]
            m13 = m13[rnd < tmp_vals]
            t = t[rnd < tmp_vals]
            m12s = cp.concatenate((m12s, m12))
            m13s = cp.concatenate((m13s, m13))
            ts = cp.concatenate((ts, t))
        return m12s[:size], m13s[:size], ts[:size]

    def generate_fixed_t(self, size, t_fix, pars):
        m12s = cp.array([])
        m13s = cp.array([])
        while m12s.size < size:
            m12, m13 = self.dcs_model.gen_phsp(size)
            t = cp.full(m12.size, t_fix)
            data = cp.vstack((m12, m13, t)).T
            tmp_vals = self.__call__(data, pars)
            pdf_max = 1.01*cp.amax(tmp_vals)
            rnd = cp.random.uniform(0, pdf_max, size=m12.size, dtype='float64')
            m12 = m12[rnd < tmp_vals]
            m13 = m13[rnd < tmp_vals]
            m12s = cp.concatenate((m12s, m12))
            m13s = cp.concatenate((m13s, m13))
        return m12s[:size], m13s[:size]


class UnbinnedLH:
    '''
    Class that creates an unbinned likelihood function given a PDF and a
    dataset.
    '''
    def __init__(self, pdf_kernel, data):
        self.pdf_kernel = pdf_kernel
        self.data = data
        
    def __call__(self, pars):
        vals = self.pdf_kernel(self.data, pars)
        nll = -2.*cp.log(vals).sum()
        return nll

class Amp3BodyLH:
    '''
    Class that creates an unbinned likelihood function given an amplitude PDF.
    '''
    def __init__(self, pdf_kernel, data):
        self.pdf_kernel = pdf_kernel
        self.data = data

    def __call__(self, pars):
        cu_pars = cp.array(pars, dtype='float64')
        vals = self.pdf_kernel(self.data, cu_pars)
        norm = self.pdf_kernel.norm(240, cu_pars)
        nll = -2.*cp.log(vals/norm).sum()
        return nll
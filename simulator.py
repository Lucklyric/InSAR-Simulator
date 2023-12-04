"""
Created on: 2018-Jun-13
File: simulator.py
'''
InSAR Simulator for generating training and testing data for "DeepInSAR: A Deep Learning Framework for SAR Interferometric Phase Restoration and Coherence Estimation" 

@author: Aaron Zimmer, Alvin(Xinyao) Sun
"""

from random import randint
from scipy.ndimage import gaussian_filter as gauss_filt
from data_utils_3vG import writeFloat, writeFloatComplex, writeShortComplex
import matplotlib.pyplot as plt
import numpy as np
import os


def normalize_slc_by_tanhmz(img, norm=False):
    phase = np.angle(img)
    points = img
    a = np.abs(points)
    shape = a.shape
    a = a.flatten()
    # a = a**0.15
    mad = np.median(np.abs(a - np.median(a)))
    mz = 0.6745 * ((a - np.median(a)) / mad)
    mz = (np.tanh(mz / 7) + 1) / 2
    if norm:
        mz = (mz - mz.min()) / (mz.max() - mz.min())
    mz = mz.reshape(shape)
    return mz * np.exp(1j * phase)

def wrap(x):
    return np.angle(np.exp(1j * x))


def rotate_grid(x, y, theta=0, p1=[0, 0]):
    c = np.cos(theta)
    s = np.sin(theta)
    x_prime = (x - p1[0]) * c - (y - p1[1]) * s
    y_prime = (x - p1[0]) * s + (y - p1[1]) * c
    return x_prime, y_prime


def eval_2d_gauss(x, y, params):
    amp, xm, ym, sx, sy, theta = params
    a = np.cos(theta)**2. / 2. / sx / sx + np.sin(theta)**2. / 2. / sy / sy
    b = -np.sin(2 * theta) / 4. / sx / sx + np.sin(2 * theta) / 4. / sy / sy
    c = np.sin(theta)**2. / 2. / sx / sx + np.cos(theta)**2. / 2. / sy / sy
    return amp * np.exp(-(a * (x - xm)**2. + 2. * b * (x - xm) * (y - ym) + c * (y - ym)**2.))


def eval_2d_building(x, y, input_mask, params):
    w, h, d, px, py = params
    x1 = x - px
    y1 = y - py
    wedge_mask = (np.abs(x1) <= w / 2.) & (np.abs(y1) <= h / 2.) & (input_mask)
    wedge = np.zeros_like(x1)
    wedge[wedge_mask] = -d / w * x1[wedge_mask] + d / 2.
    return wedge, wedge_mask


def generate_band_mask(width, height, thickness=1):
    screen = gauss_filt(np.random.normal(0, 500., (height, width)), 12.)
    return (screen < thickness) & (screen > -thickness)


class IfgSim():
    """
    stores simulated data with and without noise and allows to add specific types of signals
    Attributes:
        width and height:
        rayleigh_scale: all amplitudes are randomly drawn according to this parameter
        signal_gauss_bubbles: the phase signal
        signal: the phase signal
        signal_buildings: the phase signal model parameters
        signal_faults: the phase signal model parameters
        signal: the phase signal model parameters
        amp1: the underlying amplitude of slc1
        amp2: the underlying amplitude of slc2
        slc1: a phasor of zero phase and amp1 amplitude
        slc2: a phasor of signal phase and amp2 amplitude
        noise1: complex normal gaussian added to slc1
        noise2: complex normal gaussian added to slc2
        ifg: ifg = (slc1+noise1)*np.conj(slc2+noise2)
        x,y: 2D arrays with the x and y indices
    """

    def __init__(self, width, height, rayleigh_scale=1.0, amp_rayleigh_scale=0.3):
        np.random.seed(np.random.randint(1, 1000000 + 1))
        self.width = width
        self.height = height
        self.rayleigh_scale = rayleigh_scale
        self.x, self.y = np.meshgrid(range(self.width), range(self.height))
        self.x = self.x.astype(float)
        self.y = self.y.astype(float)
        self.signal = np.zeros((height, width))
        self.signal_gauss_bubbles = []
        self.signal_buildings = []
        self.signal_faults = []
        # amp = gauss_filt(np.random.rayleigh(self.rayleigh_scale, (self.height, self.width)), 50)
        sample_amp = np.random.rayleigh(rayleigh_scale, self.width)
        sample_amp = (sample_amp - min(sample_amp)) / (max(sample_amp) - min(sample_amp)) + 0.1
        # sample_amp = np.linspace(0,1,self.height)
        sample_amp.sort()
        amp = np.ones([self.height, self.width]) * sample_amp
        self.amp1 = amp.copy()
        self.amp2 = amp.copy()
        self.slc1 = np.exp(1j * np.zeros((height, width)))
        self.slc2 = np.zeros((height, width)).astype(complex)
        self.noise1 = np.zeros((height, width)).astype(complex)
        self.noise2 = np.zeros((height, width)).astype(complex)
        self.ifg = np.zeros((height, width)).astype(complex)
        self.noisy_ifg = np.zeros((height, width)).astype(complex)

    def add_gauss_bubble(self, sigma_range=[20, 300], amp_range=[-1, 1]):
        """
        :param sigma_range: the range of spatial scales for the gaussians
        :param amp_range: the range of amplitudes for the gaussians
        """
        amp = (np.random.random() * (amp_range[1] - amp_range[0]) + amp_range[0])
        x_mean = float(np.random.randint(int(0), int(self.width - 1)))
        y_mean = float(np.random.randint(int(0), int(self.height - 1)))
        x_std = (np.random.random() * (sigma_range[1] - sigma_range[0]) + sigma_range[0])
        y_std = (np.random.random() * (sigma_range[1] - sigma_range[0]) + sigma_range[0])
        theta = np.random.random() * 2. * np.pi - np.pi  # rotate the gaussian by a random angle
        self.signal_gauss_bubbles.append((amp, x_mean, y_mean, x_std, y_std, theta))

    def add_n_gauss_bubbles(self, sigma_range=[20, 300], amp_range=[-1, 1], nps=100):
        """
        :param sigma_range: the range of spatial scales for the gaussians
        :param amp_range: the range of amplitudes for the gaussians
        :param nps: number of random gaussians
        """
        for i in range(nps):
            self.add_gauss_bubble(sigma_range, amp_range)

    def add_building(self, width_range=[10, 100], height_range=[10, 100], depth_factor=0.2):
        """
        :param width_range: range of wedge widths 
        :param height_range: range of wedge heights
        :param depth_factor: the height of the building is proportional to the width of the wedge by this factor
        """
        w = (np.random.random() * (width_range[1] - width_range[0]) + width_range[0])
        h = (np.random.random() * (height_range[1] - height_range[0]) + height_range[0])
        d = w * depth_factor
        px = float(randint(int(0), int(self.width - 1)))
        py = float(randint(int(0), int(self.height - 1)))
        amp = np.random.rayleigh(self.rayleigh_scale)
        self.signal_buildings.append((-px + w / 2, amp, w, h, d, px, py))

    def add_n_buildings(self, width_range=[10, 100], height_range=[10, 100], depth_factor=0.2, nps=100):
        """
        :param width_range: range of wedge widths 
        :param height_range: range of wedge heights
        :param depth_factor: the height of the building is proportional to the width of the wedge by this factor
        :param nps: number of buildings to add 
        """
        for i in range(nps):
            self.add_building(width_range, height_range, depth_factor)

    def add_amp_stripe(self, thickness=1, rayleigh_scale=0.9):
        """ alters the amplitude in a band region (excluding buildings)
        :param thickness: approximate thickness of the bands
        """
        # amplitude = np.random.rayleigh(self.rayleigh_scale)
        amplitude = np.random.rayleigh(rayleigh_scale)
        mask = generate_band_mask(self.width, self.height, thickness)
        self.amp1[mask] = amplitude
        self.amp2[mask] = amplitude

    def add_amp_stripe_hard(self, thickness=1, rayleigh_scale=0.9):
        """ alters the amplitude in a band region (excluding buildings)
        :param thickness: approximate thickness of the bands
        """
        # amplitude = np.random.rayleigh(self.rayleigh_scale)
        amplitude = rayleigh_scale
        mask = generate_band_mask(self.width, self.height, thickness)
        self.amp1[mask] = amplitude
        self.amp2[mask] = amplitude

    def add_n_amp_stripes(self, thickness=1, nps=5, rayleigh_scale=0.9):
        """
        :param thickness: approximate thickness of the bands 
        :param amplitude: new amplitude in the bands
        :param nps: number of bands to add
        """
        for i in range(nps):
            self.add_amp_stripe_hard(thickness, rayleigh_scale=rayleigh_scale)

    def compile(self):
        """
        takes all the model parameters and generates the signals in the amplitude and phase based on them
        """

        # first add the gaussian bubbles
        self.signal = np.zeros((self.height, self.width))
        for params in self.signal_gauss_bubbles:
            self.signal += eval_2d_gauss(self.x, self.y, params)

        # then add the buildings
        vacant_lots = np.ones((self.height, self.width)).astype(bool)
        for params in sorted(self.signal_buildings):
            _, amp, w, h, d, px, py = params
            print(params)
            cur_building, cur_building_mask = eval_2d_building(self.x, self.y, vacant_lots, (w, h, d, px, py))
            self.signal[cur_building_mask] += cur_building[cur_building_mask]
            case = np.random.choice([1, 2, 3, 3], 1)
            if case == 1:
                self.amp1[cur_building_mask] = amp
            elif case == 2:
                self.amp2[cur_building_mask] = amp
            else:
                self.amp1[cur_building_mask] = amp
                self.amp2[cur_building_mask] = amp

            vacant_lots = (vacant_lots) & (cur_building_mask == False)

    def itoh_condition(self, phase_array):
        unwrapped = np.zeros_like(phase_array)
        unwrapped[0] = phase_array[0]

        for i in range(1, len(phase_array)):
            delta = phase_array[i] - phase_array[i - 1]
            if delta > np.pi:
                unwrapped[i] = unwrapped[i - 1] + delta - 2 * np.pi
            elif delta < -np.pi:
                unwrapped[i] = unwrapped[i - 1] + delta + 2 * np.pi
            else:
                unwrapped[i] = unwrapped[i - 1] + delta

        return unwrapped

    def unwrap(self, phase_array_2d):
        phase_array_2d = np.angle(phase_array_2d)
        # Unwrap horizontally (row-wise)
        unwrapped_range = np.zeros_like(phase_array_2d)
        for i in range(phase_array_2d.shape[0]):
            unwrapped_range[i, :] = self.itoh_condition(phase_array_2d[i, :])

        # Unwrap vertically (column-wise)
        unwrapped_azimuth = np.zeros_like(phase_array_2d)
        for j in range(phase_array_2d.shape[1]):
            unwrapped_azimuth[:, j] = self.itoh_condition(phase_array_2d[:, j])

        # Unwrap 2D (both horizontally and vertically)
        unwrapped_2d = np.zeros_like(phase_array_2d)
        for i in range(phase_array_2d.shape[0]):
            unwrapped_2d[i, :] = self.itoh_condition(unwrapped_range[i, :])
        for j in range(phase_array_2d.shape[1]):
            unwrapped_2d[:, j] = self.itoh_condition(unwrapped_2d[:, j])

        return unwrapped_range, unwrapped_azimuth, unwrapped_2d


    def update(self, sigma=0.2):
        """ 
        combines the simulated amplitudes and phases with a constant amount of noise to be added to each slc pixel
        """
        self.compile()
        self.slc1 = self.amp1 * self.slc1
        self.slc2 = self.amp2 * np.exp(1j * (self.signal))
        self.noise1 = np.random.normal(0, sigma, (self.height, self.width)) + 1j * np.random.normal(0, sigma, (self.height, self.width))
        self.noise2 = np.random.normal(0, sigma, (self.height, self.width)) + 1j * np.random.normal(0, sigma, (self.height, self.width))
        self.ifg = (self.slc1) * np.conj(self.slc2)
        self.ifg_noisy = (self.slc1 + self.noise1) * np.conj(self.slc2 + self.noise2)
        self.noisy_slc1_amp = np.absolute(self.slc1 + self.noise1)
        self.noisy_slc2_amp = np.absolute(self.slc2 + self.noise2)


def example_1():
    sim = IfgSim(width=300, height=300, rayleigh_scale=0.9)
    sim.add_n_buildings(width_range=[10, 100], height_range=[1, 40], depth_factor=0.35, nps=25)
    sim.add_n_gauss_bubbles(sigma_range=[10, 150], amp_range=[-4.5, 4.5], nps=110)
    sim.add_n_amp_stripes(thickness=9, nps=5)
    sim.add_n_amp_stripes(thickness=3, nps=50)
    sim.update(sigma=0.5)
    plt.figure()
    plt.imshow(np.angle(sim.ifg), interpolation="None")
    plt.colorbar()
    plt.figure()
    plt.imshow(np.angle(sim.ifg_noisy), interpolation="None")
    plt.colorbar()
    plt.figure()
    plt.imshow(np.abs(sim.ifg), interpolation="None")
    plt.colorbar()
    plt.figure()
    plt.imshow(np.abs(sim.ifg_noisy), interpolation="None")
    plt.colorbar()


def get_coh_dict(amp_range=[0.0, 10.0, 1000], sigma=0.2, N=100000):
    #amps = np.linspace(amp_range[0],amp_range[1],amp_range[2])
    indices = np.arange(amp_range[2])
    amps = indices * (amp_range[1] - amp_range[0]) / (amp_range[2] - 1.) + amp_range[0]
    cohs = []
    for amp in amps:
        noise1 = np.random.normal(0, sigma, N) + 1j * np.random.normal(0, sigma, N)
        noise2 = np.random.normal(0, sigma, N) + 1j * np.random.normal(0, sigma, N)
        slc1 = amp * np.exp(1j * 1) + noise1
        slc2 = amp * np.exp(1j * 1) + noise2
        cohs.append(np.abs(np.sum(slc1 * np.conj(slc2)) / np.sqrt(np.sum(np.abs(slc1)**2.) * np.sum(np.abs(slc2)**2.))))
    cohs = np.array(cohs)
    return cohs, amps


def get_coherence(amps, cohs, amp_range=[0.0, 10.0, 10000]):
    indices = ((amps - amp_range[0]) * (amp_range[2] - 1.) / (amp_range[1] - amp_range[0])).astype(int)
    return cohs[indices]


def example_coh_back():
    sim = IfgSim(width=300, height=300, rayleigh_scale=0.9)
    sim.add_n_buildings(width_range=[10, 100], height_range=[1, 40], depth_factor=0.35, nps=25)
    sim.add_n_gauss_bubbles(sigma_range=[10, 150], amp_range=[-4.5, 4.5], nps=110)
    sim.add_n_amp_stripes(thickness=9, nps=5)
    sim.add_n_amp_stripes(thickness=3, nps=50)
    sim.update(sigma=0.5)
    merge_amp = np.minimum(sim.amp1, sim.amp2)
    amp_range = [np.min(merge_amp), np.max(merge_amp), 1000]
    coh_array, amp_array = get_coh_dict(amp_range=amp_range, sigma=0.5, N=100000)
    sim_cohs = get_coherence(sim.amp1, coh_array, amp_range=amp_range)
    plt.figure()
    plt.imshow(np.angle(sim.ifg), interpolation="None", cmap="jet")
    plt.colorbar()
    plt.figure()
    plt.imshow(np.angle(sim.ifg_noisy), interpolation="None", cmap="jet")
    plt.colorbar()
    plt.figure()
    plt.imshow(np.abs(sim.ifg), interpolation="None")
    plt.colorbar()
    plt.figure()
    plt.imshow(np.abs(sim.ifg_noisy), interpolation="None")
    plt.colorbar()
    plt.figure()
    plt.imshow(sim_cohs, interpolation="None", cmap="Greys_r")
    plt.colorbar()
    plt.show()


def example_coh():
    sim = IfgSim(width=1000, height=1000, rayleigh_scale=0.9)
    sim.add_n_buildings(width_range=[10, 100], height_range=[1, 40], depth_factor=0.35, nps=25)
    sim.add_n_gauss_bubbles(sigma_range=[1, 150], amp_range=[-30, 30], nps=100)
    sim.add_n_amp_stripes(thickness=11, nps=15)
    sim.add_n_amp_stripes(thickness=10, nps=15)
    sim.add_n_amp_stripes(thickness=9, nps=15)
    sim.add_n_amp_stripes(thickness=8, nps=15)
    sim.add_n_amp_stripes(thickness=7, nps=15)
    sim.add_n_amp_stripes(thickness=6, nps=15)
    sim.add_n_amp_stripes(thickness=5, nps=15)
    sim.add_n_amp_stripes(thickness=4, nps=15)
    sim.add_n_amp_stripes(thickness=3, nps=15)
    sim.add_n_amp_stripes(thickness=2, nps=15)
    sim.add_n_amp_stripes(thickness=1, nps=15)
    sim.update(sigma=0.5)
    amp_range = [np.min(sim.amp1), np.max(sim.amp1), 1000]
    sigmas = np.linspace(0.3, 3, 1000)
    sim_cohs = np.zeros((1000, 1000))
    for i in range(1000):
        coh_array, amp_array = get_coh_dict(amp_range=amp_range, sigma=sigmas[i], N=100000)
        sim_cohs[i:] = get_coherence(sim.amp1[i, :], coh_array, amp_range=amp_range)
    plt.figure()
    plt.imshow(np.angle(sim.ifg), interpolation="None", cmap="jet")
    plt.colorbar()
    plt.figure()
    plt.imshow(np.angle(sim.ifg_noisy), interpolation="None", cmap="jet")
    plt.colorbar()
    plt.figure()
    plt.imshow(np.sqrt(np.abs(sim.ifg)), interpolation="None", cmap="gray")
    plt.colorbar()
    plt.figure()
    plt.imshow(np.sqrt(np.abs(sim.ifg_noisy)), interpolation="None", cmap="gray")
    plt.colorbar()
    plt.figure()
    plt.imshow(sim_cohs, interpolation="None", cmap="gray", vmin=0, vmax=1)
    plt.colorbar()
    plt.show()
    return sim


def generate_fix_dataset(num_of_samples=300, height=1000, width=1000, config=None):
    import os
    if not os.path.exists(config['noisy_path']):
        os.makedirs(config['noisy_path'])

    sigmas = np.random.uniform(0.3, 0.8, size=num_of_samples)
    # ray_scales = np.random.uniform(0.7, 1.0, size=num_of_samples)
    for i in range(num_of_samples):
        print("generating %d" % i)
        sim = IfgSim(width=1000, height=1000, rayleigh_scale=0.9)
        sim.add_n_buildings(width_range=[10, 100], height_range=[10, 40], depth_factor=0.35, nps=30)
        sim.add_n_gauss_bubbles(sigma_range=[1, 150], amp_range=[-6, 6], nps=150)
        # sim.add_n_amp_stripes(thickness=3, nps=10)
        # sim.add_n_amp_stripes(thickness=10, nps=15)
        # sim.add_n_amp_stripes(thickness=9, nps=3)
        # sim.add_n_amp_stripes(thickness=3, nps=5)
        sim.update(sigma=sigmas[i])
        merge_amp = np.minimum(sim.amp1, sim.amp2)
        amp_range = [np.min([sim.amp1, sim.amp2]), np.max([sim.amp1, sim.amp2]), 1000]
        coh_array, amp_array = get_coh_dict(amp_range=amp_range, sigma=sigmas[i], N=100000)
        sim_cohs = get_coherence(merge_amp, coh_array, amp_range=amp_range)
        sim_cohs[sim.amp1 != sim.amp2] = 0

        #### set ground truth to same as the noisy region
        sim.ifg[sim.amp1 != sim.amp2] = sim.ifg_noisy[sim.amp1 != sim.amp2]

        slc1_name = "%dslc1" % (i)
        slc2_name = "%dslc2" % (i)

        filename = "%s/%s_%s%s" % (config['coh_path'], slc1_name, slc2_name, config['coh_ext'])
        plt.imsave("%s.png" % filename, sim_cohs, cmap="gray")
        writeFloat(filename, sim_cohs)
        filename = "%s/%s_%s%s_minmax" % (config['coh_path'], slc1_name, slc2_name, config['coh_ext'])
        plt.imsave("%s.png" % filename, sim_cohs, cmap="gray", vmin=0, vmax=1)

        amp_slc1 = sim.noisy_slc1_amp
        filename = "%s/%s%s" % (config['rslc_path'], slc1_name, config['rslc_ext'])
        writeShortComplex(filename, amp_slc1 * np.exp(1j * 1))
        plt.imsave("%s_amp.png" % filename, amp_slc1, cmap="gray")

        amp_slc2 = sim.noisy_slc2_amp
        filename = "%s/%s%s" % (config['rslc_path'], slc2_name, config['rslc_ext'])
        writeShortComplex(filename, amp_slc2 * np.exp(1j * 1))
        plt.imsave("%s_amp.png" % filename, amp_slc2, cmap="gray")

        filename = "%s/%s_%s%s" % (config['noisy_path'], slc1_name, slc2_name, config['noisy_ext'])
        plt.imsave("%s.png" % filename, np.angle(sim.ifg_noisy), cmap="jet")
        plt.imsave("%s_amp.png" % filename, np.abs(sim.ifg_noisy), cmap="gray")
        writeFloatComplex(filename, sim.ifg_noisy)

        filename = "%s/%s_%s%s" % (config['filt_path'], slc1_name, slc2_name, config['filt_ext'])
        plt.imsave("%s.png" % filename, np.angle(sim.ifg), cmap="jet")
        plt.imsave("%s_amp.png" % filename, np.abs(sim.ifg), cmap="gray")
        writeFloatComplex(filename, sim.ifg)


def generate_fix_dataset_v2(num_of_samples=300, height=1000, width=1000, config=None):
    import os
    if not os.path.exists(config['noisy_path']):
        os.makedirs(config['noisy_path'])

    sigmas = np.random.uniform(0.3, 0.8, size=num_of_samples)
    # ray_scales = np.random.uniform(0.7, 1.0, size=num_of_samples)
    for i in range(num_of_samples):
        print("generating %d" % i)
        sim = IfgSim(width=1000, height=1000, rayleigh_scale=0.9)
        # sim.add_n_buildings(width_range=[10, 100], height_range=[10, 40], depth_factor=0.35, nps=30)
        sim.add_n_gauss_bubbles(sigma_range=[1, 150], amp_range=[-6, 6], nps=150)
        sim.add_n_amp_stripes(thickness=3, nps=10)
        # sim.add_n_amp_stripes(thickness=10, nps=15)
        # sim.add_n_amp_stripes(thickness=9, nps=3)
        # sim.add_n_amp_stripes(thickness=3, nps=5)
        sim.update(sigma=sigmas[i])
        merge_amp = np.minimum(sim.amp1, sim.amp2)
        amp_range = [np.min([sim.amp1, sim.amp2]), np.max([sim.amp1, sim.amp2]), 1000]
        coh_array, amp_array = get_coh_dict(amp_range=amp_range, sigma=sigmas[i], N=100000)
        sim_cohs = get_coherence(merge_amp, coh_array, amp_range=amp_range)
        sim_cohs[sim.amp1 != sim.amp2] = 0

        #### set ground truth to same as the noisy region
        sim.ifg[sim.amp1 != sim.amp2] = sim.ifg_noisy[sim.amp1 != sim.amp2]

        slc1_name = "%dslc1" % (i)
        slc2_name = "%dslc2" % (i)

        filename = "%s/%s_%s%s" % (config['coh_path'], slc1_name, slc2_name, config['coh_ext'])
        plt.imsave("%s.png" % filename, sim_cohs, cmap="gray")
        writeFloat(filename, sim_cohs)
        filename = "%s/%s_%s%s_minmax" % (config['coh_path'], slc1_name, slc2_name, config['coh_ext'])
        plt.imsave("%s.png" % filename, sim_cohs, cmap="gray", vmin=0, vmax=1)

        amp_slc1 = sim.noisy_slc1_amp
        filename = "%s/%s%s" % (config['rslc_path'], slc1_name, config['rslc_ext'])
        writeShortComplex(filename, amp_slc1 * np.exp(1j * 1))
        plt.imsave("%s_amp.png" % filename, amp_slc1, cmap="gray")

        amp_slc2 = sim.noisy_slc2_amp
        filename = "%s/%s%s" % (config['rslc_path'], slc2_name, config['rslc_ext'])
        writeShortComplex(filename, amp_slc2 * np.exp(1j * 1))
        plt.imsave("%s_amp.png" % filename, amp_slc2, cmap="gray")

        filename = "%s/%s_%s%s" % (config['noisy_path'], slc1_name, slc2_name, config['noisy_ext'])
        plt.imsave("%s.png" % filename, np.angle(sim.ifg_noisy), cmap="jet")
        plt.imsave("%s_amp.png" % filename, np.abs(sim.ifg_noisy), cmap="gray")
        writeFloatComplex(filename, sim.ifg_noisy)

        filename = "%s/%s_%s%s" % (config['filt_path'], slc1_name, slc2_name, config['filt_ext'])
        plt.imsave("%s.png" % filename, np.angle(sim.ifg), cmap="jet")
        plt.imsave("%s_amp.png" % filename, np.abs(sim.ifg), cmap="gray")
        writeFloatComplex(filename, sim.ifg)


def generate_fix_dataset_v3(num_of_samples=300, height=1000, width=1000, config=None):
    import os
    if not os.path.exists(config['noisy_path']):
        os.makedirs(config['noisy_path'])

    sigmas = np.random.uniform(0.3, 0.8, size=num_of_samples)
    # ray_scales = np.random.uniform(0.7, 1.0, size=num_of_samples)
    for i in range(num_of_samples):
        print("generating %d" % i)
        sim = IfgSim(width=1000, height=1000, rayleigh_scale=0.9)
        # sim.add_n_buildings(width_range=[10, 100], height_range=[10, 40], depth_factor=0.35, nps=30)
        sim.add_n_gauss_bubbles(sigma_range=[1, 150], amp_range=[-25, 25], nps=200)
        sim.add_n_amp_stripes(thickness=3, nps=2)
        # sim.add_n_amp_stripes(thickness=10, nps=15)
        # sim.add_n_amp_stripes(thickness=9, nps=3)
        # sim.add_n_amp_stripes(thickness=3, nps=5)
        sim.update(sigma=sigmas[i])
        merge_amp = np.minimum(sim.amp1, sim.amp2)
        amp_range = [np.min([sim.amp1, sim.amp2]), np.max([sim.amp1, sim.amp2]), 1000]
        coh_array, amp_array = get_coh_dict(amp_range=amp_range, sigma=sigmas[i], N=100000)
        sim_cohs = get_coherence(merge_amp, coh_array, amp_range=amp_range)
        sim_cohs[sim.amp1 != sim.amp2] = 0

        #### set ground truth to same as the noisy region
        sim.ifg[sim.amp1 != sim.amp2] = sim.ifg_noisy[sim.amp1 != sim.amp2]

        slc1_name = "%dslc1" % (i)
        slc2_name = "%dslc2" % (i)

        filename = "%s/%s_%s%s" % (config['coh_path'], slc1_name, slc2_name, config['coh_ext'])
        plt.imsave("%s.png" % filename, sim_cohs, cmap="gray")
        writeFloat(filename, sim_cohs)
        filename = "%s/%s_%s%s_minmax" % (config['coh_path'], slc1_name, slc2_name, config['coh_ext'])
        plt.imsave("%s.png" % filename, sim_cohs, cmap="gray", vmin=0, vmax=1)

        amp_slc1 = sim.noisy_slc1_amp
        filename = "%s/%s%s" % (config['rslc_path'], slc1_name, config['rslc_ext'])
        writeShortComplex(filename, amp_slc1 * np.exp(1j * 1))
        plt.imsave("%s_amp.png" % filename, amp_slc1, cmap="gray")

        amp_slc2 = sim.noisy_slc2_amp
        filename = "%s/%s%s" % (config['rslc_path'], slc2_name, config['rslc_ext'])
        writeShortComplex(filename, amp_slc2 * np.exp(1j * 1))
        plt.imsave("%s_amp.png" % filename, amp_slc2, cmap="gray")

        filename = "%s/%s_%s%s" % (config['noisy_path'], slc1_name, slc2_name, config['noisy_ext'])
        plt.imsave("%s.png" % filename, np.angle(sim.ifg_noisy), cmap="jet")
        plt.imsave("%s_amp.png" % filename, np.abs(sim.ifg_noisy), cmap="gray")
        writeFloatComplex(filename, sim.ifg_noisy)

        filename = "%s/%s_%s%s" % (config['filt_path'], slc1_name, slc2_name, config['filt_ext'])
        plt.imsave("%s.png" % filename, np.angle(sim.ifg), cmap="jet")
        plt.imsave("%s_amp.png" % filename, np.abs(sim.ifg), cmap="gray")
        writeFloatComplex(filename, sim.ifg)


def generate_fix_dataset_by_config(config):
    import os
    if not os.path.exists(config['noisy_path']):
        os.makedirs(config['noisy_path'])
    num_of_samples = config["num_of_samples"]

    sigma = config["sigma"]
    height = config["width"]
    width = config["height"]
    rayleigh_scale = config["rayleigh_scale"]
    for i in range(num_of_samples):
        print("generating %d" % i)
        sim = IfgSim(width=width, height=height, rayleigh_scale=rayleigh_scale)
        sim.add_n_buildings(
            width_range=[config["min_b_w"], config["max_b_w"]],
            height_range=[config["min_b_h"], config["max_b_h"]],
            depth_factor=0.35,
            nps=config["num_buildings"])
        sim.add_n_gauss_bubbles(
            sigma_range=[config["min_bubble_sig"], config["max_bubble_sig"]],
            amp_range=[config["min_bubble_amp"], config["max_bubble_amp"]],
            nps=config["num_bubbles"])

        for amp_config in config["type_amp_stripes"]:
            sim.add_n_amp_stripes(thickness=amp_config["thick"], nps=amp_config["num"], rayleigh_scale=amp_config["scale"])

        sim.update(sigma=sigma)
        merge_amp = np.minimum(sim.amp1, sim.amp2)
        amp_range = [np.min([sim.amp1, sim.amp2]), np.max([sim.amp1, sim.amp2]), 1000]
        coh_array, amp_array = get_coh_dict(amp_range=amp_range, sigma=sigma, N=100000)
        sim_cohs = get_coherence(merge_amp, coh_array, amp_range=amp_range)
        # sim_cohs[sim.amp1 != sim.amp2] = 0

        #### set ground truth to same as the noisy region
        # sim.ifg[sim.amp1 != sim.amp2] = sim.ifg_noisy[sim.amp1 != sim.amp2]

        slc1_name = "%dslc1" % (i)
        slc2_name = "%dslc2" % (i)

        filename = "%s/%s_%s%s" % (config['coh_path'], slc1_name, slc2_name, config['coh_ext'])
        plt.imsave("%s.png" % filename, sim_cohs, cmap="gray")
        writeFloat(filename, sim_cohs)
        filename = "%s/%s_%s%s_minmax" % (config['coh_path'], slc1_name, slc2_name, config['coh_ext'])
        plt.imsave("%s.png" % filename, sim_cohs, cmap="gray", vmin=0, vmax=1)

        filename = "%s/%s%s" % (config['rslc_path'], slc1_name, config['rslc_ext'])
        rslc1 = sim.slc1+sim.noise1
        writeFloatComplex(filename, rslc1)
        writeFloatComplex(filename + ".bar.norm", normalize_slc_by_tanhmz(rslc1, True))
        plt.imsave("%s_amp.png" % filename, np.abs(rslc1), cmap="gray")

        filename = "%s/%s%s" % (config['rslc_path'], slc2_name, config['rslc_ext'])
        rslc2 = sim.slc2+sim.noise2
        writeFloatComplex(filename, rslc2)
        writeFloatComplex(filename + ".bar.norm", normalize_slc_by_tanhmz(rslc2, True))

        filename = "%s/%s_%s%s" % (config['noisy_path'], slc1_name, slc2_name, config['noisy_ext'])
        plt.imsave("%s.png" % filename, np.angle(sim.ifg_noisy), cmap="jet")
        # plt.imsave("%s_amp.png" % filename, np.abs(sim.ifg_noisy), cmap="gray")
        writeFloatComplex(filename, sim.ifg_noisy)

        filename = "%s/%s_%s%s" % (config['filt_path'], slc1_name, slc2_name, config['filt_ext'])

        plt.imsave("%s.png" % filename, np.angle(sim.ifg), cmap="jet")
        np.save('%s.npy' % filename, np.angle(sim.ifg))
        ranged, azimuth, unwrapped = sim.unwrap(sim.ifg)
        np.save('%s_range.npy' % filename, ranged)
        np.save('%s_azimuth.npy' % filename, azimuth)
        np.save('%s_unwrapped.npy' % filename, unwrapped)
        plt.imsave("%s_range.png" % filename, ranged, cmap="jet")
        plt.imsave("%s_azimuth.png" % filename, azimuth, cmap="jet")
        plt.imsave("%s_unwrapped.png" % filename, unwrapped, cmap="jet")
        # plt.imsave("%s_amp.png" % filename, np.abs(sim.ifg), cmap="gray")
        writeFloatComplex(filename, sim.ifg)


if __name__ == "__main__":
    SIM_DIR = "./sim_data/"

    os.makedirs(SIM_DIR, exist_ok=True)

    def gen(db_name, modify={}):
        print("Starting %s db .." % (db_name))
        sample_config = {
            "noisy_path": SIM_DIR + db_name + "/ifg_fr",
            "noisy_ext": ".noisy",
            "rslc_path": SIM_DIR + db_name + "/ifg_fr",
            "rslc_ext": ".rslc",
            "filt_path": SIM_DIR + db_name + "/ifg_fr",
            "filt_ext": ".filt",
            "coh_path": SIM_DIR + db_name + "/ifg_fr",
            "coh_ext": ".filt.coh",
            "width": 512,
            "height": 512,
            "num_of_samples": 10,
            "sigma": 0.1,
            "rayleigh_scale": 0.9,
            "min_b_w": 10,
            "max_b_w": 100,
            "min_b_h": 10,
            "max_b_h": 40,
            "num_buildings": 0,
            "min_bubble_sig": 10,
            "max_bubble_sig": 150,
            "min_bubble_amp": -4,
            "max_bubble_amp": 4,
            "num_bubbles": 150,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
            }]
        }

        for key, value in modify.items():
            sample_config[key] = value
        generate_fix_dataset_by_config(sample_config)

    import concurrent.futures
    import time

    def concurrent_wrapper(db_confs):
        # print (db_confs)
        # random_seed change
        seed = 0
        for char in db_confs["db_name"]:
            seed += ord(char)
        np.random.seed(int(time.time())+seed)
        gen(db_confs["db_name"], db_confs["modify"])

    dbs = [{
        "modify": {
            "sigma": 0.1,
            "type_amp_stripes": []
        },
        "db_name": "S1-Flow-NS-Train"
    }, {
        "modify": {
            "sigma": 0.1,
            "type_amp_stripes": []
        },
        "db_name": "S1-Flow-NS-Test"
    }, {
        "modify": {
            "sigma": 0.1,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
                'scale': 0.1
            }]
        },
        "db_name": "S1-Flow-FS-Train"
    }, {
        "modify": {
            "sigma": 0.1,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
                'scale': 0.1
            }]
        },
        "db_name": "S1-Flow-FS-Test"
    }, {
        "modify": {
            "sigma": 0.2,
            "type_amp_stripes": []
        },
        "db_name": "S2-Flow-NS-Train"
    }, {
        "modify": {
            "sigma": 0.2,
            "type_amp_stripes": []
        },
        "db_name": "S2-Flow-NS-Test"
    }, {
        "modify": {
            "sigma": 0.2,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
                'scale': 0.1
            }]
        },
        "db_name": "S2-Flow-FS-Train"
    }, {
        "modify": {
            "sigma": 0.2,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
                'scale': 0.1
            }]
        },
        "db_name": "S2-Flow-FS-Test"
    }, {
        "modify": {
            "sigma": 0.3,
            "type_amp_stripes": []
        },
        "db_name": "S3-Flow-NS-Train"
    }, {
        "modify": {
            "sigma": 0.3,
            "type_amp_stripes": []
        },
        "db_name": "S3-Flow-NS-Test"
    }, {
        "modify": {
            "sigma": 0.3,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
                'scale': 0.1
            }]
        },
        "db_name": "S3-Flow-FS-Train"
    }, {
        "modify": {
            "sigma": 0.3,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
                'scale': 0.1
            }]
        },
        "db_name": "S3-Flow-FS-Test"
    }]
    # dbs = []
    dbs.extend([{
        "modify": {
            "sigma": 0.1,
            "min_bubble_amp": -20,
            "max_bubble_amp": 20,
            "type_amp_stripes": []
        },
        "db_name": "S1-FM-NS-Train"
    }, {
        "modify": {
            "sigma": 0.1,
            "min_bubble_amp": -20,
            "max_bubble_amp": 20,
            "type_amp_stripes": []
        },
        "db_name": "S1-FM-NS-Test"
    }, {
        "modify": {
            "sigma": 0.1,
            "min_bubble_amp": -20,
            "max_bubble_amp": 20,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
                'scale': 0.1
            }]
        },
        "db_name": "S1-FM-FS-Train"
    }, {
        "modify": {
            "sigma": 0.1,
            "min_bubble_amp": -20,
            "max_bubble_amp": 20,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
                'scale': 0.1
            }]
        },
        "db_name": "S1-FM-FS-Test"
    }, {
        "modify": {
            "sigma": 0.2,
            "min_bubble_amp": -20,
            "max_bubble_amp": 20,
            "type_amp_stripes": []
        },
        "db_name": "S2-FM-NS-Train"
    }, {
        "modify": {
            "sigma": 0.2,
            "min_bubble_amp": -20,
            "max_bubble_amp": 20,
            "type_amp_stripes": []
        },
        "db_name": "S2-FM-NS-Test"
    }, {
        "modify": {
            "sigma": 0.2,
            "min_bubble_amp": -20,
            "max_bubble_amp": 20,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
                'scale': 0.1
            }]
        },
        "db_name": "S2-FM-FS-Train"
    }, {
        "modify": {
            "sigma": 0.2,
            "min_bubble_amp": -20,
            "max_bubble_amp": 20,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
                'scale': 0.1
            }]
        },
        "db_name": "S2-FM-FS-Test"
    }, {
        "modify": {
            "sigma": 0.3,
            "min_bubble_amp": -20,
            "max_bubble_amp": 20,
            "type_amp_stripes": []
        },
        "db_name": "S3-FM-NS-Train"
    }, {
        "modify": {
            "sigma": 0.3,
            "min_bubble_amp": -20,
            "max_bubble_amp": 20,
            "type_amp_stripes": []
        },
        "db_name": "S3-FM-NS-Test"
    }, {
        "modify": {
            "sigma": 0.3,
            "min_bubble_amp": -20,
            "max_bubble_amp": 20,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
                'scale': 0.1
            }]
        },
        "db_name": "S3-FM-FS-Train"
    }, {
        "modify": {
            "sigma": 0.3,
            "min_bubble_amp": -20,
            "max_bubble_amp": 20,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
                'scale': 0.1
            }]
        },
        "db_name": "S3-FM-FS-Test"
    }])

    # dbs = []
    dbs.extend([{
        "modify": {
            "sigma": 0.1,
            "min_bubble_amp": -40,
            "max_bubble_amp": 40,
            "type_amp_stripes": []
        },
        "db_name": "S1-FH-NS-Train"
    }, {
        "modify": {
            "sigma": 0.1,
            "min_bubble_amp": -40,
            "max_bubble_amp": 40,
            "type_amp_stripes": []
        },
        "db_name": "S1-FH-NS-Test"
    }, {
        "modify": {
            "sigma": 0.1,
            "min_bubble_amp": -40,
            "max_bubble_amp": 40,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
                'scale': 0.1
            }]
        },
        "db_name": "S1-FH-FS-Train"
    }, {
        "modify": {
            "sigma": 0.1,
            "min_bubble_amp": -40,
            "max_bubble_amp": 40,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
                'scale': 0.1
            }]
        },
        "db_name": "S1-FH-FS-Test"
    }, {
        "modify": {
            "sigma": 0.2,
            "min_bubble_amp": -40,
            "max_bubble_amp": 40,
            "type_amp_stripes": []
        },
        "db_name": "S2-FH-NS-Train"
    }, {
        "modify": {
            "sigma": 0.2,
            "min_bubble_amp": -40,
            "max_bubble_amp": 40,
            "type_amp_stripes": []
        },
        "db_name": "S2-FH-NS-Test"
    }, {
        "modify": {
            "sigma": 0.2,
            "min_bubble_amp": -40,
            "max_bubble_amp": 40,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
                'scale': 0.1
            }]
        },
        "db_name": "S2-FH-FS-Train"
    }, {
        "modify": {
            "sigma": 0.2,
            "min_bubble_amp": -40,
            "max_bubble_amp": 40,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
                'scale': 0.1
            }]
        },
        "db_name": "S2-FH-FS-Test"
    }, {
        "modify": {
            "sigma": 0.3,
            "min_bubble_amp": -40,
            "max_bubble_amp": 40,
            "type_amp_stripes": []
        },
        "db_name": "S3-FH-NS-Train"
    }, {
        "modify": {
            "sigma": 0.3,
            "min_bubble_amp": -40,
            "max_bubble_amp": 40,
            "type_amp_stripes": []
        },
        "db_name": "S3-FH-NS-Test"
    }, {
        "modify": {
            "sigma": 0.3,
            "min_bubble_amp": -40,
            "max_bubble_amp": 40,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
                'scale': 0.1
            }]
        },
        "db_name": "S3-FH-FS-Train"
    }, {
        "modify": {
            "sigma": 0.3,
            "min_bubble_amp": -40,
            "max_bubble_amp": 40,
            "type_amp_stripes": [{
                "thick": 3,
                "num": 1,
                'scale': 0.1
            }]
        },
        "db_name": "S3-FH-FS-Test"
    }])
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=9) as executor:
        for _, _ in zip(dbs, executor.map(concurrent_wrapper, dbs)):
            print("Done")

from random import randint
from scipy.ndimage import gaussian_filter as gauss_filt
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os


def readFloatRandomPatches(fileName, width=1, num_sample=1, patch_size=1, rows=None, cols=None, height=None):
    with open(fileName, "rb") as fin:
        if rows is None:
            rows = np.random.randint(0, high=(height - patch_size), size=num_sample)
            cols = np.random.randint(0, high=(width - patch_size), size=num_sample)
            size_of_file = os.path.getsize(fileName)
            height = size_of_file / 4 / width
        patches = []
        for i in range(len(rows)):
            row = rows[i]
            col = cols[i]
            img = []
            for p_row in range(patch_size):
                fin.seek(4 * (width * (row + p_row) + col))
                img.append(np.frombuffer(fin.read(4 * patch_size), dtype=">f4").astype(np.float))
            patches.append(np.reshape(img, [patch_size, patch_size]))
    return patches, rows, cols, height


def wrap(x):
  return np.angle(np.exp(1j*x))


def rotate_grid(x,y,theta=0,p1=[0,0]):
  c = np.cos(theta)
  s = np.sin(theta)
  x_prime = (x-p1[0])*c-(y-p1[1])*s
  y_prime = (x-p1[0])*s+(y-p1[1])*c
  return x_prime, y_prime


def eval_2d_gauss(x,y,params):
  amp,xm,ym,sx,sy,theta = params
  a = np.cos(theta)**2./2./sx/sx+np.sin(theta)**2./2./sy/sy
  b = -np.sin(2*theta)/4./sx/sx+np.sin(2*theta)/4./sy/sy
  c = np.sin(theta)**2./2./sx/sx+np.cos(theta)**2./2./sy/sy
  return amp*np.exp(-(a*(x-xm)**2.+2.*b*(x-xm)*(y-ym)+c*(y-ym)**2.))


def eval_3d_ellipsoid(x,y,params):
  a, b, c, x_off, y_off = params
  x1 = x-x_off
  y1 = y-y_off
  goods =  (x1**2./b**2. + y1**2./c**2.) <= 1.0
  ellipse = np.zeros_like(x1)
  ellipse[goods] = a*np.sqrt(1 - x1[goods]**2./b**2. - y1[goods]**2./c**2.)
  return ellipse


def eval_3d_polygon(x,y,params):
  x_off, y_off, Ps, n, a, angels = params
  x1 = x-x_off
  y1 = y-y_off
  angels_0 = angels[0]
  angles = wrap(np.arctan2(y1,x1)-angels_0)
  angles[angles<0] = 2.*np.pi + angles[angles<0] 
  for i in range(len(angels)):
    angels[i] -= angels_0
    #print angels[i]
  polygon = np.zeros_like(x1)
  zeres = np.zeros_like(x1)
  for f in range(1,n):
    x2, y2 = Ps[f-1][0], Ps[f-1][1]
    x3, y3 = Ps[f][0], Ps[f][1]
    ang1 = angels[f-1]
    ang2 = angels[f]
    if ang2-ang1<np.pi:
      goods = (angles>=ang1) & (angles<ang2)
      polygon[goods] += np.maximum(zeres[goods],a-((-y2*a+y3*a)*x1[goods]+(x2*a-x3*a)*y1[goods])/(x2*y3-x3*y2))
  x2, y2 = Ps[-1][0], Ps[-1][1]
  x3, y3 = Ps[0][0], Ps[0][1]
  ang1 = angels[-1]
  ang2 = 2.*np.pi
  if ang2-ang1<np.pi:
    goods = (angles>=ang1) & (angles<ang2)
    polygon[goods] += np.maximum(zeres[goods],a-((-y2*a+y3*a)*x1[goods]+(x2*a-x3*a)*y1[goods])/(x2*y3-x3*y2))
  return polygon


def eval_2d_building(x,y,input_mask,params):
  w,h,d,px,py = params
  x1 = x-px
  y1 = y-py
  wedge_mask = (np.abs(x1) <= w/2.) & (np.abs(y1) <= h/2.) & (input_mask)
  wedge = np.zeros_like(x1)
  wedge[wedge_mask] = -d/w*x1[wedge_mask] + d/2.
  return wedge, wedge_mask


def generate_band_mask(width,height,thickness=1):
  screen = gauss_filt(np.random.normal(0, 500., (height, width)), 12.)
  return (screen<thickness) & (screen>-thickness)


class IfgSim():
  """stores simulated data with and without noise and allows to add specific types of signals

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

  def __init__(self,width,height,rayleigh_scale=1.0):
    np.random.seed(np.random.randint(1,1000000+1))
    self.width = width
    self.height = height
    self.rayleigh_scale = rayleigh_scale
    self.x, self.y = np.meshgrid(range(self.width),range(self.height))
    self.x = self.x.astype(np.float)
    self.y = self.y.astype(np.float)
    self.signal = np.zeros((height,width))
    self.signal_gauss_bubbles = []
    self.signal_ellipses = []
    self.signal_polygons = []
    self.signal_buildings = []
    self.signal_faults = []
    self.signal_band = []
    amp = gauss_filt(np.random.rayleigh(self.rayleigh_scale,(self.height,self.width)),10)
    self.amp1 = amp.copy()
    self.amp2 = amp.copy()
    self.slc1 = np.exp(1j*np.zeros((height,width)))
    self.slc2 = np.zeros((height,width)).astype(np.complex)
    self.noise1 = np.zeros((height,width)).astype(np.complex)
    self.noise2 = np.zeros((height,width)).astype(np.complex)
    self.ifg = np.zeros((height,width)).astype(np.complex)
    self.noisy_ifg = np.zeros((height,width)).astype(np.complex)
    self.add_random_dem_signal_flag = False
    self.dem_scale = 1.0


  def add_random_dem_signal(self, scale=1.0):
    self.add_random_dem_signal_flag = True
    self.dem_scale = scale


  def add_gauss_bubble(self, sigma_range=[20,300], amp_range=[-1,1]):
    """
    :param sigma_range: the range of spatial scales for the gaussians
    :param amp_range: the range of amplitudes for the gaussians
    """
    amp = (np.random.random()*(amp_range[1]-amp_range[0])+amp_range[0])
    x_mean = float(np.random.randint(int(0),int(self.width-1)))
    y_mean = float(np.random.randint(int(0),int(self.height-1)))
    x_std = (np.random.random()*(sigma_range[1]-sigma_range[0])+sigma_range[0])
    y_std = (np.random.random()*(sigma_range[1]-sigma_range[0])+sigma_range[0])
    theta = np.random.random()*2.*np.pi-np.pi # rotate the gaussian by a random angle
    self.signal_gauss_bubbles.append((amp, x_mean, y_mean, x_std, y_std, theta))


  def add_n_gauss_bubbles(self, sigma_range=[20,300], amp_range=[-1,1], nps=100):
    """
    :param sigma_range: the range of spatial scales for the gaussians
    :param amp_range: the range of amplitudes for the gaussians
    :param nps: number of random gaussians
    """
    for i in range(nps):
      self.add_gauss_bubble(sigma_range, amp_range)


  def add_ellipse(self, z_range=[1,10], x_range=[1,10], y_range=[1,10]):
    """
    :param z_range: the range of heights for the ellipses
    :param x_range: the range of radii for the first axis of the ellipses
    :param y_range: the range of radii for the second axis of the ellipses
    """
    a = (np.random.random()*(z_range[1]-z_range[0])+z_range[0])
    b = (np.random.random()*(x_range[1]-x_range[0])+x_range[0])
    c = (np.random.random()*(y_range[1]-y_range[0])+y_range[0])
    x_off = float(np.random.randint(int(0),int(self.width-1)))
    y_off = float(np.random.randint(int(0),int(self.height-1)))
    self.signal_ellipses.append((a, b, c, x_off, y_off))


  def add_n_ellipses(self, z_range=[1,10], x_range=[1,10], y_range=[1,10], nps=100):
    """
    :param z_range: the range of heights for the ellipses
    :param x_range: the range of radii for the first axis of the ellipses
    :param y_range: the range of radii for the second axis of the ellipses
    :param nps: number of random gaussians
    """
    for i in range(nps):
      self.add_ellipse(z_range, x_range, y_range)


  def add_polygon(self, z_range=[1,10], r_range=[0,100], n_range=[3,10]):
    """
    :param z_range: the range of heights for the polygons
    :param r_range: the range of radii for the polygon edges
    :param n_range: the range of number of edges for the polygons
    """
    angel = np.random.rand()*2.*np.pi
    n = np.random.randint(n_range[0],n_range[1]+1)
    x_off = float(np.random.randint(0,self.width))
    y_off = float(np.random.randint(0,self.height))
    Ps = []
    angels = []
    for j in range(n):
      r = np.random.random()*(r_range[1]-r_range[0])+r_range[0]
      Ps.append([np.cos(angel)*r,np.sin(angel)*r])
      angels.append(angel)
      angel = np.min([angels[0]+2.*np.pi,angel+np.random.rand()*(2.*np.pi/(n-1.))])
    angels[-1] = np.max([angels[0]+2.*np.pi-np.random.rand()*np.pi,angels[-1]])
    Ps[-1] = [np.cos(angels[-1])*r,np.sin(angels[-1])*r]
    a = (np.random.random()*(z_range[1]-z_range[0])+z_range[0])
    self.signal_polygons.append((x_off,y_off,Ps,n,a,angels))


  def add_n_polygons(self, z_range=[1,10], r_range=[0,100], n_range=[3,10], nps=100):
    """
    :param z_range: the range of heights for the polygons
    :param r_range: the range of radii for the polygon edges
    :param n_range: the range of number of edges for the polygons
    :param nps: number of random gaussians
    """
    for i in range(nps):
      self.add_polygon(z_range, r_range, n_range)


  def add_building(self, width_range=[10,100], height_range=[10,100], depth_factor=0.2):
    """
    :param width_range: range of wedge widths 
    :param height_range: range of wedge heights
    :param depth_factor: the height of the building is proportional to the width of the wedge by this factor
    """
    w = (np.random.random()*(width_range[1]-width_range[0])+width_range[0])
    h = (np.random.random()*(height_range[1]-height_range[0])+height_range[0])
    d = w*depth_factor
    px = float(randint(int(0),int(self.width-1)))
    py = float(randint(int(0),int(self.height-1)))
    amp = np.random.rayleigh(self.rayleigh_scale)
    self.signal_buildings.append((-px+w/2,amp,w,h,d,px,py))


  def add_n_buildings(self, width_range=[10,100], height_range=[10,100], depth_factor=0.2, nps=100):
    """
    :param width_range: range of wedge widths 
    :param height_range: range of wedge heights
    :param depth_factor: the height of the building is proportional to the width of the wedge by this factor
    :param nps: number of buildings to add 
    """
    for i in range(nps):
      self.add_building(width_range, height_range, depth_factor)

  def add_amp_stripe(self, thickness=1):
    """ alters the amplitude in a band region (excluding buildings)
    :param thickness: approximate thickness of the bands
    """
    amplitude = np.random.rayleigh(self.rayleigh_scale)
    mask = generate_band_mask(self.width,self.height,thickness)
    self.amp1[mask] = amplitude
    self.amp2[mask] = amplitude
  
  def add_phase_strip(self, amp_scale=1, thickness=1):
    mask = generate_band_mask(self.width,self.height,thickness)
    self.signal_band.append((mask, amp_scale))

  def add_n_amp_stripes(self, thickness=1, nps=5):
    """
    :param thickness: approximate thickness of the bands 
    :param amplitude: new amplitude in the bands
    :param nps: number of bands to add
    """
    for i in range(nps):
      self.add_amp_stripe(thickness)

  def add_n_phase_stripes(self, thickness=1, nps=5, amp_range=[-1, 1]):
    amps = np.random.uniform(amp_range[0], amp_range[1], nps)
    for i in range(nps):
      self.add_phase_strip(amps[i], thickness)

  def compile(self):
    """ takes all the model parameters and generates the signals in the amplitude and phase based on them
    """

    # first add the gaussian bubbles
    self.signal = np.zeros((self.height,self.width))

    if self.add_random_dem_signal_flag:
      assert(self.width==self.height)
      dems = glob('/disk/tembofallback/c003/scratch/azimmer/rdc_dems/*')
      idx = np.random.randint(len(dems))
      rdc_dem = dems[idx]
      dem_width = int(rdc_dem.split('.')[-2])
      dem_height = int(rdc_dem.split('.')[-1])
      dem, rs, cs, h = readFloatRandomPatches(rdc_dem, width=dem_width, patch_size=self.width, num_sample=1, height=dem_height)
      self.signal += dem[0] * np.random.rand() * self.dem_scale
      #TODO: do the same thing with the average rmli for the amplitudes

    for params in self.signal_gauss_bubbles:
      self.signal += eval_2d_gauss(self.x, self.y, params)

    for params in self.signal_ellipses:
      self.signal += eval_3d_ellipsoid(self.x, self.y, params)*(-1 if np.random.rand()>0.5 else 1)

    for params in self.signal_polygons:
      tmp = eval_3d_polygon(self.x, self.y, params)*(-1 if np.random.rand()>0.5 else 1)
      if np.random.rand()>0.5:
        tmp = np.transpose(tmp)
      self.signal += tmp

    for params in self.signal_band:
      self.signal[params[0]] += params[1]

    # then add the buildings
    vacant_lots = np.ones((self.height,self.width)).astype(np.bool)
    for params in sorted(self.signal_buildings):
      _,amp,w,h,d,px,py = params
      #print params
      cur_building, cur_building_mask = eval_2d_building(self.x,self.y,vacant_lots,(w,h,d,px,py))
      self.signal[cur_building_mask] += cur_building[cur_building_mask]
      self.amp1[cur_building_mask] = amp
      self.amp2[cur_building_mask] = amp
      vacant_lots = (vacant_lots) & (cur_building_mask==False)


  def update(self, sigma=0.2, sigma_correlated = 0.0):
    """ combines the simulated amplitudes and phases with a constant amount of noise to be added to each slc pixel
    """
    self.compile()
    self.slc1 = self.amp1*self.slc1
    self.slc2 = self.amp2*np.exp(1j*(self.signal))
    self.noise1 = np.random.normal(0, sigma, (self.height, self.width)) + 1j*np.random.normal(0, sigma, (self.height, self.width))
    self.noise2 = np.random.normal(0, sigma, (self.height, self.width)) + 1j*np.random.normal(0, sigma, (self.height, self.width))
    self.noise12 = 0.0 if sigma_correlated==0 else np.random.normal(0, sigma_correlated, (self.height, self.width)) + 1j*np.random.normal(0, sigma_correlated, (self.height, self.width))
    self.ifg = (self.slc1)*np.conj(self.slc2)
    self.ifg_noisy = (self.slc1+self.noise1+self.noise12)*np.conj(self.slc2+self.noise2+self.noise12)


def get_coh_dict(amp_range=[0.0,10.0,10000],sigma=0.2,sigma_correlated=0.0,N=100000):
  indices = np.arange(amp_range[2])
  amps = indices*(amp_range[1]-amp_range[0])/(amp_range[2]-1.)+amp_range[0]
  cohs = []
  for amp in amps:
    noise1 = np.random.normal(0, sigma, N) + 1j*np.random.normal(0, sigma, N)
    noise2 = np.random.normal(0, sigma, N) + 1j*np.random.normal(0, sigma, N)
    noise12 = 0.0 if sigma_correlated==0 else np.random.normal(0, sigma_correlated, N) + 1j*np.random.normal(0, sigma_correlated, N)
    slc1 = amp*np.exp(1j*1) + noise1 + noise12
    slc2 = amp*np.exp(1j*1) + noise2 + noise12
    cohs.append(np.abs(np.sum(slc1*np.conj(slc2))/np.sqrt(np.sum(np.abs(slc1)**2.)*np.sum(np.abs(slc2)**2.))))
  cohs = np.array(cohs)
  return cohs, amps


def get_coherence(amps,cohs,amp_range=[0.0,10.0,10000]):
  indices = ((amps - amp_range[0])*(amp_range[2] - 1.)/(amp_range[1]-amp_range[0])).astype(np.int)
  return cohs[indices]


def example_1():
  sim = IfgSim(width=300, height=300, rayleigh_scale=0.9)
  sim.add_n_buildings(width_range=[10,100], height_range=[1,40], depth_factor=0.35, nps=25)
  sim.add_n_gauss_bubbles(sigma_range=[10,150], amp_range=[-4.5,4.5], nps=110)
  sim.add_n_amp_stripes(thickness=9, nps=5)
  sim.add_n_amp_stripes(thickness=3, nps=50)
  sim.update(sigma=0.5)
  plt.figure(); plt.imshow(np.angle(sim.ifg),interpolation="None"); plt.colorbar()
  plt.figure(); plt.imshow(np.angle(sim.ifg_noisy),interpolation="None"); plt.colorbar()
  plt.figure(); plt.imshow(np.abs(sim.ifg),interpolation="None"); plt.colorbar()
  plt.figure(); plt.imshow(np.abs(sim.ifg_noisy),interpolation="None"); plt.colorbar()


def example_coh():
  sim = IfgSim(width=300, height=300, rayleigh_scale=0.9)
  sim.add_n_buildings(width_range=[10,100], height_range=[1,40], depth_factor=0.35, nps=25)
  sim.add_n_gauss_bubbles(sigma_range=[10,150], amp_range=[-4.5,4.5], nps=110)
  sim.add_n_amp_stripes(thickness=9, nps=5)
  sim.add_n_amp_stripes(thickness=3, nps=50)
  sim.update(sigma=0.5)
  amp_range=[np.min(sim.amp1),np.max(sim.amp1),1000]
  coh_array, amp_array = get_coh_dict(amp_range=amp_range,sigma=0.5,N=100000)
  sim_cohs = get_coherence(sim.amp1,coh_array,amp_range=amp_range)
  plt.figure(); plt.imshow(np.angle(sim.ifg),interpolation="None"); plt.colorbar()
  plt.figure(); plt.imshow(np.angle(sim.ifg_noisy),interpolation="None"); plt.colorbar()
  plt.figure(); plt.imshow(np.abs(sim.ifg),interpolation="None"); plt.colorbar()
  plt.figure(); plt.imshow(np.abs(sim.ifg_noisy),interpolation="None"); plt.colorbar()
  plt.figure(); plt.imshow(sim_cohs,interpolation="None",cmap="Greys_r"); plt.colorbar()


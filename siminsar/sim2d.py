import numpy as np
from .utils import *

class Signal_2d():
  """ just generates a 2D spatial signal with corresponding amplitude """

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
    self.amp = gauss_filt(np.random.rayleigh(self.rayleigh_scale,(self.height,self.width)),10)
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

  def add_gauss_bubble_grid(self, num_of_bubbles=4, equal=True, amp_range=[-20, 20]):
      w_step = int(self.width / num_of_bubbles)
      x_means = np.arange(int(w_step/2), int(self.width-w_step/2+1), w_step)
      y_means = np.arange(int(w_step/2), int(self.width-w_step/2+1), w_step)
      amps = np.linspace(amp_range[0], amp_range[1], num_of_bubbles)
      x_std = w_step/4 
      y_std = w_step/4
      for i in range(len(x_means)):
          for j in range(len(y_means)):
            self.signal_gauss_bubbles.append((amps[i], x_means[i], y_means[j], x_std, y_std, 0))
      # self.signal_gauss_bubbles.append((amp, 50, 50, 25, 25, 0))
      # self.signal_gauss_bubbles.append((amp, 10, 10, 10, 10, 0))

  def add_gauss_bubble_primid(self, ratio=0.50, equal=True, amp_range=[-120, 120]):
      cur_std =((self.width * ratio)/2)
      top_bound = 0
      y_mean = top_bound + cur_std
      count = 0
      while (count<5):
          print(cur_std)
          x_means = np.arange((cur_std), int(self.width), cur_std*2)
          print(x_means)
          amps = np.linspace(amp_range[0], amp_range[1], len(x_means))
          amps = np.flip(amps,0)
          for i in range(len(x_means)):
              self.signal_gauss_bubbles.append((amps[i], x_means[i], y_mean, cur_std/2, cur_std/2, 0))
          top_bound += cur_std*2
          cur_std = (self.height - top_bound)*ratio/2
          y_mean = top_bound + cur_std
          count += 1


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


  def add_amp_stripe(self, thickness_range=[1,10]):
    """ alters the amplitude in a band region (excluding buildings)
    :param thickness_range: range of approximate thicknesses of the bands
    """
    thickness = (np.random.random()*(thickness_range[1]-thickness_range[0])+thickness_range[0])
    amplitude = np.random.rayleigh(self.rayleigh_scale)
    mask = generate_band_mask(self.width,self.height,thickness)
    self.amp[mask] = amplitude


  def add_n_amp_stripes(self, thickness_range=[1,10], nps=5):
    """
    :param thickness: range of approximate thicknesses of the bands
    :param nps: number of bands to add
    """
    for i in range(nps):
      self.add_amp_stripe(thickness_range)


  def compile(self):
    """ takes all the model parameters and generates the signal and amplitude """

    # first add the gaussian bubbles
    self.signal = np.zeros((self.height,self.width))

    # if self.add_random_dem_signal_flag:
    #   assert(self.width==self.height)
    #   dems = glob('/disk/tembofallback/c003/scratch/azimmer/rdc_dems/*')
    #   idx = np.random.randint(len(dems))
    #   rdc_dem = dems[idx]
    #   dem_width = int(rdc_dem.split('.')[-2])
    #   dem_height = int(rdc_dem.split('.')[-1])
    #   dem, rs, cs, h = readFloatRandomPatches(rdc_dem, width=dem_width, patch_size=self.width, num_sample=1, height=dem_height)
    #   self.signal += dem[0] * np.random.rand() * self.dem_scale
    #   #TODO: do the same thing with the average rmli for the amplitudes

    for params in self.signal_gauss_bubbles:
      self.signal += eval_2d_gauss(self.x, self.y, params)

    for params in self.signal_ellipses:
      self.signal += eval_3d_ellipsoid(self.x, self.y, params)*(-1 if np.random.rand()>0.5 else 1)

    for params in self.signal_polygons:
      if np.random.rand()>0.5:
        self.signal += np.transpose(eval_3d_polygon(np.transpose(self.x), np.transpose(self.y), params)*(-1 if np.random.rand()>0.5 else 1))
      else:
        self.signal += eval_3d_polygon(self.x, self.y, params)*(-1 if np.random.rand()>0.5 else 1)

    # then add the buildings
    vacant_lots = np.ones((self.height,self.width)).astype(np.bool)
    for params in sorted(self.signal_buildings):
      _,amplitude,w,h,d,px,py = params
      #print params
      cur_building, cur_building_mask = eval_2d_building(self.x,self.y,vacant_lots,(w,h,d,px,py))
      self.signal[cur_building_mask] += cur_building[cur_building_mask]
      self.amp[cur_building_mask] = amplitude
      vacant_lots = (vacant_lots) & (cur_building_mask==False)

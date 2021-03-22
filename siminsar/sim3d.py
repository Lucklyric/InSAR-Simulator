import numpy as np
from .utils import *

class Signal_3d(object):
  """ takes 2d signals and generates a temporal dimension """

  def __init__(self,width,height,depth):
    np.random.seed(np.random.randint(1,1000000+1))
    self.width = width
    self.height = height
    self.depth = depth
    self.x, self.y = np.meshgrid(range(self.width),range(self.height))
    self.x = self.x.astype(np.float)
    self.y = self.y.astype(np.float)
    self.signal = np.zeros((depth,height,width))
    self.hgterr = np.zeros((height,width))
    self.rate = np.zeros((height,width))
    self.nlm = np.zeros((depth,height,width))
    self.do_hgterr = False
    self.do_rate = False
    self.do_nlm = False


  def add_hgterr(self,hgterr,bperps,conv2):
    self.hgterr = hgterr
    self.bperps = bperps
    self.conv2 = conv2
    self.do_hgterr = True


  def add_rate(self,rate,days,conv1):
    self.rate = rate
    self.days = days
    self.conv1 = conv1
    self.do_rate = True


  def add_nlm(self,nlm):
    self.nlm = nlm
    self.do_nlm = True


  def compile(self):
    if self.do_hgterr:
      assert(len(self.bperps)+1==self.depth)
      tmp = np.zeros_like(self.signal)
      for i,bperp in enumerate(self.bperps):
        self.signal[i] = bperp*self.conv2*self.hgterr
      
    if self.do_rate:
      assert(len(self.days)+1==self.depth)
      tmp = np.zeros_like(self.signal)
      for i,day in enumerate(self.days):
        tmp[i+1] = tmp[i]+day*self.conv1*self.rate
      self.signal += tmp

    if self.do_nlm:
      assert(self.nlm.shape==self.signal.shape)
      self.signal += self.nlm

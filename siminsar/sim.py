from random import randint
from scipy.ndimage import gaussian_filter as gauss_filt
import numpy as np

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

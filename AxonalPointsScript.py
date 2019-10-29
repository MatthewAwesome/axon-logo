import numpy as np
import matplotlib.pyplot as plt
import json 

# A utility function to downsample our axon bundles traces: 
def ResampleLinear1D(original, targetLen):
  original = np.array(original, dtype=np.float)
  index_arr = np.linspace(0, len(original)-1, num=targetLen, dtype=np.float)
  index_floor = np.array(index_arr, dtype=np.int) #Round down
  index_ceil = index_floor + 1
  index_rem = index_arr - index_floor #Remain
  
  val1 = original[index_floor]
  val2 = original[index_ceil % len(original)]
  interp = val1 * (1.0-index_rem) + val2 * index_rem
  assert(len(interp) == targetLen)
  return interp

#Keeping it clean:
plt.close('all')

# Some angles we seek to launch axon bundles from: (initial angles) 
superiorAngles = np.linspace(60,165,20)
inferiorAngles = np.linspace(-40,-165,20)

# The radius of the optic nerve head, in degress: 
r0 = 1.0 
# max radius, in degress. We use this for chopping the axons. 
r_max = 25


# A vector of radii along which we seek to determine the angular position of the bundles 
r_vect = np.arange(r0,45,0.01)
relative_rad = r_vect - r0

# Some beta-values, which we use to determine parameters below: 
beta_s = -1.9
beta_i = 0.71

# Some data containers to be filled in the loop below: 
superior_x= np.zeros((superiorAngles.shape[0],r_vect.shape[0]))
superior_y = np.zeros(superior_x.shape)
inferior_x = np.zeros(superior_x.shape)
inferior_y = np.zeros(superior_x.shape)

# Going downsample the traces, and plot dots as opposed to lines: 
n_interp_pts = 100
resampled_superior_x = np.zeros((superiorAngles.shape[0],n_interp_pts))
resampled_superior_y = np.zeros(resampled_superior_x.shape)
resampled_superior_r = np.zeros(resampled_superior_x.shape)
resampled_inferior_x = np.zeros(resampled_superior_x.shape)
resampled_inferior_y = np.zeros(resampled_superior_x.shape)
resampled_inferior_r = np.zeros(resampled_superior_x.shape)
resampled_r_vect = np.linspace(relative_rad.min(),relative_rad.max(),n_interp_pts)

plt.figure()
#A loop to get some values: (superior and inferior angles are the same length, 
# which means a one-loop operation)
for j in range(0,superiorAngles.shape[0]): 
  # For the initial superior angles: 
  superiorThing = (superiorAngles[j]-121)/14
  c_val_s = 1.9 + 1.4*np.tanh(superiorThing)
  b_val_s = np.exp(beta_s + 3.9 * np.tanh(-1*superiorThing))
  # And the initial inferior angles: 
  inferiorThing = (-inferiorAngles[j]-90)/25
  c_val_i = 1.0 + 0.5*np.tanh(inferiorThing)
  b_val_i = -1*np.exp(beta_i + 1.5 * np.tanh(-1*inferiorThing))
  # Some vectors we use loopwise: 
  superiorX = np.zeros(r_vect.shape)
  superiorY = np.zeros(r_vect.shape)
  inferiorX = np.zeros(r_vect.shape)
  inferiorY = np.zeros(r_vect.shape)
  # Plugging in our params and spawning bundles: 
  # Two loops; one for superior, one fo rinferior 
  for i in range(0,r_vect.shape[0]): 
    superior_angular_val = (superiorAngles[j]+ b_val_s*(r_vect[i]-r0)**c_val_s)
    inferior_angular_val = (inferiorAngles[j] + b_val_i*(r_vect[i]-r0)**c_val_i)
    # Putting in x,y cartesian coordinates; AFRAME prefers this format!
    x_prime_s = (r_vect[i]) * np.cos(superior_angular_val*np.pi/180)
    y_prime_s = (r_vect[i]) * np.sin(superior_angular_val*np.pi/180)
    x_val_s = x_prime_s + 15
    if x_val_s > 0: 
      y_val_s = y_prime_s + 2*(x_val_s/15)**2
    else: 
      y_val_s = y_prime_s
    # Check for r_max:
    cart_r_s = np.sqrt(x_val_s**2 + y_val_s**2)
    if cart_r_s <= r_max:
      superiorX[i] = x_val_s
      superiorY[i] = y_val_s
    # Now for the inferior: 
    x_prime_i = (r_vect[i]) * np.cos(inferior_angular_val*np.pi/180)
    y_prime_i = (r_vect[i]) * np.sin(inferior_angular_val*np.pi/180)
    x_val_i = x_prime_i + 15
    if x_val_i > 0: 
      y_val_i = y_prime_i + 2*(x_val_i/15)**2
    else: 
      y_val_i = y_prime_i
    cart_r_i = np.sqrt(x_val_i**2 + y_val_i**2)
    if cart_r_i <= r_max:
      inferiorX[i] = x_val_i
      inferiorY[i] = y_val_i
  # Now trim and plot: (and interpolate, too!)
  chopped_y_s = np.argmax(superiorY<0.2)
  if chopped_y_s == 0: 
    resampled_superior_x[j,:] = ResampleLinear1D(superiorX,n_interp_pts)
    resampled_superior_y[j,:] = ResampleLinear1D(superiorY,n_interp_pts)
    resampled_superior_r[j,:] = ResampleLinear1D(relative_rad,n_interp_pts)
  else: 
    resampled_superior_x[j,:] = ResampleLinear1D(superiorX[0:chopped_y_s],n_interp_pts)
    resampled_superior_y[j,:] = ResampleLinear1D(superiorY[0:chopped_y_s],n_interp_pts)
    resampled_superior_r[j,:] = ResampleLinear1D(relative_rad[0:chopped_y_s],n_interp_pts)
  # And so we can visualize what we're outputting: 
  plt.plot(resampled_superior_x[j,:],resampled_superior_y[j,:] )
  #Now trim/plot the inferior: 
  flipped_y = np.flip(inferiorY)
  flipped_max = np.argmax(flipped_y<-0.2)
  unflipped_max = flipped_y.size-flipped_max
  if unflipped_max > 500: 
    resampled_inferior_x[j,:] = ResampleLinear1D(inferiorX[0:unflipped_max],n_interp_pts)
    resampled_inferior_y[j,:] = ResampleLinear1D(inferiorY[0:unflipped_max],n_interp_pts)
    resampled_inferior_r[j,:] = ResampleLinear1D(relative_rad[0:unflipped_max],n_interp_pts)
  else: 
    resampled_inferior_x[j,:] = ResampleLinear1D(inferiorX,n_interp_pts)
    resampled_inferior_y[j,:] = ResampleLinear1D(inferiorY,n_interp_pts)
    resampled_inferior_r[j,:] = ResampleLinear1D(relative_rad,n_interp_pts)
  plt.plot(resampled_inferior_x[j,:],resampled_inferior_y[j,:] )

# Equally scale the axes: 
plt.ylim((-40, 40)) 
plt.xlim((-40, 40)) 

# Turn our numpy arrays into lists: 
sup_x_list = resampled_superior_x.tolist()
sup_y_list = resampled_superior_y.tolist()
sup_r_list = resampled_superior_r.tolist()
inf_x_list = resampled_inferior_x.tolist()
inf_y_list = resampled_inferior_y.tolist()
inf_r_list = resampled_inferior_r.tolist()

# Making dicts specific to superior and inferior:
superiorDict = {'x':sup_x_list,'y':sup_y_list,'r':sup_r_list,'angles':superiorAngles.tolist()}
inferiorDict = {'x':inf_x_list,'y':inf_y_list,'r':inf_r_list,'angles':inferiorAngles.tolist()}

#Putting it all together, jsonifying it, and outputting the axonal trajectory data: 
outputDict = {'superior':superiorDict,'inferior':inferiorDict} 
with open("fiber_points.json", "w") as ww:
    json.dump(outputDict, ww)

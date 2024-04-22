import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from corey import coreyWater, coreyOil
from conversion import daysToSeconds,secondsToDays


class Simulator3DIMPES:
    def __init__(self, grid_shape, grid_length):
        self.Ncells = grid_shape[0]*grid_shape[1]*grid_shape[2]
        self.length = grid_length
        self.grid_shape = grid_shape
        self.deltaX = grid_length[0]/grid_shape[0]
        self.deltaY = grid_length[1]/grid_shape[1]
        self.deltaZ = grid_length[2]/grid_shape[2]
        #self.delta = grid_length/grid_shape
        self.delta = np.array([self.deltaX, self.deltaY, self.deltaZ])
        self._perm = self.setPermeabilities(1.0E-13*np.ones(grid_shape))
        self.poro = 0.25*np.ones(grid_shape)
        self.pressure = 1.0E7 * np.ones(grid_shape)
        self.saturation = 0.2 * np.ones(grid_shape)
        self.saturation[2,2,2] = 0.8
        self.rightPressure = 1.0E7
        #self.leftDarcyVelocity = 2.315E-6 * self.poro[0]
        self.mobilityWeighting = 1.0
        self.deltat = daysToSeconds(0.1)
        self.time = 0.0
        self.oilViscosity = 2.0E-3
        self.waterViscosity = 1.0E-3
        self.relpermWater = coreyWater(2.0, 0.4, 0.2, 0.2)
        self.relpermOil = coreyOil(2.0, 0.2, 0.2)

    def setPermeabilities(self,permVector):
        '''
        Set permeabilities
        Args:
        permVector: A numpy array of length
        self.Ncells with perm values
        '''
        self._perm = permVector
        self._TranX = (2.0/(1.0/self._perm[:-1,:,:]+1.0/self._perm[1:,:,:]))/self.delta[0]**2
        self._TranY = (2.0/(1.0/self._perm[:,:-1,:]+1.0/self._perm[:,1:,:]))/self.delta[1]**2
        self._TranZ = (2.0/(1.0/self._perm[:,:,:-1]+1.0/self._perm[:,:,1:]))/self.delta[2]**2
        #self._TranRight = self._perm[-1]/self.deltaX**2

    # logical index to global index
    def l2g(self,i,j,k):
        return i + self.grid_shape[0] * j + self.grid_shape[0] * self.grid_shape[1] * k

    def doTimestep(self):
        '''
        Do one time step of length self.deltat
        '''
        upW = self.mobilityWeighting
        downW = 1.0-upW
        # determing the flow direction
        result_X = self.pressure[1:, :, :] - self.pressure[:-1, :, :]
        result_Y = self.pressure[:, 1:, :] - self.pressure[:, :-1, :]
        result_Z = self.pressure[:, :, 1:] - self.pressure[:, :, :-1]
        #if np.any(result_X > 0) or np.any(result_Y > 0) or np.any(result_Z > 0):
        #    print("A > 0")
        #else:
        #    print("A <= 0")

        # # Determine the flow direction and assign mobility weighting
        # if np.any(result_X > 0):
        #     mobility_weight_x = upW
        # else:
        #     mobility_weight_x = downW
          
        # if np.any(result_Y > 0):
        #     mobility_weight_y = upW
        # else:
        #     mobility_weight_y = downW
            
        # if np.any(result_Z > 0):
        #     mobility_weight_z = upW
        # else:
        #     mobility_weight_z = downW
        
              
        
        mobOil = self.relpermOil(self.saturation)/self.oilViscosity
        mobWater = self.relpermWater(self.saturation)/self.waterViscosity
        wh = np.where(result_X > 0)
        mobOilW_X = mobOil[:-1, :, :]*upW + mobOil[1:, :, :]*downW # if 'result_X is positive, the flow is in the opposite direction of increasing pressure (higher --> lower)
        mobOilW_X[wh] = mobOil[:-1, :, :][wh]*downW + mobOil[1:, :, :][wh]*upW # in the other direction (up)
        wh = np.where(result_Y > 0)
        mobOilW_Y = mobOil[:, :-1, :]*upW + mobOil[:, 1:, :]*downW
        mobOilW_Y[wh] = mobOil[:, :-1, :][wh]*downW + mobOil[:, 1:, :][wh]*upW
        wh = np.where(result_Z > 0)
        mobOilW_Z = mobOil[:, :, :-1]*upW + mobOil[:, :, 1:]*downW
        mobOilW_Z[wh] = mobOil[:, :, :-1][wh]*downW + mobOil[:, :, 1:][wh]*upW
        wh = np.where(result_X > 0)
        mobWaterW_X = mobWater[:-1, :, :]*upW + mobWater[1:, :, :]*downW
        mobWaterW_X[wh] = mobWater[:-1, :, :][wh]*upW + mobWater[1:, :, :][wh]*downW
        wh = np.where(result_Y > 0)
        mobWaterW_Y = mobWater[:, :-1, :]*upW + mobWater[:, 1:, :]*downW
        mobWaterW_Y[wh] = mobWater[:, :-1, :][wh]*upW + mobWater[:, 1:, :][wh]*downW
        wh = np.where(result_Z > 0)
        mobWaterW_Z = mobWater[:, :, :-1]*upW + mobWater[:, :, 1:]*downW
        mobWaterW_Z[wh] = mobWater[:, :, :-1][wh]*upW + mobWater[:, :, 1:][wh]*downW
        oilTrans_X = self._TranX*mobOilW_X
        oilTrans_Y = self._TranY*mobOilW_Y
        oilTrans_Z = self._TranZ*mobOilW_Z
        waterTrans_X = self._TranX*mobWaterW_X
        waterTrans_Y = self._TranY*mobWaterW_Y
        waterTrans_Z = self._TranZ*mobWaterW_Z

        # ----------------------------
        # Solve implicit for pressure:
        #
        # We solve a linear system matrixA pressure = vectorE
        #

        # --- Build matrixA:
        matrixA = np.zeros((self.Ncells,self.Ncells))
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                for k in range(self.grid_shape[2]):
                    index = self.l2g(i,j,k)
                    if not i - 1 < 0:
                        matrixA[index, index - 1] = oilTrans_X[i - 1,j,k] # Left
                        matrixA[index, index] -= oilTrans_X[i - 1,j,k] # Diagonal
                    if not i + 1 > self.grid_shape[0] - 1: # Start counting at zero
                        matrixA[index, index + 1] = oilTrans_X[i,j,k] # Right
                        matrixA[index, index] -= oilTrans_X[i,j,k]
                    if not j - 1 < 0:
                        matrixA[index, index - self.grid_shape[0]] = oilTrans_Y[i,j - 1,k]
                        matrixA[index, index] -= oilTrans_Y[i,j - 1,k]
                    if not j + 1 > self.grid_shape[1] - 1:
                        matrixA[index, index + self.grid_shape[0]] = oilTrans_Y[i,j,k]
                        matrixA[index, index] -= oilTrans_Y[i,j,k]
                    if not k - 1 < 0:
                        matrixA[index, index - self.grid_shape[0] * self.grid_shape[1]] = oilTrans_Z[i,j,k - 1]
                        matrixA[index, index] -= oilTrans_Z[i,j,k - 1]
                    if not k + 1 > self.grid_shape[2] - 1:
                        matrixA[index, index + self.grid_shape[0] * self.grid_shape[1]] = oilTrans_Z[i,j,k]
                        matrixA[index, index] -= oilTrans_Z[i,j,k]

        # Boundary conditions

        # ------
        # --- Build vectorE:
        vectorE = np.zeros(self.Ncells) # Total number of cells


        # Well 1
        index=self.l2g(0,0,0)
        matrixA[index,index]+=1
        vectorE[index]+=1E7

        index=self.l2g(2,2,2)
        matrixA[index,index]+=1
        vectorE[index]+=1.05E7

        # ------
        # --- Solve linear system:
        matrixAInv = np.linalg.inv(matrixA)

        pressure = np.dot(matrixAInv,vectorE)
        self.pressure = pressure.reshape(self.grid_shape)
        # --------------------------------
        # Solve explicitly for saturation:
        dtOverPoro = self.deltat/self.poro

        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                for k in range(self.grid_shape[2]):
                    index = self.l2g(i,j,k)

                    value = 0
                    if not i+1 > self.grid_shape[0] - 2:
                        value += waterTrans_X[i,j,k]*(pressure[(i+1)  + self.grid_shape[0] * j + self.grid_shape[0] * self.grid_shape[1] * k]-pressure[index])
                    if not i-1 < 0:
                        value += waterTrans_X[i-1,j,k]*(pressure[(i-1)  + self.grid_shape[0] * j + self.grid_shape[0] * self.grid_shape[1] * k]-pressure[index])
                    if not j+1 > self.grid_shape[1] - 2:
                        value += waterTrans_Y[i,j,k]*(pressure[i  + self.grid_shape[0] * (j+1) + self.grid_shape[0] * self.grid_shape[1] * k]-pressure[index])
                    if not j-1 < 0:
                        value += waterTrans_Y[i,j-1,k]*(pressure[i  + self.grid_shape[0] * (j-1) + self.grid_shape[0] * self.grid_shape[1] * k]-pressure[index])
                    if not k+1 > self.grid_shape[2] - 2:
                        value += waterTrans_Z[i,j,k]*(pressure[i  + self.grid_shape[0] * j + self.grid_shape[0] * self.grid_shape[1] * (k+1)]-pressure[index])
                    if not k-1 < 0:
                        value += waterTrans_Z[i,j,k-1]*(pressure[i  + self.grid_shape[0] * j + self.grid_shape[0] * self.grid_shape[1] * (k-1)]-pressure[index])

                    self.saturation[i,j,k] += dtOverPoro[i,j,k]*(value) # Substracting since we are using oil mobilities

        maxsat = 1.0-self.relpermOil.Sorw
        minsat = self.relpermOil.Swirr
        self.saturation[ self.saturation>maxsat ] = maxsat
        self.saturation[ self.saturation<minsat ] = minsat
        # --------------------------------
        #self.pressure = pressure
        self.time = self.time + self.deltat

        # Pick a cell and update (P = 1 bar and S = 0.2) - "well"

        # Production well
        self.saturation[0,0,0] = 0.2

        # Injection well
        self.saturation[2,2,2] = 0.8

    
    def simulateTo(self,time):
        '''
        Progress simulation to specific time with
        a constant timestep self.deltat
        Args:
        time: Time to advance to [s]
        '''
        baseDeltat = self.deltat
        while self.time < time:
            if self.time + baseDeltat >= time:
                self.deltat = time - self.time
                self.doTimestep()
                self.deltat = baseDeltat
                self.time = time
            else:
                self.doTimestep()

# End of the original code from the script

#Time step (for 1D) - change to 3D

# Simulation parameters
grid_shape_3d = (5, 5, 5)  # Changed to 3D grid shape
grid_length_3d = (10, 10, 10)  # Changed to 3D grid length

# Simulator for 3D simulation
simulator_3d = Simulator3DIMPES(grid_shape_3d, grid_length_3d)
# simulator_refined_3d = Simulator3DIMPES(grid_shape_3d, grid_length_3d)
# simulator_refined_3d.mobilityWeighting = 0.85

# Lists to store simulation results
pressures_3d = []
saturation_3d = []
# pressures_refined_3d = []
# saturation_refined_3d = []

# Define the times for running the simulation and compare
times = [daysToSeconds(100)]#, daysToSeconds(200), daysToSeconds(300), daysToSeconds(400), daysToSeconds(500)]

# Run simulations and store results
for time in times:
    simulator_3d.simulateTo(time)
    pressures_3d.append(simulator_3d.pressure.copy())
    saturation_3d.append(simulator_3d.saturation.copy())
    # simulator_refined_3d.simulateTo(time)
    # pressures_refined_3d.append(simulator_refined_3d.pressure.copy())
    # saturation_refined_3d.append(simulator_refined_3d.saturation.copy())

# Cell lengths for each dimension in 3D
cell_lengths_x_3d = np.linspace(0, grid_length_3d[0], grid_shape_3d[0])
cell_lengths_y_3d = np.linspace(0, grid_length_3d[1], grid_shape_3d[1])
cell_lengths_z_3d = np.linspace(0, grid_length_3d[2], grid_shape_3d[2])

# Plotting saturation in 3D (5 layers from top to the bottom)
pres = pressures_3d[-1] #Last time step
print(pres)
sat = saturation_3d[-1]
print(sat)
for i in range(grid_shape_3d[0]):
    print("Pressure:", i)
    plt.imshow(pres[:, :, i])
    plt.show()

for i in range(grid_shape_3d[0]):
    print("Saturation:", i)
    plt.imshow(sat[:, :, i])
    plt.show()

#X, Y, Z = np.meshgrid(np.arange(grid_length_3d[0]), np.arange(grid_length_3d[1]), np.arange(grid_length_3d[2]))
#X = X.reshape(np.shape(sat))
#Y = Y.reshape(np.shape(sat))
#Z = Z.reshape(np.shape(sat))
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.contourf(sat[0, :, :], Y[0, :, :], Z[0, :, :], zdir='x')
#ax.contourf(X[:, 0, :], sat, Z[:, 0, :], zdir='y')
#ax.contourf(X[:, :, 0], Y[:, :, 0], sat, zdir='z')
#plt.show()
#cmap = plt.get_cmap('plasma')

#for i, (time, sat) in enumerate(zip(times, saturation_3d)):
#    print(sat[:, :, 0])
#    ax.plot(cell_lengths_x_3d, cell_lengths_y_3d, sat)  # Adjusted dimensions
#
#ax.set_title('Saturation Profiles at Different Times')
#ax.set_xlabel('X Distance [m]')
#ax.set_ylabel('Y Distance [m]')
#ax.set_zlabel('Saturation')
#plt.show()
#
## Plotting pressure in 3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#for i, (time, pressure) in enumerate(zip(times, pressures_3d)):
#    ax.plot(cell_lengths_x_3d, cell_lengths_y_3d, pressure[:, :, 0], color=cmap(1 - i / len(times)))  # Adjusted dimensions
#
#ax.set_title('Pressure Profiles at Different Times')
#ax.set_xlabel('X Distance [m]')
#ax.set_ylabel('Y Distance [m]')
#ax.set_zlabel('Pressure')
#plt.show()
#
## Sensitivity Analysis in 3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#for i, (time, sat, sat_r) in enumerate(zip(times, saturation_3d, saturation_refined_3d)):
#    ax.plot(cell_lengths_x_3d, cell_lengths_y_3d, sat[:, :, 0], label=r'$\alpha$=1.0', color=cmap(1 - i / len(times)))
#    ax.plot(cell_lengths_x_3d, cell_lengths_y_3d, sat_r[:, :, 0], label=r'$\alpha$=0.85', linestyle='dashed', color=cmap(1 - i / len(times)))
#
#ax.set_title('Saturation Profiles at Different Mobility Weighting')
#ax.set_xlabel('X Distance [m]')
#ax.set_ylabel('Y Distance [m]')
#ax.set_zlabel('Saturation')
#ax.legend()
#plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def F(x,y):
  return np.exp(1j*y) + 2*np.exp(-1j*y/2)*np.cos(np.sqrt(3)*x/2)

def G(x,y):
   return np.conj(F(x,y))

def M(x,y):
   return np.array([[0,F(x,y)],[G(x,y),0]],dtype=complex)

N = 10
x = np.linspace (-np.pi,np.pi,N)
y = np.linspace (-np.pi,np.pi,N)
X,Y = np.meshgrid(x,y)

matrix = np.zeros((N,N,2,2),dtype=complex)

for i in range(N):
  for j in range(N):
      matrix[i,j]= M(X[i,j],Y[i,j])

eigenvalues, eigenvectors = np.linalg.eig(matrix)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot the first eigenvalue
ax.plot_surface(X, Y, eigenvalues[...,0].real, cmap='coolwarm')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('eigenvalue 1')

# plot the second eigenvalue
ax.plot_surface(X, Y, eigenvalues[...,1].real, cmap='coolwarm')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('eigenvalue 2')

plt.show()

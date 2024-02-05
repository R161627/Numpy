
""" 
NumPy is a general-purpose array-processing package. 
It provides a high-performance multidimensional array object and tools for working with these arrays.
It is the fundamental package for scientific computing with Python. 
It is open-source software.
Features of NumPy
    NumPy has various features including these important ones:
    A powerful N-dimensional array object
    Sophisticated (broadcasting) functions
    Tools for integrating C/C++ and Fortran code
    Useful linear algebra, Fourier transform, and random number capabilities 
properties:
    homogenous
    fixed item size
creating numpy arrays:
    using np.array() //1-D and 2-D
    using np.zeros()/ones()
    using np.arange()
    using np.linspace()
    using copy()
    using identity()
"""
#importing numpy
import numpy as np

# using np.array() //1-D 
arr1 = np.array([1,2,3])
print(arr1)

# using np.array() //2-D 
arr2 = np.array([[1,4],[2,3]])
print(arr2)

# using np.zeros()
arr3 = np.zeros((2,3)) # 2->row and 3 ->column
print(arr3)

# using np.ones()
arr4 = np.ones((2,3)) #  2->row and 3 ->column
print(arr4)

# using np.identity()
arr5 = np.identity(5) # returns identity matrix of order 5
print(arr5)

# using np.arange(start_index,end_index,increment/decrement)
arr6 = np.arange(10,15,2)
print(arr6)

#using np.linspace(start_index,end_index,equi_distant_points)
arr7 = np.linspace(10,20,10)
print(arr7)

# using copy()
arr8 = arr7
print(arr8)

"""
properties and attributes:
    shape --> The shape of an array is the number of elements in each dimension.
    ndim --> ndim attribute returns an integer that tells us how many dimensions the array have.
    size --> Number of elements in the array.
    itemsize --> Length of one array element in bytes.
    dtype --> Data-type of the array's elements.
    astype() --> Copy of the array, cast to a specified type.
                ndarray.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)
    reshape(x,y) --> Returns an array containing the same data with a new shape.
"""
arr9 =np.array([1,2,3,4])
print(arr9.shape) # returns (4,) --> one dimension array

arr10 = np.array([[2],[3]])
print(arr10.shape) #returns (2,1) --> two dimensional array

arr11 = np.array([[[1,2],[3,4],[5,6]]])
print(arr11.shape) #returns (1,3,2) --> three dimensional array

arr12 = np.array([1,2,3])
print(arr12.ndim)  # returns 1 --> one dimension array

arr13 = np.array([1,2,3])
print(arr13.size) #returns 3

arr14 = np.array([1,2,3])
print(arr14.itemsize)

print(arr14.dtype) #returns int64

print(arr14.astype('float'))

print(np.arange(10).reshape(2,5))

"""
    python lists vs numpy arrays
        -Faster
        -convinient
        -less memory
"""

lista = range(100)
nparr = np.arange(100)

import sys
print(sys.getsizeof(85)*len(lista)) # memory --> python list, 85 is any number in a list
print(nparr.itemsize*nparr.size) # memory --> np array

# conclusion: memory(python list) > memory(numpy array)

import time 
x = range(10000000) 
y = range(10000000,20000000)

start_time = time.time()

c =[(i+j) for i,j in zip(x,y)]

end_time = time.time() - start_time
print(end_time)

a = np.arange(10000000)
b = np.arange(10000000,20000000)
start_time_np = time.time()
c = a+b
end_time_np = time.time() -start_time_np
print(end_time_np)

# conclusion-2: operation_time(python list) > operation_time(numpy array)

"""
Slicing in python means taking elements from one given index to another given index.
We can also define the step, like this: [start:end:step]
"""
sarr = np.arange(24).reshape(6,4)
print(sarr)
"""[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]
 [16 17 18 19]
 [20 21 22 23]]"""
print(sarr[-1]) #returns [20 21 22 23]
print(sarr[:,3]) #returns all elements in the third column

"""
np.nditer(array)-->The array(s) to iterate over.
"""
for i in sarr:
    print(i) #return rows in the nd matrix

for i in np.nditer(sarr):
    print(i) #returns each element

#numpy operations
    
nparr = np.array([1,2,3,4])
nparr2 = np.array([5,6,7,8])

print(nparr+nparr2)
print(nparr-nparr2)
print(nparr*nparr2)
print(nparr/nparr2)
print(nparr%nparr2)
print(nparr*10)
print(nparr>3)

# dot product of two matrix -->multiplication of two matrix with satisfying order

nparr3 = np.arange(6).reshape(2,3)
nparr4 = np.arange(6,12).reshape(3,2)

print(nparr3.dot(nparr4))
# array.min() and array.max() -->returns max and min values in the given array
print(nparr3.min())
print(nparr3.max())
# axis=0 --> column wise , axis=1 -->row wise
print(nparr3.min(axis=0))
print(nparr3.max(axis=0))
print(nparr3.min(axis=1))
print(nparr3.max(axis=1))

#array.sum() --> returns total sum of all elements in the array
print(nparr3.sum())
print(nparr3.sum(axis=0))
print(nparr3.sum(axis=1))

#array.mean() -->returns mean of the elements
print(nparr3.mean())
print(nparr3.mean(axis=0))
print(nparr3.mean(axis=1))

# array.std() --> returns standard deviation of the array
print(nparr3.std())
print(nparr3.std(axis=0))
print(nparr3.std(axis=1))

# np.median(array) -->returns the median of the array
print(np.median(nparr3))
print(np.median(nparr3,axis=0))
print(np.median(nparr3,axis=1))

# triganometeric functions sin,cos,tan,arcsin,arccos,arctan
print(np.sin(nparr3)) #returns element wise
print(np.cos(nparr3))
print(np.tan(nparr3))
print(np.arcsin(nparr3))
print(np.arccos(nparr3))
print(np.arctan(nparr3))

# np.exp(array) --> exponential function

print(np.exp(nparr3))

# array.ravel() --> converts the n-dimentional array to 1-d array
print(nparr3.ravel())

# array.transpose() --> columns=rows, rows=columns
print(nparr3.transpose())

# np.hstack((array1,array2)) --> combines two arrays horizontally(axis should match)
# np.vstack((array1,array2)) --> combines two arrays vertically(axis should match)
a1 = np.arange(6).reshape(2,3)
a2 = np.arange(6,12).reshape(2,3)
print(np.hstack((a1,a2)))
print(np.vstack((a1,a2)))

# Splitting is reverse operation of Joining.
# print(np.hsplit(a1,3))
# print(np.vstack(a1,2))

#fancy indexing --> using this we can extract the rows and columns in any order. 
#example 2 and 8 coulmn and rows.
a3 = np.arange(24).reshape(6,4)
print(a3)
print(a3[[0,2,5]])
print(a3[:,[0,2,3]])

x = np.linspace(-40,40,100)
y = np.sin(x)
print(x.size," ",y.size)

import matplotlib.pyplot as plt 

plt.plot(x,y)



import numpy as np
# random() function generates the different random values between 0 to 1 for each run
print(np.random.random())
# seed() function generates the same random value between 0 to 1 for each run
np.random.seed(1)
print(np.random.random())
# uniform(start_index,end_index,no_of_elements) method used to get the random values in between a range
print(np.random.uniform(10,100,10).reshape(5,2)) # by deafult it returns the float values in an array
# randint(start_index,end_index,no_of_elements) method used to get the int random values between givenm range.
print(np.random.randint(10,100,10).reshape(5,2))
b = np.random.randint(10,100,10)
print(b)
print(b[np.argmax(b)])
print(np.argmax(b))
print(b[np.argmin(b)])
print(np.argmin(b))
# argmin() and argmax() returns the index of the min and max valuues of an array respectively

#replacing the odd values of array to -1
b[b%2==1]=-1
print(b) # it changing the values in original array

# np.where(condition,true_value,false_value) used to check the conditions

a = np.random.randint(1,100,10)
print(a)
c = np.where(a%2==0,a,-1)
print(c) # it can not change the original array values.
# returns the sorted array values
print(np.sort(a)) 

# refer percentile function in numpy.
# refer broadcasting in numpy







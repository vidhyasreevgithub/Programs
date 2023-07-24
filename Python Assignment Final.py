#!/usr/bin/env python
# coding: utf-8

# In[1]:


#UNIT III
#USING NUMPY


# In[ ]:


#Creating Numpy Arrays


# In[2]:


#Q1.Create the evenly spaced array using arange
import numpy as np
a = np.arange(1, 10)
print(a)
x = range(1, 10)
print(x) 
print(list(x))
x = np.arange(10.4)
print(x)
x = np.arange(0.5, 10.4, 0.8)
print(x)


# In[3]:


#Q2.Create the evenly spaced array using linespace
import numpy as np
print("50 values between 1 to 10",np.linspace(1, 10))
print("7 values between 1 to 10 ",np.linspace(1, 10, 7))
print("excluding end points",np.linspace(1, 10, 7, endpoint=False))


# In[4]:


#Q3.Createa Zero dimension array in Numpy also print its type and dimension?
import numpy as np
x = np.array(42)
print("x: ", x)
print("The type of x: ", type(x))
print("The dimension of x:", np.ndim(x))


# In[5]:


#Q4.Create two One dimension array in Numpy also print its type and dimension?
F = np.array([1, 1, 2, 3, 5, 8, 13, 21])
V = np.array([3.4, 6.9, 99.8, 12.8])
print("F: ", F)
print("V: ", V)
print("Type of F: ", F.dtype)
print("Type of V: ", V.dtype)
print("Dimension of F: ", np.ndim(F))
print("Dimension of V: ", np.ndim(V))


# In[6]:


#Q5.Write a code to create a two dimension array in Numpy and print its dimension?
A = np.array([ [3.4, 8.7, 9.9], 
               [1.1, -7.8, -0.7],
               [4.1, 12.3, 4.8]])
print(A)
print(A.ndim)


# In[8]:


#Q6.Create a multi dimension array in Numpy and print its dimension?
B = np.array([ [[111, 112], [121, 122]],
               [[211, 212], [221, 222]],
               [[311, 312], [321, 322]] ])
print(B)
print(B.ndim)


# In[9]:


#Q7.Write a code to return the shape of an array
x = np.array([ [67, 63, 87],
               [77, 69, 59],
               [85, 87, 99],
               [79, 72, 71],
               [63, 89, 93],
               [68, 92, 78]])

print(np.shape(x))


# In[10]:


#Q8.Write a code  to change the shape of the array
x.shape=(3, 6)
print(x)


# In[11]:


#Indexing and Sorting


# In[7]:


#Q9.Write a code  toprint the numbers in an array using its index value?
F = np.array([1, 1, 2, 3, 5, 8, 13, 21])
# print the first element of F
print(F[0])
# print the last element of F
print(F[-1])


# In[13]:


#Q10. Write a code  to print the number using index value in multidimensional array?
A = np.array([ [3.4, 8.7, 9.9], 
               [1.1, -7.8, -0.7],
               [4.1, 12.3, 4.8]])

print(A[1][0])


# In[14]:


#Q11.Print the index 1 of array A and return the 0th index in the 1st index of A?
tmp = A[1]
print(tmp)
print(tmp[0])


# In[15]:


#Q12.Perform slicing in single dimensional array?
S = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(S[2:5])
print(S[:4])
print(S[6:])
print(S[:])


# In[16]:


#Q13.Write a code  toPerform slicing in multi dimensional array?
A = np.array([
[11, 12, 13, 14, 15],
[21, 22, 23, 24, 25],
[31, 32, 33, 34, 35],
[41, 42, 43, 44, 45],
[51, 52, 53, 54, 55]])

print(A[:3, 2:])
print(A[3:, :])
print(A[:, 4:])


# In[8]:


#Q14.Given A= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]where the values are taken from index 2 to index 5 is stored in S and change the index 0 to 22 and index 1 to 23 in S and print A.
A = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
S = A[2:6]
S[0] = 22
S[1] = 23
print(A)


# In[18]:


#Q15.Write a code to check whether the two array A and S share the same memory space?
np.may_share_memory(A, S)


# In[19]:


#Q16.Create an array of 3 rows and 4 columns using numpy.arange function and change its 0 index value to be 42
A = np.arange(12)
B = A.reshape(3, 4)
A[0] = 42
print(B)


# In[20]:


#Numerical Operation


# In[21]:


#Q17.A list is given  lst = [2,3, 7.9, 3.3, 6.9, 0.11, 10.3, 12.9], add 2 to eachelement , multiply each element with 2.2 and subtract each element with1.38 in the given list 
import numpy as np
lst = [2,3, 7.9, 3.3, 6.9, 0.11, 10.3, 12.9]
v = np.array(lst)
v = v + 2
print(v)
print(v * 2.2)
print(v - 1.38)


# In[22]:


#Arithmethic Operation of two arrays


# In[23]:


#Q18.Create two arrays named A and B and add them, add 1 with each element in B and find the product of two arrays
import numpy as np
A = np.array([ [11, 12, 13], [21, 22, 23], [31, 32, 33] ])
B = np.ones((3,3))
print("Adding to arrays: ")
print(A + B)
print("\nMultiplying two arrays: ")
print(A * (B + 1))


# In[24]:


#Matrices Vs Two Dimensional Arrays


# In[25]:


#Q19.Write a code to perform multiplication of two arrays
import numpy as np
A = np.array([ [1, 2, 3], [2, 2, 2], [3, 3, 3] ])
B = np.array([ [3, 2, 1], [1, 2, 3], [-1, -2, -3] ])
R = A * B
print(R)


# In[26]:


#Comparison Operators


# In[27]:


#Q20.Check whether two arrays are equal using comparison operator ‘==’,where A=[ [11, 12, 13], [21, 22, 23], [31, 32, 33] ] and B=[ [11, 102, 13], [201, 22, 203], [31, 32, 303] ]
import numpy as np
A = np.array([ [11, 12, 13], [21, 22, 23], [31, 32, 33] ])
B = np.array([ [11, 102, 13], [201, 22, 203], [31, 32, 303] ])
A == B


# In[28]:


#Q21.Check whether two arrays are equal using numpy.array_equal(),using the previously created array “A” and “B”
print(np.array_equal(A, B))
print(np.array_equal(A, A))


# In[1]:


#Logical Operators


# In[4]:


#Q22.Given a=([ [True, True], [False, False]]) and b=([ [True, False], [True, False]]) check the arrays using logical_or and logical_and.
import numpy as np
a = np.array([ [True, True], [False, False]])
b = np.array([ [True, False], [True, False]])
print(np.logical_or(a, b))
print(np.logical_and(a, b))


# In[ ]:


#Operations on Arrays with its Symbols


# In[7]:


#Q23.Write a code to create multidimensional array A and a single dimensional array B .Multiply A and BAdd A and B
import numpy as np
A = np.array([ [11, 12, 13], [21, 22, 23], [31, 32, 33] ])
B = np.array([1, 2, 3])
print("Multiplication with broadcasting: ")
print(A * B)
print("... and now addition with broadcasting: ")
print(A + B)		


# In[8]:


#Q24.Given B = [1, 2, 3] , print array B with 3 rows and 3 columns
B = np.array([[1, 2, 3],] * 3)
print(B)


# In[9]:


#Q25.Consider the output of previously given array B and print thetranspose of B
np.array([[1, 2, 3],] * 3).transpose()


# In[10]:


#Q26.Given B = [1, 2, 3],consider the rows of B as columns and print the output.
B = np.array([1, 2, 3])
B[:,np.newaxis]


# In[11]:


#Flatten and Reshape arrays


# In[ ]:


#Q27.Write a code for flattening using the values "C", "F" and "A"
""" "C" means to flatten C style in row-major ordering, i.e. the rightmost index "changes the fastest".
"F" stands for Fortran column-major ordering.
"A" means preserve the the C/Fortran ordering."""


# In[12]:


import numpy as np
A = np.array([[[ 0,  1],
               [ 2,  3],
               [ 4,  5],
               [ 6,  7]],
              [[ 8,  9],
               [10, 11],
               [12, 13],
               [14, 15]],
              [[16, 17],
               [18, 19],
               [20, 21],
               [22, 23]]])
Flattened_X = A.flatten()
print(Flattened_X)
print(A.flatten(order="C"))
print(A.flatten(order="F"))
print(A.flatten(order="A"))


# In[3]:


#Q28.Write a code to create an array named X using reshape ()
X = np.array(range(24))
Y = X.reshape((3,4,2))
Y


# In[ ]:


#Random Number


# In[15]:


#Q29.Write a code to print a random number between 1 and 6 usingrandom.randint function
import random
outcome = random.randint(1,6)
print(outcome)


# In[16]:


#Q30.Write a code to print 10 random numbers between 1 and 6 usingrandom.randint function
import random
[ random.randint(1, 6) for _ in range(10) ]


# In[17]:


#Q31.Using random.randint()Random number between 1 to 7 Random number between 1 to 7 with size=1 Random number between 1 to 7 with size =10 Random number between 1 to 7 with 5 rows and 4 columns
import numpy as np	
print(np.random.randint(1, 7))
print(np.random.randint(1, 7, size=1))
print(np.random.randint(1, 7, size=10))
print(np.random.randint(1, 7, size=(5, 4)))


# In[19]:


#Q32.Create a list using string values and by using random choice function print the random string value from the list created

from random import choice
possible_destinations = ["Berlin", "Hamburg", "Munich", 
                         "Amsterdam", "London", "Paris", 
                         "Zurich", "Heidelberg", "Strasbourg", 
                         "Augsburg", "Milan", "Rome"]

print(choice(possible_destinations))


# In[20]:


#Q33.Using random.random_sample function print the random float values with 3 rows and 4 columns
import numpy as np
x = np.random.random_sample((3, 4))
print(x)


# In[ ]:


#Boolean Indexing


# In[21]:


#Q34.Given B =np.array([[42,56,89,65],[99,88,42,12],[55,42,17,18]]) print False for the values which are  greater than 42 using Boolean indexing
B = np.array([[42,56,89,65],
              [99,88,42,12],
              [55,42,17,18]])
	
print(B>=42)


# In[ ]:


#Fancy Indexing


# In[22]:


#Q35.Write a program of one array to select the corresponding index of another array A contains are of C where the corresponding value of (A<=5) is true.C= np.array([123,188,190,99,77,88,100]) and A = np.array([4,7,2,8,6,9,5])

C = np.array([123,188,190,99,77,88,100])
A = np.array([4,7,2,8,6,9,5])
R = C[A<=5]
print(R)


# In[23]:


#Q36.Extract from the array np.array([3,4,6,10,24,89,45,43,46,99,100]) with Boolean masking all the numberwhich are not divisible by 3which are divisible by 5which are divisible by 3 and 5which are divisible by 3 and set them to 42
import numpy as np
A = np.array([3,4,6,10,24,89,45,43,46,99,100])
div3 = A[A%3!=0]
print("Elements of A not divisible by 3:")
print(div3)
div5 = A[A%5==0]
print("Elements of A divisible by 5:")
print(div5)
print("Elements of A, which are divisible by 3 and 5:")
print(A[(A%3==0) & (A%5==0)])
print("------------------")
A[A%3==0] = 42
print("""New values of A after setting the elements of A, 
which are divisible by 3, to 42:""")
print(A)



# In[ ]:


#UNIT IV
#DATA MANIPULATION WITH PANDAS


# In[ ]:


#Series


# In[1]:


#Q1.   Write the program to  define a simple Series object in the following example by instantiating a Pandas Series object with a list ?
import pandas as pd
S = pd.Series([11, 28, 72, 3, 5, 8])
S


# In[2]:


#Q2.  What is the code that can directly access the index and the values of our Series S ?
print(S.index)
print(S.values) 


# In[3]:


#Q3.  If two series are taken S and S2 then write the code for addition of these two series with printing the same index ?
fruits = ['apples', 'oranges', 'cherries', 'pears']
S = pd.Series([20, 33, 52, 10], index=fruits)
S2 = pd.Series([17, 13, 31, 32], index=fruits)
print(S + S2)
print("sum of S: ", sum(S))


# In[4]:


#Q4.  What will be the program code for the above question if the indices do not have to be the same for the Series addition If an index doesn't occur in both Series, the value for this Series will be NaN ?
fruits = ['peaches', 'oranges', 'cherries', 'pears']
fruits2 = ['raspberries', 'oranges', 'cherries', 'pears']
S = pd.Series([20, 33, 52, 10], index=fruits)
S2 = pd.Series([17, 13, 31, 32], index=fruits2)
print(S + S2)


# In[5]:


#Q5. Write the code  function to extract some elements from the given series object based on the actual positions of the elements in the object ?
import pandas as pd
sr=pd.Series(['New York','Chicago','Toronto','Lisbon','Rio','Moscow'])
didx=pd.DatetimeIndex(start ='2014-08-01 10:00',freq='W',periods=6,tz='Europe/Berlin')
sr.index=didx
print(sr)


# In[6]:


#Q6. Using indexing write the code to access single values of a Series ?
print(S['apples'])


# In[7]:


#Q7. Write a code for Manipulating Pandas Data frame using Applying lambda function to a column?
import pandas as pd
values =[['Rohan', 455], ['Elvish', 250], ['Deepak', 495],
          ['Sai', 400], ['Radha', 350], ['Vansh', 450]]
df=pd.DataFrame(values, columns=['Name', 'Univ_Marks'])
df=df.assign(Percentage=lambda x: (x['Univ_Marks'] /500*100))
df


# In[8]:


#Creating Series object from Dictionary


# In[9]:


#Q8.  How to create a Series object in pandas for the resulting Series to contain the dict's keys as the indices and the values as the values?
cities = {"London":    8615246, 
          "Berlin":    3562166, 
          "Madrid":    3165235, 
          "Rome":      2874038, 
          "Paris":     2273305, 
          "Vienna":    1805681, 
          "Bucharest": 1803425, 
          "Hamburg":   1760433,
          "Budapest":  1754000,
          "Warsaw":    1740119,
          "Barcelona": 1602386,
          "Munich":    1493900,
          "Milan":     1350680}
city_series = pd.Series(cities)
print(city_series)


# In[10]:


#DataFrame


# In[11]:


#Q9. Three series are defined using pandas write the code to concantenate and show the output display ?
import pandas as pd
years = range(2014, 2018)
shop1 = pd.Series([2409.14, 2941.01, 3496.83, 3119.55], index=years)
shop2 = pd.Series([1203.45, 3441.62, 3007.83, 3619.53], index=years)
shop3 = pd.Series([3412.12, 3491.16, 3457.19, 1963.10], index=years)
pd.concat([shop1, shop2, shop3])


# In[12]:


#DataFrame from Dictionary


# In[13]:


#Q10. Give an example to derive a dataframe from a dictionary using pandas library function ?
cities = {"name": ["London", "Berlin", "Madrid", "Rome", 
                   "Paris", "Vienna", "Bucharest", "Hamburg", 
                   "Budapest", "Warsaw", "Barcelona", 
                   "Munich", "Milan"],
          "population": [8615246, 3562166, 3165235, 2874038,
                         2273305, 1805681, 1803425, 1760433,
                         1754000, 1740119, 1602386, 1493900,
                         1350680],
          "country": ["England", "Germany", "Spain", "Italy",
                      "France", "Austria", "Romania", 
                      "Germany", "Hungary", "Poland", "Spain",
                      "Germany", "Italy"]}
city_frame = pd.DataFrame(cities)
city_frame


# In[14]:


#Q11.Give an example to derive a dataframe from a dictionary using pandas function?
import pandas 
data ={'Ojaswi': {'Age': 15, 'subject': 'java', 'Address': 'Hyderabad'},
        'Rohith':  {'Age': 9, 'subject': 'python', 'Address': 'Hyderabad'},
        'Gnanesh':  {'Age': 15, 'subject': 'c/c++', 'Address': 'Guntur'},
        'divya':  {'Age': 21, 'subject': 'html', 'Address': 'ponnur'},
        'ramya':  {'Age': 15, 'subject': 'c/c++', 'Address': 'delhi'}}
data =pandas.DataFrame(data)
data


# In[15]:


#Q12.  Give the program to change both the column order and the ordering of the index with the function reindex ?
city_frame.reindex(index=[0, 2, 4, 6,  8, 10, 12, 1, 3, 5, 7, 9, 11], 
columns=['country', 'name', 'population'])


# In[16]:


#Q13. Write the code to rename a dataframe using pandas library function?
city_frame.rename(columns={"name":"Soyadı", 
                           "country":"Ülke", 
                           "population":"Nüfus"},
inplace=True)
city_frame


# In[17]:


#Q14.  Write the program for accessing row via indexing value ie select the  German cities in the following example by using 'loc' ?
city_frame = pd.DataFrame(cities, 
                          columns=("name", "population"), 
                          index=cities["country"])
print(city_frame.loc["Germany"])


# In[18]:


#Pandas Pivot Table


# In[19]:


#Q15 . Write a program to perform a pivot table format by reshaping a dataframe in pandas library function ?
import pandas as pd
d = {'A': ['kırmızı', 'yeşil', 'mavi', 'kırmızı', 'yeşil', 'mavi'],
     'B': ['bir', 'iki', 'bir',  'iki',  'bir', 'iki'],
     'C': [345, 325, 898, 989, 23, 143],
     'D': [1, 2, 3, 4, 5, 6]}
df = pd.DataFrame(d)
df


# In[20]:


#Q16.  Write a program where a Series object with an index of size nvalues. The index will not be unique, because the strings for the index are taken from the list fruits, which has less elements than nvalues ?
import pandas as pd
import numpy as np
import random
nvalues = 30
values = np.random.randint(1, 20, (nvalues,))
fruits = ["bananas", "oranges", "apples", "clementines", "cherries", "pears"]
fruits_index = np.random.choice(fruits, (nvalues,))
s = pd.Series(values, index=fruits_index)
print(s[:10])


# In[21]:


#Pandas Groupby


# In[22]:


#Q17. Write a program to get the given series in sorted label form using groupby function so that the solution is gropbyiterableform ?
grouped = s.groupby(s.index)
for fruit, s_obj in grouped:
    print(f"===== {fruit} =====")
    print(s_obj)


# In[23]:


#Q18.  The DataFrame has two columns one containing names Name and the other one coffee contains integers which are the number of cups of coffee the person drank.Write the code to sort using pandas groupby?
import pandas as pd
beverages = pd.DataFrame({'Name': ['Robert', 'Melinda', 'Brenda',
                                   'Samantha', 'Melinda', 'Robert',
                                   'Melinda', 'Brenda', 'Samantha'],
                          'Coffee': [3, 0, 2, 2, 0, 2, 0, 1, 3],
                          'Tea':    [0, 4, 2, 0, 3, 0, 3, 2, 0]})    
beverages


# In[24]:


#Q19. Write the program to calculate the average number of coffee and tea cups the persons had usinggropby function in pandas?
beverages.groupby(['Name']).mean()


# In[25]:


#Binning


# In[26]:


#Q20.  Write a program with function ‘find bin’ with two list or tuple of bins of two elements to find the index of the interval where thw vale’value’ is contained?
def find_bin(value, bins):
     for i in range(0, len(bins)):
        if bins[i][0] <= value < bins[i][1]:
            return i
        return -1
from collections import Counter
 bins = create_bins(lower_bound=50,width=4,quantity=10)
print(bins)
weights_of_persons = [73.4, 69.3, 64.9, 75.6, 74.9, 80.3, 
                      78.6, 84.1, 88.9, 90.3, 83.4, 69.3, 
                      52.4, 58.3, 67.4, 74.0, 89.3, 63.4]
binned_weights = []
for value in weights_of_persons:
bin_index = find_bin(value, bins)
print(value, bin_index, bins[bin_index])
binned_weights.append(bin_index)
    frequencies = Counter(binned_weights)
print(frequencies)


# In[27]:


#Multilevel indexing


# In[28]:


#Q21.  Write the code for multilevel indexing using Pandas data structures. It's an efficient way to store and manipulate arbitrarily high dimension data in 1-dimensional (Series) and 2-dimensional tabular (DataFrame) structures?
import pandas as pd
cities = ["Vienna", "Vienna", "Vienna",
          "Hamburg", "Hamburg", "Hamburg",
          "Berlin", "Berlin", "Berlin",
          "Zürich", "Zürich", "Zürich"]
index = [cities, ["country", "area", "population",
                  "country", "area", "population",
                  "country", "area", "population",
                  "country", "area", "population"]]
print(index)


# In[29]:


#Q22. Write the program to sort the index using slicing operation for the given series in pandas function ?
city_series = city_series.sort_index()
print("city_series with sorted index:")
print(city_series)
print("\n\n Slicing the city_series:")
city_series["Berlin":"Vienna"]


# In[30]:


#Q23.   Write a program to perform swapping multidex levels using pandas library function ?
city_series=city_series.swaplevel()
city_series.sort_index(inplace=True)
city_series


# In[31]:


#Data Visualization 


# In[32]:


#Q24 . If given a tuple data =[100, 120, 140, 180, 200, 210, 214] ,using pandas   series  function plot a line plot for the data ?
import pandas as pd
data = [100, 120, 140, 180, 200, 210, 214]
s = pd.Series(data, index=range(len(data)))
s.plot()


# In[33]:


#Q25.  For the  defineddcitionary with the population and area figures. This dictionary can be used to create the DataFrame, which we want to use for plotting a line plot ?
import pandas as pd
cities = {"name": ["London","Berlin","Madrid","Rome","Paris","Vienna","Bucharest","Hamburg","Budapest","Warsaw","Barcelona", "Munich", "Milan"],
          "population": [8615246,3562166,3165235,2874038,2273305,1805681,1803425,1760433,1754000,1740119,1602386,1493900,1350680],
          "area" : [1572,891.85,605.77,1285,105.4,414.6,228,755,525.2,517,101.9,310.4,181.8]}
city_frame = pd.DataFrame(cities,columns=["population", "area"],index=cities["name"])
print(city_frame)


# In[34]:


#Q26. Wite a program for pie chart diagram in pandas for the given series using plot function ?
import pandas as pd
fruits=['apples','pears','cherries','bananas']
series=pd.Series([20,30,40,10],
index=fruits,
name='series')
series.plot.pie(figsize=(6,6))


# In[35]:


#Date and Time


# In[36]:


#Q27.  Write a program to print the date and time using pandas date-time function?
from datetime import date
x = date(1993, 12, 14)
print(x)


# In[37]:


#Q28.  Write a program to instantiate dates in the range from January 1, 1 to December 31, 9999. This can be inquired from the attributes min and max function ?
from datetime import date
print(date.min)
print(date.max)


# In[38]:


#Q29. Write a program to show an output a dataframe of data and time ?
import pandas as pd
data =pd.date_range('1/1/2011', periods=10, freq='H')
data


# In[39]:


#Time Series


# In[40]:


#Q30. Using time series in pandas write the program to find the index consisting of time stamps?
import numpy as np
import pandas as pd
from datetime import datetime, timedelta as delta
ndays = 10
start = datetime(2017, 3, 31)
dates = [start - delta(days=x) for x in range(0, ndays)]
values = [25, 50, 15, 67, 70, 9, 28, 30, 32, 12]
ts = pd.Series(values, index=dates)
ts


# In[41]:


#Q31. Write a program to using a function to return the values lying in the given time duration?
import pandas as pd
sr=pd.Series([11, 21, 8, 18, 65, 18, 32, 10, 5, 32, None])
index_ =pd.date_range('2010-10-09 08:45', periods =11, freq='H')
sr.index=index_
print(sr)


# In[42]:


#UNIT V
#VISUALIZATION WITH MATPLOTLIB


# In[43]:


#Simple linne and scatter plot


# In[44]:


#Q1.Write a python code for simple line plot .
import matplotlib.pyplot as plt
plt.plot([-1, -4.5, 16, 23, 78, 22, 3])
plt.show()


# In[45]:


#Q2.Write a python code for scatter plot . 
import matplotlib.pyplot as plt
plt.plot([-1, -4.5, 16, 23, 78, 22, 3], "ob")
plt.show()


# In[46]:


#Q3.Write a python code for line plot and display with axis name with title. (i)x-axis named as “day”(ii)y-axis named as “Temperature in Celsius.”(iii) title named as “Temperature Graph.”
import matplotlib.pyplot as plt
days = range(1, 9)
celsius_values = [25.6, 24.1, 26.7, 28.3, 27.5, 30.5, 32.8, 33.1]
fig,ax = plt.subplots()
ax.plot(days, celsius_values)
ax.set(xlabel='Day',ylabel='Temperature in Celsius',title='Temperature Graph')


# In[48]:


#Q4.Write a python code for multiple plot(scatter,line).
import matplotlib.pyplot as plt
days = list(range(1,9))
celsius_min = [19.6, 24.1, 26.7, 28.3, 27.5, 30.5, 32.8, 33.1]
celsius_max = [24.8, 28.9, 31.3, 33.0, 34.9, 35.6, 38.4, 39.2]
fig, ax = plt.subplots()
ax.set(xlabel='Day',ylabel='Temperature in Celsius',title='Temperature Graph')
ax.plot(days, celsius_min,days, celsius_min, "oy",days, celsius_max,days, celsius_max, "or")


# In[49]:


#Bar chart 


# In[50]:


#Q5.Write a python code to display the bar plot.
import matplotlib.pyplot as plt
import numpy as np
years = [str(year) for year in range(2010, 2021)]
visitors = (1241, 50927, 162242, 222093, 
            665004, 2071987, 2460407, 3799215, 
            5399000, 5474016, 6003672)
plt.bar(years, visitors, color="green")
plt.xlabel("Years")
plt.ylabel("Values")
plt.title("Bar Chart ")
plt.plot()	
plt.show()


# In[51]:


#Q6.Write a python to find the GDP  using bar plot to display the output .
import matplotlib.pyplot as plt
import numpy as np
land_GDP_per_capita = []
with open('../data/GDP.txt') as fh:
    for line in fh:
        index, *land, gdp, population = line.split()
        land = " ".join(land)
gdp = int(gdp.replace(',', ''))
population = int(population.replace(',', ''))
per_capita = int(round(gdp * 1000000 / population, 0))
land_GDP_per_capita.append((land, per_capita))
land_GDP_per_capita.sort(key=lambda x: x[1], reverse=True)
countries, GDP_per_capita = zip(*land_GDP_per_capita)
fig = plt.figure(figsize=(6,5), dpi=200)
left, bottom, width, height = 0.1, 0.3, 0.8, 0.6
ax = fig.add_axes([left, bottom, width, height]) 
ax.bar(x=np.arange(len(GDP_per_capita)),height=GDP_per_capita,color="green", align="center",tick_label=countries)
ax.set_ylabel('in thousands of $')
ax.set_title('Largest Economies by nominal GDP in 2018')
plt.xticks(rotation=90)
plt.show()


# In[52]:


#Q7.write a python code to display the bar plot in vertical order .
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
# restore default parameters:
plt.rcdefaults() 
fig, ax = plt.subplots()
personen = ('Michael', 'Dorothea', 'Robert', 'Bea', 'Uli')
y_pos = np.arange(len(personen))
cups = (15, 22, 24, 39, 12)
ax.barh(y_pos, cups, align='center',
color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(personen)
ax.invert_yaxis()  
ax.set_xlabel('Cups')
ax.set_title('Coffee Consumption')
plt.show()


# In[53]:


#Q8.Write a python code for grouped  bar charts .
import matplotlib.pyplot as plt
import numpy as np
last_week_cups = (20, 35, 30, 35, 27)
this_week_cups = (25, 32, 34, 20, 25)
names = ['Mary', 'Paul', 'Billy', 'Franka', 'Stephan']
fig = plt.figure(figsize=(6,5), dpi=200)
left, bottom, width, height = 0.1, 0.3, 0.8, 0.6
ax = fig.add_axes([left, bottom, width, height]) 
width = 0.35   
ticks = np.arange(len(names))    
ax.bar(ticks, last_week_cups, width, label='Last week')
ax.bar(ticks + width, this_week_cups, width, align="center",label='This week')
ax.set_ylabel('Cups of Coffee')
ax.set_title('Coffee Consummation')
ax.set_xticks(ticks + width/2)
ax.set_xticklabels(names)
ax.legend(loc='best')
plt.show()


# In[54]:


#Q9.Write a python code to display the stacked bar  chart .
import matplotlib.pyplot as plt
import numpy as np
coffee = np.array([5, 5, 7, 6, 7])
tea = np.array([1, 2, 0, 2, 0])
water = np.array([10, 12, 14, 12, 15])
names = ['Mary', 'Paul', 'Billy', 'Franka', 'Stephan']
fig = plt.figure(figsize=(6,5), dpi=200)
left, bottom, width, height = 0.2, 0.1, 0.7, 0.8
ax = fig.add_axes([left, bottom, width, height]) 
width = 0.35   
ticks = np.arange(len(names))    
ax.bar(ticks, tea, width, label='Coffee', bottom=water+coffee)
ax.bar(ticks, coffee, width, align="center", label='Tea',bottom=water)
ax.bar(ticks, water, width, align="center", label='Water')


# In[55]:


#Histogram:


# In[56]:


#Q10.Write a python code  for histogram . 
import matplotlib.pyplot as plt
import numpy as np
gaussian_numbers = np.random.normal(size=10000)
gaussian_numbers
plt.hist(gaussian_numbers, bins=20)
plt.title("Gaussian Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


# In[57]:


#Q11.Write a python code to display the histogram using binnings .
n,bins,patches=plt.hist(gaussian_numbers)
print("n:",n,sum(n))
print("bins:",bins)
for i in range(len(bins)-1):
    print(bins[i+1]-bins[i])
print("patches:",patches)
print(patches[1])
print(patches[2])


# In[58]:


#Q12.Write a python code to display histogram to increase the no.of.binnings
plt.hist(gaussian_numbers, bins=100)
plt.show()


# In[59]:


#Q13.Write a python code to display the histogram in horizontal orientation.
plt.hist(gaussian_numbers, 
         bins=100, 
         orientation="horizontal")
plt.show()


# In[60]:


#Q14.Write a python code to display the histogram with edge color.
plt.hist(gaussian_numbers, 
         bins=100, 
         density=True, 
         stacked=True, 
edgecolor="#6A9662",
color="#DDFFDD")
plt.show()


# In[61]:


#Q15.Write a python code to display the histogram in cumulative using bins .
plt.hist(gaussian_numbers, 
         bins=100, 
         stacked=True,
         cumulative=True)
plt.show()


# In[62]:


#Adding markers in plots :


# In[63]:


#Q16.Write a python code to display scatter plot using  markers.  
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 11)
y1 = np.random.randint(2, 7, (11,))
y2 = np.random.randint(9, 14, (11,))
y3 = np.random.randint(15, 25, (11,))
plt.scatter(x, y1)
plt.scatter(x, y2, marker='v', color='r')
plt.scatter(x, y3, marker='^', color='m')
plt.title('Scatter Plot Example')
plt.show()


# In[64]:


#Contour plot :


# In[65]:


#Q17.Write  a python code for contour plot without charts  .
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
n, m = 7, 7
start = -3
x_vals = np.arange(start, start+n, 1)
y_vals = np.arange(start, start+m, 1)
X, Y = np.meshgrid(x_vals, y_vals)
print(X)
print(Y)


# In[66]:


#Q18.Write a  python code to display the scatter diagram using contour plot.(i)	X axis named as “x”(ii)	Y-axis named as “y”(iii)	Titled named as “Scatter  using contour plot”
fig, ax = plt.subplots()
ax.scatter(X, Y, color="green")
ax.set_title('Scatter  using contour plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


# In[67]:


#Q19.Write a python code to display contour plot using clabel method.
fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 
Z = np.sqrt(X**2 + Y**2)
cp = ax.contour(X, Y, Z)
ax.clabel(cp, inline=True, 
fontsize=10)
ax.set_title('Contour Plot Using  clabel  method')
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
plt.show()


# In[68]:


#Q20.Write a python code  for changing the linestyle= “dashed line” and color=”black” in contour plot.
import matplotlib.pyplot as plt
plt.figure()
cp = plt.contour(X, Y, Z, colors='black', linestyles='dashed')
plt.clabel(cp, inline=True, 
fontsize=10)	
plt.title('Contour Plot')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.show()


# In[69]:


#Q21.Write a python code for filled color using contour plot .
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 
start, stop, n_values = -8, 8, 800	
x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(start, stop, n_values)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.sqrt(X**2 + Y**2)
cp = plt.contourf(X, Y, Z)
plt.colorbar(cp)
ax.set_title('Contour Plot')
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
plt.show()


# In[70]:


#Q22.Write a python code for to displaty individual color using contour plot.
import numpy as np
import matplotlib.pyplot as plt
xlist = np.linspace(-3.0, 3.0, 100)
ylist = np.linspace(-3.0, 3.0, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = np.sqrt(X**2 + Y**2)
plt.figure()
contour = plt.contour(X, Y, Z)
plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
c = ('#ff0000', '#ffff00', '#0000FF', '0.6', 'c', 'm')
contour_filled = plt.contourf(X, Y, Z, colors=c)
plt.colorbar(contour_filled)
plt.title('Filled Contours Plot')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.savefig('contourplot_own_colours.png', dpi=300)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





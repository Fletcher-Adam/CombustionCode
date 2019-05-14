#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import csv
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


from openpyxl import load_workbook
def get_xcel_col(sheet, col, row_start, row_end):
    vals = []
    
    for i in range(row_start, row_end):
        vals.append(sheet[col+str(i)].value)
    return vals
R = 8.314
wb = load_workbook(r'C:\Users\DAEM-Modified.xlsx')
s1 = wb['Darrow\'s blueberry_dead']
t = get_xcel_col(s1, 'A', 11, 57)
T_c = get_xcel_col(s1, 'B', 11, 57)
T = T_c + np.array(273.15)
m_e = get_xcel_col(s1, 'D', 11, 57)
c = m_e / np.array(s1['D11'].value)
V_e = 1 - c
def r_e(V_e, t):
    dV_e = []
    dt1 = []
    dm_e = []
    for i in range(1, len(V_e)):
        dV_e.append(V_e[i] - V_e[i-1])
        dm_e.append(m_e[i]-m_e[i-1])
    for i in range(1, len(t)):
        dt1.append(t[i] - t[i-1])
    dV_e = np.array(dV_e)
    dm_e = np.array(dm_e)
    dt1 = np.array(dt1)
    r_e = dV_e / dt1
    r_em = dm_e / dt1
    return(r_e, r_em)

A1 = 7282.63
A2 = 9182476
E1 = 70372.5
E2 = 1515186
Y1 = .63007
Y2 = .98458

def r_m(A1,A2,E1,E2,Y1,Y2):
    V_m = np.zeros(len(t)-1)
    r_m = np.zeros(len(t)-1)
    dt2 = np.zeros(len(t)-1)
    dV_o = np.zeros(len(t)-1)
    dm_o = np.zeros(len(t)-1)
    for i in range(1, len(t)-1):
        dt2[i] = t[i+1] - t[i] 
        
    dm_o[0] = 1
    dV_o[0] = 0
    V_m[0] = -(A1*np.exp(-E1/R/T[0])+A2*np.exp(-E2/R/T[0]))*dm_o[0]
    r_m[0] = (Y1*A1*np.exp(-E1/R/T[0]))*dm_o[0]

    for i in range(1, len(t)-1):
        dm_o[i] = dm_o[i-1]+r_m[i-1]*dt2[i]
        V_m[i] = -(A1*np.exp(-E1/R/T[i])+A2*np.exp(-E2/R/T[i]))*dm_o[i]
        while(T[i]<230):
            r_m[0] = (Y1*A1*np.exp(-E1/R/T[0])*dm_o[0]
        while(T[i]>230):               
            r_m[0] = (Y2*A2*np.exp(-E2/R/T[0])*dm_o[0]
        dV_o[i] = dV_o[i-1]+V_m[i-1]*dt2[i]
    return(r_m, V_m, dV_o, dm_o)
def f(X):
    A1,A2,E1,E2,Y1,Y2 = X
    e1 = (r_e(V_e, t)[0] - r_m(A1,A2,E1,E2,Y1,Y2)[1])**2
    e2 = (V_e[1:46] - r_m(A1,A2,E1,E2,Y1,Y2)[2])**2
    return(sum(e1) + sum(e2))

res = minimize(f,(A1,A2,E1,E2,Y1,Y2))
A1,A2,E1,E2,Y1,Y2 = res.x
print(A1)
print(A2)
print(E1)
print(E2)
print(Y1)
print(Y2)


e1 = (V_e[0:-1] - r_m(A1,A2,E1,E2,Y1,Y2)[1])**2
e2 = (r_e(V_e, t)[0] - r_m(A1,A2,E1,E2,Y1,Y2)[0])**2


# In[4]:


plt.plot(T_c[0:45],r_e(V_e,t)[0],linestyle = ':',color = 'Green',label='exp1')
plt.xlabel('Temperature (C)')
plt.ylabel('DIG (DV/Dt)')
plt.plot(T_c[0:45],r_m(A1,A2,E1,E2,Y1,Y2)[0],color = 'Green',label='mod1')
plt.xlabel('Temperature (C)')
plt.ylabel('DIG (DV/Dt)')
plt.legend()
plt.grid()


# In[4]:


plt.plot(T_c,V_e,linestyle = ':',color = 'Green',label='exp1')
plt.xlabel('Temperature (C)')
plt.ylabel('TG (V)')
plt.plot(T_c[0:45],r_m(t,A1,A2,E1,E2,Y1,Y2)[1],'g-',label='mod1')
plt.xlabel('Temperature (C)')
plt.ylabel('TG (V)')
plt.legend()
plt.grid()


# In[ ]:





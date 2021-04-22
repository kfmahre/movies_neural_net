# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 01:37:59 2019

@author: kfmah
"""

stuff = list()

stuff.append('python')

stuff.append('chuck')

stuff.sort()

print (stuff[0])

print (stuff.__getitem__(0))

print (list.__getitem__(stuff,0))
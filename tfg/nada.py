import pandas as pd
import numpy as np
from functools import reduce

a = pd.DataFrame([1, 2, 3, 4, 5, 6])
b = pd.DataFrame([11, 12, 23, 34, 8, 7])

lista = [a, b]
l = reduce(lambda a, b: a.add(b, fill_value=0), lista)
print(l)
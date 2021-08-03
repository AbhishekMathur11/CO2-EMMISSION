#The following program has been designed to interpret the CO2 emission(grams per  mile) based on the fuel economy of various car brands
#These lists have existing reported data from https://www.eesi.org/papers/view/fact-sheet-vehicle-efficiency-and-emissions-standards

data={
"mpg_list":[30.1,31.1,32.2,33.8,35.5,36.6,38.3,40,41.7,44.7,46.8,49.4,52,54.5],
"co2_emission_list":[295,286,276,263,250,243,232,222,213,199,190,180,171,163]
}


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


x=data["mpg_list"]
y=data["co2_emission_list"]

mymodel = np.poly1d(np.polyfit(x, y, 3))

myline = np.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()

print("The above plot is the approximate reperesentation of the concerned values. Let us proceed towards finding")

y1=float(input("Enter the miles per gallon for your respective vehicle:"))
emission_out=mymodel(y1)

print("According to this data, your car emits",emission_out,"grams of CO2 per mile.")

print("Here is the data used for this calculation")

df=pd.DataFrame(data)
print(df)

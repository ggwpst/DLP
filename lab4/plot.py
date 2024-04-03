import pandas as pd
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(10,6))
Cyclical = pd.read_csv('Cyclical.csv')
Monotonic = pd.read_csv('Monotonic.csv')
Non = pd.read_csv('None.csv')
plt.plot(Cyclical['epoch'], Cyclical['loss'], label="Cyclical")
plt.plot(Monotonic['epoch'], Monotonic['loss'], label="Monotonic")
plt.plot(Non['epoch'], Non['loss'], label="None")
plt.title("KL Annealing Stragety Loss")
plt.legend(loc='best')
plt.xlabel("epoch")
plt.ylabel("Loss")
fig.savefig('KL Annealing Stragety Loss.png')
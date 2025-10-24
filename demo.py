import numpy as np
import matplotlib.pyplot as plt
from model import window 

event_w = window(event=True, length=200)
idle_w  = window(event=False, length=200)
t = np.linspace(0, 1, event_w.shape[0])

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Synthetic event (x, y, z)")
plt.plot(t, event_w)
plt.subplot(1,2,2)
plt.title("Synthetic idle (x, y, z)")
plt.plot(t, idle_w)
plt.tight_layout()
plt.show()

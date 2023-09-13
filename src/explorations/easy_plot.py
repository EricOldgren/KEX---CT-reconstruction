import matplotlib.pyplot as plt
import numpy as np


x = np.arange(10)
y = x * 10

print(x)
print(y)

plt.plot(x, y)
# plt.suptitle("testing_plot")
for i in plt.get_fignums():
    fig = plt.figure(i)
    title = fig._suptitle.get_text() if fig._suptitle is not None else f"fig{i}"
    plt.savefig(f"{title}.png")
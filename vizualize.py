import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

def main():
    plt.style.use("fivethirtyeight")
    rcParams['font.family'] = 'sans-serif'
    rcParams['figure.figsize'] = 20, 10
    rcParams['font.size'] = 13
    plt.ylabel("Time in sec")
    plt.xlabel("Number of threads")
    plt.suptitle(f"Parallel indexation ({title})")
    threads, time = data
    x = np.linspace(min(threads), max(threads), 100)
    y = max(time) / x
    plt.xticks(np.linspace(min(threads), max(threads), max(threads)))
    plt.yticks(np.linspace(0, max(time), len(threads)))
    plt.plot(x, y, ":", label=r'$Ideally\;y\;=\;\frac{c}{x}\;$', color='red')
    plt.plot(threads, time, "--", marker='o', markersize=10, markerfacecolor="blue", label=r'$Actual$', color='green')
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig('statistics.png', dpi=100)
    plt.show()
if __name__ == '__main__':
    main()
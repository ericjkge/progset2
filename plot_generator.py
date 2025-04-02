import matplotlib.pyplot as plt

def loader(filepath):
    sizes = []
    times = []
    with open(filepath, "r") as f:
        for line in f:
            size, t = line.strip().split()
            sizes.append(int(size))
            times.append(float(t))
    return sizes, times

def plotter(naive_file, strassen_file, title="Runtime Comparison"):
    # Load data
    naive_sizes, naive_times = loader(naive_file)
    strassen_sizes, strassen_times = loader(strassen_file)

    # Plot
    plt.plot(naive_sizes, naive_times, label='Naive')
    plt.plot(strassen_sizes, strassen_times, label='Strassen')
    plt.xlabel("Matrix Size (n x n)")
    plt.ylabel("Time (seconds)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Crossover point
    for size, naive_time, strassen_time in zip(naive_sizes, naive_times, strassen_times):
        if strassen_time < naive_time:
            print(f"Crossover point: n = {size}")
            return size

    print("No crossover point found.")
    return None

if __name__ == "__main__":
    plotter("naive_t1.txt", "strassen_t1.txt", title="Naive vs Strassen Runtime (t1 input)")
    plotter("naive_t2.txt", "strassen_t2.txt", title="Naive vs Strassen Runtime (t2 input)")
    plotter("naive_t3.txt", "strassen_t3.txt", title="Naive vs Strassen Runtime (t3 input)")
    plotter("naive_t4.txt", "strassen_t4.txt", title="Naive vs Strassen Runtime (t4 input)")

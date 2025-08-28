import os
import re
import matplotlib.pyplot as plt
import numpy as np

newBenchmarkNames = [ 'bay', 'cal', 'col', 'ctr', 'east', 'fla', 'lks', 'ne', 'nw', 'ny', 'usa',
                     'west' ]
def plotter():
    dict = {
        'cpu_weights': [],
        'cpu_runtime': [],
        'GPU_weights': [],
        'GPU_runtime': [],
        'GPUthst_weights': [],
        'GPUthst_runtime': [],
    }

    for file in sorted(os.listdir('runtime/')):
        print(f'Processing file {file}')
        src = open(f'runtime/{file}', 'r')
        weight = int(src.readline())
        time = float(src.readline())

        x = re.search('\_(\w+)\_', file)
        dict[f'{x.group(1)}_weights'].append(weight)
        dict[f'{x.group(1)}_runtime'].append(time)

    for i in range(len(newBenchmarkNames)):
        if dict['cpu_weights'][i] != dict['GPU_weights'][i] or dict['cpu_weights'][i] != dict['GPUthst_weights'][i]:
            print('There are some differences in the weights')

    # Parameters for bar width and positions
    bar_width = 0.25
    x = np.arange(len(newBenchmarkNames))

    # Create figure and set log scale for y-axis
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_yscale("log")

    # Plot each data series with a different color and hatch pattern
    i = 0
    for _, (label, values) in enumerate(dict.items()):
        print(values)
        print(label)
        if label == 'cpu_runtime':
            ax.bar(x + i * bar_width, values, width=bar_width, label=label, edgecolor='black')
            i += 1
        elif label == 'GPU_runtime':
            ax.bar(x + i * bar_width, values, width=bar_width, label=label, edgecolor='black')
            i += 1
        elif label == 'GPUthst_runtime':
            ax.bar(x + i * bar_width, values, width=bar_width, label=label, edgecolor='black')
            i += 1


    # Set labels, title, and legend
    ax.set_xlabel("Graph", fontsize=12)
    ax.set_ylabel("Execution Time (s)", fontsize=12)
    ax.set_xticks(x + bar_width)  # Center x-ticks
    ax.set_xticklabels(newBenchmarkNames, rotation=45)
    ax.legend(loc="upper left", fontsize=10, title="Legend", title_fontsize='13')

    # Show grid and plot
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    print(dict)








plotter()

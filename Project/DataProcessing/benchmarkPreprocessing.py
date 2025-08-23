import os
import re

def preprocessBenchmark() :
    benchmarkFolder = 'benches/'
    newBenchmarkNames = ['ny', 'bay', 'col', 'fla', 'nw', 'ne', 'cal', 'lks', 'east', 'west', 'ctr',
                         'usa']
    i = 0
    for file in os.listdir(benchmarkFolder):
        src = open(f'{benchmarkFolder}{file}', 'r')
        try:
            dst = open(f'{benchmarkFolder}{newBenchmarkNames[i]}.txt', 'x')
            dst.close()
        except:
            print('The file has already been created')
        dst = open(f'{benchmarkFolder}{newBenchmarkNames[i]}.txt', 'w')

        # Consume 4 useless lines
        for _ in range(4):
            line = src.readline();

        # Read the size of the graph
        x = re.search("(\d+\ \d+)", src.readline())
        dst.write(f'{x.group(1)}\n')

        # Consuming two lines which are useless
        src.readline()
        src.readline()

        # Copy the contents of the file in the new one
        line = src.readline()
        while (line != ''):
            x = re.search('(\d+\ \d+\ \d+)', line)
            dst.write(f'{x.group(1)}\n')
            line = src.readline()

        # Close the files
        dst.close()
        src.close()

        i += 1



preprocessBenchmark()

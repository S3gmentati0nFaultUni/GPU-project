import os
import re

def preprocessBenchmark() :
    benchmarkFolder = 'benches/'
    newBenchmarkNames = [ 'bay', 'cal', 'col', 'ctr', 'east', 'fla', 'lks', 'ne', 'nw', 'ny', 'usa',
                     'west' ]
    i = 0
    for file in sorted(os.listdir(f'{benchmarkFolder}raws/')):
        print(f'Preprocessing file {file} writing to {benchmarkFolder}{newBenchmarkNames[i]}.txt')
        src = open(f'{benchmarkFolder}raws/{file}', 'r')
        try:
            dst = open(f'{benchmarkFolder}{newBenchmarkNames[i]}.txt', 'x')
            dst.close()
        except:
            pass
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

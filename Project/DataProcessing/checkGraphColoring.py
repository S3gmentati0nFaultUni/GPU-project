def check_coloring():
    path = "/content/testing/"

    graph_file = open(path + "graph.txt", "r")
    coloring_file = open(path + "coloring.txt", "r")
    length = 0
    i = 0
    adj_lists = [];

    # Build matrix from file
    for line in graph_file:
        if i == 0:
            length = int(line)
        else:
            prep_line = file_preprocessing(line)
            adj_lists.append(prep_line)
        i = i + 1

    if length < 20:
        print(adj_lists)
        print("\n\n\n")


    # Build coloring from file
    i = 0
    col_number = 0
    coloring = []
    for line in coloring_file:
        if i == 0:
            col_number = int(line)
        else:
            coloring = file_preprocessing(line)
        i = i + 1
    if col_number < 20:
        print(coloring);


    # Check the coloring
    for i in range(0, length):
        current_color = coloring[i]
        neighbours = adj_lists[i]
        for j in range(0, len(neighbours)):
            if current_color == coloring[neighbours[j]]:
                print(f"Il nodo {j} e il nodo {i} hanno lo stesso colore e sono vicini\n")


    graph_file.close()
    coloring_file.close()


def file_preprocessing(line):
    split_line = line.split(",")
    split_line.remove("\n")
    for j in range(0, len(split_line)):
        split_line[j] = int(split_line[j])
    return split_line


check_coloring()
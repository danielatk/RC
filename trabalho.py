import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import dc_stat_think as dcst

def clean_file(filename):
    fin = open(filename, "r+", encoding = "ISO-8859-1")
    for line in fin:
        fin.write(line.replace(' +++$+++ ', ' # '))
    fin.close()

#clean_file('./datasets/cornell movie-dialogs corpus/movie_characters_metadata.csv')
#clean_file('./datasets/cornell movie-dialogs corpus/movie_conversations.csv')
#clean_file('./datasets/cornell movie-dialogs corpus/movie_lines.csv')
#clean_file('./datasets/cornell movie-dialogs corpus/movie_titles_metadata.csv')

df_characters = pd.read_csv('./datasets/cornell movie-dialogs corpus/movie_characters_metadata.csv', encoding = "ISO-8859-1", sep='\ \+\+\+\$\+\+\+\ ')
df_conversations = pd.read_csv('./datasets/cornell movie-dialogs corpus/movie_conversations.csv', encoding = "ISO-8859-1", sep='\ \+\+\+\$\+\+\+\ ')
df_lines = pd.read_csv('./datasets/cornell movie-dialogs corpus/movie_lines.csv', encoding = "ISO-8859-1", sep='\ \+\+\+\$\+\+\+\ ')
df_titles = pd.read_csv('./datasets/cornell movie-dialogs corpus/movie_titles_metadata.csv', encoding = "ISO-8859-1", sep='\ \+\+\+\$\+\+\+\ ')

columns_characters = ['character_id', 'character_name', 'movie_id', 'title', 'gender', 'position']
columns_lines = ['line_id', 'character_id', 'movie_id', 'character_name', 'text']
columns_conversations = ['character_id_1', 'character_id_2', 'movie_id', 'list']
columns_titles = ['movie_id', 'title', 'year', 'rating', 'n_votes', 'genres']

df_characters.columns = columns_characters
df_conversations.columns = columns_conversations
df_lines.columns = columns_lines
df_titles.columns = columns_titles

print(df_lines.head())

df_conversations.drop(['character_id_1', 'character_id_2'], axis='columns', inplace=True)

def select_conversations(movie_id):
    return df_conversations[df_conversations['movie_id'] == movie_id]

def select_characters(movie_id):
    return df_characters[df_characters['movie_id'] == movie_id]

def select_lines(movie_id):
    return df_lines[df_lines['movie_id'] == movie_id]

movies = ['m448', 'm301', 'm379','m445']
movie_names = []
for movie in movies:
    movie_names.append(df_titles[df_titles['movie_id'] == movie]['title'].tolist()[0])
colors = ['b','g', 'y', 'm']

graphs = []
pageranks = []
components = []
size_components = []
diameters = []
distances = []
degrees = []
weights = []

x_ecdf_pageranks = []
y_ecdf_pageranks = []
x_ecdf_distances = []
y_ecdf_distances = []
x_ecdf_degrees = []
y_ecdf_degrees = []
x_ecdf_weights = []
y_ecdf_weights = []

def ecdf(x):
    x = np.sort(x)
    def result(v):
        return np.searchsorted(x, v, side='right') / x.size
    return result

'''def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)'''

def make_label_dict(items, keys):
    label_dict = {}
    for i in range(len(keys)):
        label_dict[keys[i]] = items[i]
    return label_dict

def calc_distances(G):
    distances = np.zeros(shape=(len(G.nodes),len(G.nodes)))
    i = 0
    for source in G.nodes:
        j = 0
        for target in G.nodes:
            if nx.bidirectional_dijkstra(G,source,target):
                distances[i,j] = nx.shortest_path_length(G, source, target)
            else:
                distances[i,j] = -1
            j += 1
        i += 1
    return distances

def calc_conn_comp(G):
    conn_comps = []
    for node in G.nodes:
        conn_comp = nx.node_connected_component(G, node)
        if conn_comp not in conn_comps:
            conn_comps.append(conn_comp)
    return conn_comps

def calc_sizes_conn_comp(conn_comps):
    sizes = []
    for conn_comp in conn_comps:
        sizes.append(len(conn_comp))
    return sizes

for movie in movies:
    df_conv_local = select_conversations(movie)
    df_char_local = select_characters(movie)
    df_lines_local = select_lines(movie)
    conversations = df_conv_local['list'].tolist()
    character_ids = df_char_local['character_id'].tolist()
    character_names = df_char_local['character_name'].tolist()
    n = len(character_ids)
    A = np.zeros(shape=(n,n), dtype=float)
    for conversation in conversations:
        chars_local = []
        conversation = conversation.strip('][').replace('\'','').split(', ')
        for line in conversation:
            char = df_lines_local[df_lines_local['line_id'] == line]['character_id'].tolist()[0]
            if char not in chars_local:
                chars_local.append(char)
        indexes = []
        for char in chars_local:
            indexes.append(character_ids.index(char))
        for i in range(len(indexes)):
            for j in range(len(indexes)):
                if i != j:
                    A[indexes[i],indexes[j]] += 1

    print("Adjascency matrix: ", A)

    print("Character names :", character_names)

    G = nx.from_numpy_matrix(A)

    print("edges: ", G.edges)
    print("nodes: ", G.nodes)

    label_dict = make_label_dict(character_names, range(len(character_names)))

    weight = []

    for edge in G.edges():
        print(G.edges[edge]['weight'])
        weight.append(G.edges[edge]['weight'])

    weights.append(weight)

    max_weight = np.max(weight)
    min_weight = np.min(weight)
    mean_weight = np.mean(weight)
    median_weight = np.median(weight)
    stdev_weight = np.std(weight)

    print("Max weight:", max_weight)
    print("Min weight:", min_weight)
    print("Mean weight:", mean_weight)
    print("Standard Deviation of weights:", stdev_weight)
    print("Median weight:", median_weight)

    pagerank = nx.pagerank_numpy(G, alpha=0.85, weight='weight')

    pageranks.append(pagerank)

    diameter = []
    n_connected_components = nx.number_connected_components(G)
    if n_connected_components == 1:
        diameter.append(nx.diameter(G))
    else:
        conn_comps = calc_conn_comp(G)
        for conn_comp in conn_comps:
            diameter.append(nx.diameter(G.subgraph(conn_comp)))

    diameters.append(diameter)

    D = []
    if n_connected_components == 1:
        D.append(calc_distances(G))
        print("Distance matrix:", D)
        max_distance = np.max(D)
        min_distance = np.min(D)
        mean_distance = np.mean(D)
        median_distance = np.median(D)
        stdev_distance = np.std(D)

        print("Max distance:", max_distance)
        print("Min distance:", min_distance)
        print("Mean distance:", mean_distance)
        print("Standard Deviation of distances:", stdev_distance)
        print("Median distance:", median_distance)

    else:
        for i in range(len(conn_comps)):
            D.append(calc_distances(G.subgraph(conn_comps[i])))
            print("Distance matrix for component {}:".format(str(i)), D[i])
            max_distance = np.max(D[i])
            min_distance = np.min(D[i])
            mean_distance = np.mean(D[i])
            median_distance = np.median(D[i])
            stdev_distance = np.std(D[i])

            print("Max distance for component {}:".format(str(i)), max_distance)
            print("Min distance for component {}:".format(str(i)), min_distance)
            print("Mean distance for component {}:".format(str(i)), mean_distance)
            print("Standard Deviation of distances for component {}:".format(str(i)), stdev_distance)
            print("Median distance for component {}:".format(str(i)), median_distance)

    distances.append(D)

    if n_connected_components == 1:
        sizes = [len(G)]
    else:
        sizes = calc_sizes_conn_comp(conn_comps)

    size_components.append(sizes)

    print("Size of connected components:", sizes)
    max_size = np.max(sizes)
    min_size = np.min(sizes)
    mean_size = np.mean(sizes)
    median_size = np.median(sizes)
    stdev_size = np.std(sizes)

    print("Max size:", max_size)
    print("Min size:", min_size)
    print("Mean size:", mean_size)
    print("Standard Deviation of sizes:", stdev_size)
    print("Median size:", median_size)

    degree = G.degree

    degrees_list = []
    for i in range(len(degree)):
        degrees_list.append(degree[i])

    max_degree = np.max(degrees_list)
    min_degree = np.min(degrees_list)
    mean_degree = np.mean(degrees_list)
    stdev_degree = np.std(degrees_list)
    median_degree = np.median(degrees_list)
    
    print("Max degrees:", max_degree)
    print("Min degrees:", min_degree)
    print("Mean degrees:", mean_degree)
    print("Standard Deviation of degrees:", stdev_degree)
    print("Median degrees:", median_degree)

    pageranks_list = []
    for rank in pagerank:
        pageranks_list.append(pagerank[rank])

    max_pagerank = np.max(pageranks_list)
    min_pagerank = np.min(pageranks_list)
    mean_pagerank = np.mean(pageranks_list)
    stdev_pagerank = np.std(pageranks_list)
    median_pagerank = np.median(pageranks_list)
    
    print("Max pageranks:", max_pagerank)
    print("Min pageranks:", min_pagerank)
    print("Mean pageranks:", mean_pagerank)
    print("Standard Deviation of pageranks:", stdev_pagerank)
    print("Median pageranks:", median_pagerank)

    print("Pagerank:", pagerank)
    print("Degrees:", G.degree)
    print("Diameter:", diameter)
    print("Number of connected components:", n_connected_components)
   
    x, y = dcst.ecdf(np.array(degrees_list))
    x_ecdf_degrees.append(x)
    y_ecdf_degrees.append(y)

    x, y = dcst.ecdf(np.array(weight))
    x_ecdf_weights.append(x)
    y_ecdf_weights.append(y)

    x, y = dcst.ecdf(np.array(pageranks_list))
    x_ecdf_pageranks.append(x)
    y_ecdf_pageranks.append(y)

    graphs.append(G)
    pageranks.append(pagerank)
    distances.append(D)
    diameters.append(diameter)
    components.append(n_connected_components)
    size_components.append(sizes)

    nx.draw(G, node_size=250, labels=label_dict, with_labels=True, font_color='m', width=weight)
    #nx.draw(G, node_size=500, labels=label_dict, with_labels=True)
    plt.show()

for i in range(len(x_ecdf_weights)):
    plt.plot(x_ecdf_weights[i], y_ecdf_weights[i]*100, linestyle='--', lw = 2, color=colors[i], label=movie_names[i])
plt.xlabel("edge weight")
plt.ylabel("ECDF")
#plt.title("Empirical CDF for edge weights")
plt.legend()
plt.show()

for i in range(len(x_ecdf_weights)):
    plt.plot(x_ecdf_degrees[i], y_ecdf_degrees[i]*100, linestyle='--', lw = 2, color=colors[i], label=movie_names[i])
plt.xlabel("vertex degree")
plt.ylabel("ECDF")
#plt.title("Empirical CDF for vertex degree")
plt.legend()
plt.show()

for i in range(len(x_ecdf_weights)):
    plt.plot(x_ecdf_pageranks[i], y_ecdf_pageranks[i]*100, linestyle='--', lw = 2, color=colors[i], label=movie_names[i])
plt.xlabel("vertex pagerank")
plt.ylabel("ECDF")
#plt.title("Empirical CDF for vertex pageranks")
plt.legend()
plt.show()


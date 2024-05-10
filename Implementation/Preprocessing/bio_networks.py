import networkx as nx
import mygene
import pandas as pd
import os


def recursive_neighbour_gather(node, ppi, R, result=[], looked_nodes=[]):
    '''
    Function that finds the neighbors of a node, and starts looking at the neighbors of each of those neighbors,
    recursively.
    :param node: entrez ID of a gene
    :param ppi: graph that connects nodes
    :param R: neighborhood radius
    :param result: recursive parameter to store all looked nodes
    :param looked_nodes: recursive parameter to know which nodes we've already looked at
    :return: result, a list of all nodes in the neighborhood
    '''
    if R == 1:
        return result
    else:
        N = int(node)
        if (ppi.has_node(N)):
            neighbors = list(ppi.neighbors(N))
            looked_nodes += [N]
            if len(neighbors) > 0:
                result.extend([item for item in neighbors])
                result = list(set(result))
            for n in neighbors:
                if (int(n) not in looked_nodes) and (N != int(n)):
                    return recursive_neighbour_gather(n, ppi, R - 1, result, looked_nodes)
        return result


def get_snap(expression_DF, genes, remove_components, RADIUS):
    '''
    Function that creates snap network
    :param expression_DF: dataframe with the expression of the genes, used in order to know which are the
    genes that we need to really consider at the end, in the graph.
    :param genes: list of relevant genes that we obtained from disgenet
    :param remove_components: boolean, whether to remove components of small size
    :param RADIUS: radius through which to look at the different nodes
    :return: graph
    '''

    # We remove first line in csv with column names
    file_address = '../../Data/PPT-Ohmnet/PPT-Ohmnet_tissues-combined.edgelist'
    with open(file_address, 'r') as f:
        next(f)
        tissues_edgelist = pd.read_csv(file_address, sep='\t')

    # We prepare target to save the tissues in PPT Ohmnet that are ONLY specific to kidney
    current_directory = os.getcwd()
    kidney_specific_target = os.path.abspath(
        os.path.join(current_directory, '..', '..', 'Data', 'PPT-Ohmnet', 'PPT-Ohmnet-tissues-kidney.csv'))

    kidney_specific = tissues_edgelist[tissues_edgelist['tissue'] == 'kidney']
    kidney_specific.to_csv(kidney_specific_target, sep='\t', index=False)

    with open(kidney_specific_target, 'r') as f:
        next(f)
        G_kidney = nx.read_edgelist(f, nodetype=int, data=(('tissue', str),))

    # Genes in PPT-Ohmnet are Entrez IDs (e.g 7157), it is necessary to convert them to gene Symbols (e.g TP53, the gene name).
    # Initialize mygene object
    mg = mygene.MyGeneInfo()
    # Query gene information for list 'genes'. It specifies that the gene symbols are provided as input (scopes = symbol)
    # requests the entrezgene field to be included in the output, specifies that the genes are from the human species.
    out_entrez = list(mg.querymany(genes, scopes='symbol', fields='entrezgene', species='human', verbose=False))

    # We query for genes (DisGeNet)
    entrezgenes = []
    mapping = {}
    nodesToAdd = []
    for o in out_entrez:
        if 'entrezgene' in o:
            entrezgenes.append(int(o['entrezgene']))
            mapping[int(o['entrezgene'])] = o['query']
            res = recursive_neighbour_gather(o['entrezgene'], G_kidney, RADIUS, [], [])
            if res != []:
                nodesToAdd += [res]

    nodesToAdd = list(set([item for sublist in nodesToAdd for item in sublist]))
    addedCount = len([x for x in nodesToAdd if x not in entrezgenes])

    # we query for the added nodes through the recursive lookup
    out_symbol = list(mg.querymany(nodesToAdd, scopes='entrezgene', fields='symbol', species='human', verbose=False))
    for o in out_symbol:
        if 'symbol' in o:
            entrezgenes.append(int(o['query']))
            mapping[int(o['query'])] = o['symbol']

    # This is our expression dataframe, which we need to use in order to filter out those nodes in DisGeNet+PPI that are not in the dataframe
    entrezgenes_exp_data = []
    out_entrez_DF = list(
        mg.querymany(list(expression_DF.columns), scopes='symbol', fields='entrezgene', species='human',
                         verbose=False))
    for o in out_entrez_DF:
        if 'entrezgene' in o:
            entrezgenes_exp_data.append(int(o['entrezgene']))

    filtered_entrezgenes = list(set(entrezgenes).intersection(set(entrezgenes_exp_data)))
    filtered_mapping = {key: mapping[key] for key in mapping if key in filtered_entrezgenes}

    # we keep only nodes that are relevant to our specified genes
    original = nx.Graph(G_kidney).number_of_nodes()
    A_kidney_frozen = G_kidney.subgraph(filtered_entrezgenes)
    A_kidney = nx.Graph(A_kidney_frozen)
    subgraph_original = A_kidney.number_of_nodes()

    if remove_components == True:
        # Delete nodes from components with less than 5 nodes
        nodes_to_remove = []
        for component in list(nx.connected_components(A_kidney)):
            if len(component) < 5:
                for node in component:
                    A_kidney.remove_node(node)

    # Remove self-loops
    A_kidney.remove_edges_from(list(nx.selfloop_edges(A_kidney)))

    largest = A_kidney.number_of_nodes()
    lost = original - largest
    lost_percent = round((lost / original), 4)

    print('SNAP')
    print('Resulting network:', subgraph_original, 'nodes')
    print('Network without DF filtering: ', original, 'nodes (lost ', round((1-subgraph_original / original), 4) * 100, '%)')
    print('Added nodes through neighbor look up: ', addedCount, 'nodes')
    print('Biggest connected component:', largest, 'nodes')
    print('Total percentage of lost genes/nodes:', lost, f'({lost_percent * 100}%)')

    A_kidney_relabeled = nx.relabel_nodes(A_kidney, filtered_mapping)
    # nx.write_edgelist(A_brain_relabeled, f'data/networks/PPI_SNAP_brain_{remove_components}.edgelist')

    return A_kidney_relabeled


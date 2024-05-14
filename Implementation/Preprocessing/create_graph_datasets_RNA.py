import pickle
import pandas as pd
import networkx as nx
import os

from Preprocessing.bio_networks import get_snap


def create_tabular_dataset(ppi, clinicalFeatures = []):
   '''
   Creates a dataframe consider a PPI network. Said network needs to have as attributes a sample_id
   representing the patient, as well as a graph_label, representing the PFS value.
   :param ppi: list of graphs for each patient
   :return: dataframe
   '''
   finalRes = []
   # all graphs have the same nodes (genes)
   allGenes = list(ppi[0].nodes)
   for graph in ppi:
      res = []
      data = dict(graph.nodes(data=True))
      # first we add the sample id of the patient
      res += [graph.graph['sample_id']]
      # then we add all the gene expression values of the patient
      for gene in allGenes:
         res += [data[gene]['node_attr']]
      # finally, the PFS value of the patient

      for featName in clinicalFeatures:
         res += [graph.graph[featName]]
      # and we add it to the list of patients
      finalRes += [res]

   # Set columns, create dataframe, set index and return value
   columns = ['id'] + allGenes + clinicalFeatures
   df = pd.DataFrame(finalRes, columns=columns)
   df.set_index('id', inplace=True)
   return df

def create_samples_graphs(genData, cliData, G, clinicalFeatures = []):
   '''
   Create a graph for each sample in the dataset, using nodes and edges 
   attributes obtained previously from genetic variants information.
   '''

   samples = list(genData.index)

   print('Creating samples graphs...')

   graphs_list = []
   for sample in samples:
      sample_graph = G.copy()
      sample_data = genData.loc[sample]

      # Add graph features
      sample_graph.graph['sample_id'] = sample

      # We add some clinical features to the graph (including CENSOR and PFS)
      for featName in clinicalFeatures:
         sample_graph.graph[featName] = cliData.loc[sample][featName]

      # Add node and edge features
      for node in G.nodes:
            node_expr = sample_data[node]
            sample_graph.nodes[node]['node_attr'] = node_expr

      graphs_list.append(sample_graph)
      #print(sample_graph.nodes(data=True))
      # print(f'{sample} graph: nodes = {nx.number_of_nodes(sample_graph)}; edges = {nx.number_of_edges(sample_graph)}; label = {label}')

   return graphs_list


if __name__ == "__main__":

   CONSIDER_HISTOLOGY = False
   COHORT = 'AvelumabCohort'
   RADIUSES = [1, 2, 3]
   clinicalFeatures = ['PFS_P', 'PFS_P_CNSR', 'MATH', 'HE_TUMOR_CELL_CONTENT_IN_TUMOR_AREA',
                       'PD-L1_TOTAL_IMMUNE_CELLS_PER_TUMOR_AREA',
                       'CD8_POSITIVE_CELLS_TUMOR_CENTER', 'CD8_POSITIVE_CELLS_TOTAL_AREA', 'TRT01P']

   current_directory = os.getcwd()

   # Obtain list of genes of interest from the tsv downloaded in disgenet
   disgenet_target = os.path.abspath(
      os.path.join(current_directory, '..', '..', 'Data', 'DisGenet', 'Group_Diseases_04_4447genes.tsv'))

   G = pd.read_csv(disgenet_target, sep='\t')
   genes = list(set(G['Gene'].values))

   if CONSIDER_HISTOLOGY:
      expression_target = os.path.abspath(
         os.path.join(current_directory, '..', '..', 'Data', COHORT, 'output_GeneticDataWithHistology.csv')) # 650 patients
      clinical_target = os.path.abspath(
         os.path.join(current_directory, '..', '..', 'Data', COHORT, 'output_ClinicalDataWithHistology.csv'))
   else:
      expression_target = os.path.abspath(
         os.path.join(current_directory, '..', '..', 'Data', COHORT, 'output_GeneticData.csv')) # 738 patients
      clinical_target = os.path.abspath(
         os.path.join(current_directory, '..', '..', 'Data',COHORT, 'output_ClinicalData.csv'))
      clinicalFeatures = ['PFS_P', 'PFS_P_CNSR', 'TRT01P',
                          'HE_TUMOR_CELL_CONTENT_IN_TUMOR_AREA', 'PD-L1_TOTAL_IMMUNE_CELLS_PER_TUMOR_AREA']


   # expression data
   expression_data = pd.read_csv(expression_target, sep = ',', index_col=0)

   # we remove expressions that are very low (Q2 < 2 && Q3 < 4)
   q3_values = expression_data.quantile(0.75)
   expression_data = expression_data.loc[:, (q3_values >= 4) | (expression_data.median() >= 2)]

   clinical_data = pd.read_csv(clinical_target, sep=',', index_col=0)

   for R in RADIUSES:
      print()
      print('With radius', R, ':')

      ppi = get_snap(expression_data, genes, True, R)

      ppi_genes = list(ppi.nodes)

      filtered_expression_data = expression_data.loc[:, ppi_genes]
      # We only keep columns (genes) that are present in the network

      # Graph datasets with PPI networks and expression value per node
      variants_gd = create_samples_graphs(filtered_expression_data, clinical_data, ppi, clinicalFeatures)

      # Tabular dataset with expression data for each patient
      tabular_data = create_tabular_dataset(variants_gd, clinicalFeatures)

      addon_str = ""
      if CONSIDER_HISTOLOGY:
         addon_str = "_withHIST"

      # Next, we save it as pickle files
      output_target_graph = os.path.abspath(
         os.path.join(current_directory, '..', '..', 'Data', 'RNA_dataset_graph_R'+str(R)+ addon_str+'.pkl'))

      output_target_tabular = os.path.abspath(
         os.path.join(current_directory, '..', '..', 'Data', 'RNA_dataset_tabular_R'+str(R)+ addon_str+'.csv'))

      print("Characteristics of the tabular dataset ("+str(R)+ ") :")
      print("Number of rows: ", tabular_data.shape[0])
      print("Number of columns: ", tabular_data.shape[1])


      with open(output_target_graph, 'wb') as f:
         pickle.dump(variants_gd, f)

      # Save it as csv because we want the columns, and in pickle format it doesn't save it
      tabular_data.to_csv(output_target_tabular, index=True)



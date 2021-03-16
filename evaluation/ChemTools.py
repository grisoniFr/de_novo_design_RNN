# Contains all the necessary code to prepare the molecule:
#   - molecule sanitization (check in "import_prepare_mol" to change advanced sanitiization settings")
#   - geometry optimization (if specified by "do_geom = True"), with the specified settings

from rdkit import Chem
import pandas as pd
from rdkit.Chem import AllChem


def novelty_vs_epochs(folder, ft_path='../example/fine_tuning.csv', pt_path='../data/chembl_smiles.csv', export=False):
    # given a folder containing SMILES files, reads the files iteratively and computes the number of valid, unique and
    # novel molecules compared to the fine-tuning set (ft_path) and the pretraining set (pt_path) for each file.

    import os

    res = pd.DataFrame({"valid": [], "unique": [], "novel": []})

    folder_mols = folder + 'molecules/'
    for filename in sorted(os.listdir(folder_mols), key=numericalSort):  # imports in epoch order
        if filename.endswith(".csv"):
            print('Analysing file: ' + filename)
            # load molecules
            mols = Chem.SmilesMolSupplier(folder_mols + filename, smilesColumn=1, nameColumn=-1, titleLine=False, delimiter=',')
            # computes statistics
            stats, smis_epoch = smiles_stats(mols, ft_path=ft_path, pt_path=pt_path) # returns statistics and a dataframe with valid, unique, novel smiles
            # updates the results
            res = res.append(stats, ignore_index=True)  # appends the results of each epoch with increasing index values

            if export:
                folder_save = folder + 'molecules_novel/'
                if os.path.exists(folder_save) is False:  # checks if the folder exists, and otherwise creates it
                    os.mkdir(folder_save)

                smis_epoch.to_csv(folder_save + filename, header=None, index=False) # export smiles
    return res


def smiles_stats(suppl, ft_path='../example/fine_tuning.csv', pt_path='../data/chembl_smiles.csv'):
    # given an rdkit supplier, computes the number of valid, unique and novel molecules compared to the fine-tuning set (ft_path) and the
    # pretraining set (pt_path)

    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    smis = []
    for mol in suppl:
        if mol is not None:
            smis.append(Chem.MolToSmiles(mol))  # canon SMI

    # unique
    smis_set = list(set(smis))

    # novel
    smis_novel = novel_smiles(smis_set, ft_path, canonicalize=True)  # makes sure the FT compounds are canonicalized
    smis_novel = novel_smiles(smis_novel, pt_path)

    n_valid = len(smis)
    n_unique = len(smis_set)
    n_novel = len(smis_novel)

    res = pd.DataFrame({"valid": [n_valid], "unique": [n_unique], "novel": [n_novel]})

    return res, pd.DataFrame(smis_novel)


def novel_smiles(smi_de_novo, filepath, canonicalize=False):
    a = set(smi_de_novo)
    dataset = pd.read_csv(filepath, header=None)
    if canonicalize is True:  # canonicalize SMILES
        b = []
        for index, row in dataset.iterrows():
            b.append(Chem.MolToSmiles(Chem.MolFromSmiles(row[0])))  # canonicalize

        b = set(b)
    else:
        b = set(dataset[0].values.tolist())

    return list(a - b)


def scaffolds_vs_epochs(folder):
    # given a folder containing SMILES files, reads the files iteratively and computes the frequency of unique scaffolds

    import os

    n_unique = []
    perc_unique = []

    for filename in sorted(os.listdir(folder), key=numericalSort):  # imports in epoch order
        if filename.endswith(".csv"):
            # load molecules
            mols = Chem.SmilesMolSupplier(folder + filename, smilesColumn=0, titleLine=False, delimiter=',')

            # computes statistics
            scaff = frequent_scaffolds(mols, output_type='text')  # returns list of unique scaffolds
            n_unique.append(len(scaff))
            perc_unique.append(len(scaff)/len(mols))

    res = pd.DataFrame({'No_unique': n_unique, 'Perc_unique': perc_unique})

    return res


def morgan_vs_epochs(folder, path_ft='../example/fine_tuning.csv'):
    # given a folder containing SMILES files, reads the files iteratively and computes the frequency of unique scaffolds

    import os

    min_sim = []
    max_sim = []
    mean_sim = []

    for filename in sorted(os.listdir(folder), key=numericalSort):  # imports in epoch order
        if filename.endswith(".csv"):
            # load molecules
            sim = morgan_similarity(folder + filename, path_ft)

            # computes statistics
            min_sim.append(min(sim))
            mean_sim.append(sum(sim)/len(sim))
            max_sim.append(max(sim))

    res = pd.DataFrame({'MinSim': min_sim, 'MeanSim': mean_sim, 'MaxSim': max_sim})

    return res



def morgan_similarity(path_denovo, path_ft='../example/fine_tuning.csv'):

    from rdkit import DataStructs
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    # import mols
    denovo = Chem.SmilesMolSupplier(path_denovo, smilesColumn=0, titleLine=False, delimiter=',')
    ft = Chem.SmilesMolSupplier(path_ft,nameColumn=-1,titleLine=False)

    # compute FPs
    morgan_ft = [AllChem.GetMorganFingerprint(mol, 2) for mol in ft if mol is not None]
    morgan_denovo = [AllChem.GetMorganFingerprint(mol, 2) for mol in denovo if mol is not None]

    # the list for the dataframe
    sim = []

    # compare all fp pairwise without duplicates
    for ft_mol in morgan_ft:  # -1 so the last fp will not be used
        sim.extend(DataStructs.BulkTanimotoSimilarity(ft_mol, morgan_denovo))

    return sim

def frequent_scaffolds(suppl, output_type='supplier'):
    """
     starting from a supplier file, the function computes the most frequently recurring scaffolds and returns them as a
     supplier file (if output_type='supplier') or as a counter file.
     """
    from rdkit.Chem.Scaffolds import MurckoScaffold

    from collections import Counter
    scaff_list = []
    for mol in suppl:
        if mol is not None:
            scaff_list.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))

    freq_scaffolds = Counter()
    for scaff in scaff_list:
        freq_scaffolds[scaff] += 1

    freq_scaffolds = freq_scaffolds.most_common()

    if output_type is 'supplier':
        # converts it back in a supplier file,
        suppl_new = []
        for row in freq_scaffolds:
            mol = Chem.MolFromSmiles(row[0])
            mol.SetProp("_Name", str(round((row[1]/len(suppl))*100, 2))+'%') # assigns the molecule name as the percentage occurrence
            suppl_new.append(mol)

        freq_scaffolds = suppl_new

    return freq_scaffolds


def numericalSort(value):
    # utility to read the smile files in numerical order and not alphabetical
    import re
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts[3]
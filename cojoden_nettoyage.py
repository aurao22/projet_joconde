# %% import
from os import getcwd
from os.path import join
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import csv
from IPython.core.display import HTML
from sqlalchemy import column
from cojoden_functions import color_graph_background, get_na_columns_classement
from cojoden_functions import convert_string_to_search_string, convert_df_string_to_search_string
from collections import Counter


# ---------------------------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------------------------
# %% load_data
def load_data(data_set_path,data_set_file_name,nrows = 0, skiprows = 0, verbose=0):
    
    usecols = ['ref', 'pop_coordonnees', 
        'autr', 'bibl', 'comm', 'deno', 'desc',
        'dims', 'domn', 'dpt', 'ecol', 
        'hist',  'lieux', 'loca', 'loca2', 'mill', 'nomoff', 
        'paut', 'peri', 'pins', 'prep', 
        'region', 'repr',  'tech', 'titr', 'ville_', 'museo']

    # Suppression des colonnes avec plus de 75% de NAN 
    # 'adpt','nsda','geohi', 'manquant', 'manquant_com', 'milu','onom','refmem', 'refmer','refpal', 'retif',
    # 'peoc','peru','plieux','refmis','srep','attr','puti', 'ddpt','pdec','appl','drep','depo','epoq','decv',
    # 'expo', 'etat', 'gene', 'larc', 

    # Suppression des colonnes non nécessaire :
    # 'www','base','contient_image', 'dacq','dmaj','copy','pop_contient_geolocalisation','aptn', 
    # 'dmis', 'msgcom','museo', 'producteur', 'image','inv', 'phot', 'stat', 'label', 'historique','util','insc',


    df_origin = None
    if nrows > 0:
        print("(", skiprows, "to", nrows,"rows)")
        df_origin = pd.read_csv(join(data_set_path,data_set_file_name), skiprows=skiprows, quoting=csv.QUOTE_NONNUMERIC, sep=';', low_memory=False, usecols=usecols)
    else:
        df_origin = pd.read_csv(join(data_set_path,data_set_file_name), quoting=csv.QUOTE_NONNUMERIC, sep=';', low_memory=False, usecols=usecols)

    print(f"{df_origin.shape} données chargées ------> {list(df_origin.columns)}")

    to_rename = {
        'autr':'auteur',
        'comm':'commentaires', 
        'deno':'type_oeuvre', 
        'desc':'description',
        'dims':'dimensions', 
        'domn':'domaine', 
        'dpt':'geo_departement', 
        'ecol':'geo_ecole_pays', 
        'lieux':'creation_lieux', 
        'loca':'lieux_conservation', 
        'loca2':'geo_pays_region_ville', 
        'mill':'creation_millesime', 
        'nomoff':'nom_officiel_musee', 
        'paut':'auteur_precisions', 
        'peri':'creation_periode', 
        'pins':'inscription_precisions',
        'prep':'sujet_precisions', 
        'repr':'sujet',  
        'tech':'materiaux_technique',
        'titr':'titre',
        'ville_':'geo_ville',
        'region' : 'geo_region',
        # 'bibl':'bibliographie',
        # 'insc':'inscription',
        # 'inv':'no_inventaire', 
        # 'label':'label_musee_fr',  
        # 'phot':'credit_photo', 
        # 'producteur':'data_producteur', 
        # 'stat':'statut_juridique', 
        # 'util':'utilisation', 
    }

    df_origin = df_origin.rename(columns=to_rename)
    if verbose > 1:
        print(list(df_origin.columns))

    if verbose > 1:
        figure, _ = color_graph_background(1,1)
        sns.heatmap(df_origin.isnull(), yticklabels=False,cbar=False, cmap='viridis')
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.title("NA dans la DF")
        figure.set_size_inches(18, 5, forward=True)
        plt.show()
        display(HTML(df_origin.head().to_html()))
    return df_origin

# %% proceed_encoding
def proceed_encoding(df_origin, verbose=0):
    # L'encodage a été traité directement en ligne de commands dans le fichier de données sources.
    # Cf. script proceed_encoding.sh
    for col in df_origin.columns:
        no_na = df_origin[col].notna()
        df_origin.loc[no_na, col] = df_origin.loc[no_na, col].str.replace('Ã  ', 'à ')
    return df_origin

# %% proceed_duplicated
def proceed_duplicated(df_origin, verbose=0):
    if verbose > 0:
        print(f"[proceed_duplicated]\t INFO : Before {df_origin.shape}")
    # Suppression des 2246 rows strictement identiques
    df_clean = df_origin.drop_duplicates()

    # Suppression des doublons sur la référence
    df_clean = df_clean.drop_duplicates(subset=['ref'])
    if verbose > 0:
        print(f"[proceed_duplicated]\t INFO : after {df_clean.shape}")
    return df_clean

# %% proceed_na_values
def proceed_na_values(df_origin, verbose=0):
    df_clean = df_origin.copy()

    # Affectation de NA aux oeuvres qui n'ont pas de titre
    if verbose > 0:
        print(f"[proceed_na_values]\t INFO : {df_clean[df_clean['titre']=='(Sans titre)'].shape} oeuvres (Sans titre)")
        print(f"[proceed_na_values]\t INFO : {df_clean['titre'].isna().sum()} NA Before")
        
    df_clean.loc[df_clean['titre']=='(Sans titre)', 'titre'] = np.nan
    if verbose > 0:
        print(f"[proceed_na_values]\t INFO : {df_clean[df_clean['titre']=='(Sans titre)'].shape} oeuvres (Sans titre)")
        print(f"[proceed_na_values]\t INFO : {df_clean['titre'].isna().sum()} NA After")
    
    # A priori il serait possible de générer un titre pertient à partir des descriptions.
    # Dans un premier temps, suppression des lignes qui n'ont ni description, ni titre, ni auteur
    label_na = (df_clean["titre"].isna())&(df_clean["auteur"].isna())&(df_clean["description"].isna())&(df_clean["sujet_precisions"].isna())&(df_clean["sujet"].isna())
    df_clean = df_clean[~label_na]
    if verbose > 0:
        print(f"[proceed_na_values]\t INFO : {df_origin.shape[0]-df_clean.shape[0]} lignes sans labels supprimées")

    # On génère un titre à partir des autres colonnes textuelles.
    is_na_titre_idx = df_clean['titre'].isna()
    df_clean.loc[is_na_titre_idx, "titre"] = df_clean.loc[is_na_titre_idx,["description", 'type_oeuvre', 'sujet', 'sujet_precisions']].apply(lambda x: _nouveau_titre(description=x['description'], type_oeuvre=x['type_oeuvre'], sujet=x['sujet'], precisions=x['sujet_precisions']), axis=1)
    if verbose > 0:
        print(f"[proceed_na_values]\t INFO : {df_clean['titre'].isna().sum()} NA Titres après génération de titre")

    # On traite les types d'oeuvre
    if verbose > 0:
        print(f"[proceed_na_values]\t INFO : {df_clean['type_oeuvre'].isna().sum()} NA Type d'oeuvres")
    is_na_type_oeuvre = df_clean['type_oeuvre'].isna()
    df_clean.loc[is_na_type_oeuvre, "type_oeuvre"] = df_clean.loc[is_na_type_oeuvre,["description", 'materiaux_technique']].apply(lambda x: _nouveau_type_oeuvre(description=x['description'], materiaux_technique=x['materiaux_technique']), axis=1)
    # on traite les derniers résidus
    df_clean['type_oeuvre'] = df_clean['type_oeuvre'].fillna(df_clean['description'])
    df_clean['type_oeuvre'] = df_clean['type_oeuvre'].fillna(df_clean['materiaux_technique'])
    if verbose > 0:
        print(f"[proceed_na_values]\t INFO : {df_clean['type_oeuvre'].isna().sum()} NA Type d'oeuvres après traitement")

    # Création d'une colonne texte avec toutes les colonnes de descriptions
    df_clean["texte"] = df_clean[['sujet_precisions', "description", 'sujet']].apply(lambda x: _nouveau_texte(x=x), axis=1)
    if verbose > 0:
        print(f"[proceed_na_values]\t INFO : {df_clean['texte'].isna().sum()} NA Texte")

    return df_clean

# %% extract_villes
def extract_villes(df, dest_path, dest_file_name='villes_departement_region_pays.csv', verbose=0):
    df_cities = df[['geo_ville', 'geo_departement', 'geo_region','geo_pays_region_ville']]
    if verbose > 0:
        print(f"[extract_villes]\t INFO : {df_cities.shape} on origin df")
    df_cities = df_cities.rename(column={'geo_ville':'ville', 'geo_departement':'departement', 'geo_region':'region1','geo_pays_region_ville':'pays'})
    df_cities = df_cities.drop_duplicates()
    df_cities = df_cities.dropna('ville')
    if verbose > 0:
        print(f"[extract_villes]\t INFO : {df_cities.shape} after drop duplicates and NA")
    df_cities.to_csv(join(dest_path,dest_file_name), index=False)
    print(f"[extract_villes]\t INFO : {df_cities.shape} données sauvegardées in ------> {join(dest_path,dest_file_name)}")
    return df_cities

# %% extract_musees
def extract_musees(df, dest_path, dest_file_name='musees.csv', verbose=0):
    df_musee = df[['museo','nom_officiel_musee','pop_coordonnees','geo_ville']]
    if verbose > 0:
        print(f"[extract_musees]\t INFO : {df_musee.shape} on origin df")
    df_musee = df_musee.drop_duplicates()
    if verbose > 0:
        print(f"[extract_musees]\t INFO : {df_musee.shape} after remove duplicates")
    df_musee = df_musee[df_musee['museo'].notna()]
    if verbose > 0:
        print(f"[extract_musees]\t INFO : {df_musee.shape} after remove NA on museo")
    # Remplacement des valeurs 0 par nan
    df_musee.loc[df_musee['pop_coordonnees']=='0.0,0.0','pop_coordonnees'] = np.nan
    # Calcul des nan
    df_musee['NB_NAN_1'] = df_musee.isna().sum(axis=1)
    df_musee_sort = df_musee.sort_values(['NB_NAN_1', 'museo'])
    if verbose > 0:
        print(f"[extract_musees]\t INFO : {df_musee_sort.shape} with NB_NAN columns")
    df_musee_sort = df_musee_sort.drop_duplicates(['museo'], keep='first')
    if verbose > 0:
        print(f"[extract_musees]\t INFO : {df_musee_sort.shape} after drop duplicates on museo")
    
    # Extraction des coordonnées
    df_coordonnees = df_musee_sort['pop_coordonnees'].dropna().str.split(r",", expand=True)
    if verbose > 0:
        print(f"[extract_musees]\t INFO : {df_musee_sort.shape} musees and {df_coordonnees.shape} coordonnates")
    df_coordonnees = df_coordonnees.rename(columns={0:"latitude", 1:'longitude'})

    # Fusion des df pour affecter les coordonnées à la DF musées
    df_musee_clean = pd.merge(df_musee_sort, df_coordonnees,left_index=True, right_index=True, copy=True, indicator=True)
    df_musee_clean = df_musee_clean[['museo', 'nom_officiel_musee', 'geo_ville', 'latitude', 'longitude', 'pop_coordonnees']]
    df_musee_clean = df_musee_clean.rename(columns={
        'geo_ville' : 'ville',
        'nom_officiel_musee' : 'nom',
    })
    df_musee_clean = df_musee_clean.sort_values(by=['ville'])
    df_musee_clean = df_musee_clean.reset_index()
    df_musee_clean = df_musee_clean.drop(columns=['index'], axis=1)
    if verbose > 0:
        print(f"[extract_musees]\t INFO : {df_musee_clean.shape} musees.")
    
    df_musee_clean.to_csv(join(dest_path,dest_file_name), index=False)
    print(f"[extract_musees]\t INFO : {df_musee_clean.shape} données sauvegardées in ------> {join(dest_path,dest_file_name)}")
    return df_musee_clean
    
# %% extract_artistes
def extract_artistes(df, dest_path, dest_file_name='artistes.csv', verbose=0):
    df_aut1 = df[['auteur', 'auteur_precisions']]
    if verbose > 0:
        print(f"[extract_artistes]\t INFO : {df_aut1.shape} on origin df")
    df_aut1 = df_aut1.sort_values('auteur')
    df_aut1 = df_aut1.drop_duplicates()
    # df_aut1['auteur'] = df_aut1['auteur'].fillna(df_aut1['auteur_precisions'])
    df_aut1 = df_aut1[df_aut1['auteur'].notna()]
    if verbose > 0:
        print(f"[extract_artistes]\t INFO : {df_aut1.shape} without na and duplicates")
    
    # Suppression des termes spécifiques rencontrés lors de l'exploration des données
    to_replace = {
        '"' : "",
        "établissement " : "",
        'Établissement ' : "",
        "Établissements ": "",
        # '(fabricant)' : "",
        # '(imprimeur)': "",
        # '(constructeur)': "",
        # '(émetteur)': "",
        "\? (copie d\'après);": "",
        "? (d\'après)": "",
    }
    for str_1, str_2 in to_replace.items():
        df_aut1.loc[df_aut1['auteur'].notna(), 'auteur'] = df_aut1.loc[df_aut1['auteur'].notna(), 'auteur'].str.replace(str_1, str_2, regex=False)

    df_aut1.loc[df_aut1['auteur'].notna(), 'auteur'] = df_aut1.loc[df_aut1['auteur'].notna(), 'auteur'].str.strip()
    df_aut1.loc[(df_aut1['auteur'].notna()) & (df_aut1['auteur'])=='', 'auteur'] = np.nan
    df_aut1 = df_aut1[df_aut1['auteur'].notna()]
    df_aut1 = df_aut1.drop_duplicates()
    if verbose > 0:
        print(f"[extract_artistes]\t INFO : {df_aut1.shape} after remove specific words")

    df_aut2 = df_aut1['auteur'].dropna().str.split(r";", expand=True)
    if verbose > 0:
        print(f"[extract_artistes]\t INFO : {df_aut2.shape} after split ';'")
    
    # Extraction de tous les artistes de manière unique
    set_aut2 = None
    for col in df_aut2.columns:
        if set_aut2 is None:
            set_aut2 = set(df_aut2[col].dropna().unique())
        else:
            for v in df_aut2[col].dropna().unique():
                if len(v)>0:
                    set_aut2.add(v)
    set_aut2.remove('')
    if verbose > 0:
        print(f"[extract_artistes]\t INFO : {len(set_aut2)} unique after extraction")
    
    # Création d'une DF avec le set
    auth3 = pd.DataFrame(set_aut2)
    df_aut4 = auth3[0].dropna().str.split(r"(", expand=True)
    df_aut4 = df_aut4.sort_values(0)
    if verbose > 0:
        print(f"[extract_artistes]\t INFO : {df_aut4.shape} after split '('")
    
    # Nettoyage des données
    df_aut4['nom_naissance'] =df_aut4[0]
    df_aut4['dit'] =np.nan

    dit_2_bool = (df_aut4[1].notna()) & ((df_aut4[1].str.contains('dit)', regex=False)) | (df_aut4[1].str.contains('dite)', regex=False)))
    df_aut4.loc[dit_2_bool, 'dit'] = df_aut4.loc[dit_2_bool, 0]
    if verbose > 0:
        print(f"[extract_artistes]\t INFO : {df_aut4[dit_2_bool].shape} artistes with 'dit' name")
    
    ne_2_bool = (df_aut4[1].notna()) & ((df_aut4[2].str.contains('né)', regex=False)) | (df_aut4[2].str.contains('née)', regex=False)))
    df_aut4.loc[ne_2_bool, 'nom_naissance'] = df_aut4.loc[ne_2_bool, 1].str.replace('dit), ', '', regex=False)
    df_aut4.loc[ne_2_bool, 'nom_naissance'] = df_aut4.loc[ne_2_bool, 'nom_naissance'].str.replace('dite), ', '', regex=False)
    if verbose > 0:
        print(f"[extract_artistes]\t INFO : {df_aut4[ne_2_bool].shape} artistes with 'née' or 'né' name")
    
    # Suppression des valeurs traitées
    df_aut4.loc[df_aut4[1]=='dit)', 1]=np.nan
    df_aut4.loc[df_aut4[1]=='dite)', 1]=np.nan
    df_aut4.loc[df_aut4[1]==df_aut4.loc[8115, 1], 1]=np.nan
    df_aut4.loc[df_aut4['nom_naissance']==df_aut4.loc[8115, 'nom_naissance'], 'nom_naissance']=df_aut4.loc[df_aut4['nom_naissance']==df_aut4.loc[8115, 'nom_naissance'], 0]
    df_aut4.loc[df_aut4[1]=='dit\)', 1]=np.nan
    df_aut4.loc[df_aut4[1]=='d’après, dit)', 1]=np.nan
    df_aut4.loc[df_aut4[2]=='née)', 2]=np.nan
    df_aut4.loc[df_aut4[2]==df_aut4.loc[43882, 2], 2]=np.nan
    df_aut4.loc[df_aut4[2]==df_aut4.loc[22108, 2], 2]=np.nan
    df_aut4.loc[df_aut4[2]=='née\)', 2]=np.nan
    df_aut4.loc[df_aut4[2]=='né\)', 2]=np.nan
    if verbose > 0:
        print(f"[extract_artistes]\t INFO : {df_aut4.shape} artistes after proceed dit and né")
        print(f"[extract_artistes]\t INFO : {df_aut4.isna().sum()} NA after proceed dit and né")
    
    # 
    df_aut5 = df_aut4[['nom_naissance', 'dit', 0, 1, 2, 3]]
    df_aut5 = df_aut5.sort_values('nom_naissance')

    to_replace = {
        'attribué Ã )':"",
        'atelier, genre de)':"",
        '?)':"",
        "d'après)":"",
    }
    for str_1, str_2 in to_replace.items():
        for col in [0,1,2,3]:
            df_aut5.loc[df_aut5[col].notna(), col] = df_aut5.loc[df_aut5[col].notna(), col].str.replace(str_1, str_2, regex=False)

    if verbose > 0:
        print(f"[extract_artistes]\t INFO : {df_aut5.shape} artistes after proceed specific string to replace")

    for col in ["nom_naissance", "dit", 0,1,2,3]:
        if verbose > 0:
            print(f"[extract_artistes]\t INFO : {col}={df_aut5[col].isna().sum()} nan before ", end="")
        if col not in ["nom_naissance", "dit"]:
        # df_aut5.loc[(df_aut5[col].notna()) & (df_aut4.loc[55537, 1]==df_aut5[col]), col] = np.nan
            df_aut5.loc[(df_aut5[col].notna()) & (df_aut5['nom_naissance']==df_aut5[col]), col] = np.nan
            df_aut5.loc[(df_aut5[col].notna()) & (df_aut5['dit']==df_aut5[col]), col] = np.nan
        df_aut5.loc[(df_aut5[col].notna()), col] = df_aut5.loc[(df_aut5[col].notna()), col].str.strip()
        df_aut5.loc[(df_aut5[col].notna()) & (df_aut5[col].str.len()==0), col] = np.nan
        if verbose > 0:
            print(f"and {df_aut5[col].isna().sum()} nan after")
    if verbose > 0:
        print(f"[extract_artistes]\t INFO : {df_aut5.shape} artistes after cleaning same name and dit name")
    
    df_aut6 = df_aut5[~((df_aut5["nom_naissance"].isna()) & (df_aut5["dit"].isna()))]
    df_aut6 = df_aut6[['nom_naissance', 'dit', 1, 2, 3]]
    df_aut6 = df_aut6.rename(columns={
        1 : "metier"
    })
    df_aut6 = df_aut6.drop_duplicates()
    df_aut6.loc[(df_aut6["metier"].notna()) & (df_aut6["metier"].str.startswith("dit)")), "metier"] = np.nan
    df_aut6.loc[(df_aut6["metier"].notna()) & (df_aut6["metier"].str.startswith("dite)")), "metier"] = np.nan
    df_aut6 = df_aut6.drop_duplicates()
    if verbose > 0:
        print(f"[extract_artistes]\t INFO : {df_aut6.shape} artistes cleaning dit and dite and drop duplicates")
    
    # A ce stade il manquait des artistes, il a donc fallu les extraires en ligne de commande pour les ajouter à la DF.
    df_aut7 = _add_missing_artists(df_aut6.copy(), verbose=verbose-1)
    if verbose > 0:
        print(f"[extract_artistes]\t INFO : {df_aut7.shape} artistes after adding missing artistes")  

    df_aut7 = df_aut7.sort_values('nom_naissance')
    to_replace = {
        'attribué Ã )':"",
        'attribué à)':"",
        'atelier, genre de)':"",
        '?)':"",
        "d'après)":"",
        'dit)':'',
        ', LE GUERCHIN':'',
        'dit), GUERCINO':'',
        's))':'',
    }
    for str_1, str_2 in to_replace.items():
        for col in ['metier', 2,3]:
            df_aut7.loc[df_aut7[col].notna(), col] = df_aut7.loc[df_aut7[col].notna(), col].str.replace(str_1, str_2, regex=False)
            df_aut7.loc[df_aut7[col].notna(), col] = df_aut7.loc[df_aut7[col].notna(), col].str.replace(")", "", regex=False)

    if verbose > 0:
        print(f"[extract_artistes]\t INFO : {df_aut7.shape} artistes cleaning in other columns")  

    for col in ["nom_naissance", "dit",'metier', 2,3]:
        if verbose > 0:
            print(f"[extract_artistes]\t INFO :{col}={df_aut7[col].isna().sum()} nan before ", end="")
        if col not in ["nom_naissance", "dit"]:
            df_aut7.loc[(df_aut7[col].notna()) & (df_aut7['nom_naissance']==df_aut7[col]), col] = np.nan
            df_aut7.loc[(df_aut7[col].notna()) & (df_aut7['dit']==df_aut7[col]), col] = np.nan
        df_aut7.loc[(df_aut7[col].notna()), col] = df_aut7.loc[(df_aut7[col].notna()), col].str.strip()
        df_aut7.loc[(df_aut7[col].notna()) & (df_aut7[col].str.len()==0), col] = np.nan
        if verbose > 0: print(f"and {df_aut7[col].isna().sum()} nan after")

    df_aut7 = df_aut7.reset_index()
    df_aut7 = df_aut7.drop('index', axis=1)
    # Suppression des 7 premières lignes qui ne sont pas des auteurs
    df_aut7 = df_aut7[7:]
    df_aut8 = df_aut7.drop(range(8, 17), axis=0)
    if verbose > 0:
            print(f"[extract_artistes]\t INFO :{df_aut8.shape}")

    df_aut8.loc[(df_aut8["nom_naissance"].notna()),"nom_naissance"] = df_aut8.loc[(df_aut8["nom_naissance"].notna()),"nom_naissance"].str.replace("« ", "", regex=False)
    df_aut8.loc[(df_aut8["nom_naissance"].notna()),"nom_naissance"] = df_aut8.loc[(df_aut8["nom_naissance"].notna()),"nom_naissance"].str.replace(" »", "", regex=False)
    df_aut10 = df_aut8.drop_duplicates(["nom_naissance", "dit"])
    if verbose > 0:
            print(f"[extract_artistes]\t INFO :{df_aut10.shape}")

    df_aut10["upper_search_name"] = df_aut10["upper_name"]
    df_aut10 = convert_df_string_to_search_string(input_df=df_aut10, col_name="upper_search_name")
    if verbose > 0:
        print(f"[extract_artistes]\t INFO :{df_aut10.shape}")
    df_aut10 = df_aut10.sort_values(["upper_search_name", 'NB_NAN'])
    df_aut10 = df_aut10.drop_duplicates(['upper_search_name'], keep='first')
    if verbose > 0:
        print(f"[extract_artistes]\t INFO :{df_aut10.shape}")
    df_aut10 = df_aut10.sort_values(["upper_search_name", 'NB_NAN'])
    df_aut10 = df_aut10.drop_duplicates(['upper_search_name'], keep='first')
    if verbose > 0:
        print(f"[extract_artistes]\t INFO :{df_aut10.shape}")
    
    df_aut10.to_csv(join(dest_path,dest_file_name), index=False)
    print(f"[extract_artistes]\t INFO :{df_aut10.shape} données sauvegardées in ------> {join(dest_path,dest_file_name)}")
    return df_aut10

# %% extract_materiaux_technique
def extract_materiaux_technique(df, dest_path, dest_file_name='materiaux_techniques.csv', verbose=0):
    df1 = df[['materiaux_technique']]
    if verbose > 0:
        print(f"[extract_materiaux_technique]\t INFO : {df1.shape} on origin df")
    df1 = df1.sort_values('materiaux_technique')
    df1 = df1.drop_duplicates()
    # df_aut1['auteur'] = df_aut1['auteur'].fillna(df_aut1['auteur_precisions'])
    df1 = df1[df1['materiaux_technique'].notna()]
    if verbose > 0:
        print(f"[extract_materiaux_technique]\t INFO : {df1.shape} without na and duplicates")
    
    # Séparation des matériaux
    df2 = df1[['materiaux_technique']]
    for sep in [', ',',', ';', " (", '/', '.', ' : ']:
        df2 = df2['materiaux_technique'].dropna().str.split(sep, expand=True, regex=False)
        set_mat1 = set()
        for col in df2.columns:
            for v in df2[col].dropna().unique():
                if len(v)>1 and v !='?))'and v !='?)':
                    v2 = v.replace(')', '')
                    v2 = v2.replace('(', ' ')
                    v2 = v2.replace(':', ' ')
                    v2 = v2.replace('Ã ', 'à')
                    v2 = v2.replace(' ?', ' ')
                    v2 = v2.replace('?', ' ')
                    v2 = v2.replace('   ', ' ')
                    v2 = v2.replace('  ', ' ')
                    try:
                        int(v2)
                    except:
                        v2 = v2.strip()
                        if len(v2)>1:
                            set_mat1.add(v2)
        try:
            set_mat1.remove('')
        except:
            pass
        df2 = pd.DataFrame(set_mat1, columns=['materiaux_technique'])

    # Création du nom pour les recherches
    df_fin = df2.copy()
    df_fin["mat_search"] = df_fin['materiaux_technique']
    df_fin = convert_df_string_to_search_string(input_df=df_fin, col_name="mat_search")

    if verbose > 0:
        print(f"[extract_materiaux_technique]\t INFO : {df_fin.shape}")
    df_fin = df_fin.sort_values(["mat_search"])
    df_fin = df_fin.drop_duplicates(['mat_search'], keep='first')
    if verbose > 0:
        print(f"[extract_materiaux_technique]\t INFO : {df_fin.shape}")

    df_fin.to_csv(join(dest_path,dest_file_name), index=False)
    print(f"[extract_materiaux_technique]\t INFO : {df_fin.shape} données sauvegardées in ------> {join(dest_path,dest_file_name)}")
    return df_fin

# %% extract_oeuvres
def extract_oeuvres(df, dest_path, dest_file_name='oeuvres.csv', verbose=0):
    pass

# ----------------------------------------------------------------------------------
# PRIVATE FUNCTIONS - Text generation to fill na values
# ----------------------------------------------------------------------------------
# %% _nouveau_titre
def _nouveau_titre(description, type_oeuvre, sujet, precisions):
    """
    Génère un titre à partir du début de la description (jusqu'au premier séparateur identifié) ou à partir du type d'oeuvre ou à partir des précisions

    Args:
        description (str): _description_
        type_oeuvre (str): _description_
        sujet (str): _description_
        precisions (str): _description_

    Returns:
        str: Titre généré
    """
    titre = ""
    precision_done = False

    if precisions is not None and isinstance(precisions, str):
        titre = precisions.strip()
        precision_done = len(titre)>0
    

    if not precision_done and description is not None and isinstance(description, str):
        titre = description.split(".")[0]
        titre = titre.split(";")[0]
        titre = titre.split(",")[0]
        
    if not precision_done and type_oeuvre is not None and isinstance(type_oeuvre, str):
        start_type = type_oeuvre.split(" ")[0]
        if start_type.lower() not in titre.lower():
            titre = type_oeuvre + " " + titre

    if len(titre)==0 and sujet is not None and isinstance(sujet, str):
        start_type = sujet.split("(")[-1]
        start_type = start_type.split(")")[0]
        if len(start_type)>0:
            titre = start_type.strip()
    
    titre = titre.strip()

    return titre if len(titre)>0 else np.nan

# %% _nouveau_type_oeuvre
def _nouveau_type_oeuvre(description, materiaux_technique):
    titre = ""
    if description is not None and isinstance(description, str):
        titre = description.split(".")[0]
        titre = titre.split(";")[0]
        titre = titre.split(",")[0]
        
    if materiaux_technique is not None and isinstance(materiaux_technique, str):
        start_type = materiaux_technique.split(" ")[0]
        if start_type.lower() not in titre.lower():
            titre = materiaux_technique + " " + titre
    
    titre = titre.strip()

    return titre if len(titre)>0 else np.nan

# %% _nouveau_texte
def _nouveau_texte(x):
    titre = ""
    
    for col in x.index:
        if isinstance(x[col], str):
            sep = " " if len(titre)>0 and titre.endswith(".") else ". " if len(titre)>0 else ""
            titre = titre + sep +  x[col].strip()

    titre = titre.strip()

    return titre

# %% _add_missing_artists
def _add_missing_artists(df, verbose=0):
    # il manquait des artistes, il a donc fallu les extraires en ligne de commande pour les ajouter à la DF.
    # Le fichier name étant une simple sauvegarde de la DF
    # Ligne de commande correspondante : grep "'e : .*'" name.txt
    to_add = [("Quagliozzi","Aurélien"),
            ("Romon","Anthony"),
            ("da Rocha","François"),
            ("Dumont","Stéphane"),
            ("Collot","Patrick"),
            ("Legrand","Romain"),
            ("Defeyer","Jean-Baptiste"),
            ("Faroux","Thomas"),
            ("Giauffret","Michel"),
            ("Mattio","Jean-Philippe"),
            ("Dupont","Isabelle"),
            ("Codron","Anthony"),
            ("Courmont","Jean-Marc"),
            ("Bohée","Francis"),
            ("Cubilier","Eric"),
            ("Roger","Mylène"),
            ("Dattola","Rina"),
            ("Leroux","Rudy"),
            ("Bovis","Nadine"),
            ("Raymond","Alexis"),
            ("Jauregui","Nicolas"),
            ("Lepage","Sandra"),
            ("Delecroix","Elsa"),
            ("Denis","Daniel"),
            ("Borla","Audrey"),
            ("Stepaniak","Philippe"),
            ("Farineau","Typhanie"),
            ("Singer","Roland"),
            ("Lavagna","Richard"),
            ("Lavagna","Sabine"),
            ("Noseda","Veronica"),
            ("Collomb","Marie-Caroline"),
            ("Bertino","Eric"),
            ("Marissael","Pascal"),
            ("Desmaretz","Arnaud"),
            ("Capelain","Jean"),
            ("Bougaret","Eric"),
            ("Resegotti","Robert"),
            ("Lis","Damien"),
            ("Duhomez","Marcel")]

    for (nom, prenom) in to_add:
        new_row = {'nom_naissance':nom.upper()+' '+prenom, 'dit':np.nan, 'metier':np.nan, 2:np.nan,3:np.nan}
        #append row to the dataframe
        df = df.append(new_row, ignore_index=True)
    return df

# %% _clean_geo_text
def _clean_geo_text(input_geo_set):
    geo_clean = set()
    sorted(input_geo_set)
    for c in input_geo_set:
        low = c.lower()
        if len(c) > 0 and "musée" not in low and not low.startswith("école") \
            and not low.startswith("écomusée")\
            and not low.startswith('conseil ')\
            and not low.startswith('CAPC'.lower())\
            and not low.startswith('Ecomusée'.lower()):
            geo_clean.add(c)
    return geo_clean

# ----------------------------------------------------------------------------------
#                        MAIN
# ----------------------------------------------------------------------------------
# %% main
if __name__ == '__main__':
    verbose = 1
    run_extraction = 0

    # Récupère le répertoire du programme
    file_path = getcwd() + "\\"
    if "PROJETS" not in file_path:
        file_path = join(file_path, "PROJETS")
    if "projet_joconde" not in file_path:
        file_path = join(file_path, "projet_joconde")
    
    data_set_path = join(file_path , "dataset\\")
    data_set_file_name = "base-joconde-extrait.csv"

    print(f"Current execution path : {file_path}")
    print(f"Dataset path : {data_set_path}")

    # Chargement et nettoyage général
    df_origin = load_data(data_set_path=data_set_path,data_set_file_name=data_set_file_name, verbose=verbose)
    df_encode = proceed_encoding(df_origin, verbose=verbose)
    df_clean = proceed_duplicated(df_encode, verbose=verbose)
    df_clean_na = proceed_na_values(df_clean, verbose=verbose)

    if verbose > 1:
        figure, ax = color_graph_background(1,1)
        sns.heatmap(df_clean_na.isnull(), yticklabels=False,cbar=False, cmap='viridis')
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.title("NA dans la DF après traitement.")
        figure.set_size_inches(18, 5, forward=True)
        plt.show()

    # extraction des données
    if run_extraction > 0:
        df_villes = extract_villes(df=df_clean_na, dest_path=data_set_path, dest_file_name='villes_departement_region_pays.csv',verbose=verbose)
        df_musees = extract_musees(df=df_clean_na, dest_path=data_set_path, dest_file_name='musees.csv', verbose=verbose)
        df_artistes = extract_artistes(df=df_clean_na, dest_path=data_set_path, dest_file_name='artistes.csv', verbose=verbose)
        df_materiaux = extract_materiaux_technique(df=df_clean_na, dest_path=data_set_path, dest_file_name='materiaux_techniques.csv', verbose=verbose)
    
    

import MySQLdb
import pandas as pd
import mysql.connector
import sqlalchemy as sa
from mysql.connector import errorcode
from os.path import join, exists
from cojoden_functions import convert_df_string_to_search_string
from tqdm import tqdm


# ----------------------------------------------------------------------------------
#                        DATABASE INFORMATIONS
# ----------------------------------------------------------------------------------
db_name="cojoden"
db_user="root"
db_pwd="root"
db_host="127.0.0.1"
db_client="mysql"


# ----------------------------------------------------------------------------------
#                        DATABASE INITIALISATION
# ----------------------------------------------------------------------------------
def initialiser_bdd():
    connection = mysql.connector.connect(
        user=db_user,
        password=db_pwd,
        host=db_host)
    cursor = connection.cursor()

    with open('cojoden_dao_create.sql', 'r') as sql_file:

        for line in sql_file.split(";"):
            try:
                cursor.execute(line)
            except Exception as msg:
                print(f"[cojoden_dao>new_db] ERROR : \n\t- {line} \n\t- {msg}")

    return connection, cursor

def connecter():
    """
    Returns:
        connection
    """
    connection = None
    try:
        connection = mysql.connector.connect(
            user=db_user,
            password=db_pwd,
            host=db_host,
            database=db_name)

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print(f"[cojoden_dao > execute] ERROR : Something is wrong with your user name or password : {err}")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print(f"[cojoden_dao > execute] ERROR : Database does not exist : {err}")
        else:
            print(f"[cojoden_dao > execute] ERROR : {err}")

    return connection 


# ----------------------------------------------------------------------------------
# PEUPLEMENT DE LA BDD
# ----------------------------------------------------------------------------------
def populate_database(dataset_path=r'dataset', verbose=0):
    populate_villes(dataset_path=dataset_path, file_name=r'cojoden_villes_departement_region_pays.csv')
    populate_metiers(dataset_path=dataset_path, file_name=r'cojoden_metiers_uniques.txt')
    populate_artistes(dataset_path=dataset_path, file_name=r'cojoden_artistes.csv')
    populate_musees(dataset_path=dataset_path, file_name=r'cojoden_musees.csv')

def populate_musees(dataset_path, file_name=r'cojoden_musees.csv', verbose=0):
    nb_pop = -1
    file_path = join(dataset_path, file_name)

    if exists(file_path):
        df = pd.read_csv(file_path)
        if 'nom_search' not in list(df.columns):
            df['nom_search'] = df['nom']
            df['ville_src'] = df['ville']
            df = convert_df_string_to_search_string(df, col_name='nom_search', stop_word_to_remove=['écomusée','ecomusee','écomusé','ecomuse',"Musées", "Musees","Musée","Musee","Muse",  'Museon', 'muséum', "museum"])
            df = convert_df_string_to_search_string(df, col_name='ville')
            df = df.sort_values('nom_search')
            df.to_csv(join(dataset_path,file_name.replace(".csv", "-v2.csv")), index=False)

        dbConnection =_create_engine(verbose=verbose)
        try:
            nb_pop = df[['museo', 'nom', 'nom_search', 'ville', 'latitude', 'longitude']].to_sql(name='musee', con=dbConnection, if_exists='append', index=False, chunksize=10)
        except mysql.connector.IntegrityError as error:
            nb_pop = 0
            if verbose > 0:
                print(f"[cojoden_dao > populate_musees] WARNING : la table est déjà peuplée.\n\t- {error}")
        except Exception as error:
            if  "IntegrityError" in str(error):
                nb_pop = 0
                if verbose > 0:
                    print(f"[cojoden_dao > populate_musees] WARNING : la table est déjà peuplée.\n\t- {error}")
            else:
                raise error
    return nb_pop

def populate_metiers(dataset_path, file_name=r'cojoden_metiers_uniques.csv', verbose=0):
    nb_pop = -1
    file_path = join(dataset_path, file_name)

    if exists(file_path):
        df = pd.read_csv(file_path)
        if 'metier_search' not in list(df.columns):
            df['metier_search'] = df['metier']
            df = convert_df_string_to_search_string(df, col_name='metier_search')
            df = df[['metier_search', 'metier']]
            df = df.sort_values('metier_search')
            df = df.drop_duplicates('metier_search')
            df.to_csv(join(dataset_path,file_name.replace(".csv", "-v2.csv")), index=False)
        dbConnection =_create_engine(verbose=verbose)
        try:
            nb_pop = df.to_sql(name='metier', con=dbConnection, if_exists='append', index=False)
        except mysql.connector.IntegrityError as error:
            nb_pop = 0
            if verbose > 0:
                print(f"[cojoden_dao > populate_villes] WARNING : la table est déjà peuplée.\n\t- {error}")
        except Exception as error:
            if  "IntegrityError" in str(error):
                nb_pop = 0
                if verbose > 0:
                    print(f"[cojoden_dao > populate_villes] WARNING : la table est déjà peuplée.\n\t- {error}")
            else:
                raise error
    return nb_pop

def populate_villes(dataset_path, file_name=r'cojoden_villes_departement_region_pays.csv', verbose=0):
    nb_pop = -1
    file_path = join(dataset_path, file_name)

    if exists(file_path):
        df = pd.read_csv(file_path)
        if 'ville_search' not in list(df.columns):
            df['ville_search'] = df['ville']
            df = convert_df_string_to_search_string(df, col_name='ville_search')
            df = df[['ville_search', 'ville', 'departement', 'region1', 'region2', 'pays']]
            df = df.sort_values('ville_search')
            df = df.drop_duplicates('ville_search')
            df.to_csv(join(dataset_path,file_name.replace(".csv", "-v2.csv")), index=False)
        dbConnection =_create_engine(verbose=verbose)
        try:
            nb_pop = df[['ville_search', 'ville', 'departement', 'region1', 'region2', 'pays']].to_sql(name='ville', con=dbConnection, if_exists='append', index=False)
        except mysql.connector.IntegrityError as error:
            nb_pop = 0
            if verbose > 0:
                print(f"[cojoden_dao > populate_villes] WARNING : la table est déjà peuplée.\n\t- {error}")
        except Exception as error:
            if  "IntegrityError" in str(error):
                nb_pop = 0
                if verbose > 0:
                    print(f"[cojoden_dao > populate_villes] WARNING : la table est déjà peuplée.\n\t- {error}")
            else:
                raise error
    return nb_pop

def populate_artistes(dataset_path, file_name=r'cojoden_artistes.csv', verbose=0):
    pass


# ----------------------------------------------------------------------------------
#                        PRIVATE
# ----------------------------------------------------------------------------------
def _executer_sql(self, sql, verbose=0):
    conn = None
    cur = None
    # Séparation des try / except pour différencier les erreurs
    try:
        conn = connecter()
        cur = conn.cursor()
        if verbose > 1:
            print("[cojoden_dao > execute] INFO : Base de données crée et correctement.")
        try:
            if verbose > 1 :
                print("[cojoden_dao > execute] INFO :", sql, end="")
            cur.execute(sql)
            conn.commit()
            if "INSERT" in sql:
                res = cur.lastrowid
            else:
                res = cur.fetchall()
            if verbose:
                print(" =>",res)

        except Exception as error:
            print("[cojoden_dao > execute] ERROR : Erreur exécution SQL", error)
            raise error
    except Exception as error:
        print("[cojoden_dao > execute] ERROR : Erreur de connexion à la BDD", error)
        raise error
    finally:
        try:
            if verbose > 1:
                print("[cojoden_dao > execute] DEBUG : Le curseur est fermé")
            cur.close()
        except Exception:
            pass
        try:
            if verbose > 1:
                print("[cojoden_dao > execute] DEBUG : La connexion est fermée")
            conn.close()
        except Exception:
            pass       
    return res

def _create_sql_url(verbose=0):
    connection_url = sa.engine.URL.create(
        drivername=db_client,
        username=db_user,
        password=db_pwd,
        host=db_host,
        database=db_name
    )
    if verbose > 1:
        print(f"[cojoden_dao > sql URL] DEBUG : {connection_url}")
    return connection_url

def _create_engine(verbose=0):
    # connect_args={'ssl':{'fake_flag_to_enable_tls': True}, 'port': 3306}
    connection_url = _create_sql_url(verbose=verbose)
    db_connection = sa.create_engine(connection_url, pool_recycle=3600) # ,connect_args= connect_args)
    return db_connection

# ----------------------------------------------------------------------------------
#                        MAIN
# ----------------------------------------------------------------------------------
if __name__ == '__main__':

    nb = populate_metiers(dataset_path=r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cojoden\dataset', file_name=r'cojoden_metiers_uniques.txt', verbose=1)
    print(nb)
    nb = populate_villes(dataset_path=r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cojoden\dataset')
    print(nb)
    nb = populate_musees(dataset_path=r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cojoden\dataset', file_name=r'cojoden_musees.csv')
    print(nb)
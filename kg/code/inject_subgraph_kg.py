from neo4j import GraphDatabase
import dotenv
import os
import pandas as pd


load_status = dotenv.load_dotenv('./FinDKG_dataset/Neo4j-45e8f176-Created-2026-01-23.txt')

if load_status is False:
    raise RuntimeError('Environment variables not loaded.')

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))


df = pd.read_csv('./FinDKG_dataset/kg_companies.txt', sep=': ')
df['Name'] = df['Name'].str.replace('"', '', regex=False)
df['Tickers'] = df['Tickers'].str.replace('"', '', regex=False).str.replace(',', '', regex=False)
TARGET_COMPANIES = df['Name'].tolist()

'''
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    print("Connection established.")

    files = ['train.txt', 'valid.txt', 'test.txt']

    for filename in files:
        current_path = 'DATA_DIR/filename'

        with open(current_path, 'r') as f:


    '''


def load_inject_data(driver) -> None:

    print('Chargement des données et mapping')

    df_entities = pd.read_csv('./FinDKG_dataset/entity2id.txt', sep='\t', header=None, names=['name', 'id', 'x', 'y'])
    id_to_name = dict(zip(df_entities['id'], df_entities['name']))
    name_to_id = dict(zip(df_entities['name'], df_entities['id']))

    df_relations = pd.read_csv('FinDKG_dataset/relation2id.txt', sep='\t', header=None, names=['name', 'id'])
    id_to_rel = dict(zip(df_relations['id'], df_relations['name']))

    df_timestamp = pd.read_csv('FinDKG_dataset/time2id.txt', sep=',', header=None, names = ['id', 'time'])
    id_to_date = dict(zip(df_timestamp['id'], df_timestamp['time']))

    target_ids = set()
    target_nodes = []

    for company_name in TARGET_COMPANIES:
        if company_name in name_to_id:
            cid = name_to_id[company_name]
            target_ids.add(cid)
            target_nodes.append({'name': company_name, 'id': int(cid)})
        else:
            print(f'{company_name} is not in the extract2id.txt')

    print(f"{len(target_nodes)} entreprises identifiées et prêtes à être injectées.")

    with driver.session() as session:
        # NETTOYAGE
        print("Nettoyage de la base Neo4j existante...")
        session.run("MATCH (n) DETACH DELETE n")

        files = ['train.txt', 'valid.txt', 'test.txt']

        # CRÉATION DES NŒUDS (Batch)
        print("Création des nœuds...")

        query_nodes = """
        UNWIND $batch AS row
        CREATE (c:Company {name: row.name, original_id: row.id})
        """
        session.run(query_nodes, batch=target_nodes)

        # CRÉATION DES RELATIONS

        DATA_DIR = "./FinDKG_dataset"

        for file in files:

            df_train = pd.read_csv(
                os.path.join(DATA_DIR, file), 
                sep='\t', 
                header=None, 
                names=['h', 'r', 't', 'ts', 'x']
            )
            
            relationships_to_create = []
            count = 0
            
            # Conversion en numpy array pour vitesse
            data = df_train[['h', 'r', 't', 'ts']].values
            
            for row in data:
                h, r, t, ts = row[0], row[1], row[2], row[3]
                
                # FILTRE : Seulement si H et T sont dans notre liste
                if h in target_ids and t in target_ids:
                    rel_name = id_to_rel.get(r, "RELATED_TO")
                    rel_type = str(rel_name).replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").upper()
                    
                    # Récupération de la date réelle (si dispo), sinon l'ID
                    date_str = str(id_to_date.get(ts, ts))
                    
                    relationships_to_create.append({
                        'h_id': int(h),
                        't_id': int(t),
                        'type': rel_type,
                        'date': date_str  # <-- On ajoute la date ici
                    })
                    count += 1
            
            print(f"   -> {count} relations qualifiées trouvées.")
            
            # D. INJECTION DES RELATIONS
            from collections import defaultdict
            rels_by_type = defaultdict(list)
            for item in relationships_to_create:
                rels_by_type[item['type']].append(item)
                
            for r_type, batch in rels_by_type.items():
                print(f"      Injection de {len(batch)} liens de type :{r_type}")
                
                # Note le changement dans la requête Cypher : on ajoute {date: row.date}
                query_rel = f"""
                UNWIND $batch as row
                MATCH (h:Company {{original_id: row.h_id}})
                MATCH (t:Company {{original_id: row.t_id}})
                MERGE (h)-[r:{r_type} {{date: row.date}}]->(t)
                """
                session.run(query_rel, batch=batch)

    print("Graphe FinDKG construit AVEC dates !")
    




if __name__ == "__main__":
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
        load_inject_data(driver)
    except Exception as e:
        print(f" Erreur : {e}")
        print("Vérifie que Neo4j est lancé et que le mot de passe est bon.")
import os
import requests
from googleapiclient.discovery import build


def search_images(query, num_images=1000, output_dir="dataset"):
    # Créer ou indique un repertoire en fonction de la recherche
    output_dir = os.path.join(output_dir, query)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Remplacer 'API_KEY' par votre clé d'API
    # Remplacer 'CSE_ID' par l'ID de votre moteur de recherche personnalisé
    api_key = 'API_KEY'
    cse_id = 'CSE_ID'

    service = build("customsearch", "v1", developerKey=api_key)

    # Détermine le nombre de pages nécessaires pour atteindre num_images
    num_pages = (num_images - 1) // 10 + 1

    all_items = []  # Stocke tous les résultats de toutes les pages

    for page in range(num_pages):
        start_index = page * 10 + 1

        res = service.cse().list(
            q=query,
            cx=cse_id,
            searchType='image',
            num=10,  # Nombre maximum d'images par page
            start=start_index
            # Index de départ pour la pagination (à modifier si l'on souhaite éviter d'itérer sur les mêmes résultats)
        ).execute()

        if 'items' in res:
            all_items.extend(res['items'])  # Fusionne les résultats de la page dans la liste complète

    # Télécharge les images
    for i, item in enumerate(all_items, 1):
        image_url = item['link']
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                # Sauvegarde l'image dans le dossier souhaité
                with open(os.path.join(output_dir, f"image_{i}.jpg"), "wb") as f:
                    f.write(response.content)
                print(f"Image {i} téléchargée avec succès.")
            else:
                print(f"Échec du téléchargement de l'image {i}.")
        except Exception as e:
            print(f"Erreur lors du téléchargement de l'image {i}: {e}")


if __name__ == "__main__":
    query = input("Entrez votre recherche : ")
    search_images(query, 100)

import socket
import numpy as np
import pickle

# Initialiser le serveur
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('127.0.0.1', 5065))  # L'adresse IP '0.0.0.0' écoute sur toutes les interfaces
server_socket.listen(1)

print("En attente de la connexion du client...")
client_socket, client_address = server_socket.accept()
print(f"Connexion établie avec {client_address}")

# Recevoir les données
data = b""
while True:
    packet = client_socket.recv(4096)
    if not packet:
        break
    data += packet

# Désérialiser les données reçues en un tableau NumPy
np_array = pickle.loads(data)
print("Données reçues :")
print(np_array)

# Fermer la connexion
client_socket.close()
server_socket.close()
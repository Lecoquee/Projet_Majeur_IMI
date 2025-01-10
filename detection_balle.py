import numpy as np
import cv2
from matplotlib import pyplot as plt
import socket
import json
import time

index_camera = 0
  # Vérifie que cet index correspond à la caméra que tu veux utiliser

# Ouvrir la caméra
cap = cv2.VideoCapture(index_camera)


coord_world_tags = [[[1200,0,290], [1200,0,130], [1360,0,130], [1360,0,290]],
                    [[940,0,550], [940,0,390], [1100,0,390], [1100,0,550]],
                    [[1100,0,130], [1100,0,290], [940,0,290], [940,0,130]]]


aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_parameters = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(dictionary=aruco_dict, detectorParams=aruco_parameters)

# Vérifier si la caméra a été correctement ouverte
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Créer une fenêtre nommée de manière explicite
cv2.namedWindow('Retour vidéo', cv2.WINDOW_NORMAL)

while True:
    # Lire une image (frame) depuis la caméra
    ret, frame = cap.read()
    corners, ids, rejectedPoints = aruco_detector.detectMarkers(frame)
    I_aruco = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    i1 = frame.shape[1]/2 # Largeur de l'image en pixel
    i2 = frame.shape[0]/2 # Hauteur de l'image en pixel
    # dict_tags = {'ind_tags':ids, 'coord_world_tags':coord_world_tags, 'coord_pixels_tags':corners}
    # U1 = []
    # A = []
    # for i in range(len(ids)):
    #     for j in range(4):
    #         line = []
    #         line.append((dict_tags['coord_pixels_tags'][i][0][j][1]-i2)*dict_tags['coord_world_tags'][i][j][0])
    #         line.append((dict_tags['coord_pixels_tags'][i][0][j][1]-i2)*dict_tags['coord_world_tags'][i][j][1])
    #         line.append((dict_tags['coord_pixels_tags'][i][0][j][1]-i2)*dict_tags['coord_world_tags'][i][j][2])
    #         line.append(dict_tags['coord_pixels_tags'][i][0][j][1]-i2)
    #         line.append(-(dict_tags['coord_pixels_tags'][i][0][j][0]-i1)*dict_tags['coord_world_tags'][i][j][0])
    #         line.append(-(dict_tags['coord_pixels_tags'][i][0][j][0]-i1)*dict_tags['coord_world_tags'][i][j][1])
    #         line.append(-(dict_tags['coord_pixels_tags'][i][0][j][0]-i1)*dict_tags['coord_world_tags'][i][j][2])
    #         A.append(line)
    #         U1.append(dict_tags['coord_pixels_tags'][i][0][j][0]-i1)
    print(ids)
    print(corners)
    # Convertir l'image en espace de couleur HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Définir les bornes pour la couleur jaune (typique d'une balle de tennis)
    lower_yellow = np.array([20, 125, 125])  # Plage de couleurs jaunes (en HSV)
    upper_yellow = np.array([40, 255, 255])

    # Créer un masque pour extraire la couleur jaune (balle de tennis)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Appliquer le masque à l'image originale
    yellow_objects = cv2.bitwise_and(frame, frame, mask=mask)

    # Convertir l'image masquée en niveaux de gris
    gray = cv2.cvtColor(yellow_objects, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou pour aider à la détection des cercles
    gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Détecter les cercles avec la méthode de Hough
    circles = cv2.HoughCircles(
        gray_blurred, 
        cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=100
    )

    # Vérifier si des cercles ont été détectés
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")  # Convertir les coordonnées en entiers
        
        # Dessiner les cercles détectés sur l'image originale
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)  # Dessiner le cercle en vert
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)  # Marquer le centre en orange

    if not ret:
        print("Erreur : Impossible de lire l'image depuis la caméra.")
        break

    # Afficher l'image dans la fenêtre
    cv2.imshow('Retour vidéo', frame)

    # Quitter la boucle si l'utilisateur appuie sur la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres correctement
cap.release()
cv2.destroyAllWindows()
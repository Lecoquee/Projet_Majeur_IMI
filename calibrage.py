import numpy as np
import math
import cv2 as cv
import socket
import json

################################################################################
# Configuration
################################################################################


def configure_system():
    """
    Configure les paramètres nécessaires au fonctionnement du programme.
    """
    config = {
        # Paramètres intrinsèques webcam
        "sensor_mm": np.array([3.58, 2.685]),
        "focal_mm": 4,
        "resolution": np.array([1280, 960]),

        # Webcam
        "id_cam1": 4,
        "id_cam2": 6,
        "id_cam3": 8,

        # Serveur
        "UDP_IP": "127.0.0.1",
        "UDP_PORT1": 5065,
        "UDP_PORT2": 5066,

        # Coordonnées coins tag arucos
        "coord_world_tags": np.array([[[1200,0,810], [1200,0,650], [1360,0,650], [1360,0,810]],
                                    [[1200,0,290], [1200,0,130], [1360,0,130], [1360,0,290]],
                                    [[1200,0,550], [1200,0,390], [1360,0,390], [1360,0,550]],
                                    [[1100,0,130], [1100,0,290], [940,0,290], [940,0,130]],
                                    [[940,290,0], [1100,290,0], [1100,130,0], [940,130,0]],
                                    [[1200,290,0], [1360,290,0], [1360,130,0], [1200,130,0]],
                                    [[1100,0,390], [1100,0,550], [940,0,550], [940,0,390]]], dtype=np.float32),
        
        "order_ids_tags": [0,4,5,6,7,8,9]
    }
    resolution = config["resolution"]
    center = (resolution[0] / 2, resolution[1] / 2)
    focal_mm = config["focal_mm"]
    sensor_mm = config["sensor_mm"]

    config["m_cam"] = np.array([
        [focal_mm * resolution[0] / sensor_mm[0], 0, center[0]],
        [0, focal_mm * resolution[1] / sensor_mm[1], center[1]],
        [0, 0, 1]
    ], dtype="double")

    return config

################################################################################
# Initialisation
################################################################################

def setup_udp_server():
    """
    Configure le serveur UDP pour la communication.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return sock


def initialize_cameras(config):
    """
    Initialise les caméras avec les paramètres spécifiés.
    """
    print('opening camera ', config["id_cam1"])
    cap1 = cv.VideoCapture(config["id_cam1"])
    print('opening camera ', config["id_cam2"])
    cap2 = cv.VideoCapture(config["id_cam2"])
    print('opening camera ', config["id_cam3"])
    cap3 = cv.VideoCapture(config["id_cam3"])

    resolution = config["resolution"]
    cap1.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap1.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap2.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap2.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap3.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap3.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])

    return cap1, cap2, cap3

################################################################################
# Toolbox
################################################################################

def calibrate(frame, config):
    """
    Calibre une image de la caméra pour trouver les paramètres extrinsèques.
    """
    # Détection de codes ArUco
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    aruco_parameters = cv.aruco.DetectorParameters()
    aruco_detector = cv.aruco.ArucoDetector(dictionary=aruco_dict, detectorParams=aruco_parameters)
    corners, ids, rejectedPoints = aruco_detector.detectMarkers(frame)
    I_aruco = cv.aruco.drawDetectedMarkers(frame, corners, ids)

    if ids is not None:
        if len(ids) == len(config["coord_world_tags"]):
            print(ids)
            # Associer chaque ID détecté à ses coins
            id_to_corners = {id[0]: corner for id, corner in zip(ids, corners)}

            # Trier selon l'ordre prédéfini
            sorted_ids = []
            sorted_corners = []

            for predefined_id in config["order_ids_tags"]:
                if predefined_id in id_to_corners:
                    sorted_ids.append(predefined_id)
                    sorted_corners.append(id_to_corners[predefined_id])
            print(sorted_ids)

            if len(sorted_corners) == len(corners):
                ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(config["coord_world_tags"], sorted_corners, (frame.shape[1],frame.shape[0]), config["m_cam"], None, flags=cv.CALIB_USE_INTRINSIC_GUESS)
                return ret, mtx, dist, rvecs, tvecs
    return 0, None, None, None, None
    
def get_transformation_matrix(config, rvecs, tvecs):
    """
    Calcule la matrice de transformation pour une caméra.
    """
    rotation, _ = cv.Rodrigues(rvecs[0])
    RT = np.zeros((3,4))
    RT[:3, :3] = rotation
    RT[:3, 3] = tvecs[0].transpose()
    projection_mtx = np.dot(config["m_cam"], RT)
    return projection_mtx

def get_transformation_matrix2(r1, t1, r2, t2):
    """
    Calcule la matrice de transformation entre deux caméras.
    """
    rm1, _ = cv.Rodrigues(r1)
    rm2, _ = cv.Rodrigues(r2)
    rm12 = np.dot(rm2, rm1.T)
    r12, _ = cv.Rodrigues(rm12)
    t12 = t2 - np.dot(rm12, t1)
    return r12, t12

def compute_camera2_from_camera1(r1, t1, r12, t12):
    """
    Calcule les paramètres extrinsèques de la caméra 2 en fonction de la caméra 1.
    """
    rm1, _ = cv.Rodrigues(r1)
    rm12, _ = cv.Rodrigues(r12)
    r2, _ = cv.Rodrigues(np.dot(rm12, rm1))
    t2 = np.dot(rm12, t1) + t12
    return r2, t2

def display_results(frame, coord_world_tags, r, t, m, d):
    """
    Affiche les résultats de la calibration en dessinant les projections des points.
    """
    for i in range(len(coord_world_tags)):
        projected_points, _ = cv.projectPoints(coord_world_tags[i], r[i], t[i], m, d)
        for j in range(4):
            cv.drawMarker(frame, (int(projected_points[j][0][0]), int(projected_points[j][0][1])), color=[0,0,255])

def ball_detection(frame):
    """
    Detecte la balle si elle apparait dans le champ de la camera et renvoie son centre.
    """
    ball_detected = False
    # Convertir l'image en espace de couleur HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Définir les bornes pour la couleur jaune (typique d'une balle de tennis)
    lower_yellow = np.array([0, 100, 100])  # Plage de couleurs jaunes (en HSV)
    upper_yellow = np.array([50, 255, 255])

    # Créer un masque pour extraire la couleur jaune (balle de tennis)
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    # Appliquer le masque à l'image originale
    yellow_objects = cv.bitwise_and(frame, frame, mask=mask)

    # Convertir l'image masquée en niveaux de gris
    gray = cv.cvtColor(yellow_objects, cv.COLOR_BGR2GRAY)

    threshold = 120
    gray[gray < threshold] = 0

    # Appliquer un flou pour aider à la détection des cercles
    gray_blurred = cv.GaussianBlur(gray, (15, 15), 0)

    # Détecter les cercles avec la méthode de Hough
    circles = cv.HoughCircles(
        gray_blurred, 
        cv.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=100
    )

    # Vérifier si des cercles ont été détectés
    if circles is not None:
        ball_detected = True
        circles = np.round(circles[0, :]).astype("int")  # Convertir les coordonnées en entiers
        largest_circle = circles[0]
        largest_rayon = circles[0][2]
        i = 1
        for i in range(len(circles)):
            if circles[i][2] > largest_rayon:
                largest_rayon = circles[i][2]
                largest_circle = circles[i]

        # Dessiner les cercles détectés sur l'image originale
        (x, y, r) = largest_circle
        cv.circle(frame, (x, y), r, (0, 255, 0), 4)  # Dessiner le cercle en vert
        cv.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)  # Marquer le centre en orange
        return ball_detected, np.array([largest_circle[0:2]], dtype=np.float32)
    else:
        return ball_detected, None

def triangulation(right_projection, left_projection, circle1, circle2, right_dist, left_dist):
    """
    Triangulation de la position de la balle.
    """
    right_undist = cv.undistortPoints(circle1, 
                    config["m_cam"],
                    right_dist,
                    None,
                    config["m_cam"])

    left_undist = cv.undistortPoints(circle2, 
                    config["m_cam"],
                    left_dist,
                    None,
                    config["m_cam"])

    # Transpose to get into OpenCV's 2xN format.
    left_points_t = np.array(left_undist[0]).transpose()
    right_points_t = np.array(right_undist[0]).transpose()

    triangulation = cv.triangulatePoints(left_projection, right_projection, left_points_t, right_points_t)
    homog_points = triangulation.transpose()
    euclid_points = cv.convertPointsFromHomogeneous(homog_points)
    print('POSITION: ', euclid_points)

    return euclid_points


################################################################################
# Calibration et boucle principale
################################################################################

def calibrate_cameras(cap1, cap2, cap3, config):
    """
    Effectue la calibration des deux caméras et calcule la matrice de transformation.
    """
    calibrated1 = False
    calibrated2 = False
    calibrated3 = False
    r12, t12 = None, None

    # Tant que les caméras sont ouvertes et non calibrées
    while (not calibrated1 or not calibrated2 or not calibrated3) and cap1.isOpened() and cap2.isOpened() and cap3.isOpened: 
        ret1, frame1 = cap1.read() 
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        if ret1 and ret2 and ret3:
            # Test calibration caméra 1
            if not calibrated1:
                ret11, right_mtx, right_dist, right_rvecs, right_tvecs = calibrate(frame1, config)
                if ret11!=0:
                    calibrated1 = True
                    display_results(frame1, config["coord_world_tags"], right_rvecs, right_tvecs, config["m_cam"], right_dist)
                cv.imshow('Camera 1', frame1)

            # Test calibration caméra 2
            if not calibrated2:
                ret22, left_mtx, left_dist, left_rvecs, left_tvecs = calibrate(frame2, config)
                if ret22!=0:
                    calibrated2 = True
                    display_results(frame2, config["coord_world_tags"], left_rvecs, left_tvecs, config["m_cam"], left_dist)
                cv.imshow('Camera 2', frame2)

            # Test calibration caméra 3
            if not calibrated3:
                ret33, mtx, dist, rvecs, tvecs = calibrate(frame3, config)
                if ret33!=0:
                    calibrated3 = True
                    display_results(frame3, config["coord_world_tags"], rvecs, tvecs, config["m_cam"], dist)
                cv.imshow('Camera 3', frame3)

        # Appuyer sur ECHAP pour arrêter la calibration
        if cv.waitKey(1) == 27:
            break
        
    return right_rvecs, right_tvecs, right_dist, left_rvecs, left_tvecs, left_dist, rvecs, tvecs

def main_loop(cap1, cap2, config, sock, right_rvecs, right_tvecs, right_dist, left_rvecs, left_tvecs, left_dist, rvecs, tvecs):
    """
    Boucle principale pour capturer les images et envoyer les paramètres via UDP.
    """
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 and ret2:
            # Vecteurs rotation, translation et matrice de rotation de la camera observatrice
            r12, t12 = get_transformation_matrix2(right_rvecs[0], right_tvecs[0], rvecs[0], tvecs[0])
            r2, t2 = compute_camera2_from_camera1(right_rvecs[0], right_tvecs[0], r12, t12)
            rm, _ = cv.Rodrigues(r2)

            # Matrices de projection des 2 cameras de calibrage pour la detection de la balle
            right_projection = get_transformation_matrix(config, right_rvecs, right_tvecs)
            left_projection = get_transformation_matrix(config, left_rvecs, left_tvecs)

            # Detection de la balle sur les 2 cameras
            ball1_detected, circle1 = ball_detection(frame1)
            ball2_detected, circle2 = ball_detection(frame2)

            if ball1_detected and ball2_detected:
                # Triangulation de la position de la balle
                euclid_points = triangulation(right_projection, left_projection, circle1, circle2, right_dist, left_dist)
                
                # Construction du message JSON
                print(euclid_points.tolist()[0][0])
                message = json.dumps({
                    'B': euclid_points.tolist()[0][0],
                })
                sock.sendto(message.encode(), (config["UDP_IP"], config["UDP_PORT1"]))

            message = json.dumps({
                    'M': config["m_cam"].reshape(-1).tolist(),
                    'R': r2.T.tolist()[0],
                    'T': t2.T.tolist()[0],
                    'F': rm[:, 2].tolist(),
                    'U': rm[:, 1].T.tolist(),
            })
            sock.sendto(message.encode(), (config["UDP_IP"], config["UDP_PORT2"]))

            cv.imshow('Camera 1', frame1)
            cv.imshow('Camera 2', frame2)

        # Appuyer sur ECHAP pour arrêter la calibration
        if cv.waitKey(1) == 27:
            break

################################################################################
# Programme principal
################################################################################

if __name__ == "__main__":
    config = configure_system()
    sock = setup_udp_server()
    cap1, cap2, cap3 = initialize_cameras(config)

    try:
        right_rvecs, right_tvecs, right_dist, left_rvecs, left_tvecs, left_dist, rvecs, tvecs = calibrate_cameras(cap1, cap2, cap3, config)
        print('FIN CALIBRAGE')
        cap3.release()
        main_loop(cap1, cap2, config, sock, right_rvecs, right_tvecs, right_dist, left_rvecs, left_tvecs, left_dist, rvecs, tvecs)
    finally:
        cap1.release()
        cap2.release()
        cv.destroyAllWindows()
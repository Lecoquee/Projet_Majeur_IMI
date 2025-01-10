import numpy as np
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

        # Serveur
        "UDP_IP": "127.0.0.1",
        "UDP_PORT": 5065,

        # Coordonnées coins tag arucos
        "coord_world_tags": np.array([[[940,0,650], [940,0,810], [1100,0,810], [1100,0,650]],
                            [[940,295,0], [1100,295,0], [1100,135,0], [940,135,0]],
                            [[1100,0,390], [1100,0,550], [940,0,550], [940,0,390]],
                            [[1200,0,550], [1200,0,390], [1360,0,390], [1360,0,550]],
                            [[1200,295,0], [1360,295,0], [1360,135,0], [1200,135,0]],
                            [[1100,0,130], [1100,0,290], [940,0,290], [940,0,130]],
                            [[1200,0,290], [1200,0,130], [1360,0,130], [1360,0,290]]], dtype=np.float32)
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
    # print('opening camera ', config["id_cam2"])
    # cap2 = cv.VideoCapture(config["id_cam2"])

    resolution = config["resolution"]
    cap1.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap1.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])
    # cap2.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
    # cap2.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])

    return cap1

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
    print(ids)

    if ids is not None:
        if len(ids) == len(config["coord_world_tags"]):
            ret, right_mtx, right_dist, right_rvecs, right_tvecs = cv.calibrateCamera(config["coord_world_tags"], corners, (frame.shape[1],frame.shape[0]), config["m_cam"], None, flags=cv.CALIB_USE_INTRINSIC_GUESS)
            return ret, right_mtx, right_dist, right_rvecs, right_tvecs
        else:
            return 0, None, None, None, None
    else:
        return 0, None, None, None, None
    
def get_transformation_matrix(r1, t1, r2, t2):
    """
    Calcule la matrice de transformation entre deux caméras.
    """
    rm1, _ = cv.Rodrigues(r1)
    rm2, _ = cv.Rodrigues(r2)
    rm12 = np.dot(rm2, rm1.T)
    r12, _ = cv.Rodrigues(rm12)
    t12 = t2 - np.dot(rm12, t1)
    return r12, t12

def display_results(frame, coord_world_tags, r, t, m, d):
    """
    Affiche les résultats de la calibration en dessinant les projections des points.
    """
    for i in range(len(coord_world_tags)):
        right_projected_points, _ = cv.projectPoints(coord_world_tags[i], r[i], t[i], m, d)
        for j in range(4):
            cv.drawMarker(frame, (int(right_projected_points[j][0][0]), int(right_projected_points[j][0][1])), color=[0,0,255])

def ball_detection(frame):
    # Convertir l'image en espace de couleur HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Définir les bornes pour la couleur jaune (typique d'une balle de tennis)
    lower_yellow = np.array([20, 125, 125])  # Plage de couleurs jaunes (en HSV)
    upper_yellow = np.array([40, 255, 255])

    # Créer un masque pour extraire la couleur jaune (balle de tennis)
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    # Appliquer le masque à l'image originale
    yellow_objects = cv.bitwise_and(frame, frame, mask=mask)

    # Convertir l'image masquée en niveaux de gris
    gray = cv.cvtColor(yellow_objects, cv.COLOR_BGR2GRAY)

    # Appliquer un flou pour aider à la détection des cercles
    gray_blurred = cv.GaussianBlur(gray, (15, 15), 0)

    # Détecter les cercles avec la méthode de Hough
    circles = cv.HoughCircles(
        gray_blurred, 
        cv.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=100
    )

    # Vérifier si des cercles ont été détectés
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")  # Convertir les coordonnées en entiers
        
        # Dessiner les cercles détectés sur l'image originale
        for (x, y, r) in circles:
            cv.circle(frame, (x, y), r, (0, 255, 0), 4)  # Dessiner le cercle en vert
            cv.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)  # Marquer le centre en orange

################################################################################
# Calibration et boucle principale
################################################################################

def calibrate_cameras(cap1, config):
    """
    Effectue la calibration des deux caméras et calcule la matrice de transformation.
    """
    calibrated = False
    r12, t12 = None, None

    while not calibrated and cap1.isOpened():
        ret1, frame1 = cap1.read()
        # ret2, frame2 = cap2.read()

        if ret1:
            ret1, right_mtx, right_dist, right_rvecs, right_tvecs = calibrate(frame1, config)
            # ret2, left_mtx, left_dist, left_rvecs, left_tvecs= calibrate(frame2, config)

            if ret1:
                # r12, t12 = get_transformation_matrix(right_rvecs, right_tvecs, left_rvecs, left_tvecs)
                calibrated = True

                # Affichage des résultats
                display_results(frame1, config["coord_world_tags"], right_rvecs, right_tvecs, config["m_cam"], right_dist)
                # display_results(frame2, config["coord_world_tags"], left_rvecs, left_tvecs, config["m_cam"], left_dist)

        cv.imshow('Camera 1', frame1)
        #cv.imshow('Camera 2', frame2)

        if cv.waitKey(1) == 27:
            break

    r12 = None
    t12 = None
    return right_rvecs, right_tvecs, right_dist

def main_loop(cap1, config, right_rvecs, right_tvecs, sock, right_dist):
    """
    Boucle principale pour capturer les images et envoyer les paramètres via UDP.
    """
    while cap1.isOpened():
        ret, frame = cap1.read()
        if ret:
            # ret, right_mtx, right_dist, right_rvecs, right_tvecs = calibrate(frame, config)
            # if ret:
                # r2, t2 = compute_camera2_from_camera1(r1, t1, r12, t12)
                #rm, _ = cv.Rodrigues(r2)

                # Construction du message JSON
                # message = json.dumps({
                #     'C': config["id_cam2"],
                #     'M': config["m_cam"].reshape(-1).tolist(),
                #     'R': r2.T.tolist()[0],
                #     'T': t2.T.tolist(),
                #     'F': rm[:, 2].tolist(),
                #     'U': rm[:, 1].T.tolist(),
                #     'I': ids.T.tolist()[0],
                #     'S': len(ids),
                # })
                # sock.sendto(message.encode(), (config["UDP_IP"], config["UDP_PORT"]))

            ball_detection(frame)

            display_results(frame, config["coord_world_tags"], right_rvecs, right_tvecs, config["m_cam"], right_dist)
            cv.imshow('Camera 1', frame)

        if cv.waitKey(1) == 27:
            break

################################################################################
# Programme principal
################################################################################

if __name__ == "__main__":
    config = configure_system()
    sock = setup_udp_server()
    cap1 = initialize_cameras(config)

    try:
        r12, t12, right_dist = calibrate_cameras(cap1, config)
        # cap2.release()
        main_loop(cap1, config, r12, t12, sock, right_dist)
    finally:
        cap1.release()
        cv.destroyAllWindows()
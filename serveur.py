import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import socket
import json
import time

################################################################################
# Configuration
################################################################################

def configure_system():
    """
    Configure les paramètres nécessaires au fonctionnement du programme.
    """
    config = {
        # Paramètres grille
        "CHECKERBOARD": (2, 2),
        "grid_mm": 27.16,

        # Paramètres intrinsèques webcam
        "sensor_mm": np.array([3.58, 2.685]),
        "focal_mm": 4,
        "resolution": np.array([1280, 960]),
        "distortion": np.zeros((4, 1)),

        # Webcam
        "id_cam1": 4,
        "id_cam2": 6,

        # Serveur
        "UDP_IP": "127.0.0.1",
        "UDP_PORT": 5065,

        # Critères pour la recherche des points
        "criteria": (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    }

    # Calcul du centre et de la matrice intrinsèque de la caméra
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

def initialize_calibration_points(config):
    """
    Initialise les points de la grille utilisés pour la calibration.
    """
    CHECKERBOARD = config["CHECKERBOARD"]
    grid_mm = config["grid_mm"]
    
    objpoints = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    
    # Génère des points sur les axes X et Z
    x, z = np.mgrid[0:CHECKERBOARD[0]*grid_mm:grid_mm, 0:CHECKERBOARD[1]*grid_mm:grid_mm]
    objpoints[:, 0] = x.T.reshape(-1)  # Coordonnées X
    objpoints[:, 2] = z.T.reshape(-1)  # Coordonnées Z

    return objpoints


def setup_udp_server(config):
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

    resolution = config["resolution"]
    cap1.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap1.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap2.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap2.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])

    return cap1, cap2

################################################################################
# Toolbox
################################################################################

def calibrate(frame, config, objpoints):
    """
    Calibre une image de la caméra pour trouver les paramètres extrinsèques.
    """

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    aruco_parameters = cv.aruco.DetectorParameters()
    aruco_detector = cv.aruco.ArucoDetector(dictionary=aruco_dict, detectorParams=aruco_parameters)
    corners, ids, rejectedPoints = aruco_detector.detectMarkers(gray)

    coord_world_tags = [[[1200,0,130], [1360,0,130], [1360,0,290], [1200,0,290]]]
    coord_world_tags = np.array(coord_world_tags,dtype=np.float32) # Coordonnées monde converties en float 32 pour la fonction opencv

    if ids is not None:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(coord_world_tags[0],corners[0],(frame.shape[1],frame.shape[0]),config["m_cam"],None,flags=cv.CALIB_USE_INTRINSIC_GUESS)
        #rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(corners, config["grid_mm"], config["m_cam"], config["distortion"])
        print("code arucos: ", corners, ids, rvecs, tvecs)
        return 1, corners, ids, rvecs[0][0], tvecs[0][0]
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


def compute_camera2_from_camera1(r1, t1, r12, t12):
    """
    Calcule les paramètres extrinsèques de la caméra 2 en fonction de la caméra 1.
    """
    rm1, _ = cv.Rodrigues(r1)
    rm12, _ = cv.Rodrigues(r12)
    r2, _ = cv.Rodrigues(np.dot(rm12, rm1))
    t2 = np.dot(rm12, t1) + t12
    return r2, t2


def display_results(frame, points, r, t, m, d):
    """
    Affiche les résultats de la calibration en dessinant les projections des points.
    """
    coord_world_tags = [[[1200,0,130], [1360,0,130], [1360,0,290], [1200,0,290]]]
    coord_world_tags = np.array(coord_world_tags,dtype=np.float32) 
    projected, _ = cv.projectPoints(coord_world_tags[0], r, t, m, d)
    print("PROJECTED", projected)
    # cv.polylines(frame, projected, True, (0,255,255))

################################################################################
# Calibration et boucle principale
################################################################################

def calibrate_cameras(cap1, cap2, config, objpoints):
    """
    Effectue la calibration des deux caméras et calcule la matrice de transformation.
    """
    calibrated = False
    r12, t12 = None, None

    while not calibrated and cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 and ret2:
            ret1, corners, ids, r1, t1 = calibrate(frame1, config, objpoints)
            print("CHANGE")
            ret2, corners, _, r2, t2= calibrate(frame2, config, objpoints)

            if ret1 and ret2:
                r12, t12 = get_transformation_matrix(r1, t1, r2, t2)
                calibrated = True

                # Affichage des résultats
                display_results(frame1, objpoints, r1, t1, config["m_cam"], config["distortion"])
                display_results(frame2, objpoints, r2, t2, config["m_cam"], config["distortion"])
                r2, t2 = compute_camera2_from_camera1(r1, t1, r12, t12)
                frame3=frame2
                display_results(frame3, objpoints, r2, t2, config["m_cam"], config["distortion"])
                cv.imshow('Computed', frame3)



        cv.imshow('Camera 1', frame1)
        cv.imshow('Camera 2', frame2)

        if cv.waitKey(1) == 27:
            break

    return r12, t12


def main_loop(cap1, config, objpoints, r12, t12, sock):
    """
    Boucle principale pour capturer les images et envoyer les paramètres via UDP.
    """
    while cap1.isOpened():
        ret, frame = cap1.read()
        if ret:
            ret, corners, ids, r1, t1 = calibrate(frame, config, objpoints)
            print(ids)
            if ret:
                r2, t2 = compute_camera2_from_camera1(r1, t1, r12, t12)
                rm, _ = cv.Rodrigues(r2)

                # Construction du message JSON
                message = json.dumps({
                    'C': config["id_cam2"],
                    'M': config["m_cam"].reshape(-1).tolist(),
                    'R': r2.T.tolist()[0],
                    'T': t2.T.tolist(),
                    'F': rm[:, 2].tolist(),
                    'U': rm[:, 1].T.tolist(),
                    'I': ids.T.tolist()[0],
                    'S': len(ids),
                })
                sock.sendto(message.encode(), (config["UDP_IP"], config["UDP_PORT"]))

                display_results(frame, objpoints, r1, t1, config["m_cam"], config["distortion"])
            cv.imshow('Camera 1', frame)

        if cv.waitKey(1) == 27:
            break

################################################################################
# Programme principal
################################################################################

if __name__ == "__main__":
    config = configure_system()
    objpoints = initialize_calibration_points(config)
    sock = setup_udp_server(config)
    cap1, cap2 = initialize_cameras(config)

    try:
        r12, t12 = calibrate_cameras(cap1, cap2, config, objpoints)
        cap2.release()
        main_loop(cap1, config, objpoints, r12, t12, sock)
    finally:
        cap1.release()
        cv.destroyAllWindows()
import os
import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt

def listar_ordenar(path):
    """
    Lista y ordena los archivos de un directorio.
    
    """
    files = os.listdir(path)
    files.sort()
    return files

def load_optical_flow(of_path, filename, target_size=(1440, 1080)):
    """
    Carga y reescala el optical flow ground truth de un archivo PNG en el dataset DSEC.
    
    Parámetros:
    - of_path: Ruta donde se encuentra el archivo PNG del optical flow.
    - filename: Nombre del archivo de optical flow.
    - target_size: Tamaño objetivo (ancho, alto) al que se reescalará el flujo óptico.
    
    Retorna:
    - flow_map: Optical flow con shape (2, target_size[1], target_size[0])
    """
    # Cargar la imagen PNG con flujo óptico
    flow_16bit = cv2.imread(os.path.join(of_path, filename), cv2.IMREAD_UNCHANGED)
    flow_16bit = cv2.cvtColor(flow_16bit, cv2.COLOR_BGR2RGB)
    # Separar los canales R, G y B
    flow_x = (flow_16bit[:, :, 0].astype(np.float32) - 2**15) / 128.0
    flow_y = (flow_16bit[:, :, 1].astype(np.float32) - 2**15) / 128.0
    valid_mask = flow_16bit[:, :, 2] > 0  # Píxeles válidos

    print(flow_x.shape)


    
    # Optical Flow final
    flow_map = np.stack((flow_x, flow_y), axis=0).astype(np.float32)  # Shape (2, target_size[1], target_size[0])
    
    return flow_x, flow_y, valid_mask.astype(bool)

def compute_depth_from_disparity(disparity_path, disparity_map_filename, Q):
    """
    Calcula la profundidad a partir de un mapa de disparidad usando reproyección en 3D.

    Parámetros:
    - disparity_path: Ruta donde se encuentra el mapa de disparidad.
    - disparity_map_filename: Nombre del archivo de disparidad.
    - Q: Matriz de reproyección estéreo.

    Retorna:
    - depth_map: Mapa de profundidad con los valores reproyectados.
    - valid_mask: Máscara de píxeles válidos (donde la disparidad es mayor a 0).
    """
    # Cargar la imagen de disparidad
    disparity = cv2.imread(os.path.join(disparity_path, disparity_map_filename), cv2.IMREAD_UNCHANGED)

    # Convertir disparidad a float y escalar
    disp_float = disparity.astype(np.float32) / 256.0

    # Máscara de píxeles válidos (donde la disparidad es mayor que 0)
    valid_mask = disparity > 0

    # Calcular la profundidad usando reproyección
    depth = cv2.reprojectImageTo3D(disp_float, Q)

    # Extraer solo la profundidad (tercera dimensión)
    depth_map = depth[:, :, 2]

    # Asignar 0 a los píxeles no válidos
    depth_map[~valid_mask] = 0

    return depth_map, valid_mask

def est_self_motion_pseudo_inverse(X, Y, Dx, Dy, f, InvZ):
    """
    Estima el movimiento propio usando el método de pseudo-inversa.
    
    Parámetros:
    - flow: Diccionario con claves 'X', 'Y', 'Dx' y 'Dy'.
    - f: Longitud focal de la cámara pinhole.
    - InvZ: Inversa de la profundidad Z.
    
    Retorna:
    - motion: Diccionario con claves 'Vel' (velocidad lineal) y 'Omega' (velocidad rotacional).
    """
    # Extraer datos del diccionario flow
    X = X.flatten()
    Y = Y.flatten()
    Dx = Dx.flatten()
    Dy = Dy.flatten()
    InvZ = InvZ.flatten()
    N = np.zeros_like(X)
    
    # All constraints form an over-determined linear equation system A x = B.
    A = np.vstack([
        np.column_stack([-f * InvZ, N, X * InvZ, X * Y / f, -(f**2 + X**2) / f, Y]),
        np.column_stack([N, -f * InvZ, Y * InvZ, (f**2 + Y**2) / f, -X * Y / f, -X])
    ])
    B = np.hstack([Dx, Dy])

    print(f"Shape de A: {A.shape}")
    print(f"Shape de B: {B.shape}")
    
    # We solve the system by taking a least-squares solution (pseudo-inverse).
    X = np.linalg.pinv(A) @ B
    
    # Do not apply forward motion and unit-speed constraint to Vel, because 
    # for given depths there is no ambiguity.
    Vel = X[:3]
    Omega = X[3:]
    
    return {'Vel': Vel, 'Omega': Omega}



if __name__ == '__main__':
    
    # Nombre de los directorios
    calibration_path = "./calibration"
    depth_path = "./depth"
    disparity_path = "./disparity_event"
    of_path = "./forward"
    gt_tensors_path = "./gt_tensors"
    rectified_images_path = "./images_rectified"
    mask_path = "./mask_tensors"

    # Nombre de los archivos
    of_timestamps = "forward_timestamps.txt"
    image_timestamps = "image_timestamps.txt"
    dispartiy_timestamps = "disparity_timestamps.txt"
    reprojection_matrix = "cam_to_cam.yaml"

    # Listar y ordenar los ficheros de los directorios
    depth_map_list = listar_ordenar(depth_path)
    disparity_map_list = listar_ordenar(disparity_path)
    of_list = listar_ordenar(of_path)
    gt_tensors_list = listar_ordenar(gt_tensors_path)
    rectified_images_list = listar_ordenar(rectified_images_path)

    # Cargar los archivos de los timestamps
    of_timestamps = np.loadtxt(of_timestamps, delimiter=',', dtype='int64')
    of_timestamps = of_timestamps[:, 1]
    image_timestamps = np.loadtxt(image_timestamps, dtype='int64')
    dispartiy_timestamps = np.loadtxt(dispartiy_timestamps, dtype='int64')

    # Cargar la matriz de reproyección Q
    with open(os.path.join(calibration_path, reprojection_matrix), "r") as file:
        data = yaml.safe_load(file)

    Q = np.array(data["disparity_to_depth"]["cams_12"])

    for i in range(1):
    #for i, file in enumerate(of_list):
        # timestamp para el que se va a obtener la gt
        timestamp = of_timestamps[i]

        #### Obtenemos los optical flow ground truth ####
        Dx, Dy, mask_of = load_optical_flow(of_path, of_list[i], (1440, 1080))
        print(f"Shape del optical flow de X: {Dx.shape}")


        # Indices para las imagenes y la disparidad que tienen el mismo timestamp
        #index_image = np.where(image_timestamps == timestamp)[0][0]
        index_disparity = np.where(dispartiy_timestamps == timestamp)[0][0]

        #Cargamos la profunidad
        depth, mask_d = compute_depth_from_disparity(disparity_path, disparity_map_list[index_disparity], Q)



        # Figure 1 with grayscale colormap
        fig1 = plt.figure()
        plt.imshow(depth, cmap='gray')
        plt.colorbar()
        plt.xlabel('X Label')
        plt.ylabel('Y Label')
        plt.title('Grayscale Colormap')

        plt.show()
        print(f"Shape de la profundidad: {depth.shape}")
        invZ = 1.0 / depth[mask_d]
        print(f"Shape de la inversa de la profundidad: {invZ.shape}")

        Dx = Dx[mask_d]
        Dy = Dy[mask_d]
        print(f"Shape de Dx: {Dx.shape}")

        # Cargamos la imagen rectificada
        # image_rectified = cv2.imread(os.path.join(rectified_images_path, rectified_images_list[index_image]), cv2.IMREAD_UNCHANGED)
        # print(f"Shape de la imagen rectificada: {image_rectified.shape}")

        XY = np.array(np.where(mask_d))
        print(f"Shape de XY: {XY.shape}")

        X = XY[0]
        Y = XY[1]
        print(f"Shape de X: {X.shape}") 
        
        ego_motion = est_self_motion_pseudo_inverse(X, Y, Dx, Dy, 1, invZ)
        print(ego_motion)

        


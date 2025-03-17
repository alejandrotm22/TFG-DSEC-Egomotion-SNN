import os
import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
import matplotlib.cm as cm
import imageio.v3 as iio

def listar_ordenar(path):
    """
    Lista y ordena los archivos de un directorio.
    
    """
    files = os.listdir(path)
    files.sort()
    return files

def load_optical_flow(of_path, filename):
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


def est_self_motion_given_depth_normal_flow(X, Y, Dx, Dy, inv_z: np.array):
    # Fetch data from flow structure.
    x = X
    y = Y
    u = Dx
    v = Dy
    inv_z = inv_z

    # Get the gradient direction.
    n = np.sqrt(u ** 2 + v ** 2)
    nx = u / n
    ny = v / n

    # All constraints form an over-determined linear equation system A x = B.
    a = np.column_stack([
        -nx * inv_z,
        -ny * inv_z,
        nx * x * inv_z + ny * y * inv_z,
        nx * x * y + ny * (1 + y ** 2),
        -nx * (1 + x ** 2) - ny * x * y,
        nx * y - ny * x
    ])

    b = n

    # Solve the system by taking a least-squares solution (pseudo-inverse).
    x_solution = np.linalg.pinv(a).dot(b)

    # Format data
    vel = x_solution[:3]
    omega = x_solution[3:]


    return {'Vel': vel, 'Omega': omega}

def disp_img_to_rgb_img(disp_array: np.ndarray):
    disp_pixels = np.argwhere(disp_array > 0)
    u_indices = disp_pixels[:, 1]
    v_indices = disp_pixels[:, 0]
    disp = disp_array[v_indices, u_indices]
    max_disp = 80

    norm = mpl.colors.Normalize(vmin=0, vmax=max_disp, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='inferno')

    disp_color = mapper.to_rgba(disp)[..., :3]
    output_image = np.zeros((disp_array.shape[0], disp_array.shape[1], 3))
    output_image[v_indices, u_indices, :] = disp_color
    output_image = (255 * output_image).astype("uint8")
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    return output_image


def generate_optical_flow_video(Dx, Dy, fps, filename='optical_flow.mp4', vmax=256.0):
    """
    Genera un video a partir del flujo óptico Dx y Dy.

    Parámetros:
        Dx: np.ndarray - Componente X del flujo óptico (frames, H, W)
        Dy: np.ndarray - Componente Y del flujo óptico (frames, H, W)
        fps: int - Fotogramas por segundo para el video
        filename: str - Nombre del archivo de salida
        vmax: float - Valor máximo para la normalización
    """
    n_frames, H, W = Dx.shape  # Obtener dimensiones del flujo óptico
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Definir códec de video
    out = cv2.VideoWriter(filename, fourcc, fps, (W, H), isColor=True)  # Inicializar el escritor de video

    for i in range(n_frames):
        # Calcular la magnitud y el ángulo del flujo óptico
        V = np.sqrt(Dx[i] ** 2 + Dy[i] ** 2) / vmax  # Normalizar magnitud en [0, 1]
        Theta = np.arctan2(Dy[i], Dx[i])  # Calcular ángulo del flujo óptico

        # Crear una imagen en el espacio de color LAB
        flow_Lab = np.zeros((H, W, 3), dtype=np.float32)
        flow_Lab[:, :, 0] = 100 * V  # Asignar la magnitud al canal de luminosidad
        flow_Lab[:, :, 1] = 127 * np.cos(Theta)  # Canal A basado en el ángulo
        flow_Lab[:, :, 2] = 127 * np.sin(Theta)  # Canal B basado en el ángulo

        # Convertir la imagen de LAB a RGB para visualización
        flow_RGB = cv2.cvtColor(flow_Lab, cv2.COLOR_LAB2RGB)
        flow_RGB = (flow_RGB * 255).astype(np.uint8)  # Escalar valores a rango [0, 255]

        # Establecer en negro los píxeles sin flujo óptico
        mask = (V == 0)
        flow_RGB[mask] = [0, 0, 0]

        out.write(flow_RGB)  # Escribir el frame en el video

    out.release()  # Liberar el escritor de video
    print(f"Video guardado como {filename}")

if __name__ == '__main__':
    
    # Nombre de los directorios
    calibration_path = "./calibration"
    depth_path = "./depth"
    disparity_path = "./disparity_event"
    of_path = "./forward"
    gt_tensors_path = "./gt_tensors"
    rectified_images_path = "./images_rectified"
    mask_path = "./mask_tensors"
    events_path = "./events/left"

    # Nombre de los archivos
    of_timestamps = "forward_timestamps.txt"
    image_timestamps = "image_timestamps.txt"
    dispartiy_timestamps = "disparity_timestamps.txt"
    reprojection_matrix = "cam_to_cam.yaml"
    rectify_map_file = "rectify_map.h5"

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

    # Cargar la matriz de rectificación
    rectmap_file = h5py.File(os.path.join(events_path, rectify_map_file))
    rectmap = rectmap_file['rectify_map'][()]

    print(f"El shape de rectmap es {rectmap.shape}")

    dx_frames = []
    dy_frames = []
    mask_optflow = []

    #for i in range(32, 33):
    for i, file in enumerate(of_list):
        print("--------------------------------")
        print(f"Estamos en la iteración: {i}")

        # timestamp para el que se va a obtener la gt
        timestamp = of_timestamps[i]

        #### Obtenemos los optical flow ground truth ####
        Dx, Dy, mask_of = load_optical_flow(of_path, of_list[i])

        # Código usado para generar el video
        # if i == 0: # Uso siempre la misma máscara por simplificar
        #     mask_optflow = mask_of
        # ux = Dx
        # yx = Dy
        # ux[~mask_of] = 0
        # yx[~mask_of] = 0
        # dx_frames.append(ux)
        # dy_frames.append(yx)

        print(f"Shape del optical flow de X: {Dx.shape}")
        mask_of_zeros = (Dx != 0.0) & (Dy != 0.0)



        # Indices para las imagenes y la disparidad que tienen el mismo timestamp
        index_disparity = np.where(dispartiy_timestamps == timestamp)[0][0]

        #Cargamos la profunidad
        depth, mask_d = compute_depth_from_disparity(disparity_path, disparity_map_list[index_disparity], Q)
        # Unir las máscaras con una operación lógica AND
        mask_combined = mask_d & mask_of
        mask_combined = mask_combined & mask_of_zeros
        #Codigo para mostrar las imagenes de profundidad
        # cv2.imshow("imagen",disp_img_to_rgb_img(depth))
        # cv2.waitKey(0)

        print(f"Shape de la profundidad: {depth.shape}")
        invZ = 1.0 / depth[mask_combined]
        print(f"Shape de la inversa de la profundidad: {invZ.shape}")

        Dx = Dx[mask_combined]
        Dy = Dy[mask_combined]

        if np.any(Dx == 0.0) and np.any(Dy == 0.0):
            print("El optical flow contiene valores 0.0")
        else:
            print("No hay valores 0.0 en el optical flow")
        print(f"Shape de Dx: {Dx.shape}")

        coordenadas = np.array(np.where(mask_combined))
        X = rectmap[coordenadas[0], coordenadas[1], 0]
        Y = rectmap[coordenadas[0], coordenadas[1], 1]
        print(f"Los valores unique de X es de: {np.unique(X)}")
        print(f"Los valores unique de Y es de: {np.unique(Y)}")


        ego_motion = est_self_motion_given_depth_normal_flow(X, Y, Dx, Dy, invZ)
        print(ego_motion)

    # Continuación del código usado para generar el vídeo
    # dx_frames = np.stack(dx_frames, axis=0)
    # dy_frames = np.stack(dy_frames, axis=0)
    # generate_optical_flow_video(dx_frames, dy_frames, 11)


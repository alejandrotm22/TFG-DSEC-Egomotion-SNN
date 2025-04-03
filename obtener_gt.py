import os
import cv2
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
import matplotlib.cm as cm
import imageio.v3 as iio
import pandas

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
    valid_mask = flow_16bit[:, :, 2].astype(bool)  # Píxeles válidos
    
    # Optical Flow final
    flow_map = np.stack((flow_x, flow_y), axis=0).astype(np.float32)  # Shape (2, target_size[1], target_size[0])
    
    return flow_x, flow_y, valid_mask

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
    # Directorio global
    global_path = "./zurich_city_02_d"
    # Nombre de los directorios
    calibration_path = os.path.join(global_path, "calibration")
    disparity_path = os.path.join(global_path, "disparity_event")
    of_path = os.path.join(global_path, "flow", "forward")
    events_path = os.path.join(global_path, "events", "left")

    # Nombre de los archivos
    of_timestamps = os.path.join(global_path, "forward_timestamps.txt")
    dispartiy_timestamps = os.path.join(global_path, "disparity_timestamps.txt")
    calibration = "cam_to_cam.yaml"
    rectify_map_file = "rectify_map.h5"

    # Listar y ordenar los ficheros de los directorios
    disparity_map_list = listar_ordenar(disparity_path)
    of_list = listar_ordenar(of_path)

    # Cargar los archivos de los timestamps
    of_timestamps = np.loadtxt(of_timestamps, delimiter=',', dtype='int64')
    of_timestamps = of_timestamps[:, 1]
    dispartiy_timestamps = np.loadtxt(dispartiy_timestamps, dtype='int64')

    # Cargar la matriz de reproyección Q
    with open(os.path.join(calibration_path, calibration), "r") as file:
        data = yaml.safe_load(file)

    Q = np.array(data["disparity_to_depth"]["cams_03"])

    # # Cargar la matriz de rectificación del propio dataset
    # rectmap_file = h5py.File(os.path.join(events_path, rectify_map_file))
    # rectmap = rectmap_file['rectify_map'][()]
    # print(f"El shape de rectmap es {rectmap.shape}")

    ### Crear la matriz de rectificación a partir de los datos de calibración del dataset ###
    camera_matrix = np.array([
        [data["intrinsics"]["cam0"]["camera_matrix"][0],             0.0, data["intrinsics"]["cam0"]["camera_matrix"][2]],
        [            0.0, data["intrinsics"]["cam0"]["camera_matrix"][1], data["intrinsics"]["cam0"]["camera_matrix"][3]],
        [            0.0,             0.0,             1.0]
    ])

    distortion_coeffs = np.array(data["intrinsics"]["cam0"]["distortion_coeffs"])

    # Compute undistorted image plane.
    dist_x_map, dist_y_map = np.meshgrid(np.arange(data["intrinsics"]["cam0"]["resolution"][0], dtype=np.float32),
                                         np.arange(data["intrinsics"]["cam0"]["resolution"][1], dtype=np.float32))
    dist_x_map    = dist_x_map.flatten()
    dist_y_map    = dist_y_map.flatten()
    dist_xy_map   = np.stack((dist_x_map[:, np.newaxis], dist_y_map[:, np.newaxis]), axis=2)

    undist_xy_map = cv2.undistortPoints(dist_xy_map, camera_matrix, distortion_coeffs)
    undist_x_map  = undist_xy_map[:, :, 0].flatten()
    undist_y_map  = undist_xy_map[:, :, 1].flatten()

    # Compute velocity coeffs.
    vel_coeffs = np.zeros((len(undist_x_map), 2, 3), dtype=np.float32)

    vel_coeffs[:, 0, 0] = -1.0
    vel_coeffs[:, 1, 0] = 0.0

    vel_coeffs[:, 0, 1] = 0.0
    vel_coeffs[:, 1, 1] = -1.0

    vel_coeffs[:, 0, 2] = undist_x_map
    vel_coeffs[:, 1, 2] = undist_y_map

    # Compute omega coeffs.
    omega_coeffs = np.zeros((len(undist_x_map), 2, 3), dtype=np.float32)

    omega_coeffs[:, 0, 0] = undist_x_map * undist_y_map
    omega_coeffs[:, 1, 0] = 1.0 + undist_y_map ** 2

    omega_coeffs[:, 0, 1] = -(1.0 + undist_x_map ** 2)
    omega_coeffs[:, 1, 1] = -(undist_x_map * undist_y_map)

    omega_coeffs[:, 0, 2] = undist_y_map
    omega_coeffs[:, 1, 2] = -undist_x_map

    undist_x_map = undist_x_map.reshape(480, 640)
    undist_y_map = undist_y_map.reshape(480, 640)

    print(f"Undistorted image plane shape: {undist_y_map.shape}")
    print(f"Undistorted image plane shape: {undist_x_map.shape}")

    # Necesario para crear el video
    # dx_frames = []
    # dy_frames = []
    # mask_optflow = []
    #
    # undist_x_frames = []
    # undist_y_frames = []

    # Necesario para crear los gráficos
    # vel_x, vel_y, vel_z = [], [], []
    # omega_x, omega_y, omega_z = [], [], []
    # plt.ion()  # Habilitar el modo interactivo de Matplotlib
    # fig, axs = plt.subplots(2, 1, figsize=(10, 6))  # Dos subgráficos: uno para Vel y otro para Omega

    # for i in range(32, 33):
    for i, file in enumerate(of_list):
        print("--------------------------------")
        print(f"Estamos en la iteración: {i}")

        # timestamp para el que se va a obtener la gt
        timestamp = of_timestamps[i]

        #### Obtenemos los optical flow ground truth y su máscara ####
        Dx, Dy, mask_of = load_optical_flow(of_path, of_list[i])
        print(f"Shape del optical flow de X: {Dx.shape}")

        # Creo la mascara de donde X e Y en el optical flow son 0's
        # mask_of_zeros = (Dx != 0.0) & (Dy != 0.0)

        # Indices para las imagenes y la disparidad que tienen el mismo timestamp
        index_disparity = np.where(dispartiy_timestamps == timestamp)[0][0]

        #Cargamos la profunidad
        depth, mask_d = compute_depth_from_disparity(disparity_path, disparity_map_list[index_disparity], Q)

        # Unir las máscaras de disparidad, optical flow y ceros
        mask_combined = mask_d & mask_of

        # #Codigo para mostrar las imagenes de profundidad
        # cv2.imshow("imagen",disp_img_to_rgb_img(depth))
        # cv2.waitKey(0)

        # Código usado para generar el video
        # if i == 0:  # Uso siempre la misma máscara por simplificar
        #     mask_optflow = mask_of
        # ux = Dx
        # yx = Dy
        # ux[~mask_combined] = 0
        # yx[~mask_combined] = 0
        # dx_frames.append(ux)
        # dy_frames.append(yx)

        # Obtener la inversa de la profundidad
        print(f"Shape de la profundidad: {depth.shape}")
        invZ = 1.0 / depth[mask_combined]
        print(f"Shape de la inversa de la profundidad: {invZ.shape}")

        # Le pasamos la máscara al óptical flow
        Dx = Dx[mask_combined]
        Dy = Dy[mask_combined]

        # Código para obtener X e Y a partir de la matriz rectificada
        # coordenadas = np.array(np.where(mask_combined))
        # X = (rectmap[coordenadas[0], coordenadas[1], 0] - 335.0999870300293) / 569.7632987676102
        # Y = (rectmap[coordenadas[0], coordenadas[1], 1] - 221.23667526245117) / 569.7632987676102

        # Le pasamos la máscara a los valores de X e Y rectificados
        X = undist_x_map[mask_combined]
        Y = undist_y_map[mask_combined]

        print(f"Los valores unique de X es de: {np.unique(X)}")
        print(f"Los valores unique de Y es de: {np.unique(Y)}")

        # Calculamos el ego motion a partir de los datos
        ego_motion = est_self_motion_pseudo_inverse(X, Y, Dx, Dy, 1, invZ)
        print(ego_motion)

        inv_z = invZ.reshape(-1,1,1)
        A = vel_coeffs[mask_combined.flatten(), :, :]
        B = omega_coeffs[mask_combined.flatten(), :, :]
        vel = ego_motion['Vel']
        omega = ego_motion['Omega']
        dt = 1

        undist_uv = inv_z * A @ vel + B @ omega
        undist_u = undist_uv[:, 0]
        undist_v = undist_uv[:, 1]

        of_x = np.zeros_like(mask_combined, dtype=np.float32)
        of_x[mask_combined] = undist_u

        of_y = np.zeros_like(mask_combined, dtype=np.float32)
        of_y[mask_combined] = undist_v

        # Código necesario para crear el video del optical flow reconstruido
        # undist_x_frames.append(of_x)
        # undist_y_frames.append(of_y)

        print(f"El shape de undist_u es: {undist_u.shape}")


        # Extraer valores de Vel y Omega
    #     vel_x.append(ego_motion['Vel'][0])
    #     vel_y.append(ego_motion['Vel'][1])
    #     vel_z.append(ego_motion['Vel'][2])
    #
    #     omega_x.append(ego_motion['Omega'][0])
    #     omega_y.append(ego_motion['Omega'][1])
    #     omega_z.append(ego_motion['Omega'][2])
    #
    #     # Limpiar y actualizar el gráfico
    #     axs[0].cla()
    #     axs[0].plot(vel_x, label='Vel X', color='r')
    #     axs[0].plot(vel_y, label='Vel Y', color='g')
    #     axs[0].plot(vel_z, label='Vel Z', color='b')
    #     axs[0].set_title("Velocidad (Vel)")
    #     axs[0].legend()
    #     axs[0].set_ylabel("Valor")
    #
    #     axs[1].cla()
    #     axs[1].plot(omega_x, label='Omega X', color='r')
    #     axs[1].plot(omega_y, label='Omega Y', color='g')
    #     axs[1].plot(omega_z, label='Omega Z', color='b')
    #     axs[1].set_title("Velocidad Angular (Omega)")
    #     axs[1].legend()
    #     axs[1].set_ylabel("Valor")
    #
    #     plt.pause(0.1)  # Pausar para visualizar los cambios en tiempo real
    #
    # plt.ioff()  # Desactivar el modo interactivo
    # plt.show()  # Mostrar la gráfica final

    # Continuación del código usado para generar el vídeo
    # dx_frames = np.stack(dx_frames, axis=0)
    # dy_frames = np.stack(dy_frames, axis=0)
    # generate_optical_flow_video(dx_frames, dy_frames, 11)
    #
    # undist_x_frames = np.stack(undist_x_frames, axis=0)
    # undist_y_frames = np.stack(undist_y_frames, axis=0)
    # generate_optical_flow_video(undist_x_frames, undist_y_frames, 11, 'undistorted_of.mp4')


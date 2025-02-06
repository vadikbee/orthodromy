#################################    ПОЯСНЕНИЕ    #################################

#################################    сначала передается долгота, а потом широта (реверснуты) т.е где широта --> долгота    #################################
from pymavlink import mavutil
from flask import Flask, request, jsonify, render_template
from pyproj import Geod
import rasterio
from rasterio.warp import transform
from flask_cors import CORS
import logging
import os


# Создаем экземпляр Flask приложения
app = Flask(__name__, static_folder='static')

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,  # Уровень логирования DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',  # Формат вывода
    handlers=[logging.StreamHandler()]  # Вывод в консоль
)

# Дополнительно настраиваем логирование для werkzeug
werkzeug_log = logging.getLogger('werkzeug')
werkzeug_log.setLevel(logging.DEBUG)  # Устанавливаем уровень логирования для werkzeug
werkzeug_log.addHandler(logging.StreamHandler())  # Добавляем обработчик для вывода в консоль

# Создание подключения
master = mavutil.mavlink_connection('udp:localhost:14550')  # Замените на нужный адрес

# Разрешаем CORS для этого приложения
CORS(app, resources={r"/*": {"origins": "*"}})  # Разрешить все домены
# Включаем режим отладки
app.debug = True

# Папка для сохранения файлов в проекте (папка static)
app.config['UPLOAD_FOLDER'] = 'static'  # Папка для сохранения файлов




if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    logging.info("Папка для сохранения файлов была создана.")
else:
    logging.info("Папка для сохранения файлов уже существует.")


# Убедитесь, что папка существует
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Инициализация геоида WGS84 для вычисления ортодромии123
geod = Geod(ellps="WGS84")





# Маршрут для отображения главной HTML-страницы
@app.route('/', methods=['GET'])
def index():
    return render_template("zad.html")  # Отображаем HTML-шаблон





# Функция для проверки пересечения маршрута с запрещенными зонами
def check_intersections(route_coords, restricted_areas):
    route_line = LineString(route_coords)  # Создаем линию маршрута из координат
    intersections = []  # Список для хранения точек пересечения

    for i, area in enumerate(restricted_areas):
        if area.geom_type == 'Polygon':
            buffered_area = area  # Используем полигон напрямую
        elif area.geom_type == 'Point':
            buffered_area = area.buffer(area.radius)  # Используем радиус для круга
        else:
            continue  # Игнорируем неподдерживаемые типы

        # Проверка на пересечение маршрута с зоной
        if route_line.intersects(buffered_area):
            intersection = route_line.intersection(buffered_area)  # Получаем точки пересечения
            logging.info(f"Intersection found with restricted area {i}: {intersection}")

            # Обработка точек пересечения
            if isinstance(intersection, (LineString, Polygon)):
                for point in intersection.coords:
                    intersections.append([point[0], point[1]])  # Добавляем каждую точку пересечения
            elif not intersection.is_empty:
                intersections.append([intersection.x, intersection.y])  # Добавляем точку пересечения

    return intersections  # Возвращаем список точек пересечения


# Функция для проверки расстояния от точки до запрещенных зон
def is_too_close_to_restricted_area(point, restricted_areas, min_distance=0.01):
    for area in restricted_areas:
        if area.geom_type == 'Polygon':
            if point.distance(area.exterior) < min_distance:
                return True
    return False


# Функция для проверки, слишком ли близок маршрут к запрещенным зонам
def is_route_too_close_to_restricted_areas(route, restricted_areas, threshold=0.001):  # threshold in degrees
    route_line = LineString(route)  # Преобразуем маршрут в LineString
    for area in restricted_areas:
        if route_line.distance(area) < threshold:  # Проверяем расстояние до каждой запрещенной зоны
            return True
    return False


# Обновленная функция маршрута для обработки POST-запроса и расчета ортодромии или прямой линии
@app.route('/orthodrome', methods=['POST'])
def orthodrome():
    try:
        data = request.json
        logging.debug(f"Received data: {data}")  # Логируем полученные данные

        start_point = data.get('start_point')
        end_point = data.get('end_point')
        num_nodes = data.get('num_nodes', 1000)
        is_orthodrome = data.get('orthodrome', True)
        restricted_areas = data.get('restricted_areas', [])
        speed = data.get('speed')  # Получаем скорость
        altitude = data.get('altitude')  # Получаем высоту

        logging.debug(f"Start point: {start_point}, End point: {end_point}, Restricted areas: {restricted_areas}")

        # Проверка входных данных
        if not (isinstance(start_point, list) and len(start_point) == 2):
            logging.error("Invalid start_point")
            return jsonify({"error": "Invalid start_point"}), 400
        if not (isinstance(end_point, list) and len(end_point) == 2):
            logging.error("Invalid end_point")
            return jsonify({"error": "Invalid end_point"}), 400
        if not isinstance(num_nodes, int) or num_nodes < 2:
            logging.error("Invalid num_nodes")
            return jsonify({"error": "Invalid num_nodes"}), 400

        # Проверка формата restricted_areas
        if not isinstance(restricted_areas, list) or any(not isinstance(area, list) for area in restricted_areas):
            logging.error("Invalid restricted_areas format")
            return jsonify({"error": "Invalid restricted_areas format"}), 400

        # Преобразуем зоны запрета в объекты Polygon
        try:
            restricted_areas = [Polygon([(lat, lon) for lon, lat in area]) for area in restricted_areas if len(area) > 1]

            logging.debug(f"Restricted areas as Polygons: {restricted_areas}")
        except Exception as e:
            logging.error(f"Error occurred while creating polygons: {str(e)}")
            return jsonify({"error": f"Invalid restricted_areas: {str(e)}"}), 400

        lat1, lon1 = start_point
        lat2, lon2 = end_point

        # Создаем маршрут
        if is_orthodrome:
            points = geod.npts(lon1, lat1, lon2, lat2, num_nodes + 2000)
        else:
            lons = [lon1 + (lon2 - lon1) * i / (num_nodes - 1) for i in range(num_nodes)]
            lats = [lat1 + (lat2 - lat1) * i / (num_nodes - 1) for i in range(num_nodes)]
            points = zip(lons, lats)

        coordinates = [(lon1, lat1)] + list(points) + [(lon2, lat2)]

        # Проверка близости к запрещенным зонам
        if is_route_too_close_to_restricted_areas(coordinates, restricted_areas):
            return jsonify({"error": "Маршрут слишком близок к запрещенной зоне!"}), 400

        return jsonify({
            "coordinates": coordinates,
            "warning": None,
            "speed": speed,  # Возвращаем скорость
            "altitude": altitude  # Возвращаем высоту
        })

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500




################################################################################ обход зон запрета (расчет)
from shapely.geometry import Point, LineString, Polygon
import logging
import random

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def get_nearest_boundary_point(inter_point, area):
    """
    Находит ближайшую точку на границе зоны к точке пересечения.
    """
    inter_point_coords = inter_point.coords[0] if inter_point.geom_type == 'Point' else list(inter_point.coords)[0]
    area_boundary = area.exterior.coords[:-1]
    return min(area_boundary, key=lambda p: Point(inter_point_coords).distance(Point(p)))


def interpolate_points(start, end, num_points=5):
    """
    Интерполирует точки между двумя координатами для получения пути.
    """
    return [
        (
            start[0] + (end[0] - start[0]) * i / num_points,
            start[1] + (end[1] - start[1]) * i / num_points
        )
        for i in range(1, num_points + 1)
    ]


def find_shortest_path_with_restrictions(start_point, end_point, restricted_areas):
    """
    Находит кратчайший путь с учетом зон запрета, обходя их с адаптивным смещением и проверкой возможности вернуться на прямую траекторию после обхода.
    """
    path = [start_point]
    current_point = Point(start_point)
    max_iterations = 2000  # Ограничение на количество итераций
    buffer_distance = 0.006  # Начальное расстояние буфера вокруг запретной зоны
    visited_points = set()  # Множество для отслеживания посещённых точек

    # Создаём буферные зоны для всех запретных зон
    buffered_areas = [area.buffer(buffer_distance) for area in restricted_areas]

    for iteration_count in range(max_iterations):
        print(f"Итерация {iteration_count + 1}: Текущая точка - {current_point}")

        # Создаем линию маршрута от текущей точки до конечной
        direct_path = LineString([current_point, end_point])

        # Проверяем, пересекает ли маршрут какие-либо буферные зоны запрета
        intersects = any(direct_path.intersects(buffered_area) for buffered_area in buffered_areas)

        if not intersects:
            # Если маршрут не пересекает буферные зоны, добавляем конечную точку и выходим из цикла
            path.append(end_point)
            return path

        # Если есть пересечение, обходим буферные зоны
        for area, buffered_area in zip(restricted_areas, buffered_areas):
            if direct_path.intersects(buffered_area):
                # Находим первую точку пересечения с буферной зоной
                intersection = direct_path.intersection(buffered_area)

                if intersection.is_empty:
                    continue

                # Получаем координаты первой точки пересечения
                inter_point = list(intersection.coords)[0]
                nearest_boundary_point = get_nearest_boundary_point(Point(inter_point), buffered_area)

                # Логируем информацию о пересечении
                print(f"Пересечение с запретной зоной. Точка пересечения: {inter_point}")
                print(f"Ближайшая точка на границе: {nearest_boundary_point}")

                # Если точка уже была посещена, увеличиваем буфер и пытаемся сместиться
                if nearest_boundary_point in visited_points:
                    buffer_distance += 0.0001  # Увеличиваем буферное расстояние
                    buffered_areas = [area.buffer(buffer_distance) for area in restricted_areas]
                    print(f"Зацикливание обнаружено. Увеличиваем буфер до {buffer_distance} и пробуем сместиться.")

                    # Случайно выбираем противоположную точку на границе, чтобы изменить направление
                    random_boundary_point = random.choice(buffered_area.exterior.coords)
                    path.append(random_boundary_point)
                    current_point = Point(random_boundary_point)
                    visited_points.add(random_boundary_point)
                    break

                # Если точка не была посещена, добавляем её к маршруту
                path.append(nearest_boundary_point)
                visited_points.add(nearest_boundary_point)
                current_point = Point(nearest_boundary_point)

                # После обхода зоны создаем новую прямую к конечной точке
                direct_path_after_avoidance = LineString([current_point, end_point])

                # Проверяем, пересекает ли новая прямая с конечной точкой какие-либо запретные зоны
                if not any(direct_path_after_avoidance.intersects(buffered_area) for buffered_area in buffered_areas):
                    # Если прямая не пересекает запретные зоны, добавляем конечную точку
                    path.append(end_point)
                    return path

                # Если новая прямая снова пересекает зону, продолжаем обход вдоль границы
                parallel_offset = 0.00001  # Смещение параллельно границе
                offset_point = Point(nearest_boundary_point).buffer(parallel_offset).exterior.coords[0]
                path.append(offset_point)
                current_point = Point(offset_point)
                break  # Переходим к следующей итерации цикла

    # Если не удалось достичь конечной точки, сообщаем о проблеме
    print("Не удалось найти маршрут, обходящий зоны запрета. Попробуйте изменить начальную или конечную точку.")
    return path


@app.route('/orthodrome_with_restrictions', methods=['POST'])
def orthodrome_with_restrictions():
    try:
        data = request.json
        logging.debug(f"Received data: {data}")

        start_point = data.get('start_point')
        end_point = data.get('end_point')
        restricted_areas = data.get('restricted_areas', [])

        # Проверка корректности входных данных
        if not (isinstance(start_point, list) and len(start_point) == 2):
            return jsonify({"error": "Invalid start_point"}), 400
        if not (isinstance(end_point, list) and len(end_point) == 2):
            return jsonify({"error": "Invalid end_point"}), 400

        # Преобразуем зоны запрета в объекты Polygon
        restricted_areas = [Polygon(area) for area in restricted_areas]
        logging.debug(f"Restricted areas: {restricted_areas}")

        # Поиск кратчайшего пути с учетом запретных зон
        shortest_path = find_shortest_path_with_restrictions(start_point, end_point, restricted_areas)
        logging.debug(f"Shortest path: {shortest_path}")

        if shortest_path:
            return jsonify({
                "coordinates": shortest_path,
                "message": "Маршрут успешно построен с обходом зон запрета."
            })
        else:
            return jsonify({
                "coordinates": None,
                "message": "Не удалось найти путь, обходящий зоны запрета."
            })

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

#---------------------------------------------- KMZ --------------------------------------------
import zipfile
import xml.etree.ElementTree as ET
import os



@app.route('/create_kmz', methods=['POST'])
def create_kmz():
    try:
        # Получаем координаты маршрута из запроса
        data = request.json
        coordinates = data.get("coordinates", [])

        # Если координаты пустые, возвращаем ошибку
        if not coordinates:
            return jsonify({"error": "Нет данных для маршрута"}), 400

        # Создаем KML структуру
        kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
        document = ET.SubElement(kml, "Document")
        placemark = ET.SubElement(document, "Placemark")
        name = ET.SubElement(placemark, "name")
        name.text = "Маршрут"
        line_string = ET.SubElement(placemark, "LineString")
        coordinates_element = ET.SubElement(line_string, "coordinates")

        # Добавляем координаты маршрута в KML
        coords_text = ' '.join([f"{coord[1]},{coord[0]},0" for coord in coordinates])  #  широта, высота (0)
        coordinates_element.text = coords_text

        # Создаем дерево XML и сохраняем его как KML файл
        kml_tree = ET.ElementTree(kml)
        kml_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'route.kml')
        kml_tree.write(kml_file_path)

        # Теперь создаем KMZ архив с этим KML файлом
        kmz_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'route.kmz')
        with zipfile.ZipFile(kmz_file_path, 'w', zipfile.ZIP_DEFLATED) as kmz:
            kmz.write(kml_file_path, 'doc.kml')  # Добавляем KML в KMZ архив

        # Удаляем временный KML файл
        os.remove(kml_file_path)

        return jsonify({"message": "Маршрут сохранен в формате KMZ", "kmz_url": '/static/route.kmz'})
    except Exception as e:
        logging.error(f"Ошибка при создании KMZ: {str(e)}")
        return jsonify({"error": "Ошибка при создании KMZ"}), 500
#----------------------------------------------  --------------------------------------------
#---------------------------------------------- подключение беспилотника через кабель или по IP + порт (UDP)  --------------------------------------------
import time




# Функция для ожидания пульса с тайм-аутом
def wait_for_heartbeat(timeout=3):
    try:
        master.wait_heartbeat(timeout=timeout)
        logging.info("Соединение с беспилотником установлено!")
        return True
    except Exception as e:
        logging.error(f"Ошибка при ожидании пульса: {e}")
        return False


@app.route('/connect', methods=['GET'])
def connect_to_drone():
    logging.info("Попытка подключиться к беспилотнику...")
    if wait_for_heartbeat():
        logging.info("Беспилотник подключен успешно!")
        return jsonify({"status": "Беспилотник подключен и готов"}), 200
    else:
        logging.error("Не удалось подключиться к беспилотнику.")
        return jsonify({"error": "Не удалось подключиться к беспилотнику"}), 500


# Функция для отправки команды по MAVLink
def send_mavlink_command(speed, altitude):
    try:
        master.mav.set_position_target_local_ned_send(
            0,  # Время (0 означает немедленную отправку)
            master.target_system,  # ID системы (используем целевую систему)
            master.target_component,  # ID компонента
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # Тип кадра
            0b0000111111111000,  # Маска параметров
            0,  # x
            0,  # y
            -altitude,  # z (в MAVLink высота отрицательная)
            speed,  # скорость по оси x
            0,  # скорость по оси y
            0,  # скорость по оси z
            0,  # Поворот по оси x
            0,  # Поворот по оси y
            0  # Поворот по оси z
        )

        # Ожидание подтверждения команды
        ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=2)
        if ack:
            logging.info(f"Команда принята: {ack}")
        else:
            logging.warning("Беспилотник не подтвердил команду.")

    except Exception as e:
        logging.error(f"Ошибка при отправке команды: {e}")


# Обработчик для отправки данных на беспилотник
@app.route('/send_to_drone', methods=['POST'])
def send_to_drone():
    try:
        data = request.json
        speed = data.get('speed')
        altitude = data.get('altitude')

        print(f"Получены данные: speed={speed}, altitude={altitude}")

        if speed is None or altitude is None:
            print("Ошибка: Скорость и высота не указаны")
            return jsonify({"error": "Скорость и высота не указаны"}), 400

        if isinstance(speed, str) or isinstance(altitude, str):
            print("Ошибка: Скорость и высота должны быть числами")
            return jsonify({"error": "Скорость и высота должны быть числами"}), 400

        send_mavlink_command(speed, altitude)

        print("Команды отправлены на беспилотник")
        return jsonify({"status": "Команды отправлены на беспилотник"})
    except Exception as e:
        logging.error(f"Ошибка при отправке команд: {str(e)}")
        return jsonify({"error": f"Ошибка при отправке команд: {str(e)}"}), 500

#---------------------------------------------- подключение беспилотника через кабель или по IP + порт (UDP)  --------------------------------------------

#########################2-я страница wkt ##########################################

# Укажите путь к вашему DEM файлу (например, 'data/dem.tif')
DEM_FILE_PATH = 'data/dem.tif'


@app.route('/elevation-map')
def elevation_map():
    return render_template("elevation.html")


@app.route('/elevation', methods=['GET'])
def get_elevation():
    # Получаем WKT из параметров запроса
    wkt = request.args.get('wkt')

    if not wkt:
        return jsonify({'error': 'WKT not provided'}), 400

    # Логика обработки WKT и добавления высоты
    result_wkt = add_elevation_to_wkt(wkt)

    if result_wkt is None:
        return jsonify({"error": "Invalid WKT format."}), 400

    # Получаем границы DEM
    dem_bounds = get_dem_bounds()
    if isinstance(dem_bounds, tuple):
        return dem_bounds  # Если возникла ошибка, возвращаем ответ с ошибкой

    return jsonify({
        "result_wkt": result_wkt,
        "dem_bounds": dem_bounds
    })


def add_elevation_to_wkt(wkt):
    coords_with_elevation = []

    if wkt.startswith("POINT"):
        # Парсим координаты для POINT
        coords = parse_wkt_point(wkt)
        if coords:
            y, x = coords  # Изменяем порядок на (широта, долгота)
            elevation = get_elevation_from_dem(x, y)  # Передаем (долгота, широта)
            if elevation is not None:
                return f"POINT({int(y)} {int(x)} {elevation})"  # Поменяли порядок на (широта, долгота)
            else:
                return f"POINT({int(y)} {int(x)} 0)"  # Возвращаем высоту 0, если не удалось найти

    elif wkt.startswith("LINESTRING"):
        # Парсим координаты для LINESTRING
        coords = parse_wkt_linestring(wkt)
        if coords:
            for y, x in coords:  # Изменяем порядок на (широта, долгота)
                elevation = get_elevation_from_dem(x, y)  # Передаем (долгота, широта)
                if elevation is not None:
                    coords_with_elevation.append(
                        f"{int(y)} {int(x)} {elevation}")  # Поменяли порядок на (широта, долгота)
                else:
                    coords_with_elevation.append(f"{int(y)} {int(x)} 0")  # Если высота не найдена, ставим 0
            return f"LINESTRING({', '.join(coords_with_elevation)})"

    return None


def parse_wkt_point(wkt):
    try:
        coords = wkt[wkt.index("(") + 1:wkt.index(")")].split()
        return list(map(float, coords))
    except Exception as e:
        print(f"Error parsing POINT WKT: {e}")
        return None


def parse_wkt_linestring(wkt):
    try:
        coords = wkt[wkt.index("(") + 1:wkt.index(")")].split(",")
        return [list(map(float, coord.strip().split())) for coord in coords]
    except Exception as e:
        print(f"Error parsing LINESTRING WKT: {e}")
        return None


def is_within_bounds(lon, lat, dataset):
    """Проверяет, находятся ли координаты в пределах DEM-файла."""
    bounds = dataset.bounds
    return bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top


def get_dem_bounds():
    try:
        with rasterio.open(DEM_FILE_PATH) as dataset:
            bounds = dataset.bounds
            return {
                'left': bounds.left,
                'bottom': bounds.bottom,
                'right': bounds.right,
                'top': bounds.top
            }
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_elevation_from_dem(lon, lat):
    """Получает высоту из DEM-файла на основе долготы и широты."""
    try:
        with rasterio.open(DEM_FILE_PATH) as dataset:
            # Логируем информацию о границах DEM
            bounds = dataset.bounds
            print(f"DEM bounds: {bounds}")

            # Проверяем, что координаты находятся в пределах данных
            if not is_within_bounds(lon, lat, dataset):
                print(f"Coordinates ({lon}, {lat}) are out of DEM bounds.")
                return None

            # Преобразуем координаты из географических (lon, lat) в систему координат DEM-файла
            transformed_coords = transform('EPSG:4326', dataset.crs, [lon], [lat])
            print(f"Transformed coordinates: {transformed_coords}")

            row, col = dataset.index(transformed_coords[0][0], transformed_coords[1][0])

            # Проверяем, что координаты находятся в пределах массива данных
            if 0 <= row < dataset.height and 0 <= col < dataset.width:
                elevation = dataset.read(1)[row, col]
                print(f"Elevation at ({lon}, {lat}): {elevation}")
                return round(elevation)  # Округляем до целого
            else:
                print(f"Coordinates ({lon}, {lat}) are out of bounds of the DEM file.")
                return None
    except Exception as e:
        print(f"Error getting elevation: {e}")
        return None



if __name__ == '__main__':
    app.run(debug=True) ##
from flask import Flask, request, jsonify, render_template
from pyproj import Geod
import rasterio
from rasterio.warp import transform
from shapely.geometry import LineString, MultiLineString, Polygon
from flask_cors import CORS
import logging
from heapq import heappop, heappush
from shapely.ops import nearest_points
from shapely.geometry import Point, LineString, Polygon

# Создаем экземпляр Flask приложения
app = Flask(__name__)
# Разрешаем CORS для этого приложения
CORS(app)
# Включаем режим отладки
app.debug = True

# Инициализация геоида WGS84 для вычисления ортодромии
geod = Geod(ellps="WGS84")

# Настройка логирования
logging.basicConfig(level=logging.INFO)


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
        # Получаем данные из запроса
        data = request.json
        logging.debug(f"Received data: {data}")

        # Извлекаем начальную и конечную точки, количество узлов и тип маршрута
        start_point = data.get('start_point')
        end_point = data.get('end_point')
        num_nodes = data.get('num_nodes', 1000)  # Устанавливаем значение по умолчанию
        is_orthodrome = data.get('orthodrome', True)
        restricted_areas = data.get('restricted_areas', [])  # Получаем зоны запрета

        # Проверка входных данных
        if not (isinstance(start_point, list) and len(start_point) == 2):
            return jsonify({"error": "Invalid start_point"}), 400
        if not (isinstance(end_point, list) and len(end_point) == 2):
            return jsonify({"error": "Invalid end_point"}), 400
        if not isinstance(num_nodes, int) or num_nodes < 2:
            return jsonify({"error": "Invalid num_nodes"}), 400

        # Проверка формата restricted_areas
        if not isinstance(restricted_areas, list) or any(not isinstance(area, list) for area in restricted_areas):
            return jsonify({"error": "Invalid restricted_areas format"}), 400

        # Преобразуем зоны запрета в объекты Polygon
        try:
            restricted_areas = [Polygon(area) for area in restricted_areas if len(area) > 1]
        except Exception as e:
            logging.error(f"Error occurred while creating polygons: {str(e)}")
            return jsonify({"error": f"Invalid restricted_areas: {str(e)}"}), 400

        lon1, lat1 = start_point  # Извлекаем координаты начальной точки
        lon2, lat2 = end_point  # Извлекаем координаты конечной точки

        # Создаем маршрут
        if is_orthodrome:
            # Ортодромический маршрут с большим количеством узлов
            points = geod.npts(lon1, lat1, lon2, lat2, num_nodes + 2000)  # Больше узлов для большей точности кривой
        else:
            # Прямой маршрут
            lons = [lon1 + (lon2 - lon1) * i / (num_nodes - 1) for i in range(num_nodes)]
            lats = [lat1 + (lat2 - lat1) * i / (num_nodes - 1) for i in range(num_nodes)]
            points = zip(lons, lats)  # Соединяем долготы и широты в пары

        # Формируем полный список координат маршрута
        coordinates = [(lon1, lat1)] + list(points) + [(lon2, lat2)]

        # Проверка близости к запрещенным зонам
        if is_route_too_close_to_restricted_areas(coordinates, restricted_areas):
            return jsonify({"error": "Маршрут слишком близок к запрещенной зоне!"}), 400

        # Возвращаем координаты
        return jsonify({
            "coordinates": coordinates,
            "warning": None  # Нет предупреждений
        })

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500  # Обработка ошибок



###################вв#############################################################
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points
import logging
from flask import Flask, request, jsonify
import random
import heapq

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
    max_iterations = 1000  # Ограничение на количество итераций
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
    app.run(debug=True)
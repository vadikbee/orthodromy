from flask import Flask, request, jsonify, render_template
from pyproj import Geod
import random
import rasterio
from rasterio.plot import show
from rasterio.warp import transform
import os
from shapely.geometry import LineString, Polygon, MultiLineString
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Инициализация геоида WGS84
geod = Geod(ellps="WGS84")

# Маршрут для отображения главной HTML-страницы
@app.route('/', methods=['GET'])
def index():
    return render_template("zad.html")

# Функция для проверки пересечения маршрута с запрещенными зонами
def check_intersections(route_coords, restricted_areas):
    route_line = LineString(route_coords)  # Создаем линию маршрута
    intersections = []  # Список для хранения точек пересечения

    for area in restricted_areas:
        buffered_area = area.buffer(0.00001)  # Добавляем буфер к полигону

        # Проверка на пересечение с буфером
        if route_line.intersects(buffered_area):
            intersection = route_line.intersection(buffered_area)  # Получаем точки пересечения
            print(f"Intersection found: {intersection}")
            if isinstance(intersection, (LineString, MultiLineString)):
                intersections.append(intersection)  # Добавляем линию пересечения
            elif not intersection.is_empty:
                intersections.append([intersection.x, intersection.y])  # Добавляем точку пересечения

    return intersections  # Возвращаем список линий пересечения
import logging

logging.basicConfig(level=logging.DEBUG)
# Маршрут для обработки POST-запроса и расчета ортодромии или прямой линии
@app.route('/orthodrome', methods=['POST'])
def orthodrome():
    data = request.json
    logging.debug(f"Received data: {data}")

    start_point = data.get('start_point')
    end_point = data.get('end_point')
    num_nodes = data.get('num_nodes', 1000)
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
        restricted_areas = [Polygon(area) for area in restricted_areas]
    except Exception as e:
        logging.error(f"Error occurred while creating polygons: {str(e)}")
        return jsonify({"error": f"Invalid restricted_areas: {str(e)}"}), 400

    lon1, lat1 = start_point
    lon2, lat2 = end_point

    if is_orthodrome:
        # Ортодромический маршрут с большим количеством узлов
        points = geod.npts(lon1, lat1, lon2, lat2, num_nodes + 2000)  # Больше узлов для большей кривизны
    else:
        # Прямой маршрут
        lons = [lon1 + (lon2 - lon1) * i / (num_nodes - 1) for i in range(num_nodes)]
        lats = [lat1 + (lat2 - lat1) * i / (num_nodes - 1) for i in range(num_nodes)]
        points = zip(lons, lats)

    coordinates = [(lon1, lat1)] + list(points) + [(lon2, lat2)]

    # Проверка пересечения с запрещенными зонами
    intersections = check_intersections(coordinates, restricted_areas)
    if intersections:
        intersection_coords = []
        for line in intersections:
            if isinstance(line, LineString):
                intersection_coords.extend(list(line.coords))
            elif isinstance(line, list):
                intersection_coords.append(line)  # Добавляем точку пересечения

        return jsonify({"error": "Маршрут пересекает запрещенную зону!", "intersections": intersection_coords}), 403

    return jsonify({"coordinates": coordinates})









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
                    coords_with_elevation.append(f"{int(y)} {int(x)} {elevation}")  # Поменяли порядок на (широта, долгота)
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
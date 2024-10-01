from flask import Flask, request, jsonify, render_template
from pyproj import Geod

app = Flask(__name__)

# Инициализация геоида WGS84
geod = Geod(ellps="WGS84")


# Маршрут для отображения главной HTML-страницы
@app.route('/', methods=['GET'])
def index():
    return render_template("zad.html")


# Маршрут для новой страницы elevation
@app.route('/elevation-map', methods=['GET'])
def elevation_map():
    return render_template("elevation.html")


# Маршрут для обработки POST-запроса и расчета ортодромии или прямой линии
@app.route('/orthodrome', methods=['POST'])
def orthodrome():
    data = request.json

    start_point = data.get('start_point')
    end_point = data.get('end_point')
    num_nodes = data.get('num_nodes')
    is_orthodrome = data.get('orthodrome', True)

    if not (isinstance(start_point, list) and len(start_point) == 2):
        return jsonify({"error": "Invalid start_point"}), 400
    if not (isinstance(end_point, list) and len(end_point) == 2):
        return jsonify({"error": "Invalid end_point"}), 400
    if not isinstance(num_nodes, int) or num_nodes < 2:
        return jsonify({"error": "Invalid num_nodes"}), 400

    lon1, lat1 = start_point
    lon2, lat2 = end_point

    if is_orthodrome:
        # Ортодромический маршрут с большим количеством узлов
        points = geod.npts(lon1, lat1, lon2, lat2, num_nodes + 10)  # Больше узлов для большей кривизны
    else:
        # Прямой маршрут
        lons = [lon1 + (lon2 - lon1) * i / (num_nodes - 1) for i in range(num_nodes)]
        lats = [lat1 + (lat2 - lat1) * i / (num_nodes - 1) for i in range(num_nodes)]
        points = zip(lons, lats)

    coordinates = [(lon1, lat1)] + list(points) + [(lon2, lat2)]

    return jsonify({"coordinates": coordinates})


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
from pyproj import Geod

app = Flask(__name__)

# Инициализация геоида WGS84
geod = Geod(ellps="WGS84")


# Маршрут для отображения HTML-страницы
@app.route('/', methods=['GET'])
def index():

    return render_template("zad.html")


# Маршрут для обработки POST-запроса и расчета ортодромии
@app.route('/orthodrome', methods=['POST'])
def orthodrome():
    data = request.json  # Получаем данные

    start_point = data.get('start_point')
    end_point = data.get('end_point')
    num_nodes = data.get('num_nodes')

    # Проверяем, что все входные данные заданы корректно
    if not (isinstance(start_point, list) and len(start_point) == 2):
        return jsonify({"error": "Invalid start_point"}), 400
    if not (isinstance(end_point, list) and len(end_point) == 2):
        return jsonify({"error": "Invalid end_point"}), 400
    if not isinstance(num_nodes, int) or num_nodes < 2:
        return jsonify({"error": "Invalid num_nodes"}), 400

    lon1, lat1 = start_point
    lon2, lat2 = end_point

    # Получаем список кортежей (долгота, широта)
    points = geod.npts(lon1, lat1, lon2, lat2, num_nodes - 2)

    # Преобразуем список кортежей в два отдельных списка долгот и широт
    lons, lats = zip(*points)

    # Добавляем начальную и конечную точку в результат
    coordinates = [(lon1, lat1)] + list(zip(lons, lats)) + [(lon2, lat2)]

    return jsonify({"coordinates": coordinates})


    # Получаем список кортежей (долгота, широта)
    points = geod.npts(lon1, lat1, lon2, lat2, num_nodes - 2)

    # Преобразуем список кортежей в два отдельных списка долгот и широт
    lons, lats = zip(*points)

    # Добавляем начальную и конечную точку в результат
    coordinates = [(lon1, lat1)] + list(zip(lons, lats)) + [(lon2, lat2)]

    return jsonify({"coordinates": coordinates})


if __name__ == '__main__':
    app.run(debug=True)

import rasterio

DEM_FILE_PATH = 'data/dem.tif'  # Укажите путь к вашему DEM файлу


def check_dem_bounds():
    try:
        with rasterio.open(DEM_FILE_PATH) as dataset:
            # Получаем границы DEM файла
            bounds = dataset.bounds
            print(f"DEM bounds: {bounds}")
            print(f"Left: {bounds.left}, Bottom: {bounds.bottom}")
            print(f"Right: {bounds.right}, Top: {bounds.top}")

            return bounds
    except Exception as e:
        print(f"Error opening DEM file: {e}")
        return None


# Вызов функции
dem_bounds = check_dem_bounds()

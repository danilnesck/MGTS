import io
import exifread

def _convert_to_degress(value):
    """Конвертирует (градусы, минуты, секунды) в десятичные градусы."""
    d = float(value.values[0].num) / float(value.values[0].den)
    m = float(value.values[1].num) / float(value.values[1].den)
    s = float(value.values[2].num) / float(value.values[2].den)
    return d + (m / 60.0) + (s / 3600.0)

def get_gps_coordinates(image_bytes: bytes) -> tuple[float | None, float | None]:
    """
    Принимает байты картинки, возвращает (широта, долгота).
    Если координат нет, возвращает (None, None).
    """
    try:
        tags = exifread.process_file(io.BytesIO(image_bytes))
        
        # Проверяем, есть ли вообще GPS теги в фото
        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            lat = _convert_to_degress(tags['GPS GPSLatitude'])
            lon = _convert_to_degress(tags['GPS GPSLongitude'])

            # Проверяем полушарие (Юг/Запад означают минус)
            if tags.get('GPS GPSLatitudeRef') and tags['GPS GPSLatitudeRef'].values[0] != 'N':
                lat = -lat
            if tags.get('GPS GPSLongitudeRef') and tags['GPS GPSLongitudeRef'].values[0] != 'E':
                lon = -lon
                
            return round(lat, 6), round(lon, 6)
    except Exception as e:
        # Если файл битый или тегов нет - просто игнорируем
        pass
    
    return None, None
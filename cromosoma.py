import random, math, csv, os, json
from datetime import datetime, timedelta


def cargar_csv(ruta):
    with open(ruta, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def cargar_base_conocimiento(carpeta_csv):
    profundidad = cargar_csv(os.path.join(carpeta_csv, 'profundidad_siembra.csv'))
    separacion  = cargar_csv(os.path.join(carpeta_csv, 'separacion_minima.csv'))
    eclosion    = cargar_csv(os.path.join(carpeta_csv, 'tasa_eclosion.csv'))
    incubacion  = cargar_csv(os.path.join(carpeta_csv, 'dias_incubacion.csv'))
    corral_rows = cargar_csv(os.path.join(carpeta_csv, 'corral_incubacion.csv'))

    corral = {r['campo']: r['valor'] for r in corral_rows}
    base   = {}

    for f in profundidad:
        e = f['especie']
        base[e] = {
            'prof_min': float(f['profundidad_minima_cm']),
            'prof_opt': float(f['profundidad_optima_cm']),
            'prof_max': float(f['profundidad_maxima_cm']),
            'sigma':    float(f['sigma_cm']),
        }

    for f in separacion:
        base[f['especie']]['sep_min'] = float(f['separacion_minima_cm'])

    for f in eclosion:
        e = f['especie']
        base[e]['tasa_promedio']        = float(f['tasa_eclosion_promedio'])
        base[e]['tasa_minima']          = float(f['tasa_eclosion_minima'])
        base[e]['tasa_maxima']          = float(f['tasa_eclosion_maxima'])
        base[e]['tasa_eclosion']        = float(f['tasa_eclosion_promedio'])
        base[e]['tasa_eclosion_maxima'] = float(f['tasa_eclosion_maxima'])
        base[e]['tasa_eclosion_minima'] = float(f['tasa_eclosion_minima'])

    for f in incubacion:
        e = f['especie']
        base[e]['dias_min']  = int(f['dias_minimo'])
        base[e]['dias_prom'] = int(f['dias_promedio'])
        base[e]['dias_max']  = int(f['dias_maximo'])

    return base, corral


class Gen:
    """Un nido dentro del corral."""
    def __init__(self, id_nido, especie, x, y, prof):
        self.id_nido = id_nido
        self.especie = especie
        self.x    = round(x,    1)
        self.y    = round(y,    1)
        self.prof = round(prof, 1)

    def zona_correcta(self):
        return f"zona_{self.especie}"

    def en_zona_correcta(self, gestor):
        lim = gestor.limites_zona(self.zona_correcta())
        if not lim:
            return False
        return (lim['xmin'] <= self.x <= lim['xmax'] and
                lim['ymin'] <= self.y <= lim['ymax'])

    def copia(self):
        return Gen(self.id_nido, self.especie, self.x, self.y, self.prof)

    def __repr__(self):
        return (f"Gen(id={self.id_nido}, esp={self.especie}, "
                f"x={self.x}, y={self.y}, prof={self.prof})")


class Individuo:
    def __init__(self, genes):
        self.genes   = genes
        self.fitness = None
        self.v1 = self.v2 = self.v3 = self.v4 = None

    def num_nidos(self):
        return len(self.genes)

    def copia(self):
        ind = Individuo([g.copia() for g in self.genes])
        ind.fitness = self.fitness
        ind.v1, ind.v2, ind.v3, ind.v4 = self.v1, self.v2, self.v3, self.v4
        return ind

    def __repr__(self):
        f_str = f"{self.fitness:.4f}" if self.fitness is not None else "?"
        return f"Individuo(n={self.num_nidos()} fit={f_str})"


class GestorZonas:
    """
    Divide el corral en 3 franjas verticales proporcionales a n_nidos.

    Ejemplo (70 golfina / 25 prieta / 5 laúd, total=100):
      zona_golfina: x ∈ [0,    2800]   70% de 4000 cm
      zona_prieta:  x ∈ [2800, 3800]   25% de 4000 cm
      zona_laud:    x ∈ [3800, 4000]    5% de 4000 cm
      Todas:        y ∈ [0,    3500]
    """
    def __init__(self, corral, n_golfina, n_prieta, n_laud, base=None):
        self.largo_cm = float(corral['largo_cm'])
        self.ancho_cm = float(corral['ancho_cm'])
        self.conteo   = {'golfina': n_golfina, 'prieta': n_prieta, 'laud': n_laud}
        self.total    = n_golfina + n_prieta + n_laud
        self.base     = base
        self.limites  = self._calcular_limites()

    def _calcular_limites(self):
        if self.total == 0:
            return {}
        n_g = self.conteo['golfina']
        n_p = self.conteo['prieta']
        n_l = self.conteo['laud']

        largo_g = round(self.largo_cm * n_g / self.total, 1)
        largo_p = round(self.largo_cm * n_p / self.total, 1)
        largo_l = self.largo_cm - largo_g - largo_p

        x0 = 0.0
        x1 = largo_g
        x2 = largo_g + largo_p
        x3 = self.largo_cm

        lims = {}
        if n_g > 0:
            lims['zona_golfina'] = {'xmin': x0, 'xmax': x1,
                                    'ymin': 0.0, 'ymax': self.ancho_cm,
                                    'especie': 'golfina'}
        if n_p > 0:
            lims['zona_prieta']  = {'xmin': x1, 'xmax': x2,
                                    'ymin': 0.0, 'ymax': self.ancho_cm,
                                    'especie': 'prieta'}
        if n_l > 0:
            lims['zona_laud']    = {'xmin': x2, 'xmax': x3,
                                    'ymin': 0.0, 'ymax': self.ancho_cm,
                                    'especie': 'laud'}
        return lims

    def limites_zona(self, nombre_zona):
        return self.limites.get(nombre_zona)

    def posiciones_cuadricula(self, nombre_zona, n_nidos, sep_min=100.0,
                              nidos_ocupados=None):
        """
        Genera posiciones en cuadrícula con PASO GARANTIZADO >= sep_min.

        Algoritmo:
        1. Calcula cuántas columnas y filas caben en la zona con paso = sep_min.
           El centroide de celdas vecinas siempre estará a exactamente paso_x
           o paso_y de distancia, ambos >= sep_min.
        2. Filtra celdas que violarían sep_min con nidos previos de la misma
           especie (de jornadas anteriores).
        3. Mezcla y retorna las primeras n_nidos.
        4. Si la zona está saturada (insuficientes celdas libres), completa
           con posiciones aleatorias buscando respetar sep_min.
        """
        lim = self.limites_zona(nombre_zona)
        if not lim or n_nidos == 0:
            return []

        ancho_z  = lim['xmax'] - lim['xmin']
        alto_z   = lim['ymax'] - lim['ymin']
        especie  = lim['especie']

        # Número de celdas que caben con paso exactamente sep_min
        cols  = max(1, int(ancho_z / sep_min))
        filas = max(1, int(alto_z  / sep_min))

        # Paso real (>= sep_min siempre)
        paso_x = ancho_z / cols
        paso_y = alto_z  / filas

        # Centroides de todas las celdas
        todas = []
        for f in range(filas):
            for c in range(cols):
                x = lim['xmin'] + paso_x * (c + 0.5)
                y = lim['ymin'] + paso_y * (f + 0.5)
                todas.append((round(x, 1), round(y, 1)))

        # Filtrar con nidos previos de la misma especie
        if nidos_ocupados:
            prev_esp = [(oc['x'], oc['y'])
                        for oc in nidos_ocupados
                        if oc.get('especie') == especie]
            if prev_esp:
                libres = []
                for px, py in todas:
                    if all(math.sqrt((px - ox)**2 + (py - oy)**2) >= sep_min
                           for ox, oy in prev_esp):
                        libres.append((px, py))
                todas = libres

        random.shuffle(todas)

        if len(todas) >= n_nidos:
            return todas[:n_nidos]

        # Zona saturada — completar con muestreo aleatorio respetando sep_min
        seleccionadas = list(todas)
        intentos = 0
        max_intentos = n_nidos * 300
        prev_esp_coords = [(oc['x'], oc['y'])
                           for oc in (nidos_ocupados or [])
                           if oc.get('especie') == especie]

        while len(seleccionadas) < n_nidos and intentos < max_intentos:
            intentos += 1
            cx = round(random.uniform(lim['xmin'], lim['xmax']), 1)
            cy = round(random.uniform(lim['ymin'], lim['ymax']), 1)
            # Verificar sep_min contra ya-seleccionadas y contra previos
            valida = all(
                math.sqrt((cx - px)**2 + (cy - py)**2) >= sep_min
                for px, py in seleccionadas
            ) and all(
                math.sqrt((cx - ox)**2 + (cy - oy)**2) >= sep_min
                for ox, oy in prev_esp_coords
            )
            if valida:
                seleccionadas.append((cx, cy))

        # Si la zona está completamente saturada, añadir sin garantía
        while len(seleccionadas) < n_nidos:
            cx = round(random.uniform(lim['xmin'], lim['xmax']), 1)
            cy = round(random.uniform(lim['ymin'], lim['ymax']), 1)
            seleccionadas.append((cx, cy))

        return seleccionadas[:n_nidos]

    def __repr__(self):
        return (f"GestorZonas(dinámica)\n"
                f"  Corral: {self.largo_cm}cm × {self.ancho_cm}cm\n"
                f"  Golfina: {self.conteo['golfina']} nidos\n"
                f"  Prieta:  {self.conteo['prieta']} nidos\n"
                f"  Laúd:    {self.conteo['laud']} nidos")


class RegistroJornadas:
    def __init__(self, ruta='jornadas.json'):
        self.ruta  = ruta
        self.datos = self._cargar()

    def _cargar(self):
        if os.path.exists(self.ruta):
            with open(self.ruta, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'temporada': str(datetime.now().year), 'jornadas': []}

    def _guardar(self):
        with open(self.ruta, 'w', encoding='utf-8') as f:
            json.dump(self.datos, f, ensure_ascii=False, indent=2)

    def guardar_jornada(self, fecha, n_g, n_p, n_l, mejor, params, gestor=None):
        nidos = [{
            'id':      g.id_nido,
            'especie': g.especie,
            'x':       g.x,
            'y':       g.y,
            'prof':    g.prof,
            'zona':    g.zona_correcta()
        } for g in mejor.genes]

        # Guardar límites de zona de ESTA jornada.
        # Las zonas son proporcionales al conteo de nidos de la jornada, por lo que
        # cambian entre jornadas. Al guardarlas aquí, jornadas futuras pueden
        # dibujar los nidos previos dentro de sus zonas originales.
        zonas_jornada = {}
        if gestor:
            for nombre, lim in gestor.limites.items():
                zonas_jornada[nombre] = {k: v for k, v in lim.items()}

        jornada = {
            'fecha':    fecha,
            'entradas': {'golfina': n_g, 'prieta': n_p, 'laud': n_l},
            'params':   params,
            'zonas':    zonas_jornada,
            'resultado': {
                'fitness': round(mejor.fitness, 4),
                'v1': round(mejor.v1, 4),
                'v2': round(mejor.v2, 4),
                'v3': round(mejor.v3, 4),
                'v4': round(mejor.v4, 4),
            },
            'nidos': nidos
        }
        self.datos['jornadas'].append(jornada)
        self._guardar()
        return jornada

    def obtener_nidos_activos(self, fecha_actual_str, base):
        try:
            hoy = datetime.strptime(fecha_actual_str, '%Y-%m-%d')
        except ValueError:
            hoy = datetime.today()

        activos = []
        for jornada in self.datos['jornadas']:
            try:
                fs = datetime.strptime(jornada['fecha'], '%Y-%m-%d')
            except ValueError:
                continue
            for nido in jornada['nidos']:
                dias_max = base[nido['especie']]['dias_max']
                fecha_eclosion = fs + timedelta(days=dias_max)
                if fecha_eclosion >= hoy:
                    activos.append({
                        **nido,
                        'fecha_siembra':  jornada['fecha'],
                        'fecha_eclosion': fecha_eclosion.strftime('%Y-%m-%d'),
                        # Zonas de la jornada en que fue sembrado este nido.
                        # Permite al frontend dibujarlo dentro de su zona original.
                        'zonas_jornada':  jornada.get('zonas', {}),
                    })
        return activos

    def resumen_temporada(self):
        if not self.datos['jornadas']:
            return {}
        js = self.datos['jornadas']
        return {
            'jornadas':    len(js),
            'total_nidos': sum(j['entradas']['golfina'] +
                               j['entradas']['prieta'] +
                               j['entradas']['laud'] for j in js),
            'golfina':     sum(j['entradas']['golfina'] for j in js),
            'prieta':      sum(j['entradas']['prieta']  for j in js),
            'laud':        sum(j['entradas']['laud']    for j in js),
            'v1_promedio': round(
                sum(j['resultado']['v1'] for j in js) / len(js), 4),
        }

    def limpiar_temporada(self):
        self.datos = {'temporada': str(datetime.now().year), 'jornadas': []}
        if os.path.exists(self.ruta):
            os.remove(self.ruta)


def calcular_capacidad_restante(corral, nidos_activos, sep_min=100.0):
    largo     = float(corral['largo_cm'])
    ancho     = float(corral['ancho_cm'])
    cap_total = int(largo / sep_min) * int(ancho / sep_min)
    ocupado   = len(nidos_activos)
    libres    = max(0, cap_total - ocupado)
    pct       = (ocupado / cap_total * 100) if cap_total > 0 else 0
    return libres, round(pct, 1)

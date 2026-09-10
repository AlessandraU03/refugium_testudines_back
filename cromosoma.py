import random, math, csv, os, json
from datetime import datetime, timedelta


def cargar_csv(ruta):
    """Lee un CSV de la base de conocimiento validando que este bien formado.

    La base de conocimiento es la fuente de todos los parametros del sistema y
    cada fila declara su procedencia documental. Una coma sin comillas dentro de
    un campo de texto desplaza las columnas y csv.DictReader lo absorbe en
    silencio: los valores quedan corridos o van a parar a la clave None. Se
    valida aqui, en el unico punto de carga, para que el error salte al leer el
    archivo y no mas adelante como un numero equivocado.
    """
    with open(ruta, newline='', encoding='utf-8') as f:
        lector = csv.DictReader(f)
        filas = list(lector)
        columnas = lector.fieldnames

    if not columnas:
        raise ValueError('%s: el archivo esta vacio o no tiene encabezado.' % os.path.basename(ruta))

    nombre = os.path.basename(ruta)
    for i, fila in enumerate(filas):
        # linea real en el archivo: +1 por el encabezado, +1 porque i es base 0
        linea = i + 2
        if None in fila:
            sobrantes = fila[None]
            raise ValueError(
                '%s linea %d: hay mas campos que columnas (%d declaradas). '
                'Sobra %r. Suele deberse a una coma dentro de un texto sin '
                'comillas: encierra ese campo entre comillas dobles.'
                % (nombre, linea, len(columnas), sobrantes))
        faltantes = [c for c in columnas if fila.get(c) is None]
        if faltantes:
            raise ValueError(
                '%s linea %d: faltan las columnas %s.'
                % (nombre, linea, ', '.join(faltantes)))

    return filas


def cargar_base_conocimiento(carpeta_csv):
    profundidad = cargar_csv(os.path.join(carpeta_csv, 'profundidad_siembra.csv'))
    separacion  = cargar_csv(os.path.join(carpeta_csv, 'separacion_minima.csv'))
    eclosion    = cargar_csv(os.path.join(carpeta_csv, 'tasa_eclosion.csv'))
    incubacion  = cargar_csv(os.path.join(carpeta_csv, 'dias_incubacion.csv'))
    corral_rows = cargar_csv(os.path.join(carpeta_csv, 'corral_incubacion.csv'))

    pts_file = os.path.join(carpeta_csv, 'pts_termosensible.csv')
    pts_data = cargar_csv(pts_file) if os.path.exists(pts_file) else []

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

    for f in pts_data:
        e = f['especie']
        if e in base:
            base[e]['pts_inicio_dia']  = int(f['pts_inicio_dia'])
            base[e]['pts_fin_dia']     = int(f['pts_fin_dia'])
            base[e]['semana_critica']  = int(f['semana_critica'])
            base[e]['pivote_temp']     = float(f['pivote_temp_c'])
            base[e]['s_parameter']     = float(f['s_parameter'])

    return base, corral


class Gen:
    """Un nido dentro del corral."""
    def __init__(self, id_nido, especie, x, y, prof, num_huevas=None, orden_incubacion=None):
        self.id_nido = id_nido
        self.especie = especie
        self.x    = round(x,    1)
        self.y    = round(y,    1)
        self.prof = round(prof, 1)

        self.num_huevas = num_huevas or 0
        self.orden_incubacion = orden_incubacion or 0
        self.fecha_siembra = None
        self.semana_incubacion = None
        self.estado = "incubando"  # incubando, eclosionado, muerto

    def zona_correcta(self):
        return f"zona_{self.especie}"

    def en_zona_correcta(self, gestor):
        lim = gestor.limites_zona(self.zona_correcta())
        if not lim:
            return False
        return (lim['xmin'] <= self.x <= lim['xmax'] and
                lim['ymin'] <= self.y <= lim['ymax'])

    def copia(self):
        gen_copia = Gen(self.id_nido, self.especie, self.x, self.y, self.prof,
                       num_huevas=self.num_huevas, orden_incubacion=self.orden_incubacion)
        gen_copia.fecha_siembra = self.fecha_siembra
        gen_copia.semana_incubacion = self.semana_incubacion
        gen_copia.estado = self.estado
        return gen_copia

    def __repr__(self):
        return (f"Gen(id={self.id_nido}, esp={self.especie}, "
                f"x={self.x}, y={self.y}, prof={self.prof})")


class Individuo:
    def __init__(self, genes):
        self.genes   = genes
        self.fitness = None
        self.v1 = self.v2 = self.v3 = None
        self.orden = None
        self.indice_eclosion = None

    def num_nidos(self):
        return len(self.genes)

    def copia(self):
        ind = Individuo([g.copia() for g in self.genes])
        ind.fitness = self.fitness
        ind.v1, ind.v2, ind.v3 = self.v1, self.v2, self.v3
        ind.orden = self.orden
        ind.indice_eclosion = self.indice_eclosion
        return ind

    def __repr__(self):
        f_str = f"{self.fitness:.4f}" if self.fitness is not None else "?"
        return f"Individuo(n={self.num_nidos()} fit={f_str})"


def separacion_para_alojar(ancho_cm, alto_cm, n_nidos):
    """Mayor separacion que permite acomodar n_nidos en una rejilla regular.

    Busca el paso `s` mas grande tal que floor(ancho/s) * floor(alto/s) >= n.
    El numero de casillas crece al reducir el paso, asi que la condicion es
    monotona y se puede resolver por biseccion.

    Devolver la separacion MAS GRANDE posible importa: cada centimetro que se
    conserva es eclosion que no se pierde.
    """
    if n_nidos <= 0:
        return float(min(ancho_cm, alto_cm))

    def caben(s):
        return max(1, int(ancho_cm / s)) * max(1, int(alto_cm / s)) >= n_nidos

    lo, hi = 1.0, float(max(ancho_cm, alto_cm))
    if caben(hi):
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if caben(mid):
            lo = mid
        else:
            hi = mid
    return lo


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
        self._cache_slots = {}
        self.separacion_efectiva = {}

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

    def slots_ordenados(self, nombre_zona, sep_min=100.0, nidos_ocupados=None,
                        n_requeridos=None):
        """
        Casillas libres de la rejilla en orden de llenado, en serpentina:
        la hilera 1 se recorre de ida y la 2 de vuelta, para que el personal
        avance sin regresar al inicio de cada hilera.

        Es un criterio operativo, no biológico. La rejilla usa sep_min de la
        especie como paso, así que cualquier casilla ya respeta la separación.
        """
        lim = self.limites_zona(nombre_zona)
        if not lim:
            return []

        especie = lim['especie']
        prev = [(oc['x'], oc['y']) for oc in (nidos_ocupados or [])
                if oc.get('especie') == especie]

        # La rejilla no cambia durante una corrida y se consulta miles de veces
        # (una por evaluación y una por mutación), así que se memoriza.
        clave = (nombre_zona, sep_min, n_requeridos, len(prev),
                 round(sum(p[0] for p in prev), 1),
                 round(sum(p[1] for p in prev), 1))
        if clave in self._cache_slots:
            return self._cache_slots[clave]

        ancho_z = lim['xmax'] - lim['xmin']
        alto_z  = lim['ymax'] - lim['ymin']

        paso = float(sep_min)
        cols  = max(1, int(ancho_z / paso))
        filas = max(1, int(alto_z  / paso))

        # Si a la separacion documentada no salen suficientes casillas, se
        # aprieta la rejilla en vez de encimar nidos. Antes los sobrantes
        # terminaban apilados en coordenadas ya ocupadas: el resultado se veia
        # valido pero pedia enterrar dos nidadas en el mismo hoyo.
        #
        # Apretar tiene un costo biologico y NO se aplica en silencio: la
        # separacion realmente usada queda en self.separacion_efectiva para que
        # quien consuma la rejilla lo reporte.
        if n_requeridos and cols * filas < n_requeridos:
            paso = separacion_para_alojar(ancho_z, alto_z, n_requeridos)
            cols  = max(1, int(ancho_z / paso))
            filas = max(1, int(alto_z  / paso))

        self.separacion_efectiva[nombre_zona] = round(paso, 1)
        paso_x = ancho_z / cols
        paso_y = alto_z  / filas

        slots = []
        for f in range(filas):
            y = round(lim['ymin'] + paso_y * (f + 0.5), 1)
            recorrido = range(cols) if f % 2 == 0 else range(cols - 1, -1, -1)
            for c in recorrido:
                slots.append((round(lim['xmin'] + paso_x * (c + 0.5), 1), y))

        if prev:
            slots = [(x, y) for (x, y) in slots
                     if all(math.hypot(x - ox, y - oy) >= paso
                            for ox, oy in prev)]

        self._cache_slots[clave] = slots
        return slots

    def posiciones_cuadricula(self, nombre_zona, n_nidos, sep_min=100.0,
                              nidos_ocupados=None):
        lim = self.limites_zona(nombre_zona)
        if not lim or n_nidos == 0:
            return []

        ancho_z  = lim['xmax'] - lim['xmin']
        alto_z   = lim['ymax'] - lim['ymin']
        especie  = lim['especie']

        cols  = max(1, int(ancho_z / sep_min))
        filas = max(1, int(alto_z  / sep_min))

        paso_x = ancho_z / cols
        paso_y = alto_z  / filas

        todas = []
        for f in range(filas):
            for c in range(cols):
                x = lim['xmin'] + paso_x * (c + 0.5)
                y = lim['ymin'] + paso_y * (f + 0.5)
                todas.append((round(x, 1), round(y, 1)))

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
            valida = all(
                math.sqrt((cx - px)**2 + (cy - py)**2) >= sep_min
                for px, py in seleccionadas
            ) and all(
                math.sqrt((cx - ox)**2 + (cy - oy)**2) >= sep_min
                for ox, oy in prev_esp_coords
            )
            if valida:
                seleccionadas.append((cx, cy))

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
        for jornada in self.datos.get('jornadas', []):
            try:
                fs = datetime.strptime(jornada['fecha'], '%Y-%m-%d')
            except ValueError:
                continue
            for nido in jornada.get('nidos', []):
                esp = nido.get('especie', 'golfina')
                dias_max = base.get(esp, {}).get('dias_max', 50)
                fecha_eclosion = fs + timedelta(days=dias_max)
                # Incluir nidos activos en incubación o que eclosionaron hace menos de 15 días (descanso de arena)
                if fecha_eclosion + timedelta(days=15) >= hoy:
                    activos.append({
                        **nido,
                        'especie':        esp,
                        'fecha_siembra':  jornada['fecha'],
                        'fecha_eclosion': fecha_eclosion.strftime('%Y-%m-%d'),
                        'eclosionado':    fecha_eclosion < hoy,
                        # Zonas de la jornada en que fue sembrado este nido.
                        'zonas_jornada':  jornada.get('zonas', {}),
                    })
        return activos

    def resumen_temporada(self):
        js = self.datos.get('jornadas', [])
        if not js:
            return {}
        
        total_g = 0
        total_p = 0
        total_l = 0
        total_v1 = 0.0
        
        for j in js:
            entradas = j.get('entradas', {})
            g = entradas.get('golfina', j.get('n_golfina', 0))
            p = entradas.get('prieta', j.get('n_prieta', 0))
            l = entradas.get('laud', j.get('n_laud', 0))
            total_g += g
            total_p += p
            total_l += l
            
            res = j.get('resultado', {})
            total_v1 += res.get('v1', 0.85)
            
        return {
            'jornadas':    len(js),
            'total_nidos': total_g + total_p + total_l,
            'golfina':     total_g,
            'prieta':      total_p,
            'laud':        total_l,
            'v1_promedio': round(total_v1 / len(js), 4),
        }

    def limpiar_temporada(self):
        self.datos = {'temporada': str(datetime.now().year), 'jornadas': []}
        if os.path.exists(self.ruta):
            try:
                os.remove(self.ruta)
            except Exception:
                pass


def calcular_capacidad_restante(corral, nidos_activos, sep_min=100.0):
    largo     = float(corral['largo_cm'])
    ancho     = float(corral['ancho_cm'])
    cap_total = int(largo / sep_min) * int(ancho / sep_min)
    ocupado   = len(nidos_activos)
    libres    = max(0, cap_total - ocupado)
    pct       = (ocupado / cap_total * 100) if cap_total > 0 else 0
    return libres, round(pct, 1)

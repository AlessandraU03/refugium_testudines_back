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
    def __init__(self, genes, idx_orden=0):
        self.genes   = genes
        self.fitness = None
        self.v1 = self.v2 = self.v3 = None
        self.orden = None
        self.indice_eclosion = None

        # Cual de los ORDENES_ZONAS usa este individuo, es decir a que especie
        # le toca cada franja del corral. Es la segunda mitad del cromosoma:
        # los genes dicen donde va cada nido DENTRO de su zona, y esto dice
        # donde esta la zona. Con la malla cubriendo parte del corral, decide
        # quien recibe la sombra.
        #
        # OJO: no confundir con self.orden, que es el termino operativo de
        # llenado en serpentina.
        self.idx_orden = idx_orden

    def num_nidos(self):
        return len(self.genes)

    def copia(self):
        ind = Individuo([g.copia() for g in self.genes], self.idx_orden)
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


ESPECIES = ('golfina', 'prieta', 'laud')

# Las seis maneras de repartir tres franjas entre tres especies. El AG elige
# una: con la malla sombra cubriendo solo parte del corral, ese reparto decide
# a que especie le toca el fresco, y por tanto su proporcion sexual.
ORDENES_ZONAS = [
    ('golfina', 'prieta', 'laud'),
    ('golfina', 'laud', 'prieta'),
    ('prieta', 'golfina', 'laud'),
    ('prieta', 'laud', 'golfina'),
    ('laud', 'golfina', 'prieta'),
    ('laud', 'prieta', 'golfina'),
]


class GestorZonas:
    """
    Divide el corral en franjas verticales, una por especie presente.

    DOS COSAS QUE CAMBIARON RESPECTO A LA VERSION ANTERIOR
    ------------------------------------------------------

    1. El ancho de cada franja es proporcional al AREA que la especie
       necesita, no a su numero de nidos. Un nido de laud pide 150 cm de
       separacion contra los 100 de golfina, o sea 2.25 veces mas superficie
       (150^2 contra 100^2). Repartir por conteo le asignaba a laud una franja
       donde sus propios nidos no caben, y la rejilla tenia que apretarse.

    2. El ORDEN de las especies a lo largo del corral es un parametro, no una
       constante. Antes siempre era golfina, prieta, laud de izquierda a
       derecha. Con la malla sombra cubriendo solo una parte del corral eso
       condenaba a las dos ultimas especies al sol: medido, prieta y laud
       quedaban con sombra 0.00-0.07 y 100 % de hembras, y son precisamente
       las de pivote mas baja. Ahora el orden lo elige el algoritmo genetico.

    Ejemplo con 20 golfina / 8 prieta / 2 laud en un corral de 3000 x 800 cm,
    orden (golfina, prieta, laud):
      areas: 20*100^2 = 200 000 | 8*120^2 = 115 200 | 2*150^2 = 45 000
      zona_golfina: x en [0, 1667]
      zona_prieta:  x en [1667, 2627]
      zona_laud:    x en [2627, 3000]
    """
    def __init__(self, corral, n_golfina, n_prieta, n_laud, base=None,
                 orden=None):
        self.largo_cm = float(corral['largo_cm'])
        self.ancho_cm = float(corral['ancho_cm'])
        self.conteo   = {'golfina': n_golfina, 'prieta': n_prieta, 'laud': n_laud}
        self.total    = n_golfina + n_prieta + n_laud
        self.base     = base
        self.orden    = tuple(orden) if orden else ESPECIES
        self.limites  = self._calcular_limites()
        self._cache_slots = {}
        self.separacion_efectiva = {}

    def _area_necesaria(self, especie):
        """Superficie que pide una especie: sus nidos por su separacion al cuadrado."""
        n = self.conteo.get(especie, 0)
        if not n:
            return 0.0
        sep = 100.0
        if self.base and especie in self.base:
            sep = float(self.base[especie].get('sep_min', 100.0))
        return n * sep * sep

    def _calcular_limites(self):
        if self.total == 0:
            return {}

        presentes = [e for e in self.orden if self.conteo.get(e, 0) > 0]
        pesos = {e: self._area_necesaria(e) for e in presentes}
        total_peso = sum(pesos.values())
        if total_peso <= 0:
            return {}

        lims = {}
        x = 0.0
        for i, esp in enumerate(presentes):
            # A la ultima franja se le da el remanente exacto, para que las
            # franjas cubran el corral completo sin dejar una rendija por
            # redondeo.
            if i == len(presentes) - 1:
                ancho = self.largo_cm - x
            else:
                ancho = round(self.largo_cm * pesos[esp] / total_peso, 1)
            lims['zona_%s' % esp] = {
                'xmin': round(x, 1), 'xmax': round(x + ancho, 1),
                'ymin': 0.0, 'ymax': self.ancho_cm,
                'especie': esp,
            }
            x += ancho
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

        # TODOS los nidos ya enterrados, no solo los de esta especie.
        #
        # Antes se filtraban unicamente los previos de la misma especie, porque
        # la separacion minima esta definida por especie. Pero la separacion es
        # una norma biologica y el solapamiento es un hecho fisico: no se puede
        # excavar donde ya hay una nidada, sea de la especie que sea. Mientras
        # las franjas estuvieron fijas el error no se notaba -cada especie
        # ocupaba su propia franja-, pero desde que el AG reordena las franjas
        # una prieta nueva puede caer sobre una golfina enterrada.
        prev = [(oc['x'], oc['y']) for oc in (nidos_ocupados or [])]

        # La rejilla no cambia durante una corrida y se consulta miles de veces
        # (una por evaluación y una por mutación), así que se memoriza.
        clave = (nombre_zona, sep_min, n_requeridos, len(prev),
                 round(sum(p[0] for p in prev), 1),
                 round(sum(p[1] for p in prev), 1),
                 round(sum(p[0] * p[1] for p in prev), 1))
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

    def slots_suficientes(self, nombre_zona, sep_min, nidos_ocupados,
                          n_necesarios):
        """Casillas en orden de llenado, garantizando que alcancen para n_necesarios.

        `slots_ordenados` aprieta el paso de la rejilla para alojar
        `n_requeridos`, pero despues descarta las casillas que caen junto a un
        nido ya enterrado, asi que puede devolver menos de las pedidas. Aqui se
        vuelve a pedir con una meta mayor hasta que alcancen.

        Existe para que quien COLOCA los nidos y quien JUZGA la colocacion usen
        exactamente la misma rejilla. Cuando no era asi, el termino de orden
        evaluaba el llenado secuencial contra una rejilla de 187 casillas para
        200 nidos, y le daba cero al llenado que es, literalmente, secuencial.
        """
        lim = self.limites_zona(nombre_zona)
        previos = list(nidos_ocupados or [])
        # Cuentan todos los nidos enterrados dentro de esta franja: cada uno
        # ocupa una casilla que esta zona ya no puede usar.
        n_previos = sum(
            1 for p in previos
            if lim and lim['xmin'] <= float(p.get('x', -1)) <= lim['xmax']
            and lim['ymin'] <= float(p.get('y', -1)) <= lim['ymax'])

        objetivo = int(n_necesarios) + n_previos
        slots = self.slots_ordenados(nombre_zona, sep_min, previos,
                                     n_requeridos=objetivo)
        intentos = 0
        while len(slots) < n_necesarios and intentos < 20:
            intentos += 1
            objetivo = int(objetivo * 1.25) + 1
            slots = self.slots_ordenados(nombre_zona, sep_min, previos,
                                         n_requeridos=objetivo)
        return slots

    def __repr__(self):
        return (f"GestorZonas(dinámica)\n"
                f"  Corral: {self.largo_cm}cm × {self.ancho_cm}cm\n"
                f"  Golfina: {self.conteo['golfina']} nidos\n"
                f"  Prieta:  {self.conteo['prieta']} nidos\n"
                f"  Laúd:    {self.conteo['laud']} nidos")


class GestoresZonas:
    """Los seis repartos posibles de las franjas, listos para usar.

    El reparto de zonas es parte del cromosoma: cada individuo lleva en
    `idx_orden` cual de los ORDENES_ZONAS usa. Se precalculan los seis
    gestores en vez de construir uno por evaluacion, porque lo caro de un
    gestor no es crearlo sino la rejilla de casillas que memoriza, y esa se
    consulta miles de veces por corrida.

    Seis es el numero exacto de permutaciones de tres especies. Si algun dia
    entran mas especies habra que cambiar de estrategia: las permutaciones
    crecen como el factorial.
    """
    def __init__(self, corral, n_golfina, n_prieta, n_laud, base=None,
                 ordenes=None):
        # `ordenes` con un solo elemento deja el reparto FIJO: el AG ya no
        # puede cambiarlo. Es lo que corresponde cuando el corral ya tiene
        # nidos enterrados, porque esos no se pueden mover de franja.
        self.gestores = [
            GestorZonas(corral, n_golfina, n_prieta, n_laud, base, orden=o)
            for o in (ordenes or ORDENES_ZONAS)
        ]
        self.conteo = self.gestores[0].conteo
        self.total  = self.gestores[0].total

    def para(self, idx_orden=0):
        """El gestor que corresponde a ese individuo."""
        return self.gestores[int(idx_orden or 0) % len(self.gestores)]

    def __len__(self):
        return len(self.gestores)

    # --- Compatibilidad -------------------------------------------------
    # Quien trate a este contenedor como un gestor unico obtiene el reparto
    # por omision, que es el orden historico golfina-prieta-laud. Sirve para
    # el codigo y las pruebas que se escribieron antes de que el orden fuera
    # una decision del algoritmo.
    @property
    def limites(self):
        return self.gestores[0].limites

    @property
    def separacion_efectiva(self):
        return self.gestores[0].separacion_efectiva

    @property
    def orden(self):
        return self.gestores[0].orden

    def limites_zona(self, nombre_zona):
        return self.gestores[0].limites_zona(nombre_zona)

    def slots_ordenados(self, *a, **k):
        return self.gestores[0].slots_ordenados(*a, **k)

    def slots_suficientes(self, *a, **k):
        return self.gestores[0].slots_suficientes(*a, **k)


def orden_establecido(nidos_previos, minimo_por_especie=1):
    """Reparto de franjas que YA tiene el corral, leido de los nidos enterrados.

    Una vez sembrada la primera jornada, el reparto deja de ser una decision:
    los nidos estan enterrados en su franja y no se pueden mover. Pedirle al
    personal que la proxima jornada ponga a la golfina en el extremo opuesto
    no es una recomendacion, es una imposibilidad fisica.

    El orden se deduce de la posicion media de cada especie a lo largo del
    corral. Devuelve None si no hay nidos previos, que es el unico momento en
    que el AG puede elegir el reparto libremente.
    """
    if not nidos_previos:
        return None

    acum = {}
    for n in nidos_previos:
        esp = n.get('especie') if isinstance(n, dict) else getattr(n, 'especie', None)
        x = float(n.get('x', 0.0) if isinstance(n, dict) else getattr(n, 'x', 0.0))
        if esp in ESPECIES:
            acum.setdefault(esp, []).append(x)

    presentes = {e: sum(v) / len(v) for e, v in acum.items()
                 if len(v) >= minimo_por_especie}
    if not presentes:
        return None

    # Las especies que ya estan, en el orden en que ocupan el corral; las que
    # todavia no aparecen se anaden al final conservando el orden historico.
    ordenadas = [e for e, _ in sorted(presentes.items(), key=lambda kv: kv[1])]
    faltantes = [e for e in ESPECIES if e not in ordenadas]
    return tuple(ordenadas + faltantes)


def resolver_gestor(gestor, idx_orden=0):
    """El GestorZonas que le toca a un individuo.

    Acepta tanto un contenedor GestoresZonas como un GestorZonas suelto, para
    que todo lo escrito antes de que el orden de zonas fuera parte del
    cromosoma siga funcionando sin cambios.
    """
    return gestor.para(idx_orden) if hasattr(gestor, 'para') else gestor


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



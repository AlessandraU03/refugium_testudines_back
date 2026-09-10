"""
Modelo de capacidad y rotación de corrales de incubación.

Responde tres preguntas operativas:
  1. ¿Cuántos nidos caben con seguridad en un corral?
  2. Dada la llegada de nidos a lo largo de la temporada, ¿cuántos hay
     incubando simultáneamente cada día?
  3. ¿Qué días la demanda supera la capacidad segura, y por cuántos nidos?

Todas las constantes provienen de fuentes citadas. Ninguna es estimada.
"""

from datetime import timedelta


# Densidad máxima segura. No es una distancia sino una densidad, y el motivo
# documentado es triple: calor metabólico entre nidos vecinos, disponibilidad
# de gases respiratorios, y espacio para que el personal circule.
#   "Maintain a density of 1 nest/m2 to minimise the effects of adjacent nests
#    on temperature and respiratory gas availability, and allow space for
#    hatchery workers to move."
#   Best Practices in Sea Turtle Hatchery Management for South Asia (2018),
#   citando a Mortimer et al. (1999), Shanker et al. (2003), Ahmad et al.
#   (2004) y Maulany et al. (2012).
# Coincide con la separación de 1 m que NOM-162-SEMARNAT-2012 establece para
# golfina y que Corzo-Domínguez y Romero-Berny (2025) documentan en el corral
# de Puerto Arista.
DENSIDAD_MAX_NIDOS_M2 = 1.0

# La casilla no se libera al eclosionar sino al excavar el nido.
#   "Excavate nest 2-3 days after the majority of hatchlings have emerged"
#   Best Practices in Sea Turtle Hatchery Management for South Asia (2018).
DIAS_HASTA_EXCAVACION = 3

# La arena no se reutiliza entre temporadas. No hay un período de descanso
# documentado entre nidos individuales dentro de una misma temporada; el
# reemplazo es de temporada a temporada.
#   "we recommend sea turtle hatchery to change sand after every nesting
#    season in order to maximize hatchling productions."
#   Hoh et al., "Nest microbiota and pathogen abundance impact hatching
#   success in sea turtle conservation". FSSC 5.2% en corrales que reutilizan
#   arena contra 1.3% en playas naturales.
REEMPLAZO_ARENA = "cada_temporada"


def capacidad_segura(largo_m, ancho_m, densidad=DENSIDAD_MAX_NIDOS_M2):
    """Nidos que caben simultáneamente sin exceder la densidad segura."""
    return int(float(largo_m) * float(ancho_m) * densidad)


def separacion_implicita(largo_m, ancho_m, n_nidos):
    """
    Separación media que resultaría de meter n_nidos en esa superficie.
    Sirve para cuantificar la compresión cuando se rebasa la capacidad.
    """
    if n_nidos <= 0:
        return float('inf')
    return (float(largo_m) * float(ancho_m) / n_nidos) ** 0.5


def simular_temporada(arribos, capacidad_total, dias_incubacion,
                      dias_excavacion=DIAS_HASTA_EXCAVACION):
    """
    Simula la ocupación del corral día por día.

    Args:
        arribos: dict {date: n_nidos} nidos recolectados cada día
        capacidad_total: nidos simultáneos que caben con seguridad
        dias_incubacion: días de incubación de la especie
        dias_excavacion: días extra hasta liberar la casilla

    Returns:
        dict {date: {'ocupados', 'libres', 'deficit'}} donde `deficit` son los
        nidos que ese día no tienen lugar a densidad segura.
    """
    ocupacion_dias = dias_incubacion + dias_excavacion

    ocupados = {}
    for fecha, n in arribos.items():
        for k in range(ocupacion_dias):
            d = fecha + timedelta(days=k)
            ocupados[d] = ocupados.get(d, 0) + n

    return {
        d: {
            'ocupados': n,
            'libres':   max(0, capacidad_total - n),
            'deficit':  max(0, n - capacidad_total),
        }
        for d, n in sorted(ocupados.items())
    }


def resumen_temporada(serie, capacidad_total, largo_m=None, ancho_m=None):
    """Indicadores agregados de una simulación de temporada."""
    if not serie:
        return {}

    pico = max(v['ocupados'] for v in serie.values())
    fecha_pico = max(serie, key=lambda d: serie[d]['ocupados'])
    dias_deficit = sum(1 for v in serie.values() if v['deficit'] > 0)

    res = {
        'capacidad_segura':   capacidad_total,
        'pico_ocupacion':     pico,
        'fecha_pico':         fecha_pico,
        'deficit_maximo':     max(v['deficit'] for v in serie.values()),
        'dias_con_deficit':   dias_deficit,
        'dias_simulados':     len(serie),
        'cobertura_del_pico': capacidad_total / pico if pico else 1.0,
        'area_faltante_m2':   max(0, pico - capacidad_total) / DENSIDAD_MAX_NIDOS_M2,
    }
    if largo_m and ancho_m:
        res['separacion_en_pico_m'] = separacion_implicita(largo_m, ancho_m, pico)
    return res


def recomendacion(largo_m, ancho_m, n_nidos, tasa_base=0.75,
                  densidad_norma=DENSIDAD_MAX_NIDOS_M2, huevos_por_nido=None,
                  supervivencia_fuera=0.0, mes=9):
    """Cuantos nidos conviene alojar, y que cuesta pasarse de esa cifra.

    El corral no puede decir solamente "no caben". El personal recibe los
    nidos de madrugada y tiene que sembrarlos: lo util es saber cual es el
    limite recomendado y, si hay que rebasarlo, cuanta eclosion cuesta.

    Args:
        n_nidos: nidos que hay que alojar
        tasa_base: tasa de eclosion de la especie sin hacinamiento
        supervivencia_fuera: fraccion de crias que se salvan de los nidos que
            NO entren al corral. Por omision 0.0, el peor caso: saqueados.
            Subirlo requiere un dato de campo del sitio.

    Devuelve el diagnostico y, cuando hay excedente, la comparacion entre
    apretar todos los nidos o alojar solo los recomendados.
    """
    import termico

    area_m2 = float(largo_m) * float(ancho_m)
    huevos = float(huevos_por_nido or termico.HUEVOS_REFERENCIA)
    capacidad = capacidad_segura(largo_m, ancho_m, densidad_norma)
    sep_norma_cm = round(100.0 / (float(densidad_norma) ** 0.5), 1)

    base = {
        'area_m2':               round(area_m2, 1),
        'nidos_solicitados':     int(n_nidos),
        'capacidad_recomendada': capacidad,
        'densidad_norma_m2':     float(densidad_norma),
        'separacion_norma_cm':   sep_norma_cm,
        'huevos_por_nido':       huevos,
        'fuente_densidad':       'Best Practices IOTN (2018); NOM-162-SEMARNAT-2012',
        'fuente_costo':          "Honarvar, O'Connor y Spotila (2008), Oecologia 157:221-230",
    }

    if n_nidos <= capacidad:
        holgura = capacidad - int(n_nidos)
        base.update({
            'estado':        'dentro_de_norma',
            'excedente':     0,
            'holgura':       holgura,
            'densidad_real_m2':   round(n_nidos / area_m2, 2) if area_m2 else 0.0,
            'separacion_real_cm': round(separacion_implicita(largo_m, ancho_m, n_nidos) * 100.0, 1),
            'costo_eclosion_pct': 0.0,
            'mensaje': ('Caben los %d nidos respetando la separacion de %.0f cm. '
                        'Quedan %d espacios libres.'
                        % (n_nidos, sep_norma_cm, holgura)),
        })
        return base

    # --- Hay mas nidos que capacidad: cuantificar el precio de apretarlos ---
    excedente = int(n_nidos) - capacidad
    densidad_real = n_nidos / area_m2
    # separacion_implicita trabaja en metros; aqui todo se reporta en cm
    sep_real = round(separacion_implicita(largo_m, ancho_m, n_nidos) * 100.0, 1)

    factor = termico.factor_eclosion_por_densidad(densidad_real)
    tasa_apretada = tasa_base * factor

    crias_apretando = n_nidos * huevos * tasa_apretada
    crias_solo_norma = (capacidad * huevos * tasa_base
                        + excedente * huevos * tasa_base * supervivencia_fuera)

    base.update({
        'estado':              'sobre_capacidad',
        'excedente':           excedente,
        'holgura':             0,
        'densidad_real_m2':    round(densidad_real, 2),
        'separacion_real_cm':  sep_real,
        'factor_eclosion':     round(factor, 3),
        'tasa_sin_hacinamiento': round(tasa_base, 3),
        'tasa_con_hacinamiento': round(tasa_apretada, 3),
        'costo_eclosion_pct':  round((1.0 - factor) * 100.0, 1),
        'comparacion': {
            'alojar_todos_apretados': {
                'nidos_alojados': int(n_nidos),
                'crias_esperadas': round(crias_apretando, 0),
            },
            'alojar_solo_lo_recomendado': {
                'nidos_alojados': capacidad,
                'nidos_fuera':    excedente,
                'supervivencia_fuera': supervivencia_fuera,
                'crias_esperadas': round(crias_solo_norma, 0),
            },
            'diferencia_crias': round(crias_apretando - crias_solo_norma, 0),
            'conviene': ('apretar' if crias_apretando > crias_solo_norma
                         else 'no_apretar'),
        },
        'mensaje': ('Se recomiendan %d nidos a %.0f cm de separacion. Alojar %d '
                    'obliga a bajar la separacion a %.0f cm (%.2f nidos/m2) y '
                    'cuesta %.1f%% de la eclosion de cada nido.'
                    % (capacidad, sep_norma_cm, n_nidos, sep_real,
                       densidad_real, (1.0 - factor) * 100.0)),
    })
    return base

from evaluacion  import evaluar_poblacion
from operadores  import (seleccion_torneo, cruza_por_especie,
                         mutacion_combinada, poda_elitismo)
from inicializacion import inicializar_poblacion

# ── Parámetros internos del AG ─────────────────────────────────────────────────
#
# MEDIDOS. Eran los ultimos coeficientes del sistema elegidos a mano; se
# sometieron al mismo barrido que _PROB_REPARACION, las pasadas de reparacion y
# el peso del sexo. Escenario: 200 nidos en 30 x 8 m -corral apretado, donde la
# posicion decide algo- mas la jornada real de 100 nidos en 30 x 40 m. Tres
# repeticiones por valor, variando uno a la vez.
#
# LO PRIMERO QUE HAY QUE SABER ES EL RUIDO. La configuracion base se midio tres
# veces por separado, una en cada bloque del barrido, y dio 0.7958, 0.8008 y
# 0.7936: un rango de 0.0072 sobre configuracion IDENTICA. Cualquier diferencia
# menor que eso no es efecto del parametro.
#
#   poblacion (gen=100)      aptitud    desv
#     20                     0.7939   0.0005
#     30                     0.7957   0.0014
#     50  <- actual          0.7958   0.0014
#     80                     0.7982   0.0058
#
#   prob. de cruza (pob=50)  aptitud    desv
#     0.60                   0.7948   0.0004
#     0.85  <- actual        0.8008   0.0065
#     1.00                   0.7956   0.0014
#
#   prob. de mutacion        aptitud    desv
#     0.05                   0.7938   0.0035
#     0.15  <- actual        0.7936   0.0116
#     0.30                   0.7934   0.0076
#     0.50                   0.7839   0.0062   <- PEOR, fuera del ruido
#
# CONVERGENCIA. Una corrida de 250 generaciones guarda la mejor aptitud de cada
# una, asi que la curva dice donde deja de mejorar sin pagar corridas aparte:
#
#     gen  10   20   40   60   80  100  150  200  250
#     %   91.8 93.0 95.6 97.5 98.4 99.2 99.8 99.9  100   del valor final
#
# El AG NO ha convergido en la generacion 100: va por el 99.2 % y alcanza el
# 99.9 % en la 184. La mejora es MONOTONA, y una tendencia sostenida 150
# generaciones no es ruido, porque el ruido no tiene direccion.
#
# Eso sugeria cambiar poblacion por generaciones -menos individuos, mas
# busqueda- al mismo costo. SE PROBO Y NO FUNCIONA:
#
#   configuracion        apretado (200 nidos)     jornada real (100 nidos)
#   pob=50 gen=100       0.7988 +- 0.0109 23.1s   0.8548 +- 0.0041 10.1s
#   pob=30 gen=200       0.7974 +- 0.0008 25.4s   0.8583 +- 0.0047 11.6s
#   pob=30 gen=250       0.8011 +- 0.0050 32.4s   0.8578 +- 0.0018 14.4s
#   pob=20 gen=250       0.7994 +- 0.0064 21.3s   0.8558 +- 0.0003  9.5s
#
# Las cuatro son indistinguibles en los dos escenarios. La busqueda esta
# SATURADA: en el rango probado ningun hiperparametro cambia el resultado, y el
# unico limite que se distingue es que pm=0.50 perjudica.
#
# CONCLUSION: se dejan como estaban. No porque no se midieran, sino porque la
# medicion dice que da igual, y entre valores equivalentes no hay razon para
# cambiar. Que el resultado sea robusto a los hiperparametros es una FORTALEZA
# del proyecto: significa que no es un artefacto de haberlos ajustado.
#
# ADVERTENCIA SOBRE LOS TIEMPOS: los del primer barrido salieron contaminados
# por carga de la maquina -la misma configuracion tardo 42, 45 y 62 s en
# bloques distintos-. Los de la tabla de arriba se midieron seguidos en un solo
# proceso y si son comparables entre si.
_TAM_POB    = 50
_N_GEN      = 100
_PROB_CRUZA = 0.85
_PROB_MUT   = 0.15


def ejecutar_ag(nidos_entrada, gestor, base, corral,
                nidos_ocupados=None, callback=None, mes=None):
    """`mes` es el mes de siembra (1-12). Importa: de él dependen la
    temperatura base de la arena y el tamaño de la nidada, y por tanto la
    proporción sexual que el AG intenta alcanzar. Sin él la aptitud evaluaba
    siempre septiembre, porque los genes que crea el AG no llevan fecha."""
   
    previos_raw = nidos_ocupados or []
    
    # Preprocesar nidos_ocupados a un caché NumPy para optimizar el rendimiento del AG
    import numpy as np
    previos_cache = {"__cache__": True}
    if nidos_ocupados:
        por_esp = {}
        for n in nidos_ocupados:
            e = n['especie']
            por_esp.setdefault(e, ([], [], []))
            por_esp[e][0].append(float(n['x']))
            por_esp[e][1].append(float(n['y']))
            por_esp[e][2].append(float(n.get('prof', 0)))
        
        for esp, (xs, ys, profs) in por_esp.items():
            e = base.get(esp, {})
            prof_opt = e.get('prof_opt', 45.0)
            xs_p = np.array(xs)
            ys_p = np.array(ys)
            prfs_p = np.array([float(pf) if pf != 0 else prof_opt for pf in profs])
            previos_cache[esp] = (xs_p, ys_p, prfs_p)

    h = {k: [] for k in [
        'mejor', 'promedio',
        'v1_mejor', 'v1_promedio',
        'v2_mejor', 'v2_promedio',
        'v3_mejor', 'v3_promedio',
        'indice_mejor', 'indice_promedio'
    ]}

    # ── 1. Inicialización ──────────────────────────────────────────────────────
    poblacion = inicializar_poblacion(
        _TAM_POB, nidos_entrada, gestor, base, previos_raw)

    # ── 2. Evaluación inicial ──────────────────────────────────────────────────
    evaluar_poblacion(poblacion, gestor, base, corral, previos_cache, mes)
    mejor_global = max(poblacion, key=lambda i: i.fitness).copia()
    _registrar(h, poblacion, mejor_global)

    fits_ini = [i.fitness for i in poblacion]
    prom_ini = sum(fits_ini) / len(fits_ini)
    print(f"\n{'='*60}")
    print(f"  AG: {len(nidos_entrada)} nidos nuevos | "
          f"{len(previos_raw)} previos | pop={_TAM_POB} | gen={_N_GEN}")
    print(f"  Fitness inicial -> mejor:{mejor_global.fitness:.4f} prom:{prom_ini:.4f}")
    print(f"  V1={mejor_global.v1:.3f} V2={mejor_global.v2:.3f} "
          f"V3={mejor_global.v3:.4f}")
    print(f"{'='*60}")

    # ── 3-6. Ciclo evolutivo ───────────────────────────────────────────────────
    for gen_num in range(_N_GEN):
        parejas      = seleccion_torneo(poblacion, k=3)
        descendencia = []

        for p1, p2 in parejas:
            h1, h2 = cruza_por_especie(p1, p2, _PROB_CRUZA)
            h1 = mutacion_combinada(h1, _PROB_MUT, base, gestor, previos_cache)
            h2 = mutacion_combinada(h2, _PROB_MUT, base, gestor, previos_cache)
            descendencia.extend([h1, h2])

        evaluar_poblacion(descendencia, gestor, base, corral, previos_cache, mes)
        poblacion = poda_elitismo(poblacion, descendencia, _TAM_POB, n_elite=2)

        mejor_actual = max(poblacion, key=lambda i: i.fitness)
        if mejor_actual.fitness > mejor_global.fitness:
            mejor_global = mejor_actual.copia()

        _registrar(h, poblacion, mejor_global)

        if (gen_num + 1) % 25 == 0:
            fits = [i.fitness for i in poblacion]
            prom = sum(fits) / len(fits)
            print(f"  Gen {gen_num+1:3d} | Mejor={mejor_global.fitness:.4f} "
                  f"Prom={prom:.4f} | V1={mejor_global.v1:.3f} "
                  f"V2={mejor_global.v2:.3f}")

        if callback:
            fits = [i.fitness for i in poblacion]
            callback(gen_num + 1, mejor_global.fitness, sum(fits)/len(fits))

    fits_fin = [i.fitness for i in poblacion]
    prom_fin = sum(fits_fin) / len(fits_fin)
    print(f"\n  [OK] Finalizado | Mejor={mejor_global.fitness:.4f} Prom={prom_fin:.4f}")
    print(f"  V1={mejor_global.v1:.3f} V2={mejor_global.v2:.3f} "
          f"V3={mejor_global.v3:.4f}")

    top3 = sorted(poblacion, key=lambda i: i.fitness, reverse=True)[:3]
    return mejor_global, top3, h


def _registrar(h, pob, mejor):
    fits = [i.fitness for i in pob]
    v1s  = [i.v1 for i in pob if i.v1 is not None]
    v2s  = [i.v2 for i in pob if i.v2 is not None]
    v3s  = [i.v3 for i in pob if i.v3 is not None]
    ies  = [i.indice_eclosion for i in pob if i.indice_eclosion is not None]

    h['mejor'].append(mejor.fitness)
    h['promedio'].append(sum(fits) / len(fits) if fits else 0)
    h['v1_mejor'].append(mejor.v1 or 0)
    h['v1_promedio'].append(sum(v1s) / len(v1s) if v1s else 0)
    h['v2_mejor'].append(mejor.v2 or 0)
    h['v2_promedio'].append(sum(v2s) / len(v2s) if v2s else 0)
    h['v3_mejor'].append(mejor.v3 or 0)
    h['v3_promedio'].append(sum(v3s) / len(v3s) if v3s else 0)
    h['indice_mejor'].append(mejor.indice_eclosion if mejor.indice_eclosion is not None else 1.0)
    h['indice_promedio'].append(sum(ies) / len(ies) if ies else 1.0)

# Refugium Testudines — AG para la colocación de nidos

Algoritmo genético que decide **dónde va cada nido** de tres especies —golfina,
prieta y laúd— dentro de un corral de incubación, y a qué especie le toca la
sombra. Es un programa general: el sitio (coordenadas, orientación, malla,
riego) se declara en `csv/sitio.csv`. El ejemplo configurado es una playa cerca
de Puerto Arista, Tonalá, Chiapas (15.94521, −93.83029).

El sistema recibe los nidos que el personal recolectó esa madrugada y devuelve
una colocación concreta: coordenada, profundidad y sector físico de cada uno,
respetando los nidos que ya están enterrados de jornadas anteriores. Si no
caben todos, lo dice con el número exacto en lugar de encimarlos.

---

## Qué optimiza

La colocación de un nido decide su temperatura, y la temperatura decide dos
cosas: cuántas crías salen y de qué sexo. Ese es el circuito que cierra
`termico.py`:

```
colocación + densidad  →  temperatura  →  ┌→ tasa de eclosión → crías
   + sombra + riego                       └→ modelo de Girondot → sexo
```

Dos decisiones de modelado, documentadas en el encabezado de `termico.py`:

1. **La densidad no entra en la temperatura del período termosensible.**
   Honarvar et al. (2008) hallaron diferencias significativas entre densidades
   solo en los días 37, 40 y 47. El PTS de la golfina va del día 17 al 33: cuando
   el hacinamiento calienta la arena, el sexo ya quedó determinado. Por eso el
   apiñamiento castiga la **eclosión**, no la proporción sexual.
2. **La eclosión sale de la curva de Honarvar, no se recalcula desde la
   temperatura.** Esa curva ya incorpora temperatura, caída de O₂ y acumulación
   de CO₂; derivarla otra vez contaría dos veces el mismo efecto.

---

## Tres especies y geometría solar

**Cada especie con sus propios parámetros** (`csv/`):

| Especie | Profundidad (NOM-162) | Separación | Piso físico | Pivote | Riesgo UICN |
|---|---|---|---|---|---|
| Golfina | 40–50 cm | 100 cm | 60 cm | 29.95 °C | VU |
| Prieta | 45–60 cm | 120 cm | 60 cm | 29.2 °C | EN |
| Laúd | 75–85 cm | 150 cm | 70 cm | 29.4 °C | **CR** (Pacífico oriental) |

Pivotes: golfina de Sandoval et al., Playa Ceuta, Sinaloa; prieta de Godfrey y
Mrosovsky (2006), Suriname — **población atlántica, importada**; laúd, extremo
inferior del rango 29.4–29.8 °C documentado.

**La sombra se calcula por geometría solar** (`solar.py`), nido por nido: con
la latitud, la fecha, la orientación del corral y la altura de la malla se
sigue la trayectoria del sol cada 10 minutos, y se mide qué fracción de la
insolación directa del día intercepta la malla sobre ese nido. La geometría se
validó contra controles astronómicos: 73.99° de altura al equinoccio (esperado
74.1°) y 12.00 h de día.

Con la malla cubriendo el corral completo, el gradiente que queda es **centro
contra perímetro**: el toldo tapa por arriba pero los lados están abiertos, así
que el sol entra de lado por la mañana y por la tarde. Medido en el corral de
campo en septiembre: 96.8 % de sombra en el centro contra 48.6 % en el borde.

**Esa fracción se convierte en temperatura** con las mediciones de Hill et al.
(2015): la sombra plena baja la arena 2.2 °C a 45 cm y 1.3 °C a 75 cm; entre
ambas profundidades se interpola, y el efecto escala con la fracción de sombra
y con la opacidad de la malla.

**El riego es la segunda intervención**, de las mismas mediciones y con su curva
de dosis. Es la única que se paga cada temporada, y de ahí sale el hallazgo
económico más fuerte del proyecto:

| Dosis | Agua (corral de 240 m²) | Enfriamiento | °C por m³ |
|---|---|---|---|
| 100 mm | 24 m³ | 1.31 °C | 0.055 |
| 323 mm | 77.5 m³ | 1.81 °C | 0.023 |
| 721 mm | 173 m³ | 1.91 °C | 0.011 |

Los primeros 100 mm compran cinco veces más grados por metro cúbico que el
último tramo. Y regar **solo el perímetro** cuesta 10.5 m³ en lugar de 24, y
lleva al borde de 99.4 % de hembras y +0.37 °C de margen letal a 93 % y
+1.93 °C. `TabRecomendacion` presenta el intercambio; no decide por el
Santuario.

Cada nido de la interfaz trae **el desglose de su temperatura**: cuánto le quitó
la malla y cuánto el agua. En el perímetro el agua aporta 2.05 °C contra 1.11 °C
de la malla, o sea ahí el nido lo salva el riego, no la sombra.

**El AG decide también a qué especie le toca cada franja** del corral (seis
repartos posibles). Las franjas tienen ancho proporcional al área que la especie
necesita (nidos × separación²), no a su número de nidos.

**El reparto se decide una sola vez, con el corral vacío.** A partir de la
primera siembra queda fijo toda la temporada, y si ya hay nidos enterrados el
programa lo deduce de sus posiciones y lo bloquea. Cambiar una especie de franja
obligaría a desenterrar nidadas: no es una recomendación, es una imposibilidad.

---

## Función de aptitud

```
biológico = eclosión × profundidad × letal × densidad_norma × separación
fitness   = biológico × sexo
```

| Término | Qué mide | De dónde sale |
|---|---|---|
| `eclosión` | Crías que sobreviven al hacinamiento de esta colocación | Honarvar, O'Connor y Spotila (2008), *Oecologia* 157:221-230 |
| `profundidad` | Cumplimiento del rango de profundidad | NOM-162-SEMARNAT-2012 §6.8.4, Tabla 1 |
| `letal` | Margen por debajo de los 36 °C del último tercio | Carbonell Ellgutter et al. (2025), *Ecol Evol* 15(7):e71750 |
| `densidad_norma` | Fracción de nidos dentro de 1 nido/m² | Best Practices IOTN (2018); NOM-162-SEMARNAT-2012 |
| `separación` | Cumplimiento frente a todo vecino, contra la separación alcanzable | NOM-162-SEMARNAT-2012 |
| `sexo` | Cercanía de la cohorte a la proporción de referencia | de la Torre-Robles et al. (2017), *Agroproductividad* 10(5):39-45 |

**Los dos objetivos se maximizan juntos, como producto.** Antes la proporción
sexual entraba como desempate con peso 0.05, subordinada a la eclosión, bajo el
argumento de que una cría que no eclosiona no tiene sexo. Ese argumento sigue
siendo cierto y por eso los términos se **multiplican**: sin eclosión el
producto es cero y ninguna proporción sexual lo rescata. Pero ya no hace falta
subordinar el sexo, porque **en este corral los dos objetivos no compiten**.
Medido con 200 nidos de golfina: meterlos todos bajo la malla mantiene las
13 800 crías —la curva de Honarvar es plana por debajo de 2 nidos/m²— mejora la
proporción sexual de 99.8 % a 97.6 % de hembras y reduce los nidos en riesgo
letal de 192 a 1. Siendo la misma decisión la que mejora ambos, subordinar uno
al otro solo servía para que el AG ignorara la mitad de su trabajo.

### La meta de proporción sexual, y cuál cifra se persigue

`OBJETIVO_PCT_HEMBRAS = 55.9`. No es un 50/50 por simetría — las poblaciones de
tortuga marina son naturalmente sesgadas hacia hembras y forzar el equilibrio no
tiene respaldo. Es la proporción que implica la temperatura **medida** en un
corral mexicano comparable: de la Torre-Robles et al. (2017) registraron 30.1 °C
durante el segundo tercio de la incubación en nidos de golfina a ~45 cm en el
corral de San Juan Chacahua, Oaxaca, con 86.6 % de eclosión.

La aptitud persigue la proporción **agregada de la cohorte** —hembras sobre el
total de crías, pesada por nidada—, que es la que se libera al mar. A la
población le da igual que cada nido acierte el 55.9 % si el conjunto sale
equilibrado.

Hubo un tiempo en que la interfaz mostraba esa cifra y la aptitud optimizaba
otra: el promedio de la calidad nido por nido, elegido porque la agregada es un
solo número para toda la colocación y mover un nido la cambia 0.015 puntos sobre
170 nidos. Se **midió** si ese gradiente importaba en el resultado, con cinco
pesos entre las dos medidas: el rango completo fue 0.005 y 0.007 en dos
escenarios, sin tendencia, o sea ruido entre repeticiones. Queda en el agregado
puro, sin coeficientes de modelado injustificables.

### Por qué se retiró el término de separación como objetivo… y volvió como métrica

La aptitud anterior era `eclosión × (profundidad + separación) / 2`, donde
separación contaba qué fracción de nidos respeta los 100 cm de la NOM-162. Se
volvía perverso justo cuando más importa: con 270 nidos en 240 m² la rejilla
tiene que apretarse por debajo de la norma, de modo que **ningún** nido puede
cumplirla; el llenado uniforme —el óptimo biológico— obtenía separación 0.000 y
aptitud 0.500, mientras el AG llegaba a 0.86 apiñando nidos hasta 3 nidos/m² y
produciendo **diez crías menos**.

Ahora se mide contra la separación **alcanzable**, no contra la de la norma, así
que empuja hacia el óptimo biológico en vez de pelearse con él.

### El término de orden de llenado salió

Medía si el corral se puede sembrar recorriendo hileras en serpentina: una
comodidad para quien siembra, no un beneficio para la nidada. Cuando un criterio
humano y uno biológico compiten, gana el biológico. Medirlo lo confirmó: forzar
el llenado ordenado bajaba el cumplimiento de separación de 1.00 a 0.88 y subía
los nidos en riesgo letal.

---

## El piso físico de separación

La separación de la NOM-162 es el **objetivo**. El piso es el límite por debajo
del cual la colocación deja de ser ejecutable:

```
piso = radio_cámara(a) + radio_cámara(b) + pared_arena
```

Los dos radios son geometría: por debajo de su suma las dos excavaciones se
intersecan, y eso no es incumplir una norma, es pedir que se entierren dos
nidadas en el mismo hoyo. Los diámetros de cámara están en
`csv/camara_incubacion.csv`; la pared de arena es un **supuesto declarado**, con
su argumento: no debería ser más delgada que la parte más angosta de la
excavación que la produce, y el cuello de la cámara mide 15–20 cm.

**En este corral el piso lo fija el cerco de conteo, no la cámara.** Todos los
nidos del Santuario llevan el cerco de malla fina de 60 cm de diámetro que se
instala para contar las crías emergidas, así que la huella en superficie es el
cerco y no la cámara: golfina sube de 45 a 60 cm y prieta de 55 a 60 cm. Solo el
laúd conserva su piso por cámara, 70 cm, porque ya es más ancha que el cerco.
Cuesta el 42 % de la capacidad física del corral, y es el número correcto: no se
puede sembrar más denso de lo que permite lo que se instala encima.

Hizo falta porque el sistema apretaba la rejilla **sin fondo**:
`separacion_para_alojar` bisecaba hasta 1 cm y llegó a proponer **33.2 cm** entre
nidos de golfina, menos que el ancho de una sola cámara. Con el piso puesto, el
mismo caso da 88.9 cm.

**Es restricción, no penalización.** Metido en la aptitud, el AG lo negociaría
—cedería cumplimiento a cambio de crías—, que es exactamente el error que ya
tuvo el término de separación. Se impone al construir: la rejilla no genera
casillas por debajo del piso, las mutaciones rechazan al candidato que lo viole,
y `reparar_piso` corrige a todo hijo antes de evaluarlo.

Durante la búsqueda la reparación corre con un número acotado de pasadas, porque
se aplica a 5 000 individuos por corrida, y en corral saturado eso deja
violaciones residuales. Así que **lo que se reporta se garantiza aparte**:
`reparar_hasta_converger` itera hasta que no quede ninguna violación, y se
aplica al mejor, al Top 3 y a la referencia secuencial. Si una colocación se
reparó, se reevalúa, porque mover un nido cambia su aptitud.

### Un defecto que esto destapó

La cruza intercambia genes posición por posición, y los dos padres nacen de la
misma rejilla: un hijo podía recibir el gen *k* de un padre y el *k+1* del otro
apuntando a la misma casilla. Medido: **20 nidos del mismo individuo con
coordenadas idénticas**. El defecto ya existía; nadie lo veía porque ninguna
comprobación buscaba coordenadas repetidas y la aptitud solo mide densidad
local, que con un nido justo encima de otro apenas se mueve.

---

## Cuando no caben todos

Con el piso puesto el corral tiene un límite real, y la jornada cae en una de
tres bandas:

| Banda | Qué significa | Qué hace el programa |
|---|---|---|
| **Conforme a la NOM-162** | Caben con la separación de la norma | Los siembra todos |
| **Apretada** | Caben, pero por debajo de la norma | Los siembra todos: apretar cuesta eclosión, pero dejar un nido fuera lo expone a saqueo y depredación, que es peor |
| **Excedida** | No caben ni al límite físico | Admite los que caben y **dice cuántos no**, con otro destino: segundo corral o *in situ* con protección |

En la tercera banda, **quién entra lo decide el riesgo de extinción**
(`csv/riesgo_extincion.csv`): laúd del Pacífico oriental primero, luego prieta,
y la golfina absorbe el faltante.

**No se maximizan crías a propósito.** Sería lo natural en un optimizador y da
el resultado contrario al objetivo del proyecto: el laúd pone nidadas más
pequeñas y necesita 150 cm de separación contra los 100 de la golfina, así que
rinde muchas menos crías por metro cuadrado. Un criterio de crías puras dejaría
fuera al laúd, que está En Peligro Crítico, para meter más golfinas, que están
Vulnerables. Preservar una especie en peligro no es lo mismo que producir el
mayor número de crías, y aquí esas dos cosas se contradicen: es una decisión de
conservación, no de algoritmo, y por eso vive en un CSV discutible.

La capacidad se comprueba contra la geometría **real de las franjas**, no contra
el área. El área decía que cabían 280 nidos en 10 × 8 m mientras la franja de
prieta tenía 56 casillas para 60 nidos y la de laúd 11 para 20: el AG recibía 13
nidos sin dónde ir.

Medido, con 500/100/40 nidos en un corral de 10 × 10 m: entran los 40 laúd
completos, los 100 prieta completos, y de las 500 golfinas entran 32. Con el
piso por cámara —sin cerco— habrían entrado 186: esa es la diferencia que hace
el cerco, y es capacidad que el corral no tiene aunque el modelo anterior
dijera que sí.

---

## Contra qué se compara

Cada corrida evalúa también una **colocación de referencia**: el llenado
secuencial de la rejilla, que es el procedimiento actual del corral sin ningún
algoritmo. Se evalúa con la misma función de aptitud, el mismo modelo térmico,
el mismo corral, los mismos nidos previos y **las mismas franjas que eligió el
AG** — la única diferencia es dónde quedó cada nido dentro de su franja. Así la
comparación no le cuenta al AG la ventaja de haber elegido el reparto. Se
eligió esa referencia y no una colocación al azar porque nadie siembra al azar.

**En crías, el AG empata, y hay que presentarlo tal cual.** Con el corral sin
saturar, las dos colocaciones caben respetando la separación de la norma, así
que ninguna pierde crías por hacinamiento: 6 679 crías en ambos casos. El índice
de eclosión está acotado a 1.0 y una rejilla bien espaciada ya lo alcanza, así
que **nadie puede superarla**.

**La ventaja del AG está en la separación y en la proporción sexual.** Jornada
del reporte (70 golfina, 25 prieta y 5 laúd en 30 × 20 m, 22 de septiembre):
cumplimiento de separación 1.0000 contra 0.9800, calidad de proporción sexual
0.8455 contra 0.8326, 71.3 % de hembras contra 72.6 %, aptitud 0.8452 contra
0.8133. Todas las pruebas son simulaciones dentro del programa: la colocación
todavía no se ha aplicado en el corral del Santuario.

---

## Operadores genéticos

Los cuatro hiperparámetros están **medidos**, con el mismo barrido que el resto
de los coeficientes. La conclusión es que no discriminan: la configuración base
medida tres veces por separado dio 0.7958, 0.8008 y 0.7936 —un rango de 0.0072
sobre configuración idéntica— y ninguna diferencia entre configuraciones supera
ese ruido. El único límite que se distingue es que `pm = 0.50` perjudica.

La curva de convergencia sí dice algo: el AG **no ha convergido** en la
generación 100, va por el 99.2 % del valor final y alcanza el 99.9 % en la 184.
Eso sugería cambiar población por generaciones al mismo costo; se probó
(pob 30/gen 200, pob 30/gen 250, pob 20/gen 250) y las cuatro combinaciones son
indistinguibles en los dos escenarios. **La búsqueda está saturada.**

Se dejaron como estaban, no por falta de medición sino porque la medición dice
que da igual. Que el resultado sea robusto a los hiperparámetros es una
fortaleza: no es un artefacto de haberlos ajustado. La tabla completa está en el
encabezado de `ag.py`.

| Componente | Configuración |
|---|---|
| Población | 50 individuos, 100 generaciones (**medido**: 20–80 y 100–250 dan lo mismo) |
| Inicialización | Mitad sembrada sobre la rejilla con ruido creciente, mitad **libre** dentro de zona |
| Selección | Torneo, k = 3 |
| Cruza | Uniforme por especie, p = 0.85 (**medido**: 0.60–1.00 dan lo mismo) |
| Mutación | Orden de franjas + profundidad gaussiana + reposicionamiento local + colocación libre, p = 0.15 (**medido**: 0.05–0.30 igual, 0.50 peor) |
| Reparación del piso | 2 pasadas por hijo (**medido**), y hasta converger en lo que se reporta |
| Poda | Elitismo con reemplazo aleatorio: conserva los 2 mejores de padres + hijos y elige el resto al azar (no es μ+λ) |

**La mitad aleatoria de la población coloca libremente, no sobre la rejilla.** Si
nace sobre las mismas casillas que la mitad sembrada, la población entera arranca
en la rejilla y ninguna mutación logra sacarla: medido, las 100 colocaciones
quedaban en 27 columnas por 10 filas, idénticas al llenado secuencial. Con
colocación libre, 45 × 31.

**La reparación de orden está en 0.0, medido.** Con 100 nidos nuevos sobre 70
enterrados, aplicarla siempre pegaba a toda la población a la misma rejilla y el
cumplimiento de separación caía de 1.000 a 0.880. El código se conserva porque
el llenado en serpentina tiene un valor operativo que la aptitud no mide.

---

## Rendimiento

Medido en **proceso limpio**, que es lo que ve quien usa la aplicación (una
medición con caché caliente da números tres o cuatro veces mejores y no
representa nada):

| Jornada | Antes | Ahora |
|---|---|---|
| 100 nidos en 30 × 40 m | 41.0 s | **10.1 s** |
| 270 nidos en 30 × 8 m | 108.1 s | **36.0 s** |
| 326 admitidos en 10 × 10 m | 58.0 s | **52.5 s** |

El cuello era `solar.fraccion_sombra_dia`: recorría en Python los 144 instantes
del día para cada nido, y su memoria casi nunca acertaba porque el AG mueve los
nidos por posiciones continuas. Vectorizado como matriz de nidos × instantes,
**idéntico a 5.6e-16** en cinco fechas del año.

**Lo que no funcionó:** vectorizar `densidades_locales`, que el perfilado
señalaba como el mayor costo. Se implementó, se comprobó idéntica, y resultó
1.4 veces **más lenta** que el agrupamiento por celdas que ya había.

---

## Estructura

```
backend/
├── api.py             ← API REST Flask (7 endpoints)
├── ag.py              ← Ciclo evolutivo
├── cromosoma.py       ← Gen, Individuo, GestorZonas, piso, cupo, RegistroJornadas
├── evaluacion.py      ← Función de aptitud (única, con coeficientes citados)
├── termico.py         ← Modelo térmico: densidad, sombra y riego → crías y sexo
├── solar.py           ← Trayectoria del sol y sombra de la malla sobre cada nido
├── inicializacion.py  ← Población inicial y colocación de referencia
├── operadores.py      ← Selección, cruza, mutación, reparación del piso, poda
└── csv/               ← Base de conocimiento; cada fila declara su fuente
    ├── sitio.csv                ← Coordenadas, orientación, malla, riego, pared, cerco
    ├── camara_incubacion.csv    ← Diámetros de cámara → piso de separación
    ├── riesgo_extincion.csv     ← Categorías UICN → prioridad de cupo
    ├── corral_incubacion.csv    ← Medidas del corral de trabajo
    ├── profundidad_siembra.csv  ├─ separacion_minima.csv
    ├── pts_termosensible.csv    ├─ dias_incubacion.csv
    ├── fecundidad_mensual.csv   ├─ tasa_eclosion.csv
    └── anidacion_mensual.csv    └─ corrales.csv

frontend/src/
├── App.jsx
└── components/
    ├── PanelEntradas.jsx    ← Nidos de la jornada, fecha y medidas del corral
    ├── TarjetaCupo.jsx      ← Banda de la jornada y quién entra si no caben
    ├── TabCorral.jsx        ← Diagrama con iconos por especie, malla y zona de riego
    ├── TarjetaReparto.jsx   ← Reparto de franjas elegido y supuestos del sitio
    ├── TabRecomendacion.jsx ← Dónde, cuánto y cómo regar; reemplazo de arena
    ├── TabPTS.jsx           ← Proporción sexual nido por nido
    ├── TabEvolucion.jsx     ← Aptitud por generación
    ├── TabTop3.jsx          ← Las 3 mejores colocaciones
    ├── TabTemporada.jsx     ← Nidos ya sembrados
    ├── TabClustering.jsx    ← Densidad local del corral
    ├── TabValidacion.jsx    ← Contra tasas de eclosión empíricas
    ├── TabFechas.jsx        ← Fechas de eclosión
    └── TabVariables.jsx     ← Evolución de V1–V3
```

`dist/` está versionado a propósito: el build viaja con el código.

---

## Ejecución

Backend:

```bash
cd backend && python api.py
```

Frontend:

```bash
cd frontend && npm install && npm run dev
```

### Configuración del servidor

| Variable | Para qué |
|---|---|
| `CORS_ORIGINS` | Dominios extra (separados por comas) desde los que el navegador puede llamar a la API. Por omisión se aceptan `localhost`, `127.0.0.1` y los subdominios de `onrender.com`, `vercel.app`, `netlify.app` y `alessandrau03.github.io`. Si el frontend se publica en otro dominio, hay que agregarlo aquí o la interfaz no podrá conectarse. |
| `FLASK_DEBUG=1` | Activa el depurador de Flask al correr `python api.py`. Solo para desarrollo local: permite ejecutar código desde el navegador. |

`/api/nueva-temporada` borra todo el registro y solo actúa si la petición manda
`{"confirmar": true}`; la interfaz además pide confirmación. Las entradas
inválidas responden 400 con un mensaje, y los errores internos responden 500
sin exponer el código (el detalle queda en el registro del servidor).

**El registro de jornadas no es persistente en Render.** `jornadas.json` vive
en el disco del servidor, que Render borra en cada reinicio o despliegue. Para
una temporada real hace falta un disco persistente o una base de datos.

---

## Lo que todavía no se sostiene

Nada de esto debe presentarse como hallazgo.

1. **Diez parámetros del sitio son supuestos**, marcados `supuesto=si` en
   `csv/sitio.csv` y listados en la interfaz. Los que más pesan: orientación del
   corral (115°), altura de la malla (250 cm) y atenuación (80 %). Hill et al. no
   reportan el porcentaje de sombra de sus parcelas, así que la conversión de
   porcentaje de malla a grados también es un supuesto.

   La **pared de arena** (20 cm) sigue sin medir, pero dejó de decidir: en este
   corral el piso lo fija el cerco de conteo de 60 cm, que es dato de campo, y
   solo el laúd depende de la cámara más la pared. Sigue valiendo la pena
   preguntar al personal a qué distancia han visto derrumbes, pero la capacidad
   del corral ya no cuelga de ahí.

2. **La temperatura base de la arena es una estimación.** El modelo da 32.01 °C
   para el PTS en septiembre; Chacahua midió 30.1 °C en un corral comparable. De
   esos dos grados depende que el corral produzca 96 % o 56 % de hembras. Es el
   supuesto más grave del sistema, y la única forma de cerrarlo es medir la arena
   del corral a 45 cm durante los días 17 al 33.

3. **La temperatura pivote está pendiente de verificación.** Se usan los
   29.95 °C de Sandoval et al. (2020). Hay literatura reciente que sugiere que
   la pivote de campo es más alta, pero no fue accesible para comprobarlo.

4. **El efecto del riego corresponde a una dosis concreta.** Los coeficientes de
   Hill et al. (2015) son los de un riego promedio de 323 mm diarios durante 31
   días. Regar menos, o más espaciado, no produce ese enfriamiento. El modelo
   tampoco representa el efecto de la humedad sobre la viabilidad del nido: no
   hay coeficiente para eso y no se inventó.

5. **El modelo solo sigue la radiación directa**, sin nubosidad ni radiación
   difusa, en plena temporada de lluvias.

6. **Las profundidades tenían dos fuentes que no coincidían, y está resuelto.**
   El código usa la NOM-162 (golfina 40–50, prieta 45–60, laúd 75–85 cm). Una
   tabla de réplica de nido daba golfina 40–50 —idéntico—, pero prieta 70–85 y
   laúd 75–100. La diferencia de 25 cm en prieta no es cosmética: a 50 cm el
   modelo da 86.1 % de hembras y a 77.5 cm da 93.3 %, siete puntos en una
   especie En Peligro. **Decisión tomada: se conserva la NOM-162**, que es la
   norma vinculante y está citada con sección y tabla. De esa tabla se usan solo
   los diámetros de cámara, que es lo que la norma no especifica.

7. **El cerco de conteo está confirmado, y corrige un supuesto mío.** Yo había
   asumido que se instala por muestreo, porque sirve para contar crías
   emergidas. Va en **todos** los nidos. Su huella de 60 cm sustituye a la de la
   cámara, el piso de golfina y prieta sube a 60 cm, y cuesta el 42 % de la
   capacidad física del corral. Es el número correcto: no se puede sembrar más
   denso de lo que permite lo que se instala encima.

---

## Nota sobre el corral de presentación

`csv/corral_incubacion.csv` está en **30 × 20 m**: es el corral con el que se
**presenta** el proyecto, porque a esa escala la colocación se lee con claridad
sin quedar vacía. No es una medida de campo y no pretende serlo — la del
Corral 1 del Santuario es 30 × 8 m y sigue registrada en `csv/corrales.csv`.

A esa escala el corral admite, por área, unos 1 660 nidos de golfina al límite
físico y 600 a la separación de la NOM-162, así que una jornada normal cae
holgada en la banda «conforme a la NOM-162»: con la jornada de 70 golfina,
25 prieta y 5 laúd la separación mínima medida fue de 101.9 cm y no hay
apiñamientos que confundan la lectura del diagrama. La malla y el riego se reescalan al tamaño que se evalúe
(`solar.ajustar_a_corral`), porque el dato de campo es que el toldo cubre el
corral **completo** y esa es la relación que hay que conservar, no los
centímetros.

---

## Nota sobre el alcance

El sistema tuvo una capa de administración de corrales (capacidad instalada,
déficit estacional, recomendación de cuántos nidos alojar). Se retiró para que
el algoritmo genético vuelva a ser el método principal de acomodación de nidos,
que es el propósito del proyecto. El análisis de capacidad sigue disponible en
el historial de `git`.

# Gunicorn configuration file to increase timeout for genetic algorithm execution on Render.
# El CPU de Render es unas 10 veces mas lento que una laptop: 100 nidos tardaron
# 133 s y una jornada grande rebasaba los 240 s de antes, y gunicorn la cortaba.
timeout = 600

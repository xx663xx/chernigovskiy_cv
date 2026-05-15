import pyautogui
import cv2
import numpy as np
import mss
import time

pyautogui.PAUSE = 0  # убираем встроенную задержку между командами

# константы
INPUT_LAG   = 0.06   # задержка захвата экрана и отправки нажатия (сек)
AIR_TIME    = 0.55   # полная длительность прыжка дино (сек)
BLOCK_GAP   = 100    # максимальный пиксельный зазор между кактусами одной группы
DINO_OFFSET = 20     # отступ правее дино, где начинаем сканировать

print("Откройте игру в фуллскрине")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

sct = mss.mss()
mon = sct.monitors[1]
SW, SH = mon['width'], mon['height']

def find_game():

    # находит координаты игры на экране,
    # возвращает (ground_y, dino_end)
    # ground_y  — Y-координата линии земли
    # dino_end  — правый X-край дино (отсюда начинаем сканировать препятствия)

    pyautogui.press('space')  # запускаем игру чтобы дино начал бежать
    time.sleep(1.5)

    screen = np.array(sct.grab(mon))
    gray = cv2.cvtColor(screen, cv2.COLOR_BGRA2GRAY)

    # ищем строку земли - горизонтальную линию с множеством темных пикселей
    # на светлом фоне, растянутую на >15% ширины экрана
    ground = -1
    max_d = 0
    for y in range(50, int(SH * 0.85)):
        row = gray[y]
        dark = np.where(row < 100)[0]
        if len(dark) < 20:
            continue
        if (dark[-1] - dark[0]) > SW * 0.15 and np.sum(row > 200) > SW * 0.3 and len(dark) > max_d:
            max_d = len(dark)
            ground = y

    # ищем правый край дино
    # самый правый X-столбец с темными пикселями
    # в зоне тела дино (от ground-45 до ground-3), в левой половине экрана
    dino = 0
    if ground > 0:
        dark_xs = [x for x in range(SW // 2)
                   if np.sum(gray[ground - 45: ground - 3, x] < 100) > 8]
        if dark_xs:
            dino = dark_xs[0]
            for x in dark_xs[1:]:
                if x - dino > 15:  # разрыв более 15px - уже не дино
                    break
                dino = x
    return ground, dino


ground_y, dino_end = -1, 0
for i in range(3):
    ground_y, dino_end = find_game()
    if ground_y > 0 and dino_end > 0:
        break
    time.sleep(1)
    pyautogui.press('space')
    time.sleep(1)

if ground_y < 0 or dino_end == 0:
    print("Не удалось найти игру")
else:
    # параметры зоны сканирования
    check_x  = dino_end + DINO_OFFSET  # X откуда начинаем смотреть вправо
    scan_top = ground_y - 80           # верхняя граница (для захвата птиц)
    scan_h   = 75                      # высота зоны сканирования

    # base_row — граница нижней зоны внутри скан-полоски.
    # пиксели ниже нее принадлежат кактусам и низким птицам (стоят на земле).
    # высокие птицы в эту зону не попадают — так отличаем кактус от птицы
    base_row = scan_h - 45             # = 30px от низа полоски

    # gstrip — узкая полоска земли чуть правее дино.
    # земля постоянно скроллится и пиксели меняются.
    # если изменений нет больше двух секунд — игра остановилась (game over).
    gstrip_oy = (ground_y - 2) - scan_top  # Y смещение внутри единого grab
    gstrip_ox = 50
    gstrip_w  = 80
    gstrip_h  = 4
    combined_h = max(scan_h, gstrip_oy + gstrip_h)  # высота объединенного grab

    print(f"Земля y={ground_y}, дино до x={dino_end}, скан от x={check_x}")
    print(f"INPUT_LAG={INPUT_LAG} AIR_TIME={AIR_TIME} BLOCK_GAP={BLOCK_GAP}\n")

    t0 = time.time()        # время начала текущего раунда
    air_end = 0             # момент времени когда дино приземлится
    prev_g = None           # предыдущий кадр gstrip для детекции остановки
    last_move = time.time() # момент последнего движения земли
    last_presses = []       # лог последних прыжков (для отладки)

    try:
        while True:
            now = time.time()
            el = now - t0  # время с начала раунда (сек)

            # ширина зоны сканирования растет со временем. чем быстрее игра,
            # тем дальше вперед нужно смотреть чтобы успевать реагировать
            w = int(min(200 + el * 8, 700))
            cw = max(w, gstrip_ox + gstrip_w)  # не меньше чем нужно для gstrip

            # единый grab покрывает и зону препятствий, и gstrip
            # одной операцией вместо двух
            img = np.array(sct.grab({'left': check_x, 'top': scan_top,
                                    'width': cw, 'height': combined_h}))
            g_full = img[:, :, 0]                       # только R-канал (=gray для ч/б игры)
            g = g_full[:scan_h, :w]                     # зона препятствий
            gs = g_full[gstrip_oy:gstrip_oy + gstrip_h,
                        gstrip_ox:gstrip_ox + gstrip_w] # полоска земли

            # определяем день или ночь по верхней строке кадра
            # ночью фон темный, препятствия светлые. инвертируем маску
            night = int(np.median(g[0])) < 100
            mask = (g > 150) if night else (g < 100)

            # детектор game over
            # считаем изменения в полоске земли
            if prev_g is not None and prev_g.shape == gs.shape:
                if int(np.sum(np.abs(gs.astype(int) - prev_g.astype(int)))) > 100:
                    last_move = now
            prev_g = gs.copy()

            # столбцы, где есть хоть один препятственный пиксель
            cols = np.where(np.any(mask, axis=0))[0]

            if len(cols) > 0 and el > 1.0 and now > air_end:
                # модель скорости игры
                # около 360px/сек в начале, +3.6px/сек каждую секунду
                S = 360 + 3.6 * el

                # расстояние на котором надо начать прыжок
                # (время прыжка + inputlag) * скорость — отступ от дино
                react = int(min((0.278 + INPUT_LAG) * S - DINO_OFFSET, 250))

                # максимальная суммарная ширина группы которую дино перепрыгнет
                # за один прыжок (0.35 * S = примерно половина времени в воздухе * скорость)
                max_safe = int(0.35 * S)

                # группируем столбцы в отдельные препятствия по зазору BLOCK_GAP.
                # кактусы стоящие близко объединяются в одну группу, то есть
                # прыгаем один раз через всю группу, а не через каждый кактус
                splits = np.where(np.diff(cols) > 15)[0] + 1
                groups = np.split(cols, splits)

                bl, br = int(groups[0][0]), int(groups[0][-1]) # границы первой группы
                merged = 1
                for grp in groups[1:]:
                    gl, gr = int(grp[0]), int(grp[-1])
                    # присоединяем следующую группу если она близко и итоговая
                    # ширина не превысит то что дино способен перепрыгнуть
                    if gl - br > BLOCK_GAP or (gr - bl + 1) > max_safe:
                        break
                    br = gr
                    merged += 1

                # прыгаем когда центр группы вошел в зону реакции
                if (bl + br) // 2 < react:
                    # проверяем что в нижней части зоны есть пиксели.
                    # высокая птица в base не попадет, а кактус — попадет
                    base = mask[base_row:, bl: min(br + 5, g.shape[1])]
                    if int(np.sum(base)) >= 30:
                        pyautogui.press('space')
                        air_end = now + AIR_TIME + INPUT_LAG
                        last_presses.append((el, bl, br, merged, int(S)))
                        if len(last_presses) > 6:
                            last_presses.pop(0)

            # game over
            # земля не двигается более двух секунд
            if now - last_move > 2.0 and el > 3.0:
                print(f"Game over (раунд {el:.0f}с)")
                for p in last_presses:
                    print(f"  el={p[0]:.1f}с [{p[1]}-{p[2]}] W={p[2]-p[1]+1} "
                          f"merged={p[3]} S={p[4]}")
                last_presses = []
                time.sleep(0.5)
                for i in range(3):
                    pyautogui.press('space')
                    time.sleep(0.2)
                time.sleep(0.5)
                t0 = time.time()
                last_move = time.time()
                air_end = 0
                prev_g = None

    except KeyboardInterrupt:
        print("Стоп")  
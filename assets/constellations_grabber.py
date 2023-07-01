from collections import OrderedDict
import json
import math
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent, KeyEvent


def grab(img_fn):
    names = ['aries',
             'taurus',
             'gemini',
             'cancer',
             'leo',
             'virgo',
             'libra',
             'scorpio',
             'sagittarius',
             'capricorn',
             'aquarius',
             'pisces']
    names = ['ophiuchus']
    name_index = 0
    points = OrderedDict()
    mag = 5

    constellations = plt.imread(img_fn)
    fig, ax = plt.subplots()
    ax.imshow(constellations)

    def onclick(event: MouseEvent):
        points[names[name_index]].append({'x': event.xdata, 'y': event.ydata, 'mag': mag})
        ax.set_title(f'constellation: {names[name_index]}, mag: {mag}, #stars: {len(points[names[name_index]])}')
        fig.canvas.draw()
    
    def onkey(event: KeyEvent):
        nonlocal mag, name_index
        if event.key == ' ':
            name_index += 1
            if name_index >= len(names):
                plt.close(fig)
                return
            points[names[name_index]] = []
            ax.set_title(f'constellation: {names[name_index]}, mag: {mag}, #stars: {len(points[names[name_index]])}')
            fig.canvas.draw()
            return
        try:
            val = int(event.key)
            mag = val
            ax.set_title(f'constellation: {names[name_index]}, mag: {mag}, #stars: {len(points[names[name_index]])}')
            fig.canvas.draw()
        except:
            pass

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_release_event', onkey)

    points[names[name_index]] = []
    ax.set_title(f'constellation: {names[name_index]}, mag: {mag}, #stars: {len(points[names[name_index]])}')
    plt.show()

    return points


def main():
    img_fn = sys.argv[1]
    raw_js = sys.argv[2]
    normalized_js = sys.argv[3]

    if not os.path.exists(raw_js):
        points = grab(img_fn)
        with open(raw_js, 'w') as f:
            json.dump(points, f)
    else:
        with open(raw_js) as f:
            points = json.load(f)

    for v in points.values():
        minx = min([p['x'] for p in v])
        maxx = max([p['x'] for p in v])
        miny = min([p['y'] for p in v])
        maxy = max([p['y'] for p in v])
        diag = math.sqrt(2) * max(maxx - minx, maxy - miny)
        for p in v:
            p['x'] = (p['x'] - (maxx + minx) / 2) / diag + .5
            p['y'] = (p['y'] - (maxy + miny) / 2) / diag + .5

    fig, axs = plt.subplots(3, 4)
    for i, (k, v) in enumerate(points.items()):
        ax = axs[i // 4, i % 4]
        ax.set_aspect(1)
        ax.add_artist(plt.Circle((.5, .5), .5, fill=False))
        ax.set_title(k)
        ax.scatter([p['x'] for p in v], [p['y'] for p in v], s=[p['mag'] for p in v])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    plt.show()

    with open(normalized_js, 'w') as f:
            json.dump(points, f, indent='  ')


if __name__ == '__main__':
    main()

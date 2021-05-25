import drawSvg as svg


class Drawing:
    def __init__(self, *args, **kwargs):
        self.d = svg.Drawing(*args, **kwargs)

    def lines(self, *args, **kwargs):
        self.d.append(svg.Lines(*args, **kwargs))

    def text(self, *args, **kwargs):
        self.d.append(svg.Text(*args, **kwargs))

    def save(self, filename):
        self.d.saveSvg(filename)

    def rect(self, x, y, wid, hei, **kwargs):
        self.lines(x, y, x+wid, y, x+wid, y+hei, x, y+hei, close=True, **kwargs)

    def textBox(self, x, y, wid, hei, label, **kwargs):
        self.rect(x, y, wid, hei, **kwargs, fill="transparent", stroke="black")
        textSz = min(wid, hei) / 2
        self.text(label, textSz, x + wid/2, y + hei/2, center=0.5, **kwargs, fill="black")

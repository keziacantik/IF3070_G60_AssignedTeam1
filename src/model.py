import random
from dataclasses import dataclass

@dataclass(frozen=True)
class Vehicle:
    id: str
    w: int
    l: int
    fee: int
    weight: int

@dataclass(frozen=True)
class Ship:
    w: int
    l: int
    max_capacity: int

HORIZONTAL, VERTICAL = 0, 1

class Problem:
    def __init__(self, ship, vehicles):
        self.ship = ship
        self.vehicles = list(vehicles)
        self.n = len(self.vehicles)

    def footprint(self, i, o):
        """Ukuran (lebar_x, tinggi_y) kendaraan ke-i untuk orientasi o."""
        v = self.vehicles[i]
        if o == HORIZONTAL:
            return (v.l, v.w)
        return (v.w, v.l)

def _overlap(a, b):
    """a dan b adalah kotak (x_kiri, y_atas, x_kanan, y_bawah)."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

class State:
    def __init__(self, problem, pos):
        self.problem = problem
        self.pos = tuple(pos)

    @property
    def value(self):
        total = 0
        for i, p in enumerate(self.pos):
            if p is not None:
                total += self.problem.vehicles[i].fee
        return total

    @property
    def weight(self):
        total = 0
        for i, p in enumerate(self.pos):
            if p is not None:
                total += self.problem.vehicles[i].weight
        return total

    def rect(self, i):
        p = self.pos[i]
        if p is None:
            return None
        fw, fh = self.problem.footprint(i, p[2])
        return (p[0], p[1], p[0] + fw, p[1] + fh)

    def is_valid(self):
        prob = self.problem
        if self.weight > prob.ship.max_capacity:
            return False
        placed = []
        for i in range(prob.n):
            r = self.rect(i)
            if r is None:
                continue
            if r[0] < 0 or r[1] < 0 or r[2] > prob.ship.w or r[3] > prob.ship.l:
                return False
            placed.append(r)
        for a in range(len(placed)):
            for b in range(a + 1, len(placed)):
                if _overlap(placed[a], placed[b]):
                    return False
        return True
def random_state(problem, rng):
    W, H = problem.ship.w, problem.ship.l
    pos = [None] * problem.n
    order = list(range(problem.n))
    rng.shuffle(order)
    rects = []
    total_weight = 0
    for i in order:
        if total_weight + problem.vehicles[i].weight > problem.ship.max_capacity:
            continue
        for _ in range(30):
            o = rng.randrange(2)
            fw, fh = problem.footprint(i, o)
            if fw > W or fh > H:
                continue
            x = rng.randint(0, W - fw)
            y = rng.randint(0, H - fh)
            r = (x, y, x + fw, y + fh)
            if all(not _overlap(r, q) for q in rects):
                pos[i] = (x, y, o)
                rects.append(r)
                total_weight += problem.vehicles[i].weight
                break
    return State(problem, pos)

def random_neighbor(state, rng):
    p = state.problem
    W, H = p.ship.w, p.ship.l
    for _ in range(100):
        kind = rng.choice(["move", "rotate", "swap"])
        i = rng.randrange(p.n)
        pos = list(state.pos)
        if kind == "rotate":
            if pos[i] is None:
                continue
            x, y, o = pos[i]
            pos[i] = (x, y, 1 - o)
        elif kind == "move":
            if rng.random() < 0.15:
                pos[i] = None
            else:
                o = rng.randrange(2)
                fw, fh = p.footprint(i, o)
                if fw > W or fh > H:
                    continue
                pos[i] = (rng.randint(0, W - fw), rng.randint(0, H - fh), o)
        else:
            j = rng.randrange(p.n)
            if j == i:
                continue
            pos[i], pos[j] = pos[j], pos[i]
        kandidat = State(p, pos)
        if kandidat.pos != state.pos and kandidat.is_valid():
            return kandidat
    return None

v1 = Vehicle(id="V01", w=2, l=4, fee=30, weight=8)
v2 = Vehicle(id="V02", w=3, l=5, fee=60, weight=15)
v3 = Vehicle(id="V03", w=1, l=2, fee=10, weight=3)
kapal = Ship(w=10, l=20, max_capacity=100)
masalah = Problem(kapal, [v1, v2, v3])

print(State(masalah, [(0, 0, 0), None, (5, 5, 1)]).is_valid())  # susunan wajar
print(State(masalah, [(0, 0, 0), None, (1, 0, 0)]).is_valid())  # bertabrakan
print(State(masalah, [(8, 0, 0), None, None]).is_valid())       # keluar geladak

kapal_kecil = Ship(w=10, l=20, max_capacity=10)
masalah2 = Problem(kapal_kecil, [v1, v2, v3])
print(State(masalah2, [(0, 0, 0), None, (5, 5, 1)]).is_valid()) # kelebihan berat

for seed in range(1, 6):
    s0 = random_state(masalah, random.Random(seed))
    print(seed, s0.pos, s0.value, s0.is_valid())

s = random_state(masalah, random.Random(1))
rng = random.Random(7)
for _ in range(5):
    t = random_neighbor(s, rng)
    print(t.pos, t.value, t.is_valid())
    s = t

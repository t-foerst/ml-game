# Fensterkonfiguration
WIN_W, WIN_H = 1280, 720
FPS          = 60

# Spielparameter
MAX_HP        = 3
MINIMAP_SIZE  = 160
MINIMAP_SCALE = 1 / 20    # 1 Minimap-px = 20 Welteinheiten
INDICATOR_DIST = 90       # px vom Spieler-Mittelpunkt zu Feind-Indikatoren

# Bekannte Server
SERVERS = [
    ("Lokal",        "ws://localhost:3001/ws"),
    ("Oeffentlich",  "ws://game.foerst.haus/ws"),
]

# Farbpalette (schwarz-weiß)
BG      = (17,  17,  17)
GRID    = (28,  28,  28)
ORIGIN  = (40,  40,  40)
WHITE   = (230, 230, 230)
GRAY    = (120, 120, 120)
DARK    = (65,  65,  65)
DIMGRAY = (40,  40,  40)
HP_HIGH = (190, 190, 190)
HP_MID  = (110, 110, 110)
HP_LOW  = (65,  65,  65)

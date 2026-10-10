"""The open-vocabulary detector's label list: the scene datasets' own mpcat40 categories, fixed before any run and
identical for every method (the detector is used off the shelf, so this list is its only setting)."""
VOCABULARY = ["chair", "table", "sofa", "bed", "cabinet", "plant", "picture",
              "tv_monitor", "chest of drawers", "dresser", "desk", "counter",
              "shelf", "kitchen counter", "bathtub", "oven", "stool", "bench",
              "cushion", "curtain", "mirror", "towel", "sink", "refrigerator",
              "toilet", "stairs", "railing", "door", "window", "lamp",
              "clothes", "blinds", "board", "fireplace", "seating", "column",
              "furniture", "appliances", "shower", "bathroom cabinet"]

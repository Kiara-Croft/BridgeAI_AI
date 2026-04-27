"""
translations.py
Complete Romanian DB values -> English display mapping.
Usage: from translations import translate
"""

RO_TO_EN = {

    # Yes / No / N/A  
    "yes":                          "Yes",
    "no":                           "No",
    "Da":                           "Yes",
    "Nu":                           "No",
    "-1":                           "N/A",
    "nan":                          "N/A",
    "None":                         "N/A",
    "N/A":                          "N/A",

    # Headphone type (coloana 'type')  
    "Externe":                      "Over-ear",
    "Interne":                      "In-ear",
    "Omnidirectional":              "Omnidirectional",
    "Li-Ion":                       "Wireless",
    "Li-Polimer":                   "Wireless",
    "Lithium-ion":                  "Wireless",
    "Lithium polymer":              "Wireless",
    "Reincarcabil":                 "Wireless",

    # Headphone subtype  
    "Office/Call Center":           "Office / Call Center",
    "Open-Ear":                     "Open-ear",
    "Over-Ear":                     "Over-ear",
    "On-Ear":                       "On-ear",
    "In-Ear":                       "In-ear",

    # ANC  
    "Activ":                        "Active",
    "Adaptiv":                      "Adaptive",
    "Hybrid":                       "Hybrid",
    "Pasiv":                        "Passive",
    "Da, bidirectional":            "Yes, bidirectional",

    # Sound system  
    "Stereo":                       "Stereo",
    "Mono":                         "Mono",

    # Connectivity technology  
    "Cu fir":                       "Wired",
    "Cu fir, Wireless":             "Wired & Wireless",
    "wireless":                     "Wireless",
    "Wired":                        "Wired",
    "Wired, Bluetooth":             "Wired & Bluetooth",
    "bluetooth":                    "Bluetooth",

    # TV resolution  
    "4":                            "4K Ultra HD",
    "8":                            "8K",
    "HD Ready":                     "HD Ready",
    "Full HD":                      "Full HD",
    "4K Ultra HD":                  "4K Ultra HD",
    "8K":                           "8K",

    # TV display technology  
    "Direct (Full Array LED)":      "Full Array LED",
    "Direct LED":                   "Direct LED",
    "LED Backlit":                  "LED Backlit",
    "Mini LED":                     "Mini LED",
    "Neo QLED":                     "Neo QLED",
    "NeoQLED":                      "Neo QLED",
    "OLED":                         "OLED",
    "QD-MiniLED":                   "QD-Mini LED",
    "QLED":                         "QLED",
    "QLED Mini LED":                "QLED Mini LED",
    "QNED":                         "QNED",
    "QNED MiniLED":                 "QNED Mini LED",

    # TV OS  
    "Android Google TV":            "Android Google TV",
    "Google TV":                    "Google TV",
    "Linux":                        "Linux",
    "Titan":                        "Titan OS",
    "Tizen":                        "Tizen",
    "VIDAA":                        "VIDAA",
    "webOS":                        "webOS",

    # Monitor panel  
    "IPS":                          "IPS",
    "VA":                           "VA",
    "TN":                           "TN",
    "QD-Mini LED":                  "QD-Mini LED",
    "QD-OLED":                      "QD-OLED",
    "ADS":                          "ADS",
    "E-Paper":                      "E-Paper",
    "Mini-LED":                     "Mini-LED",

    # Monitor use case  
    "Grafica":                      "Graphics / Design",
    "Multimedia":                   "Multimedia",
    "Office":                       "Office",
    "Videoconferinta":              "Video Conferencing",
    "Digital Signage":              "Digital Signage",
    "Gaming":                       "Gaming",

    # Laptop GPU  
    "Dedicata":                     "Dedicated GPU",
    "Integrata":                    "Integrated GPU",

    # Laptop OS  
    "Fara sistem de operare":       "No OS",
    "Free DOS":                     "FreeDOS",
    "FreeDOS":                      "FreeDOS",
    "macOS":                        "macOS",
    "macOS Sonoma":                 "macOS Sonoma",

    # Laptop category  
    "Business":                     "Business",
    "Multimedia":                   "Multimedia",
    "Ultraportabil":                "Ultraportable",
    "Workstation":                  "Workstation",

    # Laptop battery 
    "Li-Polymer":                   "Li-Polymer",
    "Li-ion":                       "Li-Ion",

    # Smartphone form factor  
    "Candybar":                     "Candybar",
    "Flip":                         "Flip",
    "Foldable":                     "Foldable",

    #  Smartphone OS  
    "Android":                      "Android",
    "Android OS":                   "Android",
    "EMUI":                         "EMUI (Android)",
    "HarmonyOS":                    "HarmonyOS",
    "iOS":                          "iOS",

    #  Screen tech  
    "AMOLED":                       "AMOLED",
    "LTPO AMOLED":                  "LTPO AMOLED",
    "LTPO OLED":                    "LTPO OLED",
    "LTPO P-OLED":                  "LTPO P-OLED",
    "Super AMOLED":                 "Super AMOLED",
    "Super AMOLED+":                "Super AMOLED+",
    "P-OLED":                       "P-OLED",
    "IPS LCD":                      "IPS LCD",
    "PLS LCD":                      "PLS LCD",
    "TFT":                          "TFT",
    "LCD":                          "LCD",
    "HD+":                          "HD+",
    "LTPS":                         "LTPS",

    # Fingerprint  
    "Da, lateral":                  "Yes, side-mounted",
    "Da, spate":                    "Yes, rear",
    "Da, sub display":              "Yes, under-display",
    "Da (lateral)":                 "Yes, side-mounted",
    "Da (sub display)":             "Yes, under-display",

    # GPS  
    "Da, cu suport A-GPS":                                  "A-GPS",
    "Da, cu suport A-GPS, GALILEO":                         "A-GPS, Galileo",
    "Da, cu suport A-GPS, GLONASS":                         "A-GPS, GLONASS",
    "Da, cu suport A-GPS, GLONASS, BeiDou":                 "A-GPS, GLONASS, BeiDou",
    "Da, cu suport A-GPS, GLONASS, GALILEO":                "A-GPS, GLONASS, Galileo",
    "Da, cu suport A-GPS, GLONASS, GALILEO, BEIDOU":        "A-GPS, GLONASS, Galileo, BeiDou",
    "Da, cu suport A-GPS, GLONASS, GALILEO, BeiDou":        "A-GPS, GLONASS, Galileo, BeiDou",
    "Da, cu suport A-GPS, GLONASS, GALILEO, BeiDou, NavIC": "A-GPS, GLONASS, Galileo, BeiDou, NavIC",
    "Da, cu suport A-GPS, GLONASS, GALILEO, BeiDou, QZSS":  "A-GPS, GLONASS, Galileo, BeiDou, QZSS",
    "Da, cu suport A-GPS, GLONASS, GALILEO, NaviC, QZSS":   "A-GPS, GLONASS, Galileo, NaviC, QZSS",
    "Da, cu suport GPS, GLONASS, BeiDou, Galileo":          "GPS, GLONASS, BeiDou, Galileo",
    "Da, cu suport GPS, GLONASS, GALILEO":                  "GPS, GLONASS, Galileo",

    # Wireless charging  
    "Da, integrat":                 "Yes, built-in",
    "Wireless charging":            "Wireless charging",

    # Tablet / Smartphone processor  
    "ARM":                          "ARM",
    "Allwinner":                    "Allwinner",
    "Exynos":                       "Samsung Exynos",
    "MediaTek":                     "MediaTek",
    "Rockchip":                     "Rockchip",
    "Unisoc":                       "Unisoc",

    # Materials  
    "Aluminiu":                     "Aluminium",
    "Aluminium":                    "Aluminium",
    "Plastic":                      "Plastic",

    # Misc  
    "Specificații disponibile în descriere": "Specs available in description",
    "ore":                          "h",
    "Dual Standby":                 "Dual Standby",
    "Nano-SIM si eSIM":             "Nano-SIM & eSIM",
    "Capacitive":                   "Capacitive touchscreen",
    "Dual Pixel PDAF":              "Dual Pixel PDAF",
    "LTE":                          "4G LTE",
    "Stereo Nicam":                 "Stereo (Nicam)",
}


def translate(value) -> str:
    """
    Translates a Romanian DB value to English for display.
    Returns the original string unchanged if no translation exists.

    Usage:
        from translations import translate
        translate("Externe")   # -> "Over-ear"
        translate("yes")       # -> "Yes"
        translate("-1")        # -> "N/A"
        translate("Activ")     # -> "Active"
    """
    if value is None:
        return "N/A"
    s = str(value).strip()
    return RO_TO_EN.get(s, s)
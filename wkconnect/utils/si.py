def convert_si_units(val: int, unit_in: str, unit_out: str) -> int:
    prefix_to_exponent = {
        "yotta": 24,
        "zetta": 21,
        "exa": 18,
        "peta": 15,
        "tera": 12,
        "giga": 9,
        "mega": 6,
        "kilo": 3,
        "hecto": 2,
        "deca": 1,
        "deci": -1,
        "centi": -2,
        "milli": -3,
        "micro": -6,
        "nano": -9,
        "pico": -12,
        "femto": -15,
        "atto": -18,
        "zepto": -21,
        "yocto": -24,
    }

    def get_exponent(unit: str) -> int:
        possible_exponent = [
            exponent
            for prefix, exponent in prefix_to_exponent.items()
            if unit.startswith(prefix)
        ]
        if len(possible_exponent) == 0:
            return 0
        return possible_exponent[0]

    return val * 10 ** (get_exponent(unit_in) - get_exponent(unit_out))

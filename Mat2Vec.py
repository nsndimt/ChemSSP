import ast
import json
import os
import re
import string
from collections import defaultdict
from collections.abc import Mapping
from enum import Enum
from typing import Union, Optional, Iterator

import regex
import unidecode
from gensim.models.phrases import Phraser
from monty.fractions import gcd_float

"""
Great thanks to Original pymatgen and mat2vec author
Install the whole pymatgen makes conda environment incompatible
"""

with open(os.path.join(os.path.dirname(__file__), "models/periodic_table.json")) as f:
    _pt_data = json.load(f)

class Element(Enum):
    # This name = value convention is redundant and dumb, but unfortunately is
    # necessary to preserve backwards compatibility with a time when Element is
    # a regular object that is constructed with Element(symbol).
    H = "H"
    He = "He"
    Li = "Li"
    Be = "Be"
    B = "B"
    C = "C"
    N = "N"
    O = "O"
    F = "F"
    Ne = "Ne"
    Na = "Na"
    Mg = "Mg"
    Al = "Al"
    Si = "Si"
    P = "P"
    S = "S"
    Cl = "Cl"
    Ar = "Ar"
    K = "K"
    Ca = "Ca"
    Sc = "Sc"
    Ti = "Ti"
    V = "V"
    Cr = "Cr"
    Mn = "Mn"
    Fe = "Fe"
    Co = "Co"
    Ni = "Ni"
    Cu = "Cu"
    Zn = "Zn"
    Ga = "Ga"
    Ge = "Ge"
    As = "As"
    Se = "Se"
    Br = "Br"
    Kr = "Kr"
    Rb = "Rb"
    Sr = "Sr"
    Y = "Y"
    Zr = "Zr"
    Nb = "Nb"
    Mo = "Mo"
    Tc = "Tc"
    Ru = "Ru"
    Rh = "Rh"
    Pd = "Pd"
    Ag = "Ag"
    Cd = "Cd"
    In = "In"
    Sn = "Sn"
    Sb = "Sb"
    Te = "Te"
    I = "I"
    Xe = "Xe"
    Cs = "Cs"
    Ba = "Ba"
    La = "La"
    Ce = "Ce"
    Pr = "Pr"
    Nd = "Nd"
    Pm = "Pm"
    Sm = "Sm"
    Eu = "Eu"
    Gd = "Gd"
    Tb = "Tb"
    Dy = "Dy"
    Ho = "Ho"
    Er = "Er"
    Tm = "Tm"
    Yb = "Yb"
    Lu = "Lu"
    Hf = "Hf"
    Ta = "Ta"
    W = "W"
    Re = "Re"
    Os = "Os"
    Ir = "Ir"
    Pt = "Pt"
    Au = "Au"
    Hg = "Hg"
    Tl = "Tl"
    Pb = "Pb"
    Bi = "Bi"
    Po = "Po"
    At = "At"
    Rn = "Rn"
    Fr = "Fr"
    Ra = "Ra"
    Ac = "Ac"
    Th = "Th"
    Pa = "Pa"
    U = "U"
    Np = "Np"
    Pu = "Pu"
    Am = "Am"
    Cm = "Cm"
    Bk = "Bk"
    Cf = "Cf"
    Es = "Es"
    Fm = "Fm"
    Md = "Md"
    No = "No"
    Lr = "Lr"
    Rf = "Rf"
    Db = "Db"
    Sg = "Sg"
    Bh = "Bh"
    Hs = "Hs"
    Mt = "Mt"
    Ds = "Ds"
    Rg = "Rg"
    Cn = "Cn"
    Nh = "Nh"
    Fl = "Fl"
    Mc = "Mc"
    Lv = "Lv"
    Ts = "Ts"
    Og = "Og"

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.d = _pt_data[symbol]

    @staticmethod
    def is_valid_symbol(symbol: str) -> bool:
        """
        Returns true if symbol is a valid element symbol.
        Args:
            symbol (str): Element symbol
        Returns:
            True if symbol is a valid element (e.g., "H"). False otherwise
            (e.g., "Zebra").
        """
        return symbol in Element.__members__

    @staticmethod
    def from_Z(Z: int):
        """
        Get an element from an atomic number.
        Args:
            Z (int): Atomic number
        Returns:
            Element with atomic number Z.
        """
        for sym, data in _pt_data.items():
            if data["Atomic no"] == Z:
                return Element(sym)
        raise ValueError(f"No element with this atomic number {Z}")


class Species:
    supported_properties = ("spin",)

    def __init__(
            self,
            symbol: str,
            oxidation_state: Optional[float] = 0.0,
            properties: Optional[dict] = None,
    ):
        """
        Initializes a Species.
        Args:
            symbol (str): Element symbol, e.g., Fe
            oxidation_state (float): Oxidation state of element, e.g., 2 or -2
            properties: Properties associated with the Species, e.g.,
                {"spin": 5}. Defaults to None. Properties must be one of the
                Species supported_properties.
        .. attribute:: oxi_state
            Oxidation state associated with Species
        .. attribute:: ionic_radius
            Ionic radius of Species (with specific oxidation state).
        .. versionchanged:: 2.6.7
            Properties are now checked when comparing two Species for equality.
        """
        self._el = Element(symbol)
        self._oxi_state = oxidation_state
        self._properties = properties or {}
        for k, _ in self._properties.items():
            if k not in Species.supported_properties:
                raise ValueError(f"{k} is not a supported property")

    @staticmethod
    def from_string(species_string: str):
        """
        Returns a Species from a string representation.
        Args:
            species_string (str): A typical string representation of a
                species, e.g., "Mn2+", "Fe3+", "O2-".
        Returns:
            A Species object.
        Raises:
            ValueError if species_string cannot be interpreted.
        """

        # e.g. Fe2+,spin=5
        # 1st group: ([A-Z][a-z]*)    --> Fe
        # 2nd group: ([0-9.]*)        --> "2"
        # 3rd group: ([+\-])          --> +
        # 4th group: (.*)             --> everything else, ",spin=5"

        m = re.search(r"([A-Z][a-z]*)([0-9.]*)([+\-]*)(.*)", species_string)
        if m:

            # parse symbol
            sym = m.group(1)

            # parse oxidation state (optional)
            if not m.group(2) and not m.group(3):
                oxi = None
            else:
                oxi = 1 if m.group(2) == "" else float(m.group(2))
                oxi = -oxi if m.group(3) == "-" else oxi

            # parse properties (optional)
            properties = None
            if m.group(4):
                toks = m.group(4).replace(",", "").split("=")
                properties = {toks[0]: ast.literal_eval(toks[1])}

            # but we need either an oxidation state or a property
            if oxi is None and properties is None:
                raise ValueError("Invalid Species String")

            return Species(sym, 0 if oxi is None else oxi, properties)
        raise ValueError("Invalid Species String")


class DummySpecies(Species):
    """
    A special specie for representing non-traditional elements or species. For
    example, representation of vacancies (charged or otherwise), or special
    sites, etc.
    .. attribute:: oxi_state
        Oxidation state associated with Species.
    .. attribute:: Z
        DummySpecies is always assigned an atomic number equal to the hash
        number of the symbol. Obviously, it makes no sense whatsoever to use
        the atomic number of a Dummy specie for anything scientific. The purpose
        of this is to ensure that for most use cases, a DummySpecies behaves no
        differently from an Element or Species.
    .. attribute:: X
        DummySpecies is always assigned an electronegativity of 0.
    """

    def __init__(
            self,
            symbol: str = "X",
            oxidation_state: Optional[float] = 0,
            properties: Optional[dict] = None,
    ):
        """
        Args:
            symbol (str): An assigned symbol for the dummy specie. Strict
                rules are applied to the choice of the symbol. The dummy
                symbol cannot have any part of first two letters that will
                constitute an Element symbol. Otherwise, a composition may
                be parsed wrongly. E.g., "X" is fine, but "Vac" is not
                because Vac contains V, a valid Element.
            oxidation_state (float): Oxidation state for dummy specie.
                Defaults to zero.
        """
        # enforce title case to match other elements, reduces confusion
        # when multiple DummySpecies in a "formula" string
        symbol = symbol.title()

        for i in range(1, min(2, len(symbol)) + 1):
            if Element.is_valid_symbol(symbol[:i]):
                raise ValueError(f"{symbol} contains {symbol[:i]}, which is a valid element symbol.")

        # Set required attributes for DummySpecies to function like a Species in
        # most instances.
        self._symbol = symbol
        self._oxi_state = oxidation_state
        self._properties = properties or {}
        for k, _ in self._properties.items():
            if k not in Species.supported_properties:
                raise ValueError(f"{k} is not a supported property")


def get_el_sp(obj) -> Union[Element, Species, DummySpecies]:
    if isinstance(obj, (Element, Species, DummySpecies)):
        return obj

    try:
        c = float(obj)
        i = int(c)
        i = i if i == c else None  # type: ignore
    except (ValueError, TypeError):
        i = None

    if i is not None:
        return Element.from_Z(i)

    try:
        return Species.from_string(obj)
    except (ValueError, KeyError):
        try:
            return Element(obj)
        except (ValueError, KeyError):
            try:
                return DummySpecies.from_string(obj)
            except Exception:
                raise ValueError(f"Can't parse Element or String from type {type(obj)}: {obj}.")


class Composition(Mapping):
    # Tolerance in distinguishing different composition amounts.
    # 1e-8 is fairly tight, but should cut out most floating point arithmetic
    # errors.
    amount_tolerance = 1e-8

    # Special formula handling for peroxides and certain elements. This is so
    # that formula output does not write LiO instead of Li2O2 for example.
    special_formulas = {
        "LiO": "Li2O2",
        "NaO": "Na2O2",
        "KO": "K2O2",
        "HO": "H2O2",
        "CsO": "Cs2O2",
        "RbO": "Rb2O2",
        "O": "O2",
        "N": "N2",
        "F": "F2",
        "Cl": "Cl2",
        "H": "H2",
    }

    oxi_prob = None  # prior probability of oxidation used by oxi_state_guesses

    def __init__(self, text):
        elmap = self._parse_formula(text)  # type: ignore

        elamt = {}
        self._natoms = 0
        for k, v in elmap.items():
            if v < -Composition.amount_tolerance:
                raise ValueError("Amounts in Composition cannot be negative!")
            if abs(v) >= Composition.amount_tolerance:
                elamt[get_el_sp(k)] = v
                self._natoms += abs(v)
        self._data = elamt

    def _parse_formula(self, formula: str) -> dict[str, float]:
        """
        Args:
            formula (str): A string formula, e.g. Fe2O3, Li3Fe2(PO4)3
        Returns:
            Composition with that formula.
        Notes:
            In the case of Metallofullerene formula (e.g. Y3N@C80),
            the @ mark will be dropped and passed to parser.
        """
        # for Metallofullerene like "Y3N@C80"
        formula = formula.replace("@", "")

        def get_sym_dict(form: str, factor: Union[int, float]) -> dict[str, float]:
            sym_dict: dict[str, float] = defaultdict(float)
            for m in re.finditer(r"([A-Z][a-z]*)\s*([-*\.e\d]*)", form):
                el = m.group(1)
                amt = 1.0
                if m.group(2).strip() != "":
                    amt = float(m.group(2))
                sym_dict[el] += amt * factor
                form = form.replace(m.group(), "", 1)
            if form.strip():
                raise ValueError(f"{form} is an invalid formula!")
            return sym_dict

        m = re.search(r"\(([^\(\)]+)\)\s*([\.e\d]*)", formula)
        if m:
            factor = 1.0
            if m.group(2) != "":
                factor = float(m.group(2))
            unit_sym_dict = get_sym_dict(m.group(1), factor)
            expanded_sym = "".join([f"{el}{amt}" for el, amt in unit_sym_dict.items()])
            expanded_formula = formula.replace(m.group(), expanded_sym)
            return self._parse_formula(expanded_formula)
        return get_sym_dict(formula, 1)

    @staticmethod
    def from_string(species_string: str) -> DummySpecies:
        """
        Returns a Dummy from a string representation.
        Args:
            species_string (str): A string representation of a dummy
                species, e.g., "X2+", "X3+".
        Returns:
            A DummySpecies object.
        Raises:
            ValueError if species_string cannot be interpreted.
        """
        m = re.search(r"([A-ZAa-z]*)([0-9.]*)([+\-]*)(.*)", species_string)
        if m:
            sym = m.group(1)
            if m.group(2) == "" and m.group(3) == "":
                oxi = 0.0
            else:
                oxi = 1.0 if m.group(2) == "" else float(m.group(2))
                oxi = -oxi if m.group(3) == "-" else oxi
            properties = None
            if m.group(4):
                toks = m.group(4).split("=")
                properties = {toks[0]: float(toks[1])}
            return DummySpecies(sym, oxi, properties)
        raise ValueError("Invalid DummySpecies String")

    def get_el_amt_dict(self) -> dict[str, float]:
        """
        Returns:
            dict[str, float]: element symbol and (unreduced) amount. E.g.
            {"Fe": 4.0, "O":6.0} or {"Fe3+": 4.0, "O2-":6.0}
        """
        dic: dict[str, float] = defaultdict(float)
        for el, amt in self.items():
            dic[el.symbol] += amt
        return dic

    def __getitem__(self, item: str) -> float:
        try:
            sp = get_el_sp(item)
            return self._data.get(sp, 0)
        except ValueError as ex:
            raise TypeError(f"Invalid key {item}, {type(item)} for Composition\nValueError exception:\n{ex}")

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Union[Species, Element, DummySpecies]]:
        return self._data.__iter__()

    def __contains__(self, item) -> bool:
        try:
            sp = get_el_sp(item)
            return sp in self._data
        except ValueError as ex:
            raise TypeError(f"Invalid key {item}, {type(item)} for Composition\nValueError exception:\n{ex}")

    def __eq__(self, other: object) -> bool:
        """Defines == for Compositions."""
        if not isinstance(other, (Composition, dict)):
            return NotImplemented

        #  elements with amounts < Composition.amount_tolerance don't show up
        #  in the elmap, so checking len enables us to only check one
        #  composition's elements
        if len(self) != len(other):
            return False

        return all(abs(amt - other[el]) <= Composition.amount_tolerance for el, amt in self.items())


class MaterialsTextProcessor:
    """
    Materials Science Text Processing Tools.
    """
    ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
                "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
                "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
                "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
                "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
                "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
                "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "Uue"]

    ELEMENT_NAMES = ["hydrogen", "helium", "lithium", "beryllium", "boron", "carbon", "nitrogen", "oxygen", "fluorine",
                     "neon", "sodium", "magnesium", "aluminium", "silicon", "phosphorus", "sulfur", "chlorine", "argon",
                     "potassium", "calcium", "scandium", "titanium", "vanadium", "chromium", "manganese", "iron",
                     "cobalt", "nickel", "copper", "zinc", "gallium", "germanium", "arsenic", "selenium", "bromine",
                     "krypton", "rubidium", "strontium", "yttrium", "zirconium", "niobium", "molybdenum", "technetium",
                     "ruthenium", "rhodium", "palladium", "silver", "cadmium", "indium", "tin", "antimony", "tellurium",
                     "iodine", "xenon", "cesium", "barium", "lanthanum", "cerium", "praseodymium", "neodymium",
                     "promethium", "samarium", "europium", "gadolinium", "terbium", "dysprosium", "holmium", "erbium",
                     "thulium", "ytterbium", "lutetium", "hafnium", "tantalum", "tungsten", "rhenium", "osmium",
                     "iridium", "platinum", "gold", "mercury", "thallium", "lead", "bismuth", "polonium", "astatine",
                     "radon", "francium", "radium", "actinium", "thorium", "protactinium", "uranium", "neptunium",
                     "plutonium", "americium", "curium", "berkelium", "californium", "einsteinium", "fermium",
                     "mendelevium", "nobelium", "lawrencium", "rutherfordium", "dubnium", "seaborgium", "bohrium",
                     "hassium", "meitnerium", "darmstadtium", "roentgenium", "copernicium", "nihonium", "flerovium",
                     "moscovium", "livermorium", "tennessine", "oganesson", "ununennium"]

    ELEMENTS_AND_NAMES = ELEMENTS + ELEMENT_NAMES + [en.capitalize() for en in ELEMENT_NAMES]
    ELEMENTS_NAMES_UL = ELEMENT_NAMES + [en.capitalize() for en in ELEMENT_NAMES]

    # Elemement with the valence state in parenthesis.
    ELEMENT_VALENCE_IN_PAR = regex.compile(r"^(" + r"|".join(ELEMENTS_AND_NAMES) +
                                           r")(\(([IV|iv]|[Vv]?[Ii]{0,3})\))$")
    ELEMENT_DIRECTION_IN_PAR = regex.compile(r"^(" + r"|".join(ELEMENTS_AND_NAMES) + r")(\(\d\d\d\d?\))")

    # Exactly IV, VI or has 2 consecutive II, or roman in parenthesis: is not a simple formula.
    VALENCE_INFO = regex.compile(r"(II+|^IV$|^VI$|\(IV\)|\(V?I{0,3}\))")

    SPLIT_UNITS = ["K", "h", "V", "wt", "wt.", "MHz", "kHz", "GHz", "Hz", "days", "weeks",
                   "hours", "minutes", "seconds", "T", "MPa", "GPa", "at.", "mol.",
                   "at", "m", "N", "s-1", "vol.", "vol", "eV", "A", "atm", "bar",
                   "kOe", "Oe", "h.", "mWcm−2", "keV", "MeV", "meV", "day", "week", "hour",
                   "minute", "month", "months", "year", "cycles", "years", "fs", "ns",
                   "ps", "rpm", "g", "mg", "mAcm−2", "mA", "mK", "mT", "s-1", "dB",
                   "Ag-1", "mAg-1", "mAg−1", "mAg", "mAh", "mAhg−1", "m-2", "mJ", "kJ",
                   "m2g−1", "THz", "KHz", "kJmol−1", "Torr", "gL-1", "Vcm−1", "mVs−1",
                   "J", "GJ", "mTorr", "bar", "cm2", "mbar", "kbar", "mmol", "mol", "molL−1",
                   "MΩ", "Ω", "kΩ", "mΩ", "mgL−1", "moldm−3", "m2", "m3", "cm-1", "cm",
                   "Scm−1", "Acm−1", "eV−1cm−2", "cm-2", "sccm", "cm−2eV−1", "cm−3eV−1",
                   "kA", "s−1", "emu", "L", "cmHz1", "gmol−1", "kVcm−1", "MPam1",
                   "cm2V−1s−1", "Acm−2", "cm−2s−1", "MV", "ionscm−2", "Jcm−2", "ncm−2",
                   "Jcm−2", "Wcm−2", "GWcm−2", "Acm−2K−2", "gcm−3", "cm3g−1", "mgl−1",
                   "mgml−1", "mgcm−2", "mΩcm", "cm−2s−1", "cm−2", "ions", "moll−1",
                   "nmol", "psi", "mol·L−1", "Jkg−1K−1", "km", "Wm−2", "mass", "mmHg",
                   "mmmin−1", "GeV", "m−2", "m−2s−1", "Kmin−1", "gL−1", "ng", "hr", "w",
                   "mN", "kN", "Mrad", "rad", "arcsec", "Ag−1", "dpa", "cdm−2",
                   "cd", "mcd", "mHz", "m−3", "ppm", "phr", "mL", "ML", "mlmin−1", "MWm−2",
                   "Wm−1K−1", "Wm−1K−1", "kWh", "Wkg−1", "Jm−3", "m-3", "gl−1", "A−1",
                   "Ks−1", "mgdm−3", "mms−1", "ks", "appm", "ºC", "HV", "kDa", "Da", "kG",
                   "kGy", "MGy", "Gy", "mGy", "Gbps", "μB", "μL", "μF", "nF", "pF", "mF",
                   "A", "Å", "A˚", "μgL−1"]

    NR_BASIC = regex.compile(r"^[+-]?\d*\.?\d+\(?\d*\)?+$", regex.DOTALL)
    NR_AND_UNIT = regex.compile(r"^([+-]?\d*\.?\d+\(?\d*\)?+)([\p{script=Latin}|Ω|μ]+.*)", regex.DOTALL)

    PUNCT = list(string.punctuation) + ["\"", "“", "”", "≥", "≤", "×"]

    def __init__(self):
        self.elem_name_dict = {en: es for en, es in zip(self.ELEMENT_NAMES, self.ELEMENTS)}

    def split_token(self, token, split_oxidation=True):
        """Processes a single token, in case it needs to be split up.
        There are 2 cases when the token is split: A number with a common unit, or an
        element with a valence state.
        Args:
            token: The string to be processed.
            split_oxidation: If True, split the oxidation (valence) string. Units are always split.
        Returns:
            A list of strings.
        """
        elem_with_valence = self.ELEMENT_VALENCE_IN_PAR.match(token) if split_oxidation else None
        nr_unit = self.NR_AND_UNIT.match(token)
        if nr_unit is not None and nr_unit.group(2) in self.SPLIT_UNITS:
            # Splitting the unit from number, e.g. "5V" -> ["5", "V"].
            return [nr_unit.group(1), nr_unit.group(2)]
        elif elem_with_valence is not None:
            # Splitting element from it"s valence state, e.g. "Fe(II)" -> ["Fe", "(II)"].
            return [elem_with_valence.group(1), elem_with_valence.group(2)]
        else:
            return [token]

    def process(self, tokens, exclude_punct=False, convert_num=True, normalize_materials=True, remove_accents=True,
                make_phrases=False):
        """Processes a pre-tokenized list of strings or a string.
        Selective lower casing, material normalization, etc.
        Args:
            tokens: A list of strings or a string. If a string is supplied, will use the
                tokenize method first to split it into a list of token strings.
            exclude_punct: Bool flag to exclude all punctuation.
            convert_num: Bool flag to convert numbers (selectively) to <nUm>.
            normalize_materials: Bool flag to normalize all simple material formula.
            remove_accents: Bool flag to remove accents, e.g. Néel -> Neel.
            make_phrases: Bool flag to convert single tokens to common materials science phrases.
        Returns:
            A (processed_tokens, material_list) tuple. processed_tokens is a list of strings,
            whereas material_list is a list of (original_material_string, normalized_material_string)
            tuples.
        """
        processed, mat_list = [], []

        for i, tok in enumerate(tokens):
            if exclude_punct and tok in self.PUNCT:  # Punctuation.
                continue
            elif convert_num and self.is_number(tok):  # Number.
                # Replace all numbers with <nUm>, except if it is a crystal direction (e.g. "(111)").
                try:
                    if tokens[i - 1] == "(" and tokens[i + 1] == ")" \
                            or tokens[i - 1] == "〈" and tokens[i + 1] == "〉":
                        pass
                    else:
                        tok = "<nUm>"
                except IndexError:
                    tok = "<nUm>"
            elif tok in self.ELEMENTS_NAMES_UL:  # Chemical element name.
                # Add as a material mention.
                mat_list.append((tok, self.elem_name_dict[tok.lower()]))
                tok = tok.lower()
            elif self.is_simple_formula(tok):  # Simple chemical formula.
                normalized_formula = self.normalized_formula(tok)
                mat_list.append((tok, normalized_formula))
                if normalize_materials:
                    tok = normalized_formula
            elif (len(tok) == 1 or (len(tok) > 1 and tok[0].isupper() and tok[1:].islower())) \
                    and tok not in self.ELEMENTS and tok not in self.SPLIT_UNITS \
                    and self.ELEMENT_DIRECTION_IN_PAR.match(tok) is None:
                # To lowercase if only first letter is uppercase (chemical elements already covered above).
                tok = tok.lower()

            if remove_accents:
                tok = self.remove_accent(tok)

            processed.append(tok)

        if make_phrases:
            processed = self.make_phrases(processed, reps=2)

        return processed, mat_list

    def make_phrases(self, sentence, reps=2):
        """Generates phrases from a sentence of words.
        Args:
            sentence: A list of tokens (strings).
            reps: How many times to combine the words.
        Returns:
            A list of strings where the strings in the original list are combined
            to form phrases, separated from each other with an underscore "_".
        """
        if not hasattr(self, 'phraser'):
            self.phraser = Phraser.load(os.path.join(os.path.dirname(__file__), "models/phraser.pkl"))

        while reps > 0:
            sentence = self.phraser[sentence]
            reps -= 1
        return sentence

    def is_number(self, s):
        """Determines if the supplied string is number.
        Args:
            s: The input string.
        Returns:
            True if the supplied string is a number (both . and , are acceptable), False otherwise.
        """
        return self.NR_BASIC.match(s.replace(",", "")) is not None

    @staticmethod
    def is_element(txt):
        """Checks if the string is a chemical symbol.
        Args:
            txt: The input string.
        Returns:
            True if the string is a chemical symbol, e.g. Hg, Fe, V, etc. False otherwise.
        """
        try:
            Element(txt)
            return True
        except ValueError:
            return False

    def is_simple_formula(self, text):
        """Determines if the string is a simple chemical formula.
        Excludes some roman numbers, e.g. IV.
        Args:
            text: The input string.
        Returns:
            True if the supplied string a simple formula, e.g. IrMn, LiFePO4, etc. More complex
            formula such as LiFePxO4-x are not considered to be simple formulae.
        """
        if self.VALENCE_INFO.search(text) is not None:
            # 2 consecutive II, IV or VI should not be parsed as formula.
            # Related to valence state, so don"t want to mix with I and V elements.
            return False
        elif any(char.isdigit() or char.islower() for char in text):
            # Aas to contain at least one lowercase letter or at least one number (to ignore abbreviations).
            # Also ignores some materials like BN, but these are few and usually written in the same way,
            # so normalization won"t be crucial.
            try:
                if text in ["O2", "N2", "Cl2", "F2", "H2"]:
                    # Including chemical elements that are diatomic at room temperature and atm pressure,
                    # despite them having only a single element.
                    return True
                composition = Composition(text)
                # Has to contain more than one element, single elements are handled differently.
                if len(composition.keys()) < 2 or any([not self.is_element(key) for key in composition.keys()]):
                    return False
                return True
            except (ValueError, OverflowError):
                return False
        else:
            return False

    @staticmethod
    def get_ordered_integer_formula(el_amt, max_denominator=1000):
        """Converts a mapping of {element: stoichiometric value} to a alphabetically ordered string.
        Given a dictionary of {element : stoichiometric value, ..}, returns a string with
        elements ordered alphabetically and stoichiometric values normalized to smallest common
        integer denominator.
        Args:
            el_amt: {element: stoichiometric value} mapping.
            max_denominator: The maximum common denominator of stoichiometric values to use for
                normalization. Smaller stoichiometric fractions will be converted to the same
                integer stoichiometry.
        Returns:
            A material formula string with elements ordered alphabetically and the stoichiometry
            normalized to the smallest integer fractions.
        """
        g = gcd_float(list(el_amt.values()), 1 / max_denominator)
        d = {k: round(v / g) for k, v in el_amt.items()}
        formula = ""
        for k in sorted(d):
            if d[k] > 1:
                formula += k + str(d[k])
            elif d[k] != 0:
                formula += k
        return formula

    def normalized_formula(self, formula, max_denominator=1000):
        """Normalizes chemical formula to smallest common integer denominator, and orders elements alphabetically.
        Args:
            formula: the string formula.
            max_denominator: highest precision for the denominator (1000 by default).
        Returns:
            A normalized formula string, e.g. Ni0.5Fe0.5 -> FeNi.
        """
        try:
            formula_dict = Composition(formula).get_el_amt_dict()
            return self.get_ordered_integer_formula(formula_dict, max_denominator)
        except ValueError:
            return formula

    @staticmethod
    def remove_accent(txt):
        """Removes accents from a string.
        Args:
            txt: The input string.
        Returns:
            The de-accented string.
        """
        # There is a problem with angstrom sometimes, so ignoring length 1 strings.
        return unidecode.unidecode(txt) if len(txt) > 1 else txt

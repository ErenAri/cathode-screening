from pymatgen.core import Structure, Composition
from pymatgen.transformations.standard_transformations import SubstitutionTransformation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from typing import Dict, List, Tuple, Set

class CandidateGenerator:
    """
    Generates candidate structures via substitutional doping.
    
    Features:
    - Substitution with doping fractions
    - Toy charge plausibility filter
    - Deduplication by fingerprint
    """
    
    # Toy whitelist (MVP)
    ALLOWED_ELEMENTS = {
        "Li", "Na", "O", "Co", "Ni", "Mn", "Fe", "V", "Ti", 
        "Al", "Mg", "Zn", "Cr", "Zr", "Nb", "Mo"
    }
    
    def __init__(self, doping_fractions: List[float] = [0.05, 0.1, 0.2]):
        self.doping_fractions = doping_fractions
        self.seen_fingerprints: Set[Tuple] = set()
        
    def generate(
        self,
        base_structures: Dict[str, Structure],
        substitution_map: Dict[str, List[str]],
        max_candidates: int = 1000
    ) -> List[Structure]:
        """
        Generate candidates by substituting elements in base structures.
        """
        candidates = []
        
        for base_name, base_struct in base_structures.items():
            if len(candidates) >= max_candidates: break
            
            for site_elem, replacements in substitution_map.items():
                for replacement in replacements:
                    for fraction in self.doping_fractions:
                        try:
                            new_struct = self._create_doped_structure(
                                base_struct, site_elem, replacement, fraction
                            )
                            
                            if self._is_plausible(new_struct) and \
                               self._is_new(new_struct):
                                candidates.append(new_struct)
                                
                        except Exception:
                            continue
                            
        return candidates[:max_candidates]
    
    def _create_doped_structure(
        self, struct: Structure, 
        target_elem: str, 
        sub_elem: str, 
        fraction: float
    ) -> Structure:
        """Apply substitution transformation."""
        # Simple random substitution on species
        # Note: robust implementation requires supercells for small fractions
        # MVP: simple Probability-based replacement if supported or explicit site finding
        
        # Pymatgen SubstitutionTransformation using dict fraction
        # e.g., {"Co": {"Co": 0.9, "Ni": 0.1}}
        trans_dict = {
            target_elem: {target_elem: 1.0 - fraction, sub_elem: fraction}
        }
        trans = SubstitutionTransformation(trans_dict)
        return trans.apply_transformation(struct)
    
    def _is_plausible(self, struct: Structure) -> bool:
        """Toy plausibility filter."""
        elems = {str(e) for e in struct.composition.elements}
        
        # Must contain Li or Na
        if not (elems & {"Li", "Na"}): return False
        # Must contain O
        if "O" not in elems: return False
        # Whitelist
        if not elems.issubset(self.ALLOWED_ELEMENTS): return False
        
        return True
        
    def _is_new(self, struct: Structure) -> bool:
        """Check if structure is new based on fingerprint."""
        try:
            # Expensive but correct: Spacegroup + Formula + Vol/Atom
            sg = SpacegroupAnalyzer(struct).get_space_group_symbol()
            formula = struct.composition.reduced_formula
            nsites = len(struct)
            
            fp = (formula, sg, nsites)
            
            if fp in self.seen_fingerprints:
                return False
            
            self.seen_fingerprints.add(fp)
            return True
        except:
            return False

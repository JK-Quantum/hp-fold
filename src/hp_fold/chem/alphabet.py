from Bio.Data.PDBData import protein_letters_3to1

from hp_fold.chem.records import ChainRecord


class UnknownResidueError(ValueError):
    """A residue name outside the 20 standard amino acids."""


def one_letter(resname: str) -> str:
    try:
        return protein_letters_3to1[resname]
    except KeyError:
        raise UnknownResidueError(
            f"Unknown residue name {resname!r} ",
            "(only the 20 standard amino acids are supported)",
        ) from None


def to_sequence(chain: ChainRecord) -> str:
    """One-letter amino-acid sequence of a chain, in seq_id order."""
    letters = []
    for residue in chain.residues:
        try:
            letters.append(one_letter(residue.name))
        except UnknownResidueError as exc:
            where = f"{chain.structure_id}:{chain.chain_id}:{residue.seq_id}"
            raise UnknownResidueError(f"{exc} at {where}") from None
    return "".join(letters)

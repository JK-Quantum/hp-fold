from pathlib import Path

import pytest
from Bio.Data.PDBData import protein_letters_1to3

from hp_fold.chem.records import ChainRecord, ResidueRecord
from hp_fold.experiments.config import ExperimentConfig, RunSpec
from hp_fold.experiments.problem_source import resolve_sequence

UBIQUITIN_START = "MQIFVKTLTGKTITLEVEPS"


def chain_of(aa: str, structure_id="FAKE", chain_id="A") -> ChainRecord:
    residues = tuple(
        ResidueRecord(protein_letters_1to3[letter].upper(), i, (3.8 * i, 0.0, 0.0))
        for i, letter in enumerate(aa, start=1)
    )
    return ChainRecord(structure_id, chain_id, residues)


def cif_config(path="x.cif", length=6, pick=0) -> ExperimentConfig:
    return ExperimentConfig(
        sequence=None,
        penalty=None,
        seed=0,
        output=Path("out"),
        runs=(RunSpec("random", {"n_samples": 1}),),
        cif=Path(path),
        fragment_length=length,
        fragment_pick=pick,
    )


class FakeLoader:
    """Stands in for load_chains.

    Returns fixed chains and remembers the paths asked for.
    """

    def __init__(self, *chains: ChainRecord) -> None:
        self._chains = list(chains)
        self.calls: list[Path] = []

    def __call__(self, path: Path) -> list[ChainRecord]:
        self.calls.append(path)
        return self._chains


def test_a_given_sequence_is_returned_without_touching_any_file():
    config = ExperimentConfig(
        sequence="PHPPH",
        penalty=None,
        seed=0,
        output=Path("out"),
        runs=(RunSpec("random", {}),),
    )
    resolved = resolve_sequence(
        config, loader=FakeLoader()
    )  # loader must not be needed
    assert resolved.sequence == "PHPPH"
    assert "config" in resolved.origin


def test_first_fragment_of_a_cif_is_the_default_pick():
    resolved = resolve_sequence(
        cif_config(), loader=FakeLoader(chain_of(UBIQUITIN_START))
    )
    assert resolved.sequence == "HPHHHP"
    assert "FAKE:A:1-6" in resolved.origin and "MQIFVK" in resolved.origin


def test_pick_selects_later_fragments_in_chain_order():
    loader = FakeLoader(chain_of(UBIQUITIN_START))
    assert resolve_sequence(cif_config(pick=1), loader).sequence == "HHHPPH"
    assert resolve_sequence(cif_config(pick=2), loader).sequence == "HHPPHP"


def test_pick_counts_across_chains_in_order():
    first = chain_of("MQIFVK", chain_id="A")
    second = chain_of("MQIFVK", chain_id="B")
    resolved = resolve_sequence(cif_config(pick=1), FakeLoader(first, second))
    assert "FAKE:B:1-6" in resolved.origin


def test_the_configured_length_is_used():
    resolved = resolve_sequence(
        cif_config(length=8),
        FakeLoader(chain_of(UBIQUITIN_START)),
    )
    assert len(resolved.sequence) == 8


def test_the_loader_receives_the_configured_path():
    loader = FakeLoader(chain_of(UBIQUITIN_START))
    resolve_sequence(cif_config(path="some/dir/1UBQ.cif"), loader)
    assert loader.calls == [Path("some/dir/1UBQ.cif")]


def test_pick_out_of_range_names_the_file_and_the_available_count():
    loader = FakeLoader(chain_of("MQIFVK"))
    with pytest.raises(
        ValueError, match=r"pick = 1 is out of range.*x\.cif.*1 candidate"
    ):
        resolve_sequence(cif_config(pick=1), loader)


def test_a_chain_without_hydrophobic_fragments_gives_an_out_of_range_error():
    with pytest.raises(ValueError, match="out of range"):
        resolve_sequence(cif_config(), FakeLoader(chain_of("GGGGGGGGGG")))


def write_ca_only_cif(path, aa):
    """Smallest mmCIF the parser accepts.

    One CA atom per residue, one chain, one model.
    """
    columns = [
        "group_PDB",
        "label_atom_id",
        "label_alt_id",
        "label_comp_id",
        "label_asym_id",
        "label_seq_id",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "pdbx_PDB_model_num",
    ]
    rows = [
        f"ATOM CA . {protein_letters_1to3[letter].upper()} "
        f"A {i} {3.8 * i:.1f} 0.0 0.0 1"
        for i, letter in enumerate(aa, start=1)
    ]
    header = "".join(f"_atom_site.{c}\n" for c in columns)
    path.write_text("data_TEST\nloop_\n" + header + "\n".join(rows) + "\n")


def test_a_real_cif_file_is_read_with_the_default_loader(tmp_path):
    cif = tmp_path / "TINY.cif"
    write_ca_only_cif(cif, UBIQUITIN_START)
    resolved = resolve_sequence(cif_config(path=str(cif)))
    assert resolved.sequence == "HPHHHP"
    assert "TINY:A:1-6" in resolved.origin

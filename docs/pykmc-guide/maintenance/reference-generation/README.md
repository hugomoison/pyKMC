# Reference-generation maintenance tools

**Not part of the beginner workflow.** Each bulk WS walkthrough already provides its reference `.xyz`; load that file directly in OVITO. NiCr uses a different slab visualization workflow and no bulk reference. Do not run these scripts during normal preparation or replace the defective `initial_config.xyz` with a perfect lattice.

These tools preserve how the analysis assets were made. Use them only to rebuild a missing asset or review its construction. They refuse to overwrite an existing output. Run them with the pyKMC Python environment active; `--help` lists arguments.

| Tool | Accepted system and output |
| --- | --- |
| [make_ni_reference.py](make_ni_reference.py) | 2047-atom Ni monovacancy input; creates 2048 ideal FCC sites in a 28.16 Å cube. |
| [make_sia_reference.py](make_sia_reference.py) | 4001-atom Ni SIA trajectory; creates 4000 ideal FCC sites in a 35.2 Å cube after checking its frames. |
| [make_si_reference.py](make_si_reference.py) | 511-atom Si monovacancy input; creates 512 ideal diamond sites in a 21.72 Å cube. |
| [make_cu_reference.py](make_cu_reference.py) | 8788-atom Cu vacancy+SIA input; creates 8788 ideal FCC sites in a 46.93 Å cube (`a = 3.61 Å`). |
| [make_fe_reference.py](make_fe_reference.py) | 6746-atom Fe input with four vacancies; creates 6750 BCC sites in a 42.8295 Å cube (`a = 2.8553 Å`). |

The Ni generators use a 3.52 Å lattice parameter; the Si generator uses 5.43 Å. All require the corresponding example's aligned origin/orientation and intentionally reject other systems. In particular, the SIA generator's 4001-atom check is not suitable for a 4000-atom vacancy+SIA trajectory, even when the same perfect reference can be used after verifying alignment.

Changing cell size, strain, orientation, origin or material requires reviewing the construction and validation, not merely changing the output filename. These checks confirm geometric consistency, not the physical accuracy of a potential or a complete event catalogue.

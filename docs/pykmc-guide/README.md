# pyKMC beginner guide

Kinetic Monte Carlo (KMC) follows a system from one atomic rearrangement to the next, choosing events according to their rates and advancing a simulated clock. In pyKMC, transition searches build the event catalog as the simulation proceeds.

Start with [Installation](installation/README.md), choose one computing platform, then work through [Ni monovacancy](examples/ni-monovacancy/README.md). If pyKMC is already installed and its checks pass, go straight to that first example.

## Reading order

Each chapter contains both local and Trillium instructions, followed by result checks and visualization. Choose one route.

| Chapter | What you will learn | Learning stage / limitations |
| --- | --- | --- |
| [Ni monovacancy](examples/ni-monovacancy/README.md) | Prepare a run and follow a vacancy in OVITO | Start here |
| [Ni FCC self-interstitial](examples/ni-fcc-sia/README.md) | Connect dumbbell motion to barriers and rates | Introductory |
| [Ni vacancy + SIA](examples/ni-monovacancy-sia/README.md) | Identify a Frenkel pair and check for recombination | Structural analysis; timing needs review |
| [Si monovacancy](examples/si-monovacancy/README.md) | Follow vacancy hops in a diamond lattice | Introductory |
| [Cu vacancy + SIA](examples/cu-vacancy-sia/README.md) | Compare defect motion and competing event rates | Introductory; recombination may take longer |
| [Fe vacancy pairs](examples/fe-bivacancy/README.md) | Distinguish basin attempts, exits and fallback | Diagnostic; unresolved basin behavior |
| [NiCr surface slab](examples/nicr-slab/README.md) | Compare species-aware and species-blind event matching | Diagnostic; unresolved basin behavior |

## Before copying commands

Download the [first-edition ZIP](https://github.com/yihangliu-research/pyKMC/archive/refs/heads/docs/beginner-guide-develop.zip), extract it, and open its `docs/` directory. Copy the complete `pykmc-guide` folder from there into your home directory. This download is the contributor's guide branch while the upstream contribution is under review; it does not install or update pyKMC.

For Windows computing, use your Ubuntu home directory. The resulting `~/pykmc-guide` folder should contain `README.md`, `examples/`, `installation/` and `maintenance/` directly, without another nested guide folder. Keep an existing guide folder unchanged until you have compared the new version. This layout makes the guide paths in the commands work as written; local results go separately under `~/pykmc-runs`.

Check the location in your terminal:

```bash
cd ~/pykmc-guide &&
ls README.md examples installation maintenance
```

If you already keep the guide elsewhere, you can keep that location and use your actual path with Tab completion ([path help](paths.md)). The installation chapter separately identifies the installation directory. Keep the complete guide together because its scripts share supporting files.

The usual sequence is **prepare → check input → run → check results → visualize**. The preparation script creates a new run directory and copies the input, atomic configuration and potential together. If a step fails, keep its error output and resolve it before continuing.

## Help and further reading

- [Paths and terminal basics](paths.md): Tab completion, spaces, file locations and optional path variables.
- [Glossary](glossary.md): the terms used in the chapters.
- [Common questions](faq.md): shells, reference files and interrupted runs.
- [Validation records](maintenance/validation.md): tested versions, platforms and outstanding scientific checks.
- [References](references.md): software, methods and potentials.

## Support and references

For pyKMC problems, check the [upstream issue tracker](https://github.com/hugomoison/pyKMC/issues). Include the command, relevant input settings, software version and a short error excerpt; review logs for private information before sharing. For Trillium access, scheduling or storage, use the support contacts in the [official documentation](https://docs.alliancecan.ca/wiki/Trillium). For group-specific choices, use your existing group contact.

This is a first edition. Completed calculations and unresolved diagnostic results are distinguished in the chapters and validation records.

# Architecture

Refactorisation en cours sur un projet séparé.
Ce document résume toutes les décisions d'architecture prises.

---

## Structure des modules

```
simulation/
├── core/               # Solver(ABC), Executor
├── engine/             # Engine(ABC), un fichier par engine
│   ├── base.py         # Engine(ABC)
│   ├── lammps.py       # LammpsEngine
│   └── qe.py           # QEEngine (futur)
├── manager/            # Manager, Worker, Session
├── neighbors/          # NeighborsMethod, solvers
├── environment/        # EnvironmentMethod, solvers
├── search/             # SearchMethod, contexts
└── state/              # Conteneurs purs (dataclasses)
```

---

## Principe central : trois axes orthogonaux

Chaque module de calcul (neighbors, environment, search) est paramétré
par trois axes indépendants :

| Axe | Outil | Exprime |
|-----|-------|---------|
| Engine | `Generic[E]` | Quel moteur est requis (LAMMPS, None, QE...) |
| Algorithme | `Solver(ABC)` | Comment on calcule (KDTree, Verlet, CNA, Graph...) |
| Exécution | `Executor` | Série ou parallèle via Manager |

---

## core/

### `Solver(ABC, Generic[T_in, T_out])`
Algorithme de calcul pur. Ne connaît ni engine ni executor.
```python
class Solver(ABC, Generic[T_in, T_out]):
    @abstractmethod
    def compute(self, data: T_in) -> T_out: ...
```

### `Executor`
Stratégie d'exécution. Découple série/parallèle de la logique métier.
```python
Executor.serial()              # séquentiel local
Executor.parallel(manager.map) # distribué sur les workers
```
`run(func, items, **shared_kwargs)` — exécute func sur chaque item.

---

## engine/

### `Engine(ABC)`
Objet **passif** — expose des méthodes, pas de message loop.
Le Worker drive l'exécution.

Méthodes du contrat :
- Lifecycle : `start`, `close`, `command`
- Init : `initialize_parameters`, `initialize_system`, `initialize_potential`
- Positions : `get_positions`, `set_positions`
- Energies : `get_total_energy`, `get_potential_energy`
- Minimisation : `minimize`, `minimize_with_results`

**Pas de `search`/`refine` dans Engine** — géré par `SearchMethod`.

### `LammpsEngine(Engine)`
- Vit sur tous les ranks MPI de son communicateur
- Rank 0 retourne les résultats, autres ranks retournent None
- `engine_id` pour les logs LAMMPS
- **Pas de `as_registry()`** — logique métier pure, ignore le Manager

### Extensibilité
```
engine/base.py    → Engine(ABC)
engine/lammps.py  → LammpsEngine
engine/qe.py      → QEEngine (futur)
```

---

## manager/

Infrastructure générique — **aucune dépendance vers engine/, search/, ou tout autre module**.
`Worker` et `Manager` pourraient vivre dans une lib complètement séparée.

### Dépendance optionnelle via `try/except`

`@registrable` vit dans `manager/registry.py` mais est importé
de façon optionnelle dans `engine/base.py` :

```python
# engine/base.py
try:
    from manager.registry import registrable
except ImportError:
    # manager non installé — decorator no-op, Engine fonctionne seul
    def registrable(name=None):
        def decorator(func): return func
        return decorator
```

- **Sans** `manager/` → decorator no-op, `LammpsEngine` fonctionne normalement
- **Avec** `manager/` → méthodes marquées, `ManagerFactory` les lit via `build_registry()`

Pattern identique à `pandas` → `matplotlib`, `sklearn` → `scipy`.

### Pattern : Registry de callables + ManagerFactory

Le découplage repose sur un registre `dict[str, Callable]` construit
par `ManagerFactory` — seul endroit qui connaît le mapping nom → callable.

```
ManagerFactory.build(engines, methods)  ← seul endroit qui branche les deux
    ↓
dict[str, Callable]                     ← closures sur engine (déjà démarré)
    ↓
Worker(registry, queue)                 ← aucune dépendance engine
    ↓
Manager(local_registries, global_registry)  ← aucune dépendance engine
```

Responsabilités :
- `Engine`, `NeighborsMethod`... → logique métier pure, ignorent le Manager
- `ManagerFactory`               → seul endroit qui sait comment brancher les deux
- `Manager`, `Worker`            → infrastructure pure, ignorent tout le reste

### `ManagerFactory`
```python
class ManagerFactory:
    @staticmethod
    def build(
        local_engines:       list[Engine],
        global_engine:       Engine | None = None,
        neighbors_method:    NeighborsMethod | None = None,
        environment_method:  EnvironmentMethod | None = None,
        search_method:       SearchMethod | None = None,
    ) -> Manager:

        def make_registry(engine) -> dict[str, Callable]:
            registry = {
                "minimize":             lambda **kw: engine.minimize_with_results(**kw),
                "get_total_energy":     lambda **kw: engine.get_total_energy(**kw),
                "get_potential_energy": lambda **kw: engine.get_potential_energy(**kw),
                "set_positions":        lambda **kw: engine.set_positions(**kw),
                "get_positions":        lambda **kw: engine.get_positions(**kw),
            }
            if search_method:
                registry["search"] = lambda **kw: search_method.search(engine, **kw)
                registry["refine"] = lambda **kw: search_method.refine(engine, **kw)
            if neighbors_method:
                registry["compute_neighbors"] = lambda **kw: neighbors_method.build(engine, **kw)
            if environment_method:
                registry["compute_environment"] = lambda **kw: environment_method.compute(engine, **kw)
            return registry

        return Manager(
            local_registries=[make_registry(e) for e in local_engines],
            global_registry=make_registry(global_engine) if global_engine else {},
        )
```

### `Job`
```python
@dataclass
class Job:
    op_name: str       # nom de l'opération dans le registre
    kwargs: dict
    future: Future
```
Plus de `func` ni `use_engine` — tout passe par `op_name`.

### `Worker`
```python
class Worker:
    def __init__(self, registry: dict[str, Callable], job_queue): ...

    def _loop(self):
        job = self.job_queue.get()
        func = self.registry[job.op_name]
        result = func(**job.kwargs)
        job.future.set_result(result)
```

### `Manager`
- `local_workers` — partagent une queue (premier dispo prend le job)
- `global_worker` — queue dédiée
- `submit(op_name, **kwargs) → Future`
- `submit_global(op_name, **kwargs) → Future`
- `map(op_name, items, **shared_kwargs) → list[Future]`

### `session/`
Transport MPI séparé de la logique engine.
- `Session(ABC)`, `Channel(ABC)` — base.py
- `MpiSession`, `MpiChannel`, `RequestResponseProtocol` — mpi.py

---

## neighbors/

### `NeighborsList` (state/)
Dataclass résultat pur — aucune logique de calcul.

### `NeighborsMethod(ABC, Generic[E])`
```python
def build(self, engine: E, system: System, config) -> NeighborsList: ...
```

### `NeighborsSolver(Solver[NeighborsSolverInput, NeighborsList])`
ABC pour les algorithmes Python :
- `KDTreeSolver`
- `VerletSolver` (C++)
- `CellListSolver` (C++)

### Implémentations
```python
# Python — choisir solver + executor
PythonNeighborsMethod(solver=KDTreeSolver(), executor=Executor.serial())
PythonNeighborsMethod(solver=VerletSolver(), executor=Executor.parallel(mgr.map))

# LAMMPS — solver/executor gérés en interne par LAMMPS/MPI
LammpsNeighborsMethod()
```

---

## environment/

### `EnvironmentMethod(ABC, Generic[E])`
```python
def compute(self, engine: E, neighbors: NeighborsList, system: System) -> list: ...
def get_atoms_with_id(self, id: str | bytes) -> list[int]: ...
```

### `EnvironmentSolver(Solver[EnvironmentSolverInput, str | bytes])`
ABC pour les algorithmes Python :
- `CNASolver`   → retourne `str`   ('fcc', 'bcc', 'noncrystal'...)
- `GraphSolver` → retourne `bytes` (hash du graphe topologique)

### Implémentations
```python
# CNA Python
PythonCnaMethod(executor=Executor.serial())
PythonCnaMethod(executor=Executor.parallel(mgr.map))

# CNA LAMMPS
LammpsCnaMethod()

# Graph (toujours Python — pas d'implémentation LAMMPS)
PythonGraphMethod(executor=Executor.parallel(mgr.submit_global))

# CNA+Graph composés — CNA LAMMPS, Graph parallèle sur global comm
CnaGraphMethod(
    cna_method=LammpsCnaMethod(),
    graph_method=PythonGraphMethod(executor=Executor.parallel(mgr.submit_global)),
    neighbors_add=1,
)
```

### `CnaGraphMethod`
Composition explicite remplaçant l'ancien style `"cna/graph"`.
1. Calcule CNA sur tous les atomes
2. Identifie les non-cristallins
3. Étend optionnellement aux N shells de voisins (`neighbors_add`)
4. Calcule Graph uniquement sur ce sous-ensemble
5. Merge les résultats

---

## search/

### `SearchMethod(ABC, Generic[E])`
```python
def search(self, engine: E, context: SearchContext,
           central_atom_idx: int, positions, cell, atom_types) -> Result: ...
def refine(self, engine: E, context: RefineContext,
           central_atom_idx: int, ...) -> Result: ...
```

**Séparation engine/méthode** : `method.search(engine, ...)` et non
`engine.search(...)`. Permet de mixer librement engine et méthode.

### Contexts
```python
SearchContext(ABC)     # paramètres de search
RefineContext(ABC)     # paramètres de refine

PartnSearchContext(SearchContext)   # paramètres pARTn search
PartnRefineContext(RefineContext)   # paramètres pARTn refine
```

### Implémentations
```python
# pARTn — requiert LammpsEngine explicitement
PartnMethod(SearchMethod[LammpsEngine])
    # method_lammps.py — code pARTn + LAMMPS
    # method_qe.py     — futur

# Dimmer (futur)
DimmerMethod(SearchMethod[Engine])  # fonctionne sur tout engine
```

### Usage
```python
partn = PartnMethod()

# search distribué sur atomes
futures = manager.map(
    lambda engine, item, **kw: partn.search(engine, kw['ctx'], item, ...),
    items=list_of_atoms,
    use_engine=True,
    ctx=search_ctx,
)

# refine sur global engine
future = manager.submit_global(
    lambda engine, **kw: partn.refine(engine, kw['ctx'], ...),
    use_engine=True,
    ctx=refine_ctx,
)
```

---

## state/

Conteneurs purs — **aucune logique de calcul, aucune dépendance externe**.

```python
System           # positions, types, cell, pbc — logique ASE conservée
NeighborsList    # dataclass résultat neighbors
AtomicEnvironment # dataclass résultat environment (à simplifier)
State            # agrège System + NeighborsList + AtomicEnvironment
StateFactory     # construit un State depuis system + methods + engine
```

### `State`
Lazy recalcul — invalide neighbors/environment quand les positions changent.
```python
state.update_positions(new_positions)
# → neighbors et environment invalidés, recalculés à la prochaine demande
state.neighbors    # recalcule si invalidé
state.environment  # recalcule si invalidé
```

---

## Dépendances (sens unique strict)

```
state/            →  (rien)
core/             →  (rien)
engine/base       →  (rien)
engine/lammps     →  engine/base
manager/          →  (rien)
neighbors/base    →  state/ + core/
neighbors/python  →  neighbors/base
neighbors/lammps  →  neighbors/base + engine/lammps
environment/base  →  state/ + core/
environment/cna   →  environment/base
environment/graph →  environment/base
environment/lammps→  environment/base + engine/lammps
search/base       →  engine/base
search/partn      →  search/base + engine/lammps
```

---

## Ordre de développement recommandé

1. `state/`       — dataclasses, System existant
2. `core/`        — Solver, Executor
3. `engine/base`  — Engine(ABC)
4. `engine/lammps`— LammpsEngine + operations/
5. `manager/`     — Worker, Manager, Session
6. `neighbors/`   — solvers puis methods
7. `environment/` — solvers puis methods, CnaGraph en dernier
8. `search/`      — PartnMethod en dernier (le plus complexe)
9. `state/state.py` — State + StateFactory (branche tout ensemble)
